import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

@dataclass
class GSPOConfig:
    """Configuration for GSPO training"""
    # Core GSPO parameters
    left_clip_range: float = 3e-4
    right_clip_range: float = 4e-4
    group_size: int = 4  # G in the paper
    
    # Training parameters
    learning_rate: float = 1e-6
    batch_size: int = 8
    mini_batch_size: int = 2  # For gradient accumulation
    max_length: int = 512
    
    # Reward normalization
    reward_normalization: bool = True
    advantage_normalization: bool = True
    
    # Logging
    log_frequency: int = 10
    eval_frequency: int = 100

class GSPOTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        config: GSPOConfig,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Store old model for importance ratio computation
        self.old_model = None
        self.update_old_model()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        # Logging
        self.step = 0
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def update_old_model(self):
        """Update the old model (π_θ_old) for importance ratio computation"""
        if self.old_model is None:
            # Create a copy of the model
            self.old_model = type(self.model)(self.model.config)
            self.old_model.load_state_dict(self.model.state_dict())
        else:
            # Update the old model with current model weights
            self.old_model.load_state_dict(self.model.state_dict())
        
        self.old_model.to(self.device)
        self.old_model.eval()
    
    def compute_sequence_log_prob(
        self, 
        model: torch.nn.Module, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        response_start_idx: int
    ) -> torch.Tensor:
        """Compute log probability of sequence y given x"""
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Only consider the response part (after prompt)
            response_logits = shift_logits[:, response_start_idx:]
            response_labels = shift_labels[:, response_start_idx:]
            
            # Compute log probabilities
            log_probs = F.log_softmax(response_logits, dim=-1)
            
            # Gather log probabilities for actual tokens
            token_log_probs = log_probs.gather(
                dim=-1, 
                index=response_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Sum over sequence length to get sequence log probability
            sequence_log_prob = token_log_probs.sum(dim=-1)
            
        return sequence_log_prob
    
    def compute_importance_ratio(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_start_idx: int,
        response_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute GSPO sequence-level importance ratio s_i(θ)"""
        
        # Compute sequence log probabilities under current and old policies
        current_log_prob = self.compute_sequence_log_prob(
            self.model, input_ids, attention_mask, response_start_idx
        )
        old_log_prob = self.compute_sequence_log_prob(
            self.old_model, input_ids, attention_mask, response_start_idx
        )
        
        # Compute length-normalized importance ratio
        # s_i(θ) = (π_θ(y_i|x) / π_θ_old(y_i|x))^(1/|y_i|)
        log_ratio = (current_log_prob - old_log_prob) / response_lengths.float()
        importance_ratio = torch.exp(log_ratio)
        
        return importance_ratio
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute group-based advantages as in GSPO"""
        batch_size = rewards.size(0)
        group_size = self.config.group_size
        
        # Reshape to group format
        grouped_rewards = rewards.view(-1, group_size)
        
        # Compute group statistics
        group_means = grouped_rewards.mean(dim=1, keepdim=True)
        group_stds = grouped_rewards.std(dim=1, keepdim=True) + 1e-8
        
        # Compute advantages: (r - mean) / std
        advantages = (grouped_rewards - group_means) / group_stds
        
        # Reshape back to original format
        advantages = advantages.view(batch_size)
        
        return advantages
    
    def gspo_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
        response_start_idx: int,
        response_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute GSPO loss"""
        
        # Compute importance ratios
        importance_ratios = self.compute_importance_ratio(
            input_ids, attention_mask, response_start_idx, response_lengths
        )
        
        # Compute advantages
        advantages = self.compute_advantages(rewards)
        
        # Apply clipping to importance ratios
        clipped_ratios = torch.clamp(
            importance_ratios,
            1 - self.config.left_clip_range,
            1 + self.config.right_clip_range
        )
        
        # Compute GSPO objective terms
        unclipped_objective = importance_ratios * advantages
        clipped_objective = clipped_ratios * advantages
        
        # Take minimum (pessimistic bound)
        objective = torch.min(unclipped_objective, clipped_objective)
        
        # GSPO loss is negative of objective (since we minimize)
        loss = -objective.mean()
        
        # Compute statistics for logging
        clip_fraction = (importance_ratios != clipped_ratios).float().mean()
        
        stats = {
            "loss": loss.item(),
            "clip_fraction": clip_fraction.item(),
            "importance_ratio_mean": importance_ratios.mean().item(),
            "importance_ratio_std": importance_ratios.std().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            "reward_mean": rewards.mean().item()
        }
        
        return loss, stats
    
    def generate_responses(
        self, 
        prompts: List[str], 
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True
    ) -> List[str]:
        """Generate responses for given prompts"""
        self.model.eval()
        
        responses = []
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length - max_new_tokens
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the new tokens (response)
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.size(1):],
                skip_special_tokens=True
            )
            responses.append(response)
        
        return responses
    
    def train_step(
        self,
        queries: List[str],
        reward_function: callable
    ) -> Dict[str, float]:
        """Single GSPO training step"""
        
        # Generate multiple responses per query (group size)
        all_responses = []
        all_rewards = []
        all_input_data = []
        
        for query in queries:
            # Generate G responses for this query
            responses = self.generate_responses(
                [query] * self.config.group_size,
                max_new_tokens=256
            )
            
            # Compute rewards for responses
            query_rewards = []
            for response in responses:
                reward = reward_function(query, response)
                query_rewards.append(reward)
            
            # Prepare input data for loss computation
            for response in responses:
                full_text = query + response
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length
                ).to(self.device)
                
                response_start_idx = len(self.tokenizer.encode(query, add_special_tokens=False))
                response_length = len(self.tokenizer.encode(response, add_special_tokens=False))
                
                all_input_data.append({
                    'input_ids': inputs.input_ids.squeeze(0),
                    'attention_mask': inputs.attention_mask.squeeze(0),
                    'response_start_idx': response_start_idx,
                    'response_length': response_length
                })
            
            all_responses.extend(responses)
            all_rewards.extend(query_rewards)
        
        # Convert to tensors
        rewards_tensor = torch.tensor(all_rewards, device=self.device, dtype=torch.float)
        
        # Batch the input data
        input_ids = torch.stack([item['input_ids'] for item in all_input_data])
        attention_mask = torch.stack([item['attention_mask'] for item in all_input_data])
        response_lengths = torch.tensor(
            [item['response_length'] for item in all_input_data],
            device=self.device,
            dtype=torch.float
        )
        
        # Assume same response_start_idx for simplicity (can be generalized)
        response_start_idx = all_input_data[0]['response_start_idx']
        
        # Compute loss
        self.model.train()
        loss, stats = self.gspo_loss(
            input_ids, attention_mask, rewards_tensor, 
            response_start_idx, response_lengths
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.step += 1
        
        # Log statistics
        if self.step % self.config.log_frequency == 0:
            self.logger.info(f"Step {self.step}: {stats}")
            # Optionally log to wandb
            # wandb.log(stats, step=self.step)
        
        return stats

def create_math_reward_function():
    """Simple reward function for math problems"""
    def reward_fn(query: str, response: str) -> float:
        # This is a placeholder - implement actual math evaluation
        # Could use a separate verifier model or exact match checking
        
        # Simple heuristic: longer responses that contain numbers get higher rewards
        if any(char.isdigit() for char in response):
            base_reward = 0.6
        else:
            base_reward = 0.3
        
        # Add small random noise for demonstration
        noise = np.random.normal(0, 0.1)
        reward = np.clip(base_reward + noise, 0, 1)
        
        return reward
    
    return reward_fn

def create_code_reward_function():
    """Simple reward function for coding problems"""
    def reward_fn(query: str, response: str) -> float:
        # Placeholder - implement actual code execution/testing
        
        # Simple heuristic: responses with code-like patterns
        code_indicators = ['def ', 'class ', 'import ', 'return ', 'if ', 'for ']
        code_score = sum(1 for indicator in code_indicators if indicator in response)
        
        base_reward = min(code_score * 0.15, 0.8)
        noise = np.random.normal(0, 0.1)
        reward = np.clip(base_reward + noise, 0, 1)
        
        return reward
    
    return reward_fn

# Example usage
if __name__ == "__main__":
    # Configuration
    config = GSPOConfig(
        group_size=4,
        batch_size=8,
        learning_rate=1e-6
    )
    
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-7B"  # Adjust based on your choice
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize trainer
    trainer = GSPOTrainer(model, tokenizer, config)
    
    # Example training data
    math_queries = [
        "Solve: 2x + 5 = 13. What is x?",
        "Calculate the area of a circle with radius 4.",
        "Find the derivative of f(x) = x^2 + 3x - 2.",
    ]
    
    # Create reward function
    reward_fn = create_math_reward_function()
    
    # Training loop
    for epoch in range(10):  # Small number for testing
        print(f"\n=== Epoch {epoch + 1} ===")
        
        stats = trainer.train_step(math_queries, reward_fn)
        
        # Update old model periodically
        if (epoch + 1) % 5 == 0:
            trainer.update_old_model()
            print("Updated old model")
    
    print("Training completed!") 