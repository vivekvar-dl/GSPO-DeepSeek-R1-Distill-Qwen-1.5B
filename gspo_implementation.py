import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

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
        
        # Ensure model is on correct device and in training mode
        self.model.to(self.device)
        self.model.train()
        
        # Ensure model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad_(True)
        
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
        
        # Clear old model from memory first
        if self.old_model is not None:
            del self.old_model
            torch.cuda.empty_cache()
        
        # Create a copy of the model
        self.old_model = type(self.model)(self.model.config)
        self.old_model.load_state_dict(self.model.state_dict())
        
        self.old_model.to(self.device)
        self.old_model.eval()
        
        # Freeze old model parameters to save memory
        for param in self.old_model.parameters():
            param.requires_grad_(False)
    
    def compute_sequence_log_prob(
        self, 
        model: torch.nn.Module, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        response_start_idx: int,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """Compute log probability of sequence y given x"""
        
        def _compute_log_prob():
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
            
            # Apply attention mask to ignore padded tokens
            response_mask = attention_mask[:, response_start_idx+1:response_start_idx+1+token_log_probs.shape[1]]
            if response_mask.shape[1] != token_log_probs.shape[1]:
                # Adjust mask size if needed
                min_len = min(response_mask.shape[1], token_log_probs.shape[1])
                response_mask = response_mask[:, :min_len]
                token_log_probs = token_log_probs[:, :min_len]
            
            sequence_log_prob = (token_log_probs * response_mask).sum(dim=-1)
            return sequence_log_prob
        
        if requires_grad:
            return _compute_log_prob()
        else:
            with torch.no_grad():
                return _compute_log_prob()
    
    def compute_importance_ratio(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_start_idx: int,
        response_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute GSPO sequence-level importance ratio s_i(θ)"""
        
        # Compute sequence log probabilities under current and old policies
        # Current model needs gradients for backprop
        current_log_prob = self.compute_sequence_log_prob(
            self.model, input_ids, attention_mask, response_start_idx, requires_grad=True
        )
        # Old model doesn't need gradients
        old_log_prob = self.compute_sequence_log_prob(
            self.old_model, input_ids, attention_mask, response_start_idx, requires_grad=False
        )
        
        # Compute length-normalized importance ratio
        # s_i(θ) = (π_θ(y_i|x) / π_θ_old(y_i|x))^(1/|y_i|)
        log_ratio = (current_log_prob - old_log_prob) / response_lengths.float()
        
        # Clamp log_ratio to prevent extreme values
        log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
        
        importance_ratio = torch.exp(log_ratio)
        
        # Additional stability check
        importance_ratio = torch.clamp(importance_ratio, min=1e-8, max=1e8)
        
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
        
        # Check for numerical issues
        if torch.isnan(importance_ratios).any() or torch.isinf(importance_ratios).any():
            self.logger.warning("NaN or Inf detected in importance ratios, skipping step")
            # Return dummy loss that won't affect training
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                "loss": 0.0, "clip_fraction": 0.0, "importance_ratio_mean": 1.0,
                "importance_ratio_std": 0.0, "advantage_mean": 0.0, 
                "advantage_std": 0.0, "reward_mean": rewards.mean().item()
            }
        
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
        
        # Check for numerical issues in loss
        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.warning("NaN or Inf detected in loss, skipping step")
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                "loss": 0.0, "clip_fraction": 0.0, "importance_ratio_mean": 1.0,
                "importance_ratio_std": 0.0, "advantage_mean": 0.0, 
                "advantage_std": 0.0, "reward_mean": rewards.mean().item()
            }
        
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
            try:
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
                        temperature=max(temperature, 0.1),  # Prevent temperature from being too low
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,  # Prevent repetition
                        no_repeat_ngram_size=3
                    )
                
                # Decode only the new tokens (response)
                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.size(1):],
                    skip_special_tokens=True
                )
                responses.append(response)
                
            except Exception as e:
                self.logger.warning(f"Generation failed for prompt: {e}")
                # Return a safe fallback response
                responses.append(" The answer is 42.")
        
        return responses
    
    def train_step(
        self,
        queries: List[str],
        reward_function: callable
    ) -> Dict[str, float]:
        """Single GSPO training step with memory optimization"""
        
        # Generate multiple responses per query (group size)
        all_responses = []
        all_rewards = []
        all_input_data = []
        
        for query in queries:
            # Generate G responses for this query
            responses = self.generate_responses(
                [query] * self.config.group_size,
                max_new_tokens=128  # Reduced token length for memory
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
        
        # Pad sequences to same length for batching
        max_length = max(item['input_ids'].size(0) for item in all_input_data)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for item in all_input_data:
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            
            # Pad to max_length
            pad_length = max_length - input_ids.size(0)
            if pad_length > 0:
                input_ids = F.pad(input_ids, (0, pad_length), value=self.tokenizer.pad_token_id)
                attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
            
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
        
        # Stack padded tensors
        input_ids = torch.stack(padded_input_ids)
        attention_mask = torch.stack(padded_attention_mask)
        response_lengths = torch.tensor(
            [item['response_length'] for item in all_input_data],
            device=self.device,
            dtype=torch.float
        )
        
        # Assume same response_start_idx for simplicity (can be generalized)
        response_start_idx = all_input_data[0]['response_start_idx']
        
        # Clear intermediate variables to save memory
        del all_input_data, padded_input_ids, padded_attention_mask
        
        # Compute loss
        self.model.train()
        loss, stats = self.gspo_loss(
            input_ids, attention_mask, rewards_tensor, 
            response_start_idx, response_lengths
        )
        
        # Skip step if loss is problematic
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0.0:
            self.logger.warning("Skipping step due to problematic loss")
            torch.cuda.empty_cache()
            return {
                "loss": 0.0, "clip_fraction": 0.0, "importance_ratio_mean": 1.0,
                "importance_ratio_std": 0.0, "advantage_mean": 0.0, 
                "advantage_std": 0.0, "reward_mean": rewards_tensor.mean().item()
            }
        
        # Backward pass with gradient scaling for stability
        self.optimizer.zero_grad()
        
        # Scale loss to prevent gradient explosion
        scaled_loss = loss / 10.0
        scaled_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        self.step += 1
        
        # Log statistics
        if self.step % self.config.log_frequency == 0:
            self.logger.info(f"Step {self.step}: {stats}")
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log(stats, step=self.step)
        
        # Clear cache to free memory
        torch.cuda.empty_cache()
        
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