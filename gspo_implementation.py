import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional imports for memory optimization
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    bnb = None

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
    group_size: int = 2  # Reduced default for memory
    
    # Training parameters - memory optimized defaults
    learning_rate: float = 1e-6
    batch_size: int = 1  # Very small for memory
    mini_batch_size: int = 1
    max_length: int = 256  # Reduced sequence length
    gradient_accumulation_steps: int = 4  # Accumulate gradients
    
    # Memory optimization
    use_8bit_optimizer: bool = True
    use_gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3  # Aggressive clipping
    
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
        
        # Enable gradient checkpointing if requested
        if config.use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")
        
        # Ensure model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad_(True)
        
        # Store old model for importance ratio computation
        self.old_model = None
        self.update_old_model()
        
        # Use memory-efficient optimizer
        if config.use_8bit_optimizer and BNB_AVAILABLE:
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(), 
                lr=config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0
            )
            print("✓ Using 8-bit AdamW optimizer")
        else:
            # Fallback to regular AdamW with memory optimizations
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=config.learning_rate,
                betas=(0.9, 0.95),  # More stable betas
                eps=1e-8,
                weight_decay=0.0,
                foreach=False  # Disable foreach for memory
            )
            print("✓ Using regular AdamW optimizer")
        
        # Gradient accumulation
        self.accumulation_steps = 0
        
        # Logging - Enable debug for verification
        self.step = 0
        logging.basicConfig(level=logging.DEBUG)  # Enable debug logging for verification
        self.logger = logging.getLogger(__name__)
        self.logger.info("GSPO Trainer initialized with debug logging enabled for verification")
    
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
        """Compute log probability of sequence y given x with robust error handling"""
        
        def _compute_log_prob():
            try:
                # Use autocast for memory efficiency
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    # Ensure we have valid dimensions
                    if logits.size(1) <= 1:
                        self.logger.warning("Sequence too short for log prob computation")
                        return torch.tensor(-10.0, device=self.device, dtype=torch.float32)
                    
                    # Shift logits and labels for next-token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    
                    # Ensure response_start_idx is valid
                    max_response_start = shift_logits.size(1) - 1
                    safe_response_start = min(response_start_idx, max_response_start)
                    
                    if safe_response_start >= shift_logits.size(1):
                        self.logger.warning("Response start index out of bounds")
                        return torch.tensor(-10.0, device=self.device, dtype=torch.float32)
                    
                    # Only consider the response part (after prompt)
                    response_logits = shift_logits[:, safe_response_start:]
                    response_labels = shift_labels[:, safe_response_start:]
                    
                    if response_logits.size(1) == 0:
                        self.logger.warning("Empty response for log prob computation")
                        return torch.tensor(-10.0, device=self.device, dtype=torch.float32)
                    
                    # Compute log probabilities with numerical stability
                    log_probs = F.log_softmax(response_logits, dim=-1)
                    
                    # Check for NaN/Inf in log_probs
                    if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                        self.logger.warning("NaN/Inf in log_probs")
                        return torch.tensor(-10.0, device=self.device, dtype=torch.float32)
                    
                    # Gather log probabilities for actual tokens
                    token_log_probs = log_probs.gather(
                        dim=-1, 
                        index=response_labels.unsqueeze(-1)
                    ).squeeze(-1)
                    
                    # Apply attention mask to ignore padded tokens
                    response_mask_start = safe_response_start + 1
                    response_mask_end = response_mask_start + token_log_probs.shape[1]
                    
                    if response_mask_end <= attention_mask.size(1):
                        response_mask = attention_mask[:, response_mask_start:response_mask_end]
                    else:
                        # Truncate if mask is too short
                        available_mask_len = attention_mask.size(1) - response_mask_start
                        if available_mask_len <= 0:
                            self.logger.warning("No valid response mask")
                            return torch.tensor(-10.0, device=self.device, dtype=torch.float32)
                        
                        response_mask = attention_mask[:, response_mask_start:response_mask_start + available_mask_len]
                        token_log_probs = token_log_probs[:, :available_mask_len]
                    
                    # Ensure mask and token_log_probs have same length
                    min_len = min(response_mask.shape[1], token_log_probs.shape[1])
                    if min_len <= 0:
                        self.logger.warning("No valid tokens for log prob computation")
                        return torch.tensor(-10.0, device=self.device, dtype=torch.float32)
                    
                    response_mask = response_mask[:, :min_len]
                    token_log_probs = token_log_probs[:, :min_len]
                    
                    # Compute sequence log probability
                    masked_log_probs = token_log_probs * response_mask
                    sequence_log_prob = masked_log_probs.sum(dim=-1)
                    
                    # Final stability check
                    sequence_log_prob = torch.clamp(sequence_log_prob, min=-50.0, max=10.0)
                    
                    if torch.isnan(sequence_log_prob).any() or torch.isinf(sequence_log_prob).any():
                        self.logger.warning("NaN/Inf in final sequence log prob")
                        return torch.tensor(-10.0, device=self.device, dtype=torch.float32)
                    
                    return sequence_log_prob
                    
            except Exception as e:
                self.logger.warning(f"Error in log prob computation: {e}")
                return torch.tensor(-10.0, device=self.device, dtype=torch.float32)
        
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
        """Compute GSPO sequence-level importance ratio s_i(θ) with better numerical stability"""
        
        # Compute sequence log probabilities under current and old policies
        # Current model needs gradients for backprop
        current_log_prob = self.compute_sequence_log_prob(
            self.model, input_ids, attention_mask, response_start_idx, requires_grad=True
        )
        # Old model doesn't need gradients
        old_log_prob = self.compute_sequence_log_prob(
            self.old_model, input_ids, attention_mask, response_start_idx, requires_grad=False
        )
        
        # Add small epsilon to prevent division by zero in lengths
        safe_lengths = torch.clamp(response_lengths.float(), min=1.0)
        
        # Compute length-normalized importance ratio
        # s_i(θ) = (π_θ(y_i|x) / π_θ_old(y_i|x))^(1/|y_i|)
        log_ratio = (current_log_prob - old_log_prob) / safe_lengths
        
        # Debug logging for GSPO verification
        if self.step % 10 == 0:  # Log every 10 steps to avoid spam
            self.logger.debug(f"GSPO Debug - Current log prob: {current_log_prob.mean().item():.4f}")
            self.logger.debug(f"GSPO Debug - Old log prob: {old_log_prob.mean().item():.4f}")
            self.logger.debug(f"GSPO Debug - Log ratio before length norm: {(current_log_prob - old_log_prob).mean().item():.4f}")
            self.logger.debug(f"GSPO Debug - Response lengths: {safe_lengths.mean().item():.2f}")
            self.logger.debug(f"GSPO Debug - Length normalized log ratio: {log_ratio.mean().item():.6f}")
        
        # More aggressive clamping to prevent extreme values
        log_ratio = torch.clamp(log_ratio, min=-5.0, max=5.0)
        
        importance_ratio = torch.exp(log_ratio)
        
        # Additional stability check with tighter bounds
        importance_ratio = torch.clamp(importance_ratio, min=0.1, max=10.0)
        
        # Debug logging for importance ratios
        if self.step % 10 == 0:
            ratio_mean = importance_ratio.mean().item()
            ratio_std = importance_ratio.std().item()
            self.logger.debug(f"GSPO Debug - Importance ratio mean: {ratio_mean:.6f}, std: {ratio_std:.6f}")
        
        # Final check for numerical issues
        if torch.isnan(importance_ratio).any() or torch.isinf(importance_ratio).any():
            self.logger.warning("NaN/Inf in importance ratio, using neutral ratios")
            # Return neutral ratios (1.0) to avoid breaking training
            return torch.ones_like(importance_ratio)
        
        return importance_ratio
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute group-based advantages as in GSPO with better numerical stability"""
        batch_size = rewards.size(0)
        group_size = self.config.group_size
        
        # Handle case where batch size is smaller than group size
        if batch_size < group_size:
            # For small batches, use simple normalization with better stability
            if batch_size == 1:
                # Single element - return zero advantage (neutral)
                return torch.zeros_like(rewards)
            
            reward_mean = rewards.mean()
            # Use unbiased std with better numerical stability
            reward_std = torch.sqrt(rewards.var(unbiased=False) + 1e-8)
            advantages = (rewards - reward_mean) / reward_std
            return advantages
        
        # Ensure batch size is divisible by group size
        num_complete_groups = batch_size // group_size
        if num_complete_groups == 0:
            # Fallback to simple normalization
            if batch_size == 1:
                return torch.zeros_like(rewards)
            
            reward_mean = rewards.mean()
            reward_std = torch.sqrt(rewards.var(unbiased=False) + 1e-8)
            advantages = (rewards - reward_mean) / reward_std
            return advantages
        
        # Take only complete groups
        complete_batch_size = num_complete_groups * group_size
        rewards_subset = rewards[:complete_batch_size]
        
        # Reshape to group format
        grouped_rewards = rewards_subset.view(-1, group_size)
        
        # Compute group statistics with better numerical stability
        group_means = grouped_rewards.mean(dim=1, keepdim=True)
        group_vars = grouped_rewards.var(dim=1, unbiased=False, keepdim=True)
        group_stds = torch.sqrt(group_vars + 1e-8)
        
        # Compute advantages: (r - mean) / std
        advantages = (grouped_rewards - group_means) / group_stds
        
        # Reshape back to original format
        advantages = advantages.view(complete_batch_size)
        
        # If we had incomplete groups, handle remaining rewards
        if complete_batch_size < batch_size:
            remaining_rewards = rewards[complete_batch_size:]
            remaining_size = remaining_rewards.size(0)
            
            if remaining_size == 1:
                # Single element - neutral advantage
                remaining_advantages = torch.zeros_like(remaining_rewards)
            else:
                remaining_mean = remaining_rewards.mean()
                remaining_var = remaining_rewards.var(unbiased=False)
                remaining_std = torch.sqrt(remaining_var + 1e-8)
                remaining_advantages = (remaining_rewards - remaining_mean) / remaining_std
            
            # Concatenate
            advantages = torch.cat([advantages, remaining_advantages])
        
        return advantages
    
    def gspo_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
        response_start_idx: int,
        response_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute GSPO loss with debug logging"""
        
        self.logger.debug(f"GSPO Loss Debug - Input shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}, rewards={rewards.shape}, response_lengths={response_lengths.shape}")
        self.logger.debug(f"Response start idx: {response_start_idx}")
        
        # Compute importance ratios
        try:
            importance_ratios = self.compute_importance_ratio(
                input_ids, attention_mask, response_start_idx, response_lengths
            )
            self.logger.debug(f"Importance ratios computed: {importance_ratios}")
        except Exception as e:
            self.logger.warning(f"Error computing importance ratios: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                "loss": 0.0, "clip_fraction": 0.0, "importance_ratio_mean": 1.0,
                "importance_ratio_std": 0.0, "advantage_mean": 0.0, 
                "advantage_std": 0.0, "reward_mean": rewards.mean().item()
            }
        
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
        try:
            advantages = self.compute_advantages(rewards)
            self.logger.debug(f"Advantages computed: {advantages}")
        except Exception as e:
            self.logger.warning(f"Error computing advantages: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                "loss": 0.0, "clip_fraction": 0.0, "importance_ratio_mean": 1.0,
                "importance_ratio_std": 0.0, "advantage_mean": 0.0, 
                "advantage_std": 0.0, "reward_mean": rewards.mean().item()
            }
        
        # Apply clipping to importance ratios
        clipped_ratios = torch.clamp(
            importance_ratios,
            1 - self.config.left_clip_range,
            1 + self.config.right_clip_range
        )
        self.logger.debug(f"Clipped ratios: {clipped_ratios}")
        
        # Debug logging for clipping verification
        if self.step % 10 == 0:
            clipping_occurred = not torch.equal(importance_ratios, clipped_ratios)
            num_clipped = (importance_ratios != clipped_ratios).sum().item()
            self.logger.debug(f"GSPO Clipping Debug - Any clipping occurred: {clipping_occurred}")
            self.logger.debug(f"GSPO Clipping Debug - Number of ratios clipped: {num_clipped}/{importance_ratios.numel()}")
            self.logger.debug(f"GSPO Clipping Debug - Clipping bounds: [{1 - self.config.left_clip_range:.6f}, {1 + self.config.right_clip_range:.6f}]")
        
        # Compute GSPO objective terms
        unclipped_objective = importance_ratios * advantages
        clipped_objective = clipped_ratios * advantages
        
        self.logger.debug(f"Unclipped objective: {unclipped_objective}")
        self.logger.debug(f"Clipped objective: {clipped_objective}")
        
        # Take minimum (pessimistic bound)
        objective = torch.min(unclipped_objective, clipped_objective)
        self.logger.debug(f"Final objective: {objective}")
        
        # Debug logging for objective computation
        if self.step % 10 == 0:
            pessimistic_bound_used = not torch.equal(unclipped_objective, objective)
            self.logger.debug(f"GSPO Objective Debug - Pessimistic bound used: {pessimistic_bound_used}")
            self.logger.debug(f"GSPO Objective Debug - Objective mean: {objective.mean().item():.6f}")
        
        # GSPO loss is negative of objective (since we minimize)
        loss = -objective.mean()
        self.logger.debug(f"Loss before checks: {loss}")
        
        # Check for numerical issues in loss
        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.warning("NaN or Inf detected in loss, skipping step")
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                "loss": 0.0, "clip_fraction": 0.0, "importance_ratio_mean": 1.0,
                "importance_ratio_std": 0.0, "advantage_mean": 0.0, 
                "advantage_std": 0.0, "reward_mean": rewards.mean().item()
            }
        
        # Check if loss is exactly zero (which might indicate a problem)
        if loss.item() == 0.0:
            self.logger.warning("Loss is exactly zero, this might indicate a problem")
        
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
        
        self.logger.debug(f"Final stats: {stats}")
        
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
        """Single GSPO training step with aggressive memory optimization"""
        
        # Run verification on first step and periodically
        if self.step == 0 or self.step % 50 == 0:
            print(f"\n🔍 Running GSPO verification at step {self.step}")
            verification_results = self.verify_gspo_working()
            clipping_results = self.verify_clipping_bounds()
            
            # Log verification results to wandb if available
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "verification/old_model_exists": verification_results.get("old_model_exists", False),
                    "verification/params_different": verification_results.get("params_different", False),
                    "verification/model_requires_grad": verification_results.get("model_requires_grad", False),
                    "verification/clipping_range_width": clipping_results.get("range_width", 0.0)
                }, step=self.step)
        
        # Generate multiple responses to get better batch statistics
        all_responses = []
        all_rewards = []
        all_input_data = []
        
        # Process multiple queries if available, or repeat single query
        effective_queries = queries[:2] if len(queries) >= 2 else queries * 2
        
        for query in effective_queries:
            # Generate responses
            group_size = min(self.config.group_size, 2)
            responses = self.generate_responses(
                [query] * group_size,
                max_new_tokens=64  # Very short responses
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
                    max_length=min(self.config.max_length, 256)
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
        
        # Ensure we have at least 2 samples for stable statistics
        if len(all_rewards) < 2:
            # Duplicate the data to avoid single-element issues
            all_rewards = all_rewards * 2
            all_input_data = all_input_data * 2
            all_responses = all_responses * 2
        
        # Process in micro-batches
        batch_size = len(all_input_data)
        micro_batch_size = 2  # Process 2 samples at a time for better stability
        
        accumulated_loss = 0.0
        accumulated_stats = {}
        valid_micro_batches = 0
        
        # Track importance ratio test for first micro-batch
        first_micro_batch_tested = False
        
        for i in range(0, batch_size, micro_batch_size):
            end_idx = min(i + micro_batch_size, batch_size)
            micro_batch = all_input_data[i:end_idx]
            micro_rewards = all_rewards[i:end_idx]
            
            # Skip if micro-batch is too small
            if len(micro_batch) < 1:
                continue
            
            # Convert to tensors
            rewards_tensor = torch.tensor(micro_rewards, device=self.device, dtype=torch.float)
            
            # Pad sequences to same length for batching
            max_length = max(item['input_ids'].size(0) for item in micro_batch)
            
            padded_input_ids = []
            padded_attention_mask = []
            
            for item in micro_batch:
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
                [item['response_length'] for item in micro_batch],
                device=self.device,
                dtype=torch.float
            )
            
            response_start_idx = micro_batch[0]['response_start_idx']
            
            # Test importance ratio computation on first micro-batch of first few steps
            if not first_micro_batch_tested and self.step < 3:
                print(f"\n🧮 Testing importance ratio computation at step {self.step}")
                test_results = self.test_importance_ratio_computation(
                    input_ids, attention_mask, response_start_idx, response_lengths
                )
                first_micro_batch_tested = True
                
                # Log test results to wandb
                if WANDB_AVAILABLE and wandb.run is not None and "error" not in test_results:
                    wandb.log({
                        "debug/current_log_prob": test_results["current_log_prob"],
                        "debug/old_log_prob": test_results["old_log_prob"],
                        "debug/log_difference": test_results["log_difference"],
                        "debug/importance_ratio": test_results["importance_ratio"],
                        "debug/clipping_occurred": test_results["clipping_occurred"]
                    }, step=self.step)
            
            # Clear intermediate variables
            del micro_batch, padded_input_ids, padded_attention_mask
            
            # Compute loss with autocast
            self.model.train()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss, stats = self.gspo_loss(
                    input_ids, attention_mask, rewards_tensor, 
                    response_start_idx, response_lengths
                )
            
            # Skip step if loss is problematic
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0.0:
                self.logger.warning(f"Skipping micro-batch {i//micro_batch_size} due to problematic loss")
                continue
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            scaled_loss.backward()
            
            accumulated_loss += loss.item()
            valid_micro_batches += 1
            
            if not accumulated_stats:
                accumulated_stats = stats.copy()
            else:
                # Average the stats
                for key in stats:
                    accumulated_stats[key] = (accumulated_stats[key] + stats[key]) / 2
            
            # Clear cache
            torch.cuda.empty_cache()
        
        # Only proceed with optimizer step if we had valid micro-batches
        if valid_micro_batches == 0:
            self.logger.warning("No valid micro-batches processed, skipping optimizer step")
            return {
                "loss": 0.0, "clip_fraction": 0.0, "importance_ratio_mean": 1.0,
                "importance_ratio_std": 0.0, "advantage_mean": 0.0, 
                "advantage_std": 0.0, "reward_mean": np.mean(all_rewards)
            }
        
        # Update accumulation counter
        self.accumulation_steps += 1
        
        # Only step optimizer after accumulation_steps
        if self.accumulation_steps >= self.config.gradient_accumulation_steps:
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.accumulation_steps = 0
            
            self.step += 1
        
        # Log statistics
        if self.step % self.config.log_frequency == 0 and self.step > 0:
            self.logger.info(f"Step {self.step}: {accumulated_stats}")
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log(accumulated_stats, step=self.step)
        
        # Aggressive cache clearing
        torch.cuda.empty_cache()
        
        return accumulated_stats if accumulated_stats else {
            "loss": 0.0, "clip_fraction": 0.0, "importance_ratio_mean": 1.0,
            "importance_ratio_std": 0.0, "advantage_mean": 0.0, 
            "advantage_std": 0.0, "reward_mean": np.mean(all_rewards)
        }

    def verify_gspo_working(self) -> Dict[str, bool]:
        """Debug function to verify GSPO components are working correctly"""
        verification_results = {}
        
        print("\n" + "="*50)
        print("🔍 GSPO VERIFICATION CHECK")
        print("="*50)
        
        # 1. Check if old model exists and is different
        old_model_exists = self.old_model is not None
        verification_results["old_model_exists"] = old_model_exists
        print(f"✓ Old model exists: {old_model_exists}")
        
        if old_model_exists:
            # Check if models are different objects
            models_different_objects = id(self.model) != id(self.old_model)
            verification_results["models_different_objects"] = models_different_objects
            print(f"✓ Models are different objects: {models_different_objects}")
            
            # Check if model parameters are actually different
            try:
                current_param = next(self.model.parameters()).flatten()[:100]
                old_param = next(self.old_model.parameters()).flatten()[:100]
                params_different = not torch.allclose(current_param, old_param, atol=1e-6)
                verification_results["params_different"] = params_different
                print(f"✓ Model parameters different: {params_different}")
                
                if params_different:
                    param_diff = torch.mean(torch.abs(current_param - old_param)).item()
                    print(f"  → Parameter difference magnitude: {param_diff:.8f}")
                else:
                    print("  ⚠️  Parameters are identical - this might indicate a problem!")
                    
            except Exception as e:
                print(f"  ❌ Error checking parameter differences: {e}")
                verification_results["params_different"] = False
        
        # 2. Check GSPO configuration
        print(f"\n📋 GSPO Configuration:")
        print(f"  → Left clip range: {self.config.left_clip_range}")
        print(f"  → Right clip range: {self.config.right_clip_range}")
        print(f"  → Group size: {self.config.group_size}")
        print(f"  → Learning rate: {self.config.learning_rate}")
        
        # 3. Check if gradients are enabled properly
        model_requires_grad = any(p.requires_grad for p in self.model.parameters())
        old_model_requires_grad = any(p.requires_grad for p in self.old_model.parameters()) if old_model_exists else False
        
        verification_results["model_requires_grad"] = model_requires_grad
        verification_results["old_model_frozen"] = not old_model_requires_grad
        
        print(f"\n🎯 Gradient Configuration:")
        print(f"  → Current model requires_grad: {model_requires_grad}")
        print(f"  → Old model frozen (good): {not old_model_requires_grad}")
        
        # 4. Training step verification
        training_step = self.step
        print(f"\n📈 Training Progress:")
        print(f"  → Current training step: {training_step}")
        print(f"  → Accumulation steps: {self.accumulation_steps}/{self.config.gradient_accumulation_steps}")
        
        return verification_results
    
    def test_importance_ratio_computation(self, test_input_ids, test_attention_mask, response_start_idx, response_lengths):
        """Test importance ratio computation with debug output"""
        print("\n" + "="*50)
        print("🧮 TESTING IMPORTANCE RATIO COMPUTATION")
        print("="*50)
        
        try:
            # Compute current model log prob
            current_log_prob = self.compute_sequence_log_prob(
                self.model, test_input_ids, test_attention_mask, response_start_idx, requires_grad=True
            )
            print(f"✓ Current model log prob: {current_log_prob}")
            
            # Compute old model log prob
            old_log_prob = self.compute_sequence_log_prob(
                self.old_model, test_input_ids, test_attention_mask, response_start_idx, requires_grad=False
            )
            print(f"✓ Old model log prob: {old_log_prob}")
            
            # Compute difference
            log_diff = current_log_prob - old_log_prob
            print(f"✓ Log probability difference: {log_diff}")
            
            # Length normalization
            safe_lengths = torch.clamp(response_lengths.float(), min=1.0)
            normalized_log_ratio = log_diff / safe_lengths
            print(f"✓ Length normalized log ratio: {normalized_log_ratio}")
            print(f"✓ Response lengths: {response_lengths}")
            
            # Final importance ratio
            importance_ratio = torch.exp(normalized_log_ratio)
            print(f"✓ Raw importance ratio: {importance_ratio}")
            
            # Clipped importance ratio
            clipped_ratio = torch.clamp(importance_ratio, min=0.1, max=10.0)
            print(f"✓ Clipped importance ratio: {clipped_ratio}")
            
            # Check if clipping occurred
            clipping_occurred = not torch.equal(importance_ratio, clipped_ratio)
            print(f"✓ Clipping occurred: {clipping_occurred}")
            
            return {
                "current_log_prob": current_log_prob.item(),
                "old_log_prob": old_log_prob.item(), 
                "log_difference": log_diff.item(),
                "importance_ratio": importance_ratio.item(),
                "clipping_occurred": clipping_occurred
            }
            
        except Exception as e:
            print(f"❌ Error in importance ratio test: {e}")
            return {"error": str(e)}
    
    def verify_clipping_bounds(self):
        """Verify GSPO clipping bounds are reasonable"""
        print("\n" + "="*50)
        print("✂️ VERIFYING CLIPPING BOUNDS")
        print("="*50)
        
        left_bound = 1 - self.config.left_clip_range  # Should be ~0.9997
        right_bound = 1 + self.config.right_clip_range  # Should be ~1.0004
        
        print(f"✓ Left clipping bound: {left_bound:.6f}")
        print(f"✓ Right clipping bound: {right_bound:.6f}")
        print(f"✓ Clipping range width: {right_bound - left_bound:.6f}")
        
        if self.config.left_clip_range > 0.01 or self.config.right_clip_range > 0.01:
            print("⚠️  Clipping ranges seem large - GSPO uses very conservative clipping!")
        
        if self.config.left_clip_range == self.config.right_clip_range:
            print("⚠️  Left and right clip ranges are equal - this is unusual for GSPO")
        
        return {
            "left_bound": left_bound,
            "right_bound": right_bound,
            "range_width": right_bound - left_bound
        }

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