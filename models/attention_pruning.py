                     
                       
"""
Dynamic attention head pruning module.

Implements dynamic head pruning from section 6.2:
- PrunableMultiHeadAttention: base class for prunable multi-head attention
- HeadImportanceTracker: attention head importance tracker
- Utility functions

Reference: COAAOA_Optimization_Proposals.md line 682-763
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class HeadImportanceTracker(nn.Module):
    """
    Attention head importance tracker.

    Maintains a learnable importance weight per head and sparsifies via L1.

    Math:
        H^(l) = Σ_{i=1}^h s_i · Head_i(H^(l-1))
        where s_i = σ(w_i) is the learnable head importance.

    Training strategy:
        1. Early: train all heads (s_i ≈ 1)
        2. Mid: sparsify s_i via L1 regularization
        3. Late: remove heads with s_i < ε
    """

    def __init__(self, num_heads: int, init_value: float = 1.0):
        """
        Args:
            num_heads: number of attention heads
            init_value: initial weight value (default 1.0, all heads equal)
        """
        super().__init__()
        self.num_heads = num_heads

                          
        self.head_importance = nn.Parameter(
            torch.ones(num_heads) * init_value
        )

    def forward(self, multi_head_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply importance weights to multi-head outputs.

        Args:
            multi_head_output: [N, num_heads, head_dim] multi-head output

        Returns:
            weighted_output: [N, num_heads * head_dim] weighted output
            importance: [num_heads] importance after sigmoid
        """
                                  
        importance = torch.sigmoid(self.head_importance)               

                                                               
        weighted_output = multi_head_output * importance.view(1, -1, 1)

                                          
        weighted_output = weighted_output.reshape(
            weighted_output.size(0), -1
        )

        return weighted_output, importance

    def get_pruning_mask(self, threshold: float = 0.1) -> torch.Tensor:
        """
        Get pruning mask (which heads to keep).

        Args:
            threshold: importance threshold; heads below are pruned

        Returns:
            mask: [num_heads] True means keep, False means prune
        """
        importance = torch.sigmoid(self.head_importance)
        mask = importance > threshold
        return mask

    def prune_heads(self, threshold: float = 0.1) -> Dict[str, any]:
        """
        Prune heads and return pruning info.

        Args:
            threshold: importance threshold

        Returns:
            pruning_info: dict with pruning statistics
        """
        mask = self.get_pruning_mask(threshold)
        importance = torch.sigmoid(self.head_importance)

        num_kept = mask.sum().item()
        num_pruned = (~mask).sum().item()

        pruning_info = {
            'num_heads_original': self.num_heads,
            'num_heads_kept': num_kept,
            'num_heads_pruned': num_pruned,
            'pruning_ratio': num_pruned / self.num_heads,
            'importance_values': importance.detach().cpu().numpy(),
            'mask': mask.detach().cpu().numpy()
        }

        return pruning_info

    def get_l1_loss(self) -> torch.Tensor:
        """
        Compute L1 regularization loss for head importance.

        Returns:
            l1_loss: L1 norm (for sparsification)
        """
        return torch.abs(self.head_importance).sum()


class PrunableMultiHeadAttention(nn.Module):
    """
    Base class for prunable multi-head attention.

    Provides generic head pruning features for COA and AOA attention layers.

    Features:
        - Maintain importance weights per head
        - L1 regularization for sparsification
        - Post-training pruning
        - Pruning statistics
    """

    def __init__(self, num_heads: int, head_dim: int, enable_pruning: bool = True):
        """
        Args:
            num_heads: number of attention heads
            head_dim: per-head dimension
            enable_pruning: whether to enable head pruning
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.enable_pruning = enable_pruning

        if enable_pruning:
            self.head_tracker = HeadImportanceTracker(num_heads)
            print(f"[PrunableMultiHeadAttention] Pruning enabled, initial heads: {num_heads}")
        else:
            self.head_tracker = None
            print(f"[PrunableMultiHeadAttention] Pruning disabled")

    def apply_head_weights(self, multi_head_output: torch.Tensor) -> torch.Tensor:
        """
        Apply importance weights to multi-head outputs (if pruning enabled).

        Args:
            multi_head_output: [N, num_heads, head_dim] multi-head output

        Returns:
            output: [N, num_heads * head_dim] processed output
        """
        if self.enable_pruning and self.head_tracker is not None:
                      
            weighted_output, importance = self.head_tracker(multi_head_output)
            return weighted_output
        else:
                        
            return multi_head_output.reshape(multi_head_output.size(0), -1)

    def get_l1_loss(self) -> torch.Tensor:
        """
        Get L1 regularization loss for head importance.

        Returns:
            l1_loss: L1 loss if pruning enabled, else 0
        """
        if self.enable_pruning and self.head_tracker is not None:
            return self.head_tracker.get_l1_loss()
        else:
            return torch.tensor(0.0)

    def get_pruning_info(self, threshold: float = 0.1) -> Optional[Dict]:
        """
        Get pruning info.

        Args:
            threshold: importance threshold

        Returns:
            pruning_info: pruning stats dict, or None if disabled
        """
        if self.enable_pruning and self.head_tracker is not None:
            return self.head_tracker.prune_heads(threshold)
        else:
            return None

    def get_device(self) -> torch.device:
        """Get the device of the module."""
        try:
            return next(self.parameters()).device
        except StopIteration:
                           
            return torch.device('cpu')


def collect_head_importance_losses(model: nn.Module) -> torch.Tensor:
    """
    Collect L1 losses from all prunable attention layers in a model.

    Args:
        model: PyTorch model

    Returns:
        total_l1_loss: sum of L1 losses for head importance weights
    """
    total_l1_loss = 0.0
    count = 0

    for module in model.modules():
        if isinstance(module, PrunableMultiHeadAttention):
            l1_loss = module.get_l1_loss()
            if l1_loss is not None and torch.is_tensor(l1_loss):
                total_l1_loss += l1_loss
                count += 1

    if count > 0:
        return total_l1_loss
    else:
        return torch.tensor(0.0)


def print_pruning_statistics(model: nn.Module, threshold: float = 0.1) -> None:
    """
    Print pruning statistics for all prunable attention layers in a model.

    Args:
        model: PyTorch model
        threshold: importance threshold
    """
    print("\n" + "=" * 60)
    print("Attention Head Pruning Statistics")
    print("=" * 60)

    total_original = 0
    total_kept = 0
    layer_count = 0

    for name, module in model.named_modules():
        if isinstance(module, PrunableMultiHeadAttention):
            info = module.get_pruning_info(threshold)
            if info is not None:
                layer_count += 1
                total_original += info['num_heads_original']
                total_kept += info['num_heads_kept']

                print(f"\n[{name}]")
                print(f"  original heads: {info['num_heads_original']}")
                print(f"  kept heads: {info['num_heads_kept']}")
                print(f"  pruned heads: {info['num_heads_pruned']}")
                print(f"  pruning ratio: {info['pruning_ratio']*100:.2f}%")
                print(f"  head importance: {info['importance_values']}")

    if layer_count > 0:
        print("\n" + "-" * 60)
        print(f"Overall:")
        print(f"  layers: {layer_count}")
        print(f"  total original heads: {total_original}")
        print(f"  total kept heads: {total_kept}")
        print(f"  total pruned heads: {total_original - total_kept}")
        print(f"  overall pruning ratio: {(total_original - total_kept) / total_original * 100:.2f}%")
    else:
        print("\nWarning: no prunable attention layers found")

    print("=" * 60 + "\n")


def get_pruning_summary(model: nn.Module, threshold: float = 0.1) -> Dict:
    """
    Get a pruning summary for a model.

    Args:
        model: PyTorch model
        threshold: importance threshold

    Returns:
        summary: summary dict
    """
    total_original = 0
    total_kept = 0
    layer_infos = []

    for name, module in model.named_modules():
        if isinstance(module, PrunableMultiHeadAttention):
            info = module.get_pruning_info(threshold)
            if info is not None:
                total_original += info['num_heads_original']
                total_kept += info['num_heads_kept']
                layer_infos.append({
                    'name': name,
                    **info
                })

    summary = {
        'num_layers': len(layer_infos),
        'total_original_heads': total_original,
        'total_kept_heads': total_kept,
        'total_pruned_heads': total_original - total_kept,
        'overall_pruning_ratio': (total_original - total_kept) / total_original if total_original > 0 else 0.0,
        'layer_infos': layer_infos
    }

    return summary
