                      
                             
                                                       

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassAwareEdgeWeighting(nn.Module):
    """
    Class-aware edge-type importance weighting module.

    Key ideas:
    - Learn a separate weight for each (class, edge type, attention head) triple
    - Different classes can assign different importance to edge types
    - Mitigate class imbalance where tail classes are dominated by head-class weights

    Math:
        Original HGT: w_{etype} is a global scalar shared by all classes
        Here: w_{class, etype, head} is class-specific

    Args:
        num_classes: number of classes
        num_etypes: number of edge types
        num_heads: number of attention heads
        init_value: initial weight value (default: 1.0)
        use_softmax: whether to apply softmax across edge-type weights (default: False)
    """

    def __init__(self, num_classes, num_etypes, num_heads, init_value=1.0, use_softmax=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_etypes = num_etypes
        self.num_heads = num_heads
        self.use_softmax = use_softmax

                                                   
                                               
        self.class_edge_weights = nn.Parameter(
            torch.ones(num_classes, num_etypes, num_heads) * init_value
        )

        print(f"[ClassAwareEdgeWeighting] Initialized")
        print(f"  - classes: {num_classes}, edge types: {num_etypes}, heads: {num_heads}")
        print(f"  - parameters: {num_classes * num_etypes * num_heads:,} weights")
        print(f"  - normalization: {'Softmax' if use_softmax else 'None'}")

    def forward(self, etype, target_class, head_idx):
        """
        Return weights based on edge type, target class, and head index.

        Args:
            etype: [E] int tensor, edge type IDs (0..num_etypes-1)
            target_class: [E] int tensor, target class labels (0..num_classes-1, or -1 for non-target nodes)
            head_idx: int, attention head index (0..num_heads-1)

        Returns:
            weights: [E] float tensor, per-edge weights

        Example:
            >>> weighting = ClassAwareEdgeWeighting(num_classes=3, num_etypes=2, num_heads=4)
            >>> etype = torch.tensor([0, 1, 0, 1])  4 edges
            >>> target_class = torch.tensor([0, 0, 2, -1])  first 2 -> class 0, third -> class 2, fourth is non-target
            >>> weights = weighting(etype, target_class, head_idx=0)
            >>> print(weights.shape)  torch.Size([4])
        """
                 
        assert etype.dim() == 1, f"etype must be 1D tensor, got {etype.dim()}D"
        assert target_class.dim() == 1, f"target_class must be 1D tensor, got {target_class.dim()}D"
        assert etype.size(0) == target_class.size(0),\
            f"etype and target_class length mismatch: {etype.size(0)} vs {target_class.size(0)}"
        assert 0 <= head_idx < self.num_heads,\
            f"head_idx out of range: {head_idx} (expected 0..{self.num_heads-1})"

                                
        target_class_min = target_class.min().item()
        target_class_max = target_class.max().item()
        if target_class_max >= self.num_classes:
            print(f"[Error] target_class max {target_class_max} out of range [0, {self.num_classes-1}]")
            print(f"  target_class stats: min={target_class_min}, max={target_class_max}")
            print(f"  num_classes: {self.num_classes}")
                     
            target_class = torch.clamp(target_class, -1, self.num_classes - 1)
            print(f"  Clamped to valid range")

                                                                
                                               
        weights_for_head = self.class_edge_weights[:, :, head_idx]                             

        if self.use_softmax:
                              
            weights_for_head = F.softmax(weights_for_head, dim=1)                             

                                   
                            
        valid_mask = target_class >= 0       

                             
        target_class_safe = torch.where(valid_mask, target_class, torch.zeros_like(target_class))

                                                           
        weights = weights_for_head[target_class_safe, etype]       

                                            
        weights = torch.where(valid_mask, weights, torch.ones_like(weights))

        return weights

    def get_class_weights(self, class_idx, head_idx):
        """
        Get all edge-type weights for a given class and head (for analysis/visualization).

        Args:
            class_idx: class index
            head_idx: attention head index

        Returns:
            weights: [num_etypes] edge-type weights for this class/head
        """
        weights = self.class_edge_weights[class_idx, :, head_idx]
        if self.use_softmax:
            weights = F.softmax(weights, dim=0)
        return weights.detach()

    def print_top_classes_weights(self, top_k=5, head_idx=0):
        """
        Print edge-type weight distributions for the top-K classes (debugging).

        Args:
            top_k: number of classes to show
            head_idx: attention head index
        """
        print(f"\n[ClassAwareEdgeWeighting] Top {top_k} classes, head {head_idx} edge-type weights:")
        print(f"{'Class':<8} | ", end='')
        for etype_id in range(self.num_etypes):
            print(f"EdgeType{etype_id:<6} | ", end='')
        print()
        print("-" * (10 + 12 * self.num_etypes))

        for class_idx in range(min(top_k, self.num_classes)):
            weights = self.get_class_weights(class_idx, head_idx)
            print(f"Class{class_idx:<4} | ", end='')
            for w in weights:
                print(f"{w.item():<10.4f} | ", end='')
            print()


class DynamicEdgeWeighting(nn.Module):
    """
    Dynamic, sample-adaptive edge-type weighting (option B, unused).

    Predict edge-type weights from node features without explicit class labels.
    Useful for inference or when class labels are unavailable.

    Args:
        hidden_dim: node feature dimension
        num_etypes: number of edge types
        num_heads: number of attention heads
    """

    def __init__(self, hidden_dim, num_etypes, num_heads):
        super().__init__()
        self.num_etypes = num_etypes
        self.num_heads = num_heads

                        
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_etypes * num_heads)
        )

        print(f"[DynamicEdgeWeighting] Initialized")
        print(f"  - input dim: {hidden_dim}, edge types: {num_etypes}, heads: {num_heads}")
        print(f"  - parameters: ~{hidden_dim*128 + 128*num_etypes*num_heads:,}")

    def forward(self, node_embedding, etype, head_idx):
        """
        Predict edge-type weights from node features.

        Args:
            node_embedding: [E, hidden_dim] target node embeddings
            etype: [E] edge type IDs
            head_idx: attention head index

        Returns:
            weights: [E] dynamic edge weights
        """
                    
        all_weights = self.weight_predictor(node_embedding)                             
        all_weights = all_weights.view(-1, self.num_heads, self.num_etypes)                              

                                  
        all_weights = F.softmax(all_weights, dim=2)                              

                        
        weights_for_head = all_weights[:, head_idx, :]                   
        weights = weights_for_head.gather(1, etype.unsqueeze(-1)).squeeze(-1)       

        return weights


             
if __name__ == '__main__':
    print("="*60)
    print("Test ClassAwareEdgeWeighting")
    print("="*60)

          
    weighting = ClassAwareEdgeWeighting(
        num_classes=107,
        num_etypes=4,
        num_heads=8,
        use_softmax=False
    )

             
    batch_size = 16
    etype = torch.randint(0, 4, (batch_size,))         
    target_class = torch.randint(0, 107, (batch_size,))          

    print(f"\nInput:")
    print(f"  etype: {etype}")
    print(f"  target_class: {target_class}")

          
    weights = weighting(etype, target_class, head_idx=0)
    print(f"\nOutput weights: {weights}")
    print(f"  Shape: {weights.shape}")

                 
    weighting.print_top_classes_weights(top_k=3, head_idx=0)

    print("\n" + "="*60)
    print("Test passed!")
    print("="*60)
