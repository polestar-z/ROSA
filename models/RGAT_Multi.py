"""
RGAT_Multi: Relational Graph Attention Network for Multi-Label Node Classification

This is a modified version of RGAT specifically designed for multi-label node classification
on heterogeneous graphs. The key modifications are:
1. Only returns logits for the target node type (instead of all node types)
2. Supports multi-label classification via proper output dimensions
3. Automatically detects the target node type from the graph

Author: Based on OpenHGNN RGAT implementation
Date: 2026-01-19
"""

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from . import BaseModel, register_model


@register_model('RGAT_Multi')
class RGAT_Multi(BaseModel):
    """
    RGAT model for multi-label node classification on heterogeneous graphs.

    Key Features:
    - Supports heterogeneous graphs with multiple node and edge types
    - Uses relational graph attention mechanism
    - Designed for multi-label classification (returns only target node logits)
    - Automatically detects target node type

    Parameters:
    -----------
    in_dim : int
        Input feature dimension
    out_dim : int
        Output dimension (number of classes for multi-label classification)
    h_dim : int
        Hidden dimension
    etypes : list
        List of edge types in the heterogeneous graph
    num_heads : int
        Number of attention heads
    num_hidden_layers : int
        Number of hidden layers (default: 1)
    dropout : float
        Dropout rate (default: 0)
    target_ntype : str
        Target node type for classification
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        """
        Build RGAT_Multi model from arguments.

        Automatically detects the target node type if not specified in args.

        Parameters:
        -----------
        args : Namespace
            Arguments containing model hyperparameters
        hg : DGLHeteroGraph
            Heterogeneous graph

        Returns:
        --------
        RGAT_Multi
            Initialized model instance
        """
                                                              
        if hasattr(args, 'category'):
            target_ntype = args.category
        else:
            target_ntype = cls._detect_target_ntype(hg)

        return cls(
            in_dim=args.in_dim,
            out_dim=args.out_dim,
            h_dim=args.hidden_dim,
            etypes=hg.etypes,
            num_heads=args.num_heads,
            num_hidden_layers=args.num_layers - 2,
            dropout=args.dropout,
            target_ntype=target_ntype
        )

    def __init__(self,
                 in_dim,
                 out_dim,
                 h_dim,
                 etypes,
                 num_heads,
                 num_hidden_layers=1,
                 dropout=0,
                 target_ntype='paper'):
        """
        Initialize RGAT_Multi model.

        Parameters:
        -----------
        in_dim : int
            Input feature dimension
        out_dim : int
            Output dimension (number of classes)
        h_dim : int
            Hidden dimension
        etypes : list
            List of edge types
        num_heads : int
            Number of attention heads
        num_hidden_layers : int
            Number of hidden layers
        dropout : float
            Dropout rate
        target_ntype : str
            Target node type for classification
        """
        super(RGAT_Multi, self).__init__()
        self.rel_names = etypes
        self.target_ntype = target_ntype
        self.layers = nn.ModuleList()

                               
        self.layers.append(RGATLayer(
            in_dim, h_dim, num_heads, self.rel_names,
            activation=F.relu, dropout=dropout, last_layer_flag=False
        ))

                       
        for i in range(num_hidden_layers):
            self.layers.append(RGATLayer(
                h_dim * num_heads, h_dim, num_heads, self.rel_names,
                activation=F.relu, dropout=dropout, last_layer_flag=False
            ))

                      
        self.layers.append(RGATLayer(
            h_dim * num_heads, out_dim, num_heads, self.rel_names,
            activation=None, last_layer_flag=True
        ))

    def forward(self, hg, h_dict=None):
        """
        Forward propagation.

        CRITICAL: This method only returns the logits for the target node type,
        not all node types. This is required for multi-label classification.

        Parameters:
        -----------
        hg : DGLHeteroGraph or list of blocks
            Input graph or blocks for mini-batch training
        h_dict : dict
            Node feature dictionary {node_type: features}

        Returns:
        --------
        dict
            Dictionary containing only the target node type logits
            Format: {target_ntype: logits}
        """
        if hasattr(hg, 'ntypes'):
                                 
            for layer in self.layers:
                h_dict = layer(hg, h_dict)
        else:
                                             
            for layer, block in zip(self.layers, hg):
                h_dict = layer(block, h_dict)

                                                             
                                                                               
        target_logits = h_dict[self.target_ntype]

        return {self.target_ntype: target_logits}

    @staticmethod
    def _detect_target_ntype(hg):
        """
        Automatically detect which node type has labels.

        Parameters:
        -----------
        hg : DGLHeteroGraph
            Heterogeneous graph

        Returns:
        --------
        str
            Node type that has 'label' in its data

        Raises:
        -------
        ValueError
            If no node type with labels is found
        """
        for ntype in hg.ntypes:
            if 'label' in hg.nodes[ntype].data:
                return ntype
        raise ValueError("No node type with 'label' found in graph")


class RGATLayer(nn.Module):
    """
    Relational Graph Attention Layer.

    This layer applies graph attention mechanism on heterogeneous graphs
    with multiple relation types.

    Parameters:
    -----------
    in_feat : int
        Input feature dimension
    out_feat : int
        Output feature dimension
    num_heads : int
        Number of attention heads
    rel_names : list
        List of relation (edge) types
    activation : callable, optional
        Activation function (default: None)
    dropout : float
        Dropout rate (default: 0.0)
    last_layer_flag : bool
        Whether this is the last layer (affects head aggregation)
    bias : bool
        Whether to use bias (default: True)
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 num_heads,
                 rel_names,
                 activation=None,
                 dropout=0.0,
                 last_layer_flag=False,
                 bias=True):
        super(RGATLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.last_layer_flag = last_layer_flag

                                                                     
        self.conv = dglnn.HeteroGraphConv({
            rel: dgl.nn.pytorch.GATConv(
                in_feat, out_feat, num_heads=num_heads,
                bias=bias, allow_zero_in_degree=True
            )
            for rel in rel_names
        })

    def forward(self, g, h_dict):
        """
        Forward propagation for RGAT layer.

        Parameters:
        -----------
        g : DGLHeteroGraph
            Input heterogeneous graph
        h_dict : dict
            Node feature dictionary {node_type: features}

        Returns:
        --------
        dict
            Output feature dictionary {node_type: features}
        """
                                               
        h_dict = self.conv(g, h_dict)

        out_put = {}
        for n_type, h in h_dict.items():
            if self.last_layer_flag:
                                                     
                h = h.mean(1)
            else:
                                                            
                h = h.flatten(1)

            out_put[n_type] = h.squeeze()

        return out_put
