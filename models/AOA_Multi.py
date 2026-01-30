import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
from . import BaseModel, register_model
from ..layers.macro_layer.SemanticConv import SemanticAttention
from ..layers.MetapathConv import MetapathConv
from ..utils.utils import extract_metapaths, get_ntypes_from_canonical_etypes


@register_model('AOA_Multi')
class AOA_Multi(BaseModel):
    r"""
    AOA model for multi-label node classification.

    This is a modified version of AOA that supports multi-label classification on heterogeneous graphs.
    The key difference from the original AOA is that this model only returns outputs for the target node type
    instead of all node types, which is required for multi-label classification tasks.

    Based on the paper: Heterogeneous Graph Attention Network <https://arxiv.org/pdf/1903.07293.pdf>

    Key modifications:
    - Automatic target node type detection
    - Only returns output for the target node type (required for multi-label classification)
    - Compatible with multi-label classification datasets (your_dataset_filtered, persona, imdb)

    Parameters
    ------------
    ntype_meta_paths_dict : dict[str, dict[str, list[etype]]]
        Dict from node type to dict from meta path name to meta path.
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden layer dimension.
    out_dim : int
        Output feature dimension (number of classes for classification).
    num_heads : list[int]
        Number of attention heads for each layer.
    dropout : float
        Dropout probability.
    target_ntype : str
        Target node type for classification.
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        """
        Build model from arguments.

        Key feature: Automatically detects target node type from the graph.
        """
                                    
        if hasattr(args, 'category'):
            target_ntype = args.category
        else:
            target_ntype = cls._detect_target_ntype(hg)

                                                                 
        ntypes = set()
        if hasattr(args, 'target_link'):
            ntypes = get_ntypes_from_canonical_etypes(args.target_link)
        elif hasattr(args, 'category'):
            ntypes.add(args.category)
        else:
                                               
            ntypes.add(target_ntype)

        ntype_meta_paths_dict = {}
        for ntype in ntypes:
            ntype_meta_paths_dict[ntype] = {}
            if hasattr(args, 'meta_paths_dict') and args.meta_paths_dict:
                for meta_path_name, meta_path in args.meta_paths_dict.items():
                                                            
                    if meta_path[0][0] == ntype:
                        ntype_meta_paths_dict[ntype][meta_path_name] = meta_path

                                           
        for ntype, meta_paths_dict in ntype_meta_paths_dict.items():
            if len(meta_paths_dict) == 0:
                ntype_meta_paths_dict[ntype] = extract_metapaths(ntype, hg.canonical_etypes)

        return cls(ntype_meta_paths_dict=ntype_meta_paths_dict,
                   in_dim=args.hidden_dim,
                   hidden_dim=args.hidden_dim,
                   out_dim=args.out_dim,
                   num_heads=args.num_heads,
                   dropout=args.dropout,
                   target_ntype=target_ntype)

    def __init__(self, ntype_meta_paths_dict, in_dim, hidden_dim, out_dim, num_heads, dropout, target_ntype):
        """
        Initialize AOA_Multi model.

        Key differences from original AOA:
        - Stores target_ntype
        - Only creates _AOA module for the target node type
        """
        super(AOA_Multi, self).__init__()
        self.out_dim = out_dim
        self.target_ntype = target_ntype
        self.mod_dict = nn.ModuleDict()

                                                   
                                                               
        for ntype, meta_paths_dict in ntype_meta_paths_dict.items():
            if ntype == target_ntype:
                self.mod_dict[ntype] = _AOA(meta_paths_dict, in_dim, hidden_dim, out_dim, num_heads, dropout)

    def forward(self, g, h_dict):
        r"""
        Forward propagation.

        Key difference from original AOA: Only returns output for target node type.

        Parameters
        -----------
        g : DGLHeteroGraph or dict[str, dict[str, DGLBlock]]
            For full batch, it is a heterogeneous graph. For mini batch, it is a dict from node type to dict from
            meta path name to DGLBlock.
        h_dict : dict[str, Tensor] or dict[str, dict[str, dict[str, Tensor]]]
            The input features. For full batch, it is a dict from node type to node features. For mini batch, it is
            a dict from node type to dict from meta path name to dict from node type to node features.

        Returns
        --------
        out_dict : dict[str, Tensor]
            The output features. Only contains the target node type.
        """
        out_dict = {}

                                           
        if self.target_ntype in self.mod_dict:
            aoa = self.mod_dict[self.target_ntype]

            if isinstance(g, dict):
                            
                if self.target_ntype not in g:
                    raise ValueError(f"Target node type '{self.target_ntype}' not found in mini-batch graph")
                _g = g[self.target_ntype]
                _in_h = h_dict[self.target_ntype]
            else:
                            
                _g = g
                _in_h = h_dict

            _out_h = aoa(_g, _in_h)

                                                 
            if self.target_ntype in _out_h:
                out_dict[self.target_ntype] = _out_h[self.target_ntype]

        return out_dict

    @staticmethod
    def _detect_target_ntype(hg):
        """
        Automatically detect which node type has labels.

        This is useful when the category is not explicitly specified in args.

        Parameters
        -----------
        hg : DGLHeteroGraph
            The input heterogeneous graph.

        Returns
        --------
        target_ntype : str
            The node type that has labels.
        """
        for ntype in hg.ntypes:
            if 'label' in hg.nodes[ntype].data:
                return ntype
        raise ValueError("No node type with 'label' found in graph")


class _AOA(nn.Module):
    """
    Internal AOA module for a single node type.

    This class remains unchanged from the original AOA implementation.
    """

    def __init__(self, meta_paths_dict, in_dim, hidden_dim, out_dim, num_heads, dropout):
        super(_AOA, self).__init__()
        self.layers = nn.ModuleList()

                     
        self.layers.append(AOALayer(meta_paths_dict, in_dim, hidden_dim, num_heads[0], dropout))

                          
        for l in range(1, len(num_heads)):
            self.layers.append(AOALayer(meta_paths_dict, hidden_dim * num_heads[l - 1],
                                        hidden_dim, num_heads[l], dropout))

                      
        self.linear = nn.Linear(hidden_dim * num_heads[-1], out_dim)

    def forward(self, g, h_dict):
        """
        Forward propagation through all AOA layers.

        Parameters
        -----------
        g : DGLHeteroGraph or dict[str, DGLBlock]
            The input graph.
        h_dict : dict[str, Tensor] or dict[str, dict[str, Tensor]]
            The input features.

        Returns
        --------
        out_dict : dict[str, Tensor]
            The output features after all layers and linear transformation.
        """
        for gnn in self.layers:
            h_dict = gnn(g, h_dict)

        out_dict = {}
        for ntype, h in h_dict.items():
            out_dict[ntype] = self.linear(h_dict[ntype])

        return out_dict

    def get_emb(self, g, h_dict):
        """
        Get embeddings before the final linear layer.

        Parameters
        -----------
        g : DGLHeteroGraph or dict[str, DGLBlock]
            The input graph.
        h_dict : dict[str, Tensor]
            The input features.

        Returns
        --------
        emb_dict : dict[str, numpy.ndarray]
            The embeddings.
        """
        h = h_dict[self.category]
        for gnn in self.layers:
            h = gnn(g, h)

        return {self.category: h.detach().cpu().numpy()}


class AOALayer(nn.Module):
    """
    AOA layer with meta-path based attention.

    This class remains unchanged from the original AOA implementation.

    Parameters
    ------------
    meta_paths_dict : dict[str, list[etype]]
        Dict from meta path name to meta path.
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    layer_num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.

    Attributes
    ------------
    _cached_graph : dgl.DGLHeteroGraph
        A cached graph.
    _cached_coalesced_graph : dict
        Cached coalesced graph dict generated by dgl.metapath_reachable_graph().
    """

    def __init__(self, meta_paths_dict, in_dim, out_dim, layer_num_heads, dropout):
        super(AOALayer, self).__init__()
        self.meta_paths_dict = meta_paths_dict

                                      
        semantic_attention = SemanticAttention(in_size=out_dim * layer_num_heads)

                                     
        mods = nn.ModuleDict({
            mp: GATConv(in_dim, out_dim, layer_num_heads,
                       dropout, dropout, activation=F.elu,
                       allow_zero_in_degree=True)
            for mp in meta_paths_dict
        })

                               
        self.model = MetapathConv(meta_paths_dict, mods, semantic_attention)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        r"""
        Forward propagation.

        Parameters
        -----------
        g : DGLHeteroGraph or dict[str, DGLBlock]
            For full batch, it is a heterogeneous graph. For mini batch, it is a dict from meta path name to DGLBlock.
        h : dict[str, Tensor] or dict[str, dict[str, Tensor]]
            The input features. For full batch, it is a dict from node type to node features. For mini batch, it is
            a dict from meta path name to dict from node type to node features.

        Returns
        --------
        h : dict[str, Tensor]
            The output features. Dict from node type to node features.
        """
                    
        if isinstance(g, dict):
            h = self.model(g, h)
                    
        else:
                                                            
            if self._cached_graph is None or self._cached_graph is not g:
                self._cached_graph = g
                self._cached_coalesced_graph.clear()
                for mp, mp_value in self.meta_paths_dict.items():
                    self._cached_coalesced_graph[mp] = dgl.metapath_reachable_graph(g, mp_value)
            h = self.model(self._cached_coalesced_graph, h)

        return h
