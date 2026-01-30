                      
                       
                                                                

import math
import dgl
import torch
import torch.nn as nn

from dgl import function as fn
from dgl.nn.pytorch.linear import TypedLinear
from dgl.nn.pytorch.softmax import edge_softmax
from . import BaseModel, register_model
from ..utils import to_hetero_feat


@register_model('COA_Multi')
class COA_Multi(BaseModel):
    r"""
    COA (Heterogeneous Graph Transformer) adapted for multi-label classification

    Key Features:
    1. Supports multi-label classification with FloatTensor labels
    2. Uses BCEWithLogitsLoss for training
    3. Compatible with multiple datasets (your_dataset_filtered, persona, imdb)
    4. Automatically detects target node type

    Differences from original COA:
    - Output layer is specific to target node type only
    - Returns dict with only target node type predictions
    - Configurable target node type via args or auto-detection

    Parameters
    ----------
    in_dim: int
        Input feature dimension
    out_dim: int
        Output dimension (number of classes)
    num_heads: list
        Number of attention heads in each layer
    num_etypes: int
        Number of edge types
    ntypes: list
        List of all node types
    num_layers: int
        Number of COA layers
    dropout: float
        Dropout rate
    use_norm: bool
        Whether to use layer normalization
    target_ntype: str
        Target node type for classification (e.g., 'paper', 'user', 'movie')
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        """
        Build COA_Multi model from arguments

        Args:
            args: Configuration arguments, should contain:
                - hidden_dim: Hidden dimension
                - out_dim: Output dimension (number of classes)
                - num_heads: Number of attention heads
                - num_layers: Number of layers
                - dropout: Dropout rate
                - norm: Whether to use normalization
                - category (optional): Target node type
            hg: DGLHeteroGraph

        Returns:
            COA_Multi instance
        """
                                      
        if hasattr(args, 'category'):
            target_ntype = args.category
        else:
                                                          
            target_ntype = cls._detect_target_ntype(hg)
            print(f"[COA_Multi] Auto-detected target node type: '{target_ntype}'")

        return cls(
            in_dim=args.hidden_dim,
            out_dim=args.out_dim,
            num_heads=args.num_heads,
            num_etypes=len(hg.etypes),
            ntypes=hg.ntypes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_norm=args.norm,
            target_ntype=target_ntype
        )

    @staticmethod
    def _detect_target_ntype(hg):
        """
        Auto-detect target node type by finding which node type has labels

        Args:
            hg: DGLHeteroGraph

        Returns:
            target_ntype: str
        """
        for ntype in hg.ntypes:
            if 'label' in hg.nodes[ntype].data or 'labels' in hg.nodes[ntype].data:
                return ntype

                                       
        print(f"[Warning] No node type with 'label' found, using first node type: {hg.ntypes[0]}")
        return hg.ntypes[0]

    def __init__(self, in_dim, out_dim, num_heads, num_etypes, ntypes,
                 num_layers, dropout=0.2, use_norm=False, target_ntype='paper'):
        """
        Initialize COA_Multi model

        Args:
            in_dim: Input feature dimension
            out_dim: Output dimension (number of classes for multi-label)
            num_heads: Number of attention heads
            num_etypes: Number of edge types
            ntypes: List of all node types
            num_layers: Number of COA layers
            dropout: Dropout rate
            use_norm: Whether to use layer normalization
            target_ntype: Target node type for classification
        """
        super(COA_Multi, self).__init__()
        self.num_layers = num_layers
        self.ntypes = ntypes
        self.target_ntype = target_ntype

        print(f"[COA_Multi] Initializing model with target node type: '{target_ntype}'")
        print(f"[COA_Multi] Output dimension (num_classes): {out_dim}")

                              
        self.coa_layers = nn.ModuleList()

                     
        self.coa_layers.append(
            COAConv(
                in_dim,
                in_dim // num_heads,
                num_heads,
                len(ntypes),
                num_etypes,
                dropout,
                use_norm
            )
        )

                       
        for _ in range(1, num_layers - 1):
            self.coa_layers.append(
                COAConv(
                    in_dim,
                    in_dim // num_heads,
                    num_heads,
                    len(ntypes),
                    num_etypes,
                    dropout,
                    use_norm
                )
            )

                    
        self.coa_layers.append(
            COAConv(
                in_dim,
                out_dim,
                1,
                len(ntypes),
                num_etypes,
                dropout,
                use_norm
            )
        )

        print(f"[COA_Multi] Model initialized with {num_layers} layers")

    def forward(self, hg, h_dict):
        """
        Forward pass of COA_Multi

        Args:
            hg: DGLHeteroGraph or list of blocks (for mini-batch)
            h_dict: Dict of node features {ntype: Tensor[N, in_dim]}

        Returns:
            out_dict: Dict with only target node type
                {target_ntype: Tensor[N, out_dim]} - raw logits for BCEWithLogitsLoss
        """
        if hasattr(hg, 'ntypes'):
                                 
            with hg.local_scope():
                hg.ndata['h'] = h_dict
                g = dgl.to_homogeneous(hg, ndata='h')
                h = g.ndata['h']

                                  
                for l in range(self.num_layers):
                    h = self.coa_layers[l](
                        g, h,
                        g.ndata['_TYPE'],
                        g.edata['_TYPE'],
                        presorted=True
                    )

                                                        
                h_dict = to_hetero_feat(h, g.ndata['_TYPE'], self.ntypes)
        else:
                                 
            h = h_dict
            for layer, block in zip(self.coa_layers, hg):
                h = layer(
                    block, h,
                    block.ndata['_TYPE']['_N'],
                    block.edata['_TYPE'],
                    presorted=False
                )

                                      
            h_dict = to_hetero_feat(
                h,
                block.ndata['_TYPE']['_N'][:block.num_dst_nodes()],
                self.ntypes
            )

                                                               
        return {self.target_ntype: h_dict[self.target_ntype]}

    @property
    def to_homo_flag(self):
        return True


class COAConv(nn.Module):
    r"""
    Heterogeneous Graph Transformer Convolution

    This is a copy of the COAConv from COA.py to ensure independence.
    Implements the core COA message passing mechanism.
    """

    def __init__(self,
                 in_size,
                 head_size,
                 num_heads,
                 num_ntypes,
                 num_etypes,
                 dropout=0.2,
                 use_norm=False):
        super().__init__()
        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.sqrt_d = math.sqrt(head_size)
        self.use_norm = use_norm

                                              
        self.linear_k = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_q = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_v = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_a = TypedLinear(head_size * num_heads, head_size * num_heads, num_ntypes)

                                      
        self.relation_pri = nn.ParameterList([
            nn.Parameter(torch.ones(num_etypes))
            for i in range(num_heads)
        ])
        self.relation_att = nn.ModuleList([
            TypedLinear(head_size, head_size, num_etypes)
            for i in range(num_heads)
        ])
        self.relation_msg = nn.ModuleList([
            TypedLinear(head_size, head_size, num_etypes)
            for i in range(num_heads)
        ])

        self.drop = nn.Dropout(dropout)

        if use_norm:
            self.norm = nn.LayerNorm(head_size * num_heads)

    def forward(self, g, x, ntype, etype, *, presorted=False):
        """
        Forward computation

        Parameters
        ----------
        g : DGLGraph
            The input graph
        x : torch.Tensor
            Node features, shape: [|V|, in_size]
        ntype : torch.Tensor
            Node type IDs, shape: [|V|]
        etype : torch.Tensor
            Edge type IDs, shape: [|E|]
        presorted : bool
            Whether nodes and edges are pre-sorted by type

        Returns
        -------
        torch.Tensor
            New node features, shape: [|V|, head_size * num_heads]
        """
        self.presorted = presorted

        if g.is_block:
                        
            x_src = x
            x_dst = x[:g.num_dst_nodes()]
            srcntype = ntype
            dstntype = ntype[:g.num_dst_nodes()]
        else:
                        
            x_src = x
            x_dst = x
            srcntype = ntype
            dstntype = ntype

        with g.local_scope():
                                    
            k = self.linear_k(x_src, srcntype, presorted).view(-1, self.num_heads, self.head_size)
            q = self.linear_q(x_dst, dstntype, presorted).view(-1, self.num_heads, self.head_size)
            v = self.linear_v(x_src, srcntype, presorted).view(-1, self.num_heads, self.head_size)

                            
            g.srcdata['k'] = k
            g.dstdata['q'] = q
            g.srcdata['v'] = v
            g.edata['etype'] = etype

                                            
            g.apply_edges(self.message)

                                                
            g.edata['m'] = g.edata['m'] * edge_softmax(g, g.edata['a']).unsqueeze(-1)

                                
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h'))

                                       
            h = g.dstdata['h'].view(-1, self.num_heads * self.head_size)

                                         
            h = self.drop(self.linear_a(h, dstntype, presorted))

                                 
            if self.use_norm:
                h = self.norm(h)

            return h

    def message(self, edges):
        """
        Compute attention scores and messages for each edge
        """
        a, m = [], []
        etype = edges.data['etype']

                        
        k = torch.unbind(edges.src['k'], dim=1)
        q = torch.unbind(edges.dst['q'], dim=1)
        v = torch.unbind(edges.src['v'], dim=1)

        for i in range(self.num_heads):
                             
            kw = self.relation_att[i](k[i], etype, self.presorted)
            qw = self.relation_att[i](q[i], etype, self.presorted)
            a.append((kw * qw).sum(-1) * self.relation_pri[i][etype] / self.sqrt_d)

                     
            m.append(self.relation_msg[i](v[i], etype, self.presorted))

        return {
            'a': torch.stack(a, dim=1),                  
            'm': torch.stack(m, dim=1)                              
        }
