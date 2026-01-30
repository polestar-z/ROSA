import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from . import BaseModel, register_model
from .HGT import HGTConv
from .HAN import HANLayer
from ..utils.utils import to_hetero_feat, extract_metapaths, get_ntypes_from_canonical_etypes


class FusionModule_Original(nn.Module):
    def __init__(self, hidden_dim, ntypes, output_dim, dropout):
        super().__init__()
        self.linear_dict = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, output_dim)
            )
            for ntype in ntypes
        })

    def forward(self, h_hgt_dict, h_han_dict):
        h_fused_dict = {}
        for ntype in h_hgt_dict.keys():
            if ntype in h_han_dict:
                h_concat = torch.cat([h_hgt_dict[ntype], h_han_dict[ntype]], dim=-1)
                h_fused_dict[ntype] = self.linear_dict[ntype](h_concat)
            else:
                h_fused_dict[ntype] = h_hgt_dict[ntype]
        return h_fused_dict


class GatedFusionModule(nn.Module):
    def __init__(self, hidden_dim, ntypes, output_dim, dropout, gate_type='bidirectional'):
        super().__init__()
        assert output_dim == hidden_dim, \
            "GatedFusion requires output_dim == hidden_dim"

        self.gate_type = gate_type
        self.hidden_dim = hidden_dim

        if gate_type == 'basic':
            self.gate_dict = nn.ModuleDict({
                ntype: nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.Sigmoid()
                )
                for ntype in ntypes
            })

        elif gate_type == 'bidirectional':
            self.gate1_dict = nn.ModuleDict({
                ntype: nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.Sigmoid()
                )
                for ntype in ntypes
            })
            self.gate2_dict = nn.ModuleDict({
                ntype: nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.Sigmoid()
                )
                for ntype in ntypes
            })
        else:
            raise ValueError(f"Unsupported gate_type: {gate_type}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, h_hgt_dict, h_han_dict):
        h_fused_dict = {}

        for ntype in h_hgt_dict.keys():
            if ntype not in h_han_dict:
                h_fused_dict[ntype] = h_hgt_dict[ntype]
                continue

            h_hgt = h_hgt_dict[ntype]
            h_han = h_han_dict[ntype]

            h_concat = torch.cat([h_hgt, h_han], dim=-1)

            if self.gate_type == 'basic':
                gate = self.gate_dict[ntype](h_concat)
                h_fused = gate * h_hgt + (1 - gate) * h_han

            elif self.gate_type == 'bidirectional':
                gate1 = self.gate1_dict[ntype](h_concat)
                gate2 = self.gate2_dict[ntype](h_concat)
                h_fused = gate1 * h_hgt + gate2 * h_han

            h_fused_dict[ntype] = self.dropout(h_fused)

        return h_fused_dict


FusionModule = GatedFusionModule


class AdaptiveResidual(nn.Module):
    def __init__(self, hidden_dim, ntypes):
        super().__init__()
        self.gate_dict = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
            for ntype in ntypes
        })

    def forward(self, h_new_dict, h_old_dict):
        h_out_dict = {}

        for ntype in h_new_dict.keys():
            if ntype not in h_old_dict:
                h_out_dict[ntype] = h_new_dict[ntype]
                continue

            h_new = h_new_dict[ntype]
            h_old = h_old_dict[ntype]

            h_concat = torch.cat([h_new, h_old], dim=-1)

            gate = self.gate_dict[ntype](h_concat)

            h_out_dict[ntype] = gate * h_new + (1 - gate) * h_old

        return h_out_dict


class BranchConsistencyLoss(nn.Module):
    def __init__(self, loss_type='cosine', temperature=0.1):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature

    def forward(self, h_hgt_dict, h_han_dict, mask=None):
        total_loss = 0.0
        num_nodes = 0

        for ntype in h_hgt_dict.keys():
            if ntype not in h_han_dict:
                continue

            h_hgt = h_hgt_dict[ntype]
            h_han = h_han_dict[ntype]

            if mask is not None and ntype in mask:
                h_hgt = h_hgt[mask[ntype]]
                h_han = h_han[mask[ntype]]

            if h_hgt.size(0) == 0:
                continue

            if self.loss_type == 'cosine':
                cosine_sim = F.cosine_similarity(h_hgt, h_han, dim=-1)
                loss = (1 - cosine_sim).mean()

            elif self.loss_type == 'l2':
                loss = F.mse_loss(h_hgt, h_han)

            elif self.loss_type == 'contrastive':
                h_hgt = F.normalize(h_hgt, dim=-1)
                h_han = F.normalize(h_han, dim=-1)

                sim_matrix = torch.matmul(h_hgt, h_han.T) / self.temperature

                labels = torch.arange(h_hgt.size(0), device=h_hgt.device)
                loss = F.cross_entropy(sim_matrix, labels)
            else:
                raise ValueError(f"Unsupported loss type: {self.loss_type}")

            total_loss += loss * h_hgt.size(0)
            num_nodes += h_hgt.size(0)

        return total_loss / num_nodes if num_nodes > 0 else torch.tensor(0.0, device=h_hgt.device)


class HybridLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, hgt_num_heads, han_num_heads,
                 num_etypes, ntypes, meta_paths_dict, dropout, use_norm,
                 exclude_persona_in_hgt=False, fusion_type='gated', gate_type='bidirectional'):
        super().__init__()
        self.ntypes = ntypes
        self.exclude_persona_in_hgt = exclude_persona_in_hgt
        self.fusion_type = fusion_type
        self.gate_type = gate_type

        head_size = hidden_dim // hgt_num_heads
        self.hgt_conv = HGTConv(
            in_size=in_dim,
            head_size=head_size,
            num_heads=hgt_num_heads,
            num_ntypes=len(ntypes),
            num_etypes=num_etypes,
            dropout=dropout,
            use_norm=use_norm
        )

        han_out_dim = hidden_dim // han_num_heads
        self.han_dict = nn.ModuleDict()
        for ntype, meta_paths in meta_paths_dict.items():
            self.han_dict[ntype] = HANLayer(
                meta_paths_dict=meta_paths,
                in_dim=in_dim,
                out_dim=han_out_dim,
                layer_num_heads=han_num_heads,
                dropout=dropout
            )

        if fusion_type == 'gated':
            self.fusion = GatedFusionModule(
                hidden_dim=hidden_dim,
                ntypes=ntypes,
                output_dim=hidden_dim,
                dropout=dropout,
                gate_type=gate_type
            )
        elif fusion_type == 'original':
            self.fusion = FusionModule_Original(
                hidden_dim=hidden_dim,
                ntypes=ntypes,
                output_dim=hidden_dim,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def forward(self, hg, h_dict, return_branch_features=False):
        h_hgt_dict = self._hgt_forward(hg, h_dict)
        h_han_dict = self._han_forward(hg, h_dict)
        h_fused_dict = self.fusion(h_hgt_dict, h_han_dict)

        if return_branch_features:
            return h_fused_dict, h_hgt_dict, h_han_dict
        else:
            return h_fused_dict

    def _hgt_forward(self, hg, h_dict):
        if self.exclude_persona_in_hgt and 'persona' in hg.ntypes:
            return self._hgt_forward_exclude_persona(hg, h_dict)
        else:
            return self._hgt_forward_original(hg, h_dict)

    def _hgt_forward_original(self, hg, h_dict):
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            g_homo = dgl.to_homogeneous(hg, ndata='h')
            h_homo = g_homo.ndata['h']

            h_homo = self.hgt_conv(
                g_homo,
                h_homo,
                g_homo.ndata['_TYPE'],
                g_homo.edata['_TYPE'],
                presorted=True
            )

            h_hgt_dict = to_hetero_feat(h_homo, g_homo.ndata['_TYPE'], self.ntypes)

        return h_hgt_dict

    def _hgt_forward_exclude_persona(self, hg, h_dict):
        exclude_dst_ntype = 'persona'
        kept_etypes = []
        for canonical_etype in hg.canonical_etypes:
            src_type, etype, dst_type = canonical_etype
            if dst_type != exclude_dst_ntype:
                kept_etypes.append(canonical_etype)

        hg_sub = dgl.edge_type_subgraph(hg, kept_etypes)
        kept_ntypes = hg_sub.ntypes
        h_dict_sub = {nt: h_dict[nt] for nt in kept_ntypes if nt in h_dict}

        with hg_sub.local_scope():
            hg_sub.ndata['h'] = h_dict_sub
            g_homo = dgl.to_homogeneous(hg_sub, ndata='h')
            h_homo = g_homo.ndata['h']

            h_homo = self.hgt_conv(
                g_homo,
                h_homo,
                g_homo.ndata['_TYPE'],
                g_homo.edata['_TYPE'],
                presorted=True
            )

            h_hgt_dict = to_hetero_feat(h_homo, g_homo.ndata['_TYPE'], kept_ntypes)

        for ntype in self.ntypes:
            if ntype not in h_hgt_dict and ntype in h_dict:
                h_hgt_dict[ntype] = h_dict[ntype]

        return h_hgt_dict


    def _han_forward(self, hg, h_dict):
        h_han_dict = {}
        for ntype, han_layer in self.han_dict.items():
            h_out = han_layer(hg, h_dict)
            h_han_dict.update(h_out)
        return h_han_dict


@register_model('ROSA')
class ROSA(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        ntypes = set()
        if hasattr(args, 'target_link'):
            ntypes = get_ntypes_from_canonical_etypes(args.target_link)
        elif hasattr(args, 'category'):
            ntypes.add(args.category)
        else:
            raise ValueError("Require 'target_link' or 'category'")

        ntype_meta_paths_dict = {}
        for ntype in ntypes:
            ntype_meta_paths_dict[ntype] = {}
            for meta_path_name, meta_path in args.meta_paths_dict.items():
                if meta_path[0][0] == ntype:
                    ntype_meta_paths_dict[ntype][meta_path_name] = meta_path

        for ntype, meta_paths_dict in ntype_meta_paths_dict.items():
            if len(meta_paths_dict) == 0:
                ntype_meta_paths_dict[ntype] = extract_metapaths(ntype, hg.canonical_etypes)

        exclude_persona_in_hgt = 'persona' in hg.ntypes

        fusion_type = getattr(args, 'fusion_type', 'gated')
        gate_type = getattr(args, 'gate_type', 'bidirectional')
        residual_type = getattr(args, 'residual_type', 'adaptive')

        use_consistency_loss = getattr(args, 'use_consistency_loss', False)
        consistency_loss_type = getattr(args, 'consistency_loss_type', 'cosine')
        consistency_temperature = getattr(args, 'consistency_temperature', 0.1)

        return cls(
            in_dim=args.hidden_dim,
            hidden_dim=args.hidden_dim,
            out_dim=args.out_dim,
            num_layers=args.num_layers,
            hgt_num_heads=args.hgt_num_heads,
            han_num_heads=args.han_num_heads,
            num_etypes=len(hg.etypes),
            ntypes=hg.ntypes,
            meta_paths_dict=ntype_meta_paths_dict,
            dropout=args.dropout,
            use_norm=args.norm,
            exclude_persona_in_hgt=exclude_persona_in_hgt,
            fusion_type=fusion_type,
            gate_type=gate_type,
            residual_type=residual_type,
            use_consistency_loss=use_consistency_loss,
            consistency_loss_type=consistency_loss_type,
            consistency_temperature=consistency_temperature
        )

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers,
                 hgt_num_heads, han_num_heads, num_etypes, ntypes,
                 meta_paths_dict, dropout, use_norm, exclude_persona_in_hgt=False,
                 fusion_type='gated', gate_type='bidirectional', residual_type='adaptive',
                 use_consistency_loss=False, consistency_loss_type='cosine', consistency_temperature=0.1):
        super().__init__()
        self.ntypes = ntypes
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.fusion_type = fusion_type
        self.gate_type = gate_type
        self.residual_type = residual_type
        self.use_consistency_loss = use_consistency_loss

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_dim = in_dim if i == 0 else hidden_dim
            self.layers.append(
                HybridLayer(
                    in_dim=layer_in_dim,
                    hidden_dim=hidden_dim,
                    hgt_num_heads=hgt_num_heads,
                    han_num_heads=han_num_heads,
                    num_etypes=num_etypes,
                    ntypes=ntypes,
                    meta_paths_dict=meta_paths_dict,
                    dropout=dropout,
                    use_norm=use_norm,
                    exclude_persona_in_hgt=exclude_persona_in_hgt,
                    fusion_type=fusion_type,
                    gate_type=gate_type
                )
            )

        if residual_type == 'adaptive':
            self.adaptive_residual = AdaptiveResidual(
                hidden_dim=hidden_dim,
                ntypes=ntypes
            )
        elif residual_type == 'fixed':
            self.adaptive_residual = None
        else:
            raise ValueError(f"Unsupported residual_type: {residual_type}")

        if use_consistency_loss:
            self.consistency_loss = BranchConsistencyLoss(
                loss_type=consistency_loss_type,
                temperature=consistency_temperature
            )
        else:
            self.consistency_loss = None

        self.output_linear = nn.ModuleDict({
            ntype: nn.Linear(hidden_dim, out_dim)
            for ntype in ntypes
        })

    def forward(self, hg, h_dict, return_branch_features=False):
        h_hgt_last = None
        h_han_last = None

        for i, layer in enumerate(self.layers):
            h_dict_input = h_dict

            if return_branch_features:
                h_dict, h_hgt, h_han = layer(hg, h_dict, return_branch_features=True)
                h_hgt_last = h_hgt
                h_han_last = h_han
            else:
                h_dict = layer(hg, h_dict)

            if i > 0 or all(h_dict_input[ntype].shape[-1] == h_dict[ntype].shape[-1]
                           for ntype in h_dict.keys() if ntype in h_dict_input):

                if self.residual_type == 'adaptive':
                    h_dict = self.adaptive_residual(h_dict, h_dict_input)
                else:
                    for ntype in h_dict.keys():
                        if ntype in h_dict_input and h_dict_input[ntype].shape[-1] == h_dict[ntype].shape[-1]:
                            h_dict[ntype] = h_dict[ntype] + h_dict_input[ntype]

        out_dict = {}
        for ntype, h in h_dict.items():
            out_dict[ntype] = self.output_linear[ntype](h)

        if return_branch_features:
            return out_dict, h_hgt_last, h_han_last
        else:
            return out_dict