import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from . import BaseModel, register_model
from .COA import COAConv
from .AOA import AOALayer
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

    def forward(self, h_coa_dict, h_aoa_dict):
        h_fused_dict = {}
        for ntype in h_coa_dict.keys():
            if ntype in h_aoa_dict:
                h_concat = torch.cat([h_coa_dict[ntype], h_aoa_dict[ntype]], dim=-1)
                h_fused_dict[ntype] = self.linear_dict[ntype](h_concat)
            else:
                h_fused_dict[ntype] = h_coa_dict[ntype]
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

    def forward(self, h_coa_dict, h_aoa_dict):
        h_fused_dict = {}

        for ntype in h_coa_dict.keys():
            if ntype not in h_aoa_dict:
                h_fused_dict[ntype] = h_coa_dict[ntype]
                continue

            h_coa = h_coa_dict[ntype]
            h_aoa = h_aoa_dict[ntype]

            h_concat = torch.cat([h_coa, h_aoa], dim=-1)

            if self.gate_type == 'basic':
                gate = self.gate_dict[ntype](h_concat)
                h_fused = gate * h_coa + (1 - gate) * h_aoa

            elif self.gate_type == 'bidirectional':
                gate1 = self.gate1_dict[ntype](h_concat)
                gate2 = self.gate2_dict[ntype](h_concat)
                h_fused = gate1 * h_coa + gate2 * h_aoa

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

    def forward(self, h_coa_dict, h_aoa_dict, mask=None):
        total_loss = 0.0
        num_nodes = 0

        for ntype in h_coa_dict.keys():
            if ntype not in h_aoa_dict:
                continue

            h_coa = h_coa_dict[ntype]
            h_aoa = h_aoa_dict[ntype]

            if mask is not None and ntype in mask:
                h_coa = h_coa[mask[ntype]]
                h_aoa = h_aoa[mask[ntype]]

            if h_coa.size(0) == 0:
                continue

            if self.loss_type == 'cosine':
                cosine_sim = F.cosine_similarity(h_coa, h_aoa, dim=-1)
                loss = (1 - cosine_sim).mean()

            elif self.loss_type == 'l2':
                loss = F.mse_loss(h_coa, h_aoa)

            elif self.loss_type == 'contrastive':
                h_coa = F.normalize(h_coa, dim=-1)
                h_aoa = F.normalize(h_aoa, dim=-1)

                sim_matrix = torch.matmul(h_coa, h_aoa.T) / self.temperature

                labels = torch.arange(h_coa.size(0), device=h_coa.device)
                loss = F.cross_entropy(sim_matrix, labels)
            else:
                raise ValueError(f"Unsupported loss type: {self.loss_type}")

            total_loss += loss * h_coa.size(0)
            num_nodes += h_coa.size(0)

        return total_loss / num_nodes if num_nodes > 0 else torch.tensor(0.0, device=h_coa.device)


class HybridLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, coa_num_heads, aoa_num_heads,
                 num_etypes, ntypes, meta_paths_dict, dropout, use_norm,
                 exclude_persona_in_coa=False, fusion_type='gated', gate_type='bidirectional'):
        super().__init__()
        self.ntypes = ntypes
        self.exclude_persona_in_coa = exclude_persona_in_coa
        self.fusion_type = fusion_type
        self.gate_type = gate_type

        head_size = hidden_dim // coa_num_heads
        self.coa_conv = COAConv(
            in_size=in_dim,
            head_size=head_size,
            num_heads=coa_num_heads,
            num_ntypes=len(ntypes),
            num_etypes=num_etypes,
            dropout=dropout,
            use_norm=use_norm
        )

        aoa_out_dim = hidden_dim // aoa_num_heads
        self.aoa_dict = nn.ModuleDict()
        for ntype, meta_paths in meta_paths_dict.items():
            self.aoa_dict[ntype] = AOALayer(
                meta_paths_dict=meta_paths,
                in_dim=in_dim,
                out_dim=aoa_out_dim,
                layer_num_heads=aoa_num_heads,
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
        h_coa_dict = self._coa_forward(hg, h_dict)
        h_aoa_dict = self._aoa_forward(hg, h_dict)
        h_fused_dict = self.fusion(h_coa_dict, h_aoa_dict)

        if return_branch_features:
            return h_fused_dict, h_coa_dict, h_aoa_dict
        else:
            return h_fused_dict

    def _coa_forward(self, hg, h_dict):
        if self.exclude_persona_in_coa and 'persona' in hg.ntypes:
            return self._coa_forward_exclude_persona(hg, h_dict)
        else:
            return self._coa_forward_original(hg, h_dict)

    def _coa_forward_original(self, hg, h_dict):
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            g_homo = dgl.to_homogeneous(hg, ndata='h')
            h_homo = g_homo.ndata['h']

            h_homo = self.coa_conv(
                g_homo,
                h_homo,
                g_homo.ndata['_TYPE'],
                g_homo.edata['_TYPE'],
                presorted=True
            )

            h_coa_dict = to_hetero_feat(h_homo, g_homo.ndata['_TYPE'], self.ntypes)

        return h_coa_dict

    def _coa_forward_exclude_persona(self, hg, h_dict):
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

            h_homo = self.coa_conv(
                g_homo,
                h_homo,
                g_homo.ndata['_TYPE'],
                g_homo.edata['_TYPE'],
                presorted=True
            )

            h_coa_dict = to_hetero_feat(h_homo, g_homo.ndata['_TYPE'], kept_ntypes)

        for ntype in self.ntypes:
            if ntype not in h_coa_dict and ntype in h_dict:
                h_coa_dict[ntype] = h_dict[ntype]

        return h_coa_dict


    def _aoa_forward(self, hg, h_dict):
        h_aoa_dict = {}
        for ntype, aoa_layer in self.aoa_dict.items():
            h_out = aoa_layer(hg, h_dict)
            h_aoa_dict.update(h_out)
        return h_aoa_dict


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

        exclude_persona_in_coa = 'persona' in hg.ntypes

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
            coa_num_heads=args.coa_num_heads,
            aoa_num_heads=args.aoa_num_heads,
            num_etypes=len(hg.etypes),
            ntypes=hg.ntypes,
            meta_paths_dict=ntype_meta_paths_dict,
            dropout=args.dropout,
            use_norm=args.norm,
            exclude_persona_in_coa=exclude_persona_in_coa,
            fusion_type=fusion_type,
            gate_type=gate_type,
            residual_type=residual_type,
            use_consistency_loss=use_consistency_loss,
            consistency_loss_type=consistency_loss_type,
            consistency_temperature=consistency_temperature
        )

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers,
                 coa_num_heads, aoa_num_heads, num_etypes, ntypes,
                 meta_paths_dict, dropout, use_norm, exclude_persona_in_coa=False,
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
                    coa_num_heads=coa_num_heads,
                    aoa_num_heads=aoa_num_heads,
                    num_etypes=num_etypes,
                    ntypes=ntypes,
                    meta_paths_dict=meta_paths_dict,
                    dropout=dropout,
                    use_norm=use_norm,
                    exclude_persona_in_coa=exclude_persona_in_coa,
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
        h_coa_last = None
        h_aoa_last = None

        for i, layer in enumerate(self.layers):
            h_dict_input = h_dict

            if return_branch_features:
                h_dict, h_coa, h_aoa = layer(hg, h_dict, return_branch_features=True)
                h_coa_last = h_coa
                h_aoa_last = h_aoa
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
            return out_dict, h_coa_last, h_aoa_last
        else:
            return out_dict