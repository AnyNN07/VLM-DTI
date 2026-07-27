import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

                                                
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VLMNET'))

from VLMNET.model import QSGNNLayer, MultiScale1DCNN, BANLayer, entropic_fgw_loss, _masked_cdist
from VLMNET.VLM import VirtualLearnableMapping

class UniQCM2Net_Modality(nn.Module):
       
                                                          
                                                                  
       
    def __init__(self, drug_in: int = 120, prot_in: int = 41,
                 gnn_hidden: int = 256, k: int = 5,
                 drug_llm_dim: int = 768, prot_llm_dim: int = 320,
                 gw_epsilon: float = 0.01,
                 gw_outer_iters: int = 10, gw_inner_iters: int = 5,
                 protein_gw_cap: int = 300,
                 fusion_mode: str = 'VLM-XA',
                 vlm_ablation: str = 'full',
                 input_modality: str = 'full'):                              
        super().__init__()
        self.k = k
        self.gnn_hidden = gnn_hidden
        self.gw_epsilon = gw_epsilon
        self.gw_outer_iters = gw_outer_iters
        self.gw_inner_iters = gw_inner_iters
        self.protein_gw_cap = protein_gw_cap
        self.fusion_mode = fusion_mode
        self.input_modality = input_modality
        self.fgw_alpha = 0.5                                           

        if self.input_modality not in ['full', 'graph_only', 'llm_only']:
            raise ValueError(f"Unknown input_modality: {self.input_modality}")

                                                                   
        if self.input_modality == 'full':
            if self.fusion_mode in ('Cross-Attn', 'Hybrid', 'VLM-XA', 'VLM-GW'):
                self.cross_attn_d = nn.MultiheadAttention(gnn_hidden, num_heads=4, batch_first=True)
                self.cross_attn_p = nn.MultiheadAttention(gnn_hidden, num_heads=4, batch_first=True)
            if self.fusion_mode in ('VLM', 'VLM-XA', 'VLM-GW'):
                self.vlm = VirtualLearnableMapping(hidden_dim=gnn_hidden, ablation=vlm_ablation)

                                                                
        if self.input_modality in ['full', 'graph_only']:
            self.drug_gnn = nn.ModuleList([
                QSGNNLayer(drug_in if i == 0 else gnn_hidden, gnn_hidden,
                           edge_dim=12, edge_feat_name='bond')
                for i in range(k)
            ])
            self.prot_gnn = nn.ModuleList([
                QSGNNLayer(prot_in if i == 0 else gnn_hidden, gnn_hidden,
                           edge_dim=5, edge_feat_name='feats')
                for i in range(k)
            ])
            self.drug_jk = nn.Linear(gnn_hidden * k, gnn_hidden)
            self.prot_jk = nn.Linear(gnn_hidden * k, gnn_hidden)

                                                            
        if self.input_modality in ['full', 'llm_only']:
            self.drug_norm = nn.LayerNorm(drug_llm_dim)
            self.prot_norm = nn.LayerNorm(prot_llm_dim)
            self.drug_llm_encoder = MultiScale1DCNN(drug_llm_dim, gnn_hidden)
            self.prot_llm_encoder = MultiScale1DCNN(prot_llm_dim, gnn_hidden)

                                          
        self.v_norm = nn.LayerNorm(gnn_hidden)
        self.q_norm = nn.LayerNorm(gnn_hidden)

                                            
        if self.input_modality == 'full':
            self.alpha_d = nn.Parameter(torch.tensor(0.5))
            self.alpha_p = nn.Parameter(torch.tensor(0.5))

                                          
        self.ban = BANLayer(gnn_hidden, gnn_hidden, num_glimpses=4)

                          
        self.fc = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
                                                 
        init_list = []
        if self.input_modality in ['full', 'graph_only']:
            init_list.extend([self.drug_jk, self.prot_jk])
        if self.input_modality in ['full', 'llm_only']:
            init_list.extend([self.drug_llm_encoder.proj, self.prot_llm_encoder.proj])
            
        for m in init_list:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
            
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _gnn_jk_forward(self, gnn_layers, g, h, jk_proj):
        layer_outputs = []
        for layer in gnn_layers:
            h = layer(g, h)
            layer_outputs.append(h)
        h_jk = torch.cat(layer_outputs, dim=-1)
        h_jk = F.relu(jk_proj(h_jk))
        return h_jk

    def _unbatch(self, flat, batch_num_nodes, max_len, B):
        feat_dim = flat.size(-1)
        padded = torch.zeros(B, max_len, feat_dim,
                             device=flat.device, dtype=flat.dtype)
        idx = 0
        for i, num_nodes in enumerate(batch_num_nodes):
            n = num_nodes.item()
            padded[i, :n] = flat[idx:idx + n]
            idx += n
        return padded

    def forward(self, bg_drug, bg_prot, d_feat, p_feat, d_lens, p_lens):
        B = bg_drug.batch_size if bg_drug is not None else d_feat.shape[0]
        cl_loss = torch.tensor(0.0, device=d_feat.device if d_feat is not None else bg_drug.device)

                                        
        if self.input_modality in ['full', 'graph_only']:
            d_nodes = bg_drug.batch_num_nodes()
            max_d = d_nodes.max().item()
            h_d = bg_drug.ndata['atom'].float()
            h_d_out = self._gnn_jk_forward(self.drug_gnn, bg_drug, h_d, self.drug_jk)
            v_gnn = self._unbatch(h_d_out, d_nodes, max_d, B)
            
            p_nodes = bg_prot.batch_num_nodes()
            max_p = p_nodes.max().item()
            h_p = bg_prot.ndata['feats'].float()
            h_p_out = self._gnn_jk_forward(self.prot_gnn, bg_prot, h_p, self.prot_jk)
            q_gnn = self._unbatch(h_p_out, p_nodes, max_p, B)

                                           
        if self.input_modality in ['full', 'llm_only']:
            d_feat = self.drug_norm(d_feat)
            p_feat = self.prot_norm(p_feat)
            v_llm = F.relu(self.drug_llm_encoder(d_feat))
            q_llm = F.relu(self.prot_llm_encoder(p_feat))

                                     
        if self.input_modality == 'graph_only':
            v_fused, q_fused = v_gnn, q_gnn
            v_len_ban, p_len_ban = d_nodes, p_nodes
            
        elif self.input_modality == 'llm_only':
            v_fused, q_fused = v_llm, q_llm
            v_len_ban, p_len_ban = d_lens, p_lens
            
        elif self.input_modality == 'full':
            v_len_ban, p_len_ban = d_nodes, p_nodes
            d_pad_mask = torch.arange(v_llm.shape[1], device=v_llm.device).expand(B, -1) >= d_lens.unsqueeze(1)
            p_pad_mask = torch.arange(q_llm.shape[1], device=q_llm.device).expand(B, -1) >= p_lens.unsqueeze(1)
            
            if self.fusion_mode == 'VLM-XA':
                p_aa_idx = self._unbatch(bg_prot.ndata['feats'].float()[:,:35].argmax(-1).unsqueeze(-1).float(), p_nodes, max_p, B).long().squeeze(-1)
                d_atom_idx = self._unbatch(bg_drug.ndata['atom'].float()[:,:71].argmax(-1).unsqueeze(-1).float(), d_nodes, max_d, B).long().squeeze(-1)
                q_enh, v_enh, _ = self.vlm(p_aa_idx, d_atom_idx, p_lens=p_nodes, d_lens=d_nodes, graph_feat=q_gnn, llm_feat=v_gnn)
                v_fused, _ = self.cross_attn_d(v_enh, v_llm, v_llm, key_padding_mask=d_pad_mask)
                q_fused, _ = self.cross_attn_p(q_enh, q_llm, q_llm, key_padding_mask=p_pad_mask)
                cl_loss = 0.0001 * (self.vlm.P_virt.norm(p=2) + self.vlm.D_virt.norm(p=2))
            else:
                raise ValueError("For full modality, only VLM-XA is fully implemented in this snippet. Adapt for others if needed.")

                  
        v_fused = self.v_norm(v_fused)
        q_fused = self.q_norm(q_fused)
        out = self.ban(v_fused, q_fused, v_len_ban, p_len_ban)

                 
        preds = torch.sigmoid(self.fc(out)).squeeze(-1)
        return preds, cl_loss
