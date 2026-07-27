import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from torch.nn.utils import weight_norm
from VLM import VirtualLearnableMapping

class SafeInstanceNorm1d(nn.InstanceNorm1d):
       
                                                                                       
                                                                                         
                                                                    
       
    def forward(self, x):
        if x.shape[-1] == 1:
                                                                           
            out = super().forward(torch.cat([x, x], dim=-1))
            return out[..., :1]
        return super().forward(x)

                                                                                 
                                         
                                                                                 

class QSGNNLayer(nn.Module):
       
                                                           
                                                                            

                                                                       
                                                      
       

    def __init__(self, in_dim: int, out_dim: int,
                 edge_dim: int = None, edge_feat_name: str = 'bond'):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_neigh = nn.Linear(in_dim, out_dim)
        att_in = out_dim * 2
        if edge_dim is not None:
            att_in += edge_dim
        self.att_fc = nn.Linear(att_in, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.edge_dim = edge_dim
        self.edge_feat_name = edge_feat_name

    def forward(self, g, h):
        h_self = self.W_self(h)
        h_neigh = self.W_neigh(h)
        g = g.local_var()
        g.ndata['h'] = h_neigh

        if self.edge_dim is not None:
            if self.edge_feat_name in g.edata and g.num_edges() > 0:
                g.edata['_tmp_edge'] = g.edata[self.edge_feat_name].float()
            else:
                g.edata['_tmp_edge'] = torch.zeros(g.num_edges(), self.edge_dim, device=g.device)
            g.apply_edges(lambda edges: {
                'a': self.leaky_relu(self.att_fc(torch.cat([
                    edges.src['h'], edges.dst['h'], edges.data['_tmp_edge']], dim=-1)))
            })
            g.edata.pop('_tmp_edge', None)
        else:
            g.apply_edges(lambda edges: {
                'a': self.leaky_relu(self.att_fc(torch.cat([edges.src['h'], edges.dst['h']], dim=-1)))
            })

        g.edata['a'] = dgl.nn.functional.edge_softmax(g, g.edata['a'])
        g.update_all(fn.u_mul_e('h', 'a', 'm'), fn.sum('m', 'neigh_agg'))
        return F.relu(h_self + g.ndata['neigh_agg'])

                                                                                 
                                                     
                                                                                 

class MultiScale1DCNN(nn.Module):
                                                                             
    def __init__(self, in_dim, out_dim, kernel_sizes=(3, 5, 7, 9)):
        super().__init__()
        n_scales = len(kernel_sizes)
        per_scale = out_dim // n_scales
        self.per_scale_dims = []
        remaining = out_dim
        for i in range(n_scales):
            dim = per_scale if i < n_scales - 1 else remaining
            self.per_scale_dims.append(dim)
            remaining -= dim
        self.convs = nn.ModuleList([
            nn.Conv1d(in_dim, dim, k, padding=k//2, bias=False)
            for k, dim in zip(kernel_sizes, self.per_scale_dims)])
        self.bns = nn.ModuleList([SafeInstanceNorm1d(d, affine=True) for d in self.per_scale_dims])
        self.proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        outs = [F.relu(bn(conv(x_t))) for conv, bn in zip(self.convs, self.bns)]
        x_out = torch.cat(outs, dim=1).transpose(1, 2)
        return self.dropout(self.proj(x_out))

                                                                                 
                                                                   
                                                                                 

def _masked_cdist(x, lengths):
       
                                                                 

         
                                          
                                             
            
                                                                        
                                                
                                                        
       
    B, L, D = x.shape
    device = x.device

                         
    len_mask = torch.arange(L, device=device).expand(B, L) < lengths.unsqueeze(1)

                                                               
    pair_mask = len_mask.unsqueeze(2) & len_mask.unsqueeze(1)             

                                          
                                                              
    x_norm_sq = (x ** 2).sum(dim=-1)          
    C = (x_norm_sq.unsqueeze(2) + x_norm_sq.unsqueeze(1)
         - 2 * torch.bmm(x, x.transpose(1, 2)))             

                                     
    C = C * pair_mask.float()

    return C, pair_mask

def entropic_fgw_loss(C1, C2, p, q, M=None, alpha=0.5, epsilon=0.01,
                      outer_iters=10, inner_iters=5):
       
                                               

                                                                   

                                              
                                                     
                                                             

         
                                                                     
                                                                        
                                                       
                                                      
                                                                                              
                                                              
                                                
                                             
                                                 
            
                                           
                             
       
    B, N, _ = C1.shape
    M_len = C2.shape[1]
    device = C1.device
    eps_clamp = max(epsilon, 1e-4)

    T = torch.einsum('bn,bm->bnm', p, q)             

                                                                               
    C1 = C1 / (C1.amax(dim=(1, 2), keepdim=True) + 1e-8)
    C2 = C2 / (C2.amax(dim=(1, 2), keepdim=True) + 1e-8)
    if M is not None:
        M = M / (M.amax(dim=(1, 2), keepdim=True) + 1e-8)

    C1_sq = C1 ** 2
    C2_sq = C2 ** 2

    for _ in range(outer_iters):
        t1 = T.sum(dim=2)          
        t2 = T.sum(dim=1)          

                                                
        term1 = torch.bmm(C1_sq, t1.unsqueeze(2))                         
        term2 = torch.bmm(C2_sq, t2.unsqueeze(2))                         
        cross = torch.bmm(torch.bmm(C1, T), C2.transpose(1, 2))             
        gw_cost = term1 + term2.transpose(1, 2) - 2.0 * cross              

                                                        
        if M is not None and alpha < 1.0:
            cost = (1.0 - alpha) * M + alpha * gw_cost
        else:
            cost = gw_cost

        K = torch.exp(-cost / eps_clamp)
        K = torch.clamp(K, min=1e-30, max=1e30)

        u = torch.ones(B, N, device=device)
        v = torch.ones(B, M_len, device=device)
        for _ in range(inner_iters):
            Kv = torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2)
            v = q / (Kv + 1e-8)
            Ku = torch.bmm(K, v.unsqueeze(2)).squeeze(2)
            u = p / (Ku + 1e-8)
        T = u.unsqueeze(2) * K * v.unsqueeze(1)
        T = T / (T.sum(dim=(1, 2), keepdim=True) + 1e-8)

    fgw_loss = (T * cost).sum(dim=(1, 2)).mean()
    return T, fgw_loss

                                                                                 
                                          
                                                                                 

class BANLayer(nn.Module):
       
                                                       

                                                                  
                                                
       

    def __init__(self, v_dim: int, q_dim: int, num_glimpses: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_glimpses = num_glimpses

        self.v_att = weight_norm(nn.Linear(v_dim, num_glimpses))
        self.q_att = weight_norm(nn.Linear(q_dim, num_glimpses))
        self.b_net = nn.ModuleList([
            weight_norm(nn.Linear(v_dim + q_dim, num_glimpses))
            for _ in range(num_glimpses)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, v, q, v_lens, q_lens):
        B, L_v, _ = v.shape
        _, L_q, _ = q.shape

        v_mask = torch.arange(L_v, device=v.device).expand(B, L_v) >= v_lens.unsqueeze(1)
        q_mask = torch.arange(L_q, device=q.device).expand(B, L_q) >= q_lens.unsqueeze(1)

        v_att = self.v_att(v)               
        q_att = self.q_att(q)               

        att = torch.einsum('bvk,bqk->bvqk', v_att, q_att)                    
        att.masked_fill_(v_mask.unsqueeze(2).unsqueeze(3), -1e4)
        att.masked_fill_(q_mask.unsqueeze(1).unsqueeze(3), -1e4)
        att = F.softmax(att.view(B, -1, self.num_glimpses), dim=1).view(
            B, L_v, L_q, self.num_glimpses)

        out = []
        for k in range(self.num_glimpses):
            att_k = att[:, :, :, k]                                         
            att_v = att_k.sum(dim=2)                                   
            att_q = att_k.sum(dim=1)                                   
            v_pool = torch.bmm(att_v.unsqueeze(1), v).squeeze(1)              
            q_pool = torch.bmm(att_q.unsqueeze(1), q).squeeze(1)              
            joint = torch.cat([v_pool, q_pool], dim=-1)                             
            out.append(self.b_net[k](joint))                              

        return self.dropout(torch.cat(out, dim=-1))                    

                                                                                 
                                                   
                                                                                 

class UniQCM2Net(nn.Module):
       
                                                                      
                                                                        
                       

                                        
                                                              
                                                               
                
                                                                       
                                                                       
                                                                 
                                                                   
                                                               
                                          
                                            
                                  
       

    def __init__(self, drug_in: int = 120, prot_in: int = 41,
                 gnn_hidden: int = 256, k: int = 5,
                 drug_llm_dim: int = 768, prot_llm_dim: int = 320,
                 gw_epsilon: float = 0.01,
                 gw_outer_iters: int = 10, gw_inner_iters: int = 5,
                 protein_gw_cap: int = 300,
                 fusion_mode: str = 'VLM-XA',
                 vlm_ablation: str = 'full'):
        super().__init__()
        self.k = k
        self.gnn_hidden = gnn_hidden
        self.gw_epsilon = gw_epsilon
        self.gw_outer_iters = gw_outer_iters
        self.gw_inner_iters = gw_inner_iters
        self.protein_gw_cap = protein_gw_cap
        self.fusion_mode = fusion_mode

                                   
        if self.fusion_mode in ('Cross-Attn', 'Hybrid', 'VLM-XA', 'VLM-GW'):
            self.cross_attn_d = nn.MultiheadAttention(gnn_hidden, num_heads=4, batch_first=True)
            self.cross_attn_p = nn.MultiheadAttention(gnn_hidden, num_heads=4, batch_first=True)
        if self.fusion_mode in ('VLM', 'VLM-XA', 'VLM-GW'):
            self.vlm = VirtualLearnableMapping(hidden_dim=gnn_hidden, ablation=vlm_ablation)
        if self.fusion_mode in ('VLM', 'VLM-Lite', 'VLM-Attn'):
                                                                        
            pass
        if self.fusion_mode == 'Concat':
            self.concat_proj_d = nn.Linear(gnn_hidden * 2, gnn_hidden)
            self.concat_proj_p = nn.Linear(gnn_hidden * 2, gnn_hidden)

                                                                
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

                                                       
        self.drug_norm = nn.LayerNorm(drug_llm_dim)
        self.prot_norm = nn.LayerNorm(prot_llm_dim)

                                                            
        self.drug_llm_encoder = MultiScale1DCNN(drug_llm_dim, gnn_hidden)
        self.prot_llm_encoder = MultiScale1DCNN(prot_llm_dim, gnn_hidden)

                                                          
        self.v_norm = nn.LayerNorm(gnn_hidden)
        self.q_norm = nn.LayerNorm(gnn_hidden)

                                                          
                                                                         
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
        for m in [self.drug_jk, self.prot_jk, self.drug_llm_encoder.proj, self.prot_llm_encoder.proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for attr in ['concat_proj_d', 'concat_proj_p']:
            if hasattr(self, attr):
                m = getattr(self, attr)
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

    def _gw_ota_align(self, h_graph, h_llm, g_lengths, l_lengths, name='drug'):
           
                                                                      

             
                                                            
                                                             
                                                  
                                                 
                                                                 
                
                                                                               
                                   
           
        B, N, hidden = h_graph.shape
        M = h_llm.shape[1]
        device = h_graph.device

                                                                     
        max_n, max_m = N, M
        if name == 'protein':
            if N > self.protein_gw_cap:
                max_n = self.protein_gw_cap
                g_lengths = torch.clamp(g_lengths, max=max_n)
            if M > self.protein_gw_cap:
                max_m = self.protein_gw_cap
                l_lengths = torch.clamp(l_lengths, max=max_m)

        h_g = h_graph[:, :max_n, :]                       
        h_l = h_llm[:, :max_m, :]                         

                                                                      
        C_graph, mask_g = _masked_cdist(h_g, g_lengths)                     
        C_text,  mask_t = _masked_cdist(h_l, l_lengths)                      

                                     
        p = mask_g.diagonal(dim1=1, dim2=2).float()                    
        q = mask_t.diagonal(dim1=1, dim2=2).float()                    
        p = p / (p.sum(dim=1, keepdim=True) + 1e-8)                   
        q = q / (q.sum(dim=1, keepdim=True) + 1e-8)                   

                                                                             
                                                                   
        h_g_norm = (h_g ** 2).sum(dim=-1)              
        h_l_norm = (h_l ** 2).sum(dim=-1)              
        M = h_g_norm.unsqueeze(2) + h_l_norm.unsqueeze(1) \
            - 2.0 * torch.bmm(h_g, h_l.transpose(1, 2))                     
        T, gw_loss = entropic_fgw_loss(
            C_graph, C_text, p, q, M=M, alpha=self.fgw_alpha,
            epsilon=self.gw_epsilon,
            outer_iters=self.gw_outer_iters,
            inner_iters=self.gw_inner_iters
        )                        

                                                                  
                                                      
        h_llm_aligned = torch.bmm(T, h_l)

                                                     
        alpha = torch.sigmoid(self.alpha_d if name == 'drug' else self.alpha_p)
        h_fused = alpha * h_g + (1.0 - alpha) * h_llm_aligned                      

                                                                                  
        if max_n < N:
            padding = h_graph[:, max_n:, :]
            h_fused = torch.cat([h_fused, padding], dim=1)

        return h_fused, gw_loss

    def forward(self, bg_drug, bg_prot, d_feat, p_feat, d_lens, p_lens):
           
             
                                                                               
                                                                                
                                                     
                                                       
                                              
                                              
                
                                                
                                                                 
           
        B = bg_drug.batch_size
        d_nodes = bg_drug.batch_num_nodes()
        p_nodes = bg_prot.batch_num_nodes()
        max_d = d_nodes.max().item()                            
        max_p = p_nodes.max().item()                               

                                                                                
                                        
                                                                                
        h_d = bg_drug.ndata['atom'].float()
        h_d_out = self._gnn_jk_forward(self.drug_gnn, bg_drug, h_d, self.drug_jk)
        v_gnn = self._unbatch(h_d_out, d_nodes, max_d, B)

                                                                                
                                           
                                                                                
        h_p = bg_prot.ndata['feats'].float()
        h_p_out = self._gnn_jk_forward(self.prot_gnn, bg_prot, h_p, self.prot_jk)
        q_gnn = self._unbatch(h_p_out, p_nodes, max_p, B)

                                                                                
                                                              
                                                                                
        d_feat = self.drug_norm(d_feat)
        p_feat = self.prot_norm(p_feat)
        v_llm = F.relu(self.drug_llm_encoder(d_feat))                  
        q_llm = F.relu(self.prot_llm_encoder(p_feat))                  

                                                                                
                               
                                                                                
        d_pad_mask = torch.arange(v_llm.shape[1], device=v_llm.device).expand(B, -1) >= d_lens.unsqueeze(1)
        p_pad_mask = torch.arange(q_llm.shape[1], device=q_llm.device).expand(B, -1) >= p_lens.unsqueeze(1)

        if self.fusion_mode == 'GW-OTA':
            v_fused, gw_d = self._gw_ota_align(v_gnn, v_llm, d_nodes, d_lens, name='drug')
            q_fused, gw_p = self._gw_ota_align(q_gnn, q_llm, p_nodes, p_lens, name='protein')
            cl_loss = gw_d + gw_p
        elif self.fusion_mode == 'Concat':
                                                                          
            N_d, M_d = v_gnn.shape[1], v_llm.shape[1]
            if M_d < N_d:
                v_llm = torch.cat([v_llm, torch.zeros(v_llm.shape[0], N_d-M_d, v_llm.shape[2], device=v_llm.device)], dim=1)
            elif N_d < M_d:
                v_gnn = torch.cat([v_gnn, torch.zeros(v_gnn.shape[0], M_d-N_d, v_gnn.shape[2], device=v_gnn.device)], dim=1)
            N_p, M_p = q_gnn.shape[1], q_llm.shape[1]
            if M_p < N_p:
                q_llm = torch.cat([q_llm, torch.zeros(q_llm.shape[0], N_p-M_p, q_llm.shape[2], device=q_llm.device)], dim=1)
            elif N_p < M_p:
                q_gnn = torch.cat([q_gnn, torch.zeros(q_gnn.shape[0], M_p-N_p, q_gnn.shape[2], device=q_gnn.device)], dim=1)
            v_fused = self.concat_proj_d(torch.cat([v_gnn, v_llm], dim=-1))
            q_fused = self.concat_proj_p(torch.cat([q_gnn, q_llm], dim=-1))
            cl_loss = torch.tensor(0.0, device=v_fused.device)
        elif self.fusion_mode == 'Cross-Attn':
            v_fused, _ = self.cross_attn_d(v_gnn, v_llm, v_llm, key_padding_mask=d_pad_mask)
            q_fused, _ = self.cross_attn_p(q_gnn, q_llm, q_llm, key_padding_mask=p_pad_mask)
            cl_loss = torch.tensor(0.0, device=v_fused.device)
        elif self.fusion_mode == 'Hybrid':
                                                                   
            v_fused, _ = self.cross_attn_d(v_gnn, v_llm, v_llm, key_padding_mask=d_pad_mask)
            q_fused, _ = self.cross_attn_p(q_gnn, q_llm, q_llm, key_padding_mask=p_pad_mask)
            _, gw_d = self._gw_ota_align(v_gnn, v_llm, d_nodes, d_lens, name='drug')
            _, gw_p = self._gw_ota_align(q_gnn, q_llm, p_nodes, p_lens, name='protein')
            cl_loss = gw_d + gw_p
        elif self.fusion_mode == 'VLM':
            p_aa_idx = self._unbatch(bg_prot.ndata['feats'].float()[:,:35].argmax(-1).unsqueeze(-1).float(), p_nodes, max_p, B).long().squeeze(-1)
            d_atom_idx = self._unbatch(bg_drug.ndata['atom'].float()[:,:71].argmax(-1).unsqueeze(-1).float(), d_nodes, max_d, B).long().squeeze(-1)
            q_fused, v_fused, _ = self.vlm(p_aa_idx, d_atom_idx, p_lens=p_nodes, d_lens=d_nodes, graph_feat=q_gnn, llm_feat=v_gnn)
            cl_loss = 0.0001 * (self.vlm.P_virt.norm(p=2) + self.vlm.D_virt.norm(p=2))
        elif self.fusion_mode == 'VLM-XA':
                                                                           
            p_aa_idx = self._unbatch(bg_prot.ndata['feats'].float()[:,:35].argmax(-1).unsqueeze(-1).float(), p_nodes, max_p, B).long().squeeze(-1)
            d_atom_idx = self._unbatch(bg_drug.ndata['atom'].float()[:,:71].argmax(-1).unsqueeze(-1).float(), d_nodes, max_d, B).long().squeeze(-1)
            q_enh, v_enh, _ = self.vlm(p_aa_idx, d_atom_idx, p_lens=p_nodes, d_lens=d_nodes, graph_feat=q_gnn, llm_feat=v_gnn)
            v_fused, _ = self.cross_attn_d(v_enh, v_llm, v_llm, key_padding_mask=d_pad_mask)
            q_fused, _ = self.cross_attn_p(q_enh, q_llm, q_llm, key_padding_mask=p_pad_mask)
            cl_loss = 0.0001 * (self.vlm.P_virt.norm(p=2) + self.vlm.D_virt.norm(p=2))
        elif self.fusion_mode == 'VLM-GW':
                                                             
            p_aa_idx = self._unbatch(bg_prot.ndata['feats'].float()[:,:35].argmax(-1).unsqueeze(-1).float(), p_nodes, max_p, B).long().squeeze(-1)
            d_atom_idx = self._unbatch(bg_drug.ndata['atom'].float()[:,:71].argmax(-1).unsqueeze(-1).float(), d_nodes, max_d, B).long().squeeze(-1)
            q_enh, v_enh, vlm_loss = self.vlm(p_aa_idx, d_atom_idx, p_lens=p_nodes, d_lens=d_nodes, graph_feat=q_gnn, llm_feat=v_gnn)
            v_fused, gw_d = self._gw_ota_align(v_enh, v_llm, d_nodes, d_lens, name='drug')
            q_fused, gw_p = self._gw_ota_align(q_enh, q_llm, p_nodes, p_lens, name='protein')
            cl_loss = vlm_loss + gw_d + gw_p
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

                                                                                
                                            
                                                                                
        v_fused = self.v_norm(v_fused)
        q_fused = self.q_norm(q_fused)
        out = self.ban(v_fused, q_fused, d_nodes, p_nodes)

                                                                                
                    
                                                                                
        preds = torch.sigmoid(self.fc(out)).squeeze(-1)

        return preds, cl_loss
