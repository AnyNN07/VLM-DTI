import torch
import torch.nn as nn
import torch.nn.functional as F

class VirtualLearnableMapping(nn.Module):
    def __init__(self, num_aa=35, num_atoms=71, hidden_dim=128,
                 ablation='full'):
        super().__init__()
        self.ablation = ablation
        self.P_virt = nn.Parameter(torch.randn(num_aa, hidden_dim) * 0.02)
        self.D_virt = nn.Parameter(torch.randn(num_atoms, hidden_dim) * 0.02)
        self.gate_p = nn.Parameter(torch.tensor(0.5))
        self.gate_d = nn.Parameter(torch.tensor(0.5))

    def forward(self, aa_indices, atom_indices, p_lens=None, d_lens=None, graph_feat=None, llm_feat=None):
        B, N_p = aa_indices.shape
        _, N_d = atom_indices.shape

        if p_lens is not None:
            p_mask = torch.arange(N_p, device=aa_indices.device).expand(B, N_p) >= p_lens.unsqueeze(1)
        else:
            p_mask = torch.zeros(B, N_p, dtype=torch.bool, device=aa_indices.device)
            
        if d_lens is not None:
            d_mask = torch.arange(N_d, device=atom_indices.device).expand(B, N_d) >= d_lens.unsqueeze(1)
        else:
            d_mask = torch.zeros(B, N_d, dtype=torch.bool, device=atom_indices.device)

                                                            
        p_v = self.P_virt[aa_indices] if self.ablation != 'no_P' \
              else torch.zeros(B, N_p, self.P_virt.shape[1], device=aa_indices.device)
        d_v = self.D_virt[atom_indices] if self.ablation != 'no_D' \
              else torch.zeros(B, N_d, self.D_virt.shape[1], device=atom_indices.device)

                                                  
        p_v = p_v.masked_fill(p_mask.unsqueeze(-1), 0.0)
        d_v = d_v.masked_fill(d_mask.unsqueeze(-1), 0.0)

                                                                            
        p_base = graph_feat + p_v if graph_feat is not None else p_v
        d_base = llm_feat   + d_v if llm_feat is not None else d_v

        if self.ablation == 'no_attn':
                                                                                       
            p_out, d_out = p_base, d_base
        else:
                                                              
            scale = self.P_virt.shape[1] ** 0.5
            attn   = torch.bmm(p_v, d_v.transpose(1, 2)) / scale
            
            attn.masked_fill_(d_mask.unsqueeze(1), -1e4)
            attn_p = F.softmax(attn, dim=2)
            cross_p = torch.bmm(attn_p, d_v)
            
            attn_t = attn.transpose(1, 2)
            attn_t.masked_fill_(p_mask.unsqueeze(1), -1e4)
            attn_d = F.softmax(attn_t, dim=2)
            cross_d = torch.bmm(attn_d, p_v)

            if self.ablation == 'fixed_gate':
                                                            
                p_out = 0.5 * p_base + 0.5 * cross_p
                d_out = 0.5 * d_base + 0.5 * cross_d
            else:
                                                                 
                gp = torch.sigmoid(self.gate_p)
                gd = torch.sigmoid(self.gate_d)
                p_out = gp * p_base + (1 - gp) * cross_p
                d_out = gd * d_base + (1 - gd) * cross_d

        vlm_loss = 0.0001 * (self.P_virt.norm(p=2) + self.D_virt.norm(p=2))
        return p_out, d_out, vlm_loss
