import os
import torch
import numpy as np
import pandas as pd
import dgl
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

_global_p_lm = None
_global_d_lm = None

class UnifiedDtiDatasetV2(Dataset):
    def __init__(self, csv_path_or_df, smiles_npy, p_lm_pkl, d_lm_pkl, seq_to_pdb_json, comp_graph_dir, prot_graph_dir, load_llm=True):
        self.load_llm = load_llm
        if isinstance(csv_path_or_df, pd.DataFrame):
            df = csv_path_or_df.reset_index(drop=True)
        else:
            df = pd.read_csv(csv_path_or_df)
            
                                                                          
        
        self.smiles_col = [str(x).strip() for x in (df['SMILES'].tolist() if 'SMILES' in df.columns else df['smiles'].
  tolist())]
        self.protein_col = [str(x).strip() for x in (df['Protein'].tolist() if 'Protein' in df.columns else df['protein'].
  tolist())]
        self.label_col = df['Y'].tolist() if 'Y' in df.columns else df['label'].tolist()
        self.length = len(df)
        
                                                                                  
        if os.path.exists(smiles_npy):
            self.smiles_all = np.load(smiles_npy, allow_pickle=True)
            self.smiles_dict = {str(s): i for i, s in enumerate(self.smiles_all)}
        else:
            self.smiles_dict = {}
            
        if self.load_llm:
            import pickle
            
            with open(p_lm_pkl, 'rb') as f:
                raw_p = pickle.load(f)
            self.p_lm = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw_p.items()}

            with open(d_lm_pkl, 'rb') as f:
                raw_d = pickle.load(f)
            self.d_lm = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw_d.items()}
            
            
                      
                                             
                                             
        
        import json
        with open(seq_to_pdb_json, 'r') as f:
            raw_s2p = json.load(f)
            self.seq_to_pdb = {}
                                                                  
            for pdb, seqs in raw_s2p.items():
                clean_pdb = str(pdb).strip()
                if isinstance(seqs, list):
                    for s in seqs:
                        self.seq_to_pdb[str(s).strip()] = clean_pdb
                else:
                    self.seq_to_pdb[str(seqs).strip()] = clean_pdb
            
        self.comp_graph_dir = comp_graph_dir
        self.prot_graph_dir = prot_graph_dir
        
                                                             
        self._comp_cache = {}
        self._prot_cache = {}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        smiles = self.smiles_col[idx]
        protein = self.protein_col[idx]
        label = self.label_col[idx]

                                             
        s_idx = self.smiles_dict.get(smiles, -1)
        if s_idx not in self._comp_cache:
            try:
                bg_drug, _ = dgl.load_graphs(os.path.join(self.comp_graph_dir, f"{s_idx}.bin"))
                self._comp_cache[s_idx] = bg_drug[0]
            except Exception as e:
                print(f'图加载失败:{s_idx},错误原因:{e}')
                bg_drug = dgl.graph(([0], [0]))
                bg_drug.ndata['atom'] = torch.zeros(1, 120)
                self._comp_cache[s_idx] = bg_drug
        bg_drug = self._comp_cache[s_idx]
            
                                            
        pdb_id = self.seq_to_pdb.get(protein, '')
        if pdb_id not in self._prot_cache:
            try:
                bg_prot, _ = dgl.load_graphs(os.path.join(self.prot_graph_dir, f"{pdb_id}.bin"))
                self._prot_cache[pdb_id] = bg_prot[0]
            except Exception as e:
                                   
                bg_prot = dgl.graph(([0], [0]))

                                           
                bg_prot.ndata['feats'] = torch.zeros(1, 41, dtype=torch.float64)
                bg_prot.ndata['_ID'] = torch.zeros(1, dtype=torch.int64)
                bg_prot.ndata['lap_pos_enc'] = torch.zeros(1, 8, dtype=torch.float32)

                                                            
                bg_prot.edata['feats'] = torch.zeros(1, 5, dtype=torch.float64)
                bg_prot.edata['_ID'] = torch.zeros(1, dtype=torch.int64)

                self._prot_cache[pdb_id] = bg_prot
                
        bg_prot = self._prot_cache[pdb_id]

                                                        
        if self.load_llm:
            d_feat = self.d_lm.get(smiles, torch.zeros((1, 768), dtype=torch.float32))
            p_feat = self.p_lm.get(protein, torch.zeros((1, 320), dtype=torch.float32))
        else:
            d_feat = torch.zeros((1, 768), dtype=torch.float32)
            p_feat = torch.zeros((1, 320), dtype=torch.float32)
        
        return bg_drug, bg_prot, d_feat, p_feat, label
        
        
def unified_collate_fn_v2(batch):
    bg_drugs, bg_prots, d_feats, p_feats, labels = zip(*batch)

    bg_drug_batch = dgl.batch(bg_drugs)
    bg_prot_batch = dgl.batch(bg_prots)

                                        
                     
    if 'atom' in bg_drug_batch.ndata:
        bg_drug_batch.ndata['atom'] = torch.nan_to_num(bg_drug_batch.ndata['atom'], nan=0.0)
    if 'feats' in bg_prot_batch.ndata:
        bg_prot_batch.ndata['feats'] = torch.nan_to_num(bg_prot_batch.ndata['feats'], nan=0.0)

                    
    if 'bond' in bg_drug_batch.edata:
        bg_drug_batch.edata['bond'] = torch.nan_to_num(bg_drug_batch.edata['bond'], nan=0.0)
    if 'feats' in bg_prot_batch.edata:
        bg_prot_batch.edata['feats'] = torch.nan_to_num(bg_prot_batch.edata['feats'], nan=0.0)
                                                              

    d_lens = torch.tensor([x.size(0) for x in d_feats], dtype=torch.long)
    p_lens = torch.tensor([x.size(0) for x in p_feats], dtype=torch.long)

    d_feat_pad = pad_sequence(d_feats, batch_first=True)
    p_feat_pad = pad_sequence(p_feats, batch_first=True)

                                 
    d_feat_pad = torch.nan_to_num(d_feat_pad, nan=0.0)
    p_feat_pad = torch.nan_to_num(p_feat_pad, nan=0.0)

    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    return bg_drug_batch, bg_prot_batch, d_feat_pad, p_feat_pad, labels_tensor, d_lens, p_lens

