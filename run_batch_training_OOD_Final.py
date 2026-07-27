import os, csv, gc, itertools, torch, numpy as np, pandas as pd, sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'VLMNET'))
from unified_dataset_v2 import UnifiedDtiDatasetV2, unified_collate_fn_v2
import torch.nn as nn
from model import UniQCM2Net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_dir = 'D:/Paper2Data/testData4'

def set_all_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    

def get_data_paths(dataset, split):
    base = os.path.join(data_dir, dataset, split)
    if split == 'random':
        train_paths = [os.path.join(base, 'train', 'samples.csv')]
        valid_path  = os.path.join(base, 'valid', 'samples.csv')
        test_path   = os.path.join(base, 'test', 'samples.csv')
    else:
        target_train = os.path.join(base, 'target_train.csv')
        source_train = os.path.join(base, 'source_train.csv')
        train_paths  = [target_train]
        if os.path.exists(source_train): train_paths.append(source_train)
        valid_path = os.path.join(base, 'target_valid.csv')
        test_path  = os.path.join(base, 'target_test.csv')
    return train_paths, valid_path, test_path

def _normalize_columns(df):
    rename = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl == 'smiles': rename[col] = 'SMILES'
        elif cl == 'protein': rename[col] = 'Protein'
        elif cl in ('label', 'y'): rename[col] = 'Y'
    drop = [c for c in df.columns if c.lower().startswith('unnamed')]
    if rename: df = df.rename(columns=rename)
    if drop: df = df.drop(columns=drop)
    return df

def load_dataframes(train_paths, valid_path, test_path):
    train_dfs = [_normalize_columns(pd.read_csv(p)) for p in train_paths if os.path.exists(p)]
    if not train_dfs: raise ValueError("No train files found.")
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df  = _normalize_columns(pd.read_csv(valid_path)) if os.path.exists(valid_path) else None
    test_df = _normalize_columns(pd.read_csv(test_path)) if os.path.exists(test_path) else None
    return train_df, val_df, test_df

def test_epoch(model, dataloader):
    y_pred, y_label = [], []
    model.eval()
    with torch.no_grad():
        for bg_drug, bg_prot, d_feat, p_feat, labels, d_lens, p_lens in dataloader:
            bg_drug, bg_prot = bg_drug.to(device), bg_prot.to(device)
            d_feat, p_feat = d_feat.to(device), p_feat.to(device)
            d_lens, p_lens = d_lens.to(device), p_lens.to(device)
            preds, _ = model(bg_drug, bg_prot, d_feat, p_feat, d_lens, p_lens)
            y_pred.extend(preds.cpu().numpy())
            y_label.extend(labels.numpy())
            del bg_drug, bg_prot, d_feat, p_feat, labels, d_lens, p_lens, preds, _
    y_pred = np.array(y_pred).flatten()
    y_label = np.array(y_label)
    y_pred_c = np.round(y_pred).astype(int)
    try:
        if len(np.unique(y_label)) < 2: return 0.5, 0.5, 0.5, 0.0, 0.0, 0.5
        AUC = roc_auc_score(y_label, y_pred)
        precision, recall, _ = precision_recall_curve(y_label, y_pred)
        PRAUC = auc(recall, precision)
        AUPRC = average_precision_score(y_label, y_pred)
        prec = precision_score(y_label, y_pred_c, zero_division=0)
        rec  = recall_score(y_label, y_pred_c, zero_division=0)
        acc  = accuracy_score(y_label, y_pred_c)
    except:
        AUC = PRAUC = AUPRC = prec = rec = acc = 0.0
    return float(AUC), float(PRAUC), float(AUPRC), float(prec), float(rec), float(acc)

def run_training():
    datasets = ['Patent', 'celegans', 'human', 'Article']
    ood_splits = ['cluster', 'cold_smiles', 'cold_protein', 'cold_both']
    random_split = ['random']
    all_splits = ood_splits + random_split
    
    phase1 = list(itertools.product(datasets, ood_splits, [42]))
    phase2 = list(itertools.product(datasets, random_split, [42]))
    phase3 = list(itertools.product(datasets, all_splits, [1234, 2025, 12345, 666]))
    
    schedule = phase1 + phase2 + phase3

    bs, h, k = 16, 128, 5
    accum_steps = 8                                   
    lr, wd, max_epochs = 5e-5, 1e-5, 30

    base_out = 'E:/LINUX_code/solve_OOD_to24G_LocalFinal/output/batch_results_OOD_Final'
    os.makedirs(base_out, exist_ok=True)
    out_csv = os.path.join(base_out, 'results.csv')

    completed = set()
    if os.path.exists(out_csv):
        with open(out_csv, 'r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                completed.add((row['Dataset'], row['Split'], int(row['Seed'])))
    else:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['Dataset','Split','Seed','BestEpoch','AUC_Test','PRAUC_Test',
                                     'AUPRC_Test','Precision_Test','Recall_Test','Accuracy_Test',
                                     'F1_Test','ModelPath'])

    total = len(schedule)
    print(f"Resume: {len(completed)}/{total} done, {total-len(completed)} pending")

    for dataset, split, seed in schedule:
        if (dataset, split, seed) in completed:
            print(f"SKIP {dataset}/{split}/seed={seed} (done)")
            continue

        print(f"\n=> {dataset}/{split}/seed={seed}", flush=True)
        set_all_seeds(seed)

        tp, vp, tep = get_data_paths(dataset, split)
        tr_df, vl_df, ts_df = load_dataframes(tp, vp, tep)

        smiles_npy = os.path.join(data_dir, dataset, 'Modalities', 'smiles_all.npy')
        p_lm_pkl   = os.path.join(data_dir, dataset, 'Modalities', 'p_LM.pkl')
        d_lm_pkl   = os.path.join(data_dir, dataset, 'Modalities', 'd_LM.pkl')
        s2p_json   = os.path.join(data_dir, dataset, 'sequence_to_pdb.json')
        cg_dir     = os.path.join(data_dir, dataset, 'CompoundGraph')
        pg_dir     = os.path.join(data_dir, dataset, 'ProteinGraph', 'package_bin')

        train_ds = UnifiedDtiDatasetV2(tr_df, smiles_npy, p_lm_pkl, d_lm_pkl, s2p_json, cg_dir, pg_dir, load_llm=True)
        val_ds   = UnifiedDtiDatasetV2(vl_df, smiles_npy, p_lm_pkl, d_lm_pkl, s2p_json, cg_dir, pg_dir, load_llm=True)
        test_ds  = UnifiedDtiDatasetV2(ts_df, smiles_npy, p_lm_pkl, d_lm_pkl, s2p_json, cg_dir, pg_dir, load_llm=True)

        tl  = DataLoader(train_ds, batch_size=bs, shuffle=True,  collate_fn=unified_collate_fn_v2, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=False)
        vl2 = DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=unified_collate_fn_v2, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=False)
        tsl = DataLoader(test_ds,  batch_size=bs, shuffle=False, collate_fn=unified_collate_fn_v2, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=False)

        current_vlm_ablation = 'no_attn'
        model = UniQCM2Net(gnn_hidden=h, fusion_mode='VLM-XA', vlm_ablation=current_vlm_ablation).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10, min_lr=1e-6)
        warmup_epochs = 5; base_lr = lr
        bce = torch.nn.BCELoss()   
        best_val, best_ep = 0.0, -1
        best_metrics = None
        model_dir = os.path.join(base_out, f'models/{dataset}/{split}')
        weight_path = os.path.join(model_dir, f'seed_{seed}.pth')

        opt.zero_grad(set_to_none=True)
        for ep in range(1, max_epochs + 1):
            if ep <= warmup_epochs:
                for g in opt.param_groups: g['lr'] = base_lr * ep / warmup_epochs
            model.train(); ep_loss = 0.0
            pbar = tqdm(tl, desc=f"Epoch {ep:2d}/{max_epochs}", leave=False)
            for i, (bg_d, bg_p, df_, pf_, labels, dl_, pl_) in enumerate(pbar):
                bg_d, bg_p = bg_d.to(device), bg_p.to(device)
                df_, pf_ = df_.to(device), pf_.to(device)
                labels = labels.to(device).float()
                dl_, pl_ = dl_.to(device), pl_.to(device)
                
                preds, cl = model(bg_d, bg_p, df_, pf_, dl_, pl_)
                preds = torch.nan_to_num(torch.clamp(preds, 1e-7, 1 - 1e-7))
                
                loss = (bce(preds, labels) + 1e-04 * cl) / accum_steps
                loss.backward()
                
                ep_loss += loss.item() * accum_steps
                
                if (i + 1) % accum_steps == 0 or (i + 1) == len(tl):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                
                pbar.set_postfix({"Loss": f"{ep_loss/(i+1):.4f}"})
                del bg_d, bg_p, df_, pf_, labels, dl_, pl_, preds, cl, loss

            val_auc, val_prauc, val_auprc, val_pre, val_rec, val_acc = test_epoch(model, vl2)
            if ep > warmup_epochs: scheduler.step(val_auc)
            t_auc, t_prauc, t_auprc, t_pre, t_rec, t_acc = test_epoch(model, tsl)
            val_f1 = (2*val_pre*val_rec)/(val_pre+val_rec+1e-5) if (val_pre+val_rec)>0 else 0.0
            print(f"E{ep:2d} Loss={ep_loss/len(tl):.4f} Val={val_auc:.4f} F1={val_f1:.4f}")

            if val_auc > best_val:
                best_val, best_ep = val_auc, ep
                t_f1 = (2*t_pre*t_rec)/(t_pre+t_rec+1e-5) if (t_pre+t_rec)>0 else 0.0
                best_metrics = {'auc':t_auc,'prauc':t_prauc,'auprc':t_auprc,
                                'pre':t_pre,'rec':t_rec,'acc':t_acc,'f1':t_f1}
                os.makedirs(model_dir, exist_ok=True)
                torch.save(model.state_dict(), weight_path)
                print(f">>> Best! Test AUC={t_auc:.4f} Saved: {weight_path}")

        if best_metrics is None:
            best_metrics = {'auc':0,'prauc':0,'auprc':0,'pre':0,'rec':0,'acc':0,'f1':0}

        with open(out_csv, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([dataset, split, seed, best_ep,
                f"{best_metrics['auc']:.4f}", f"{best_metrics['prauc']:.4f}",
                f"{best_metrics['auprc']:.4f}", f"{best_metrics['pre']:.4f}",
                f"{best_metrics['rec']:.4f}", f"{best_metrics['acc']:.4f}",
                f"{best_metrics['f1']:.4f}", weight_path])
        print(f"DONE {dataset}/{split}/seed={seed}: Best Test={best_metrics['auc']:.4f}")

        del model, opt, train_ds, val_ds, test_ds, tl, vl2, tsl
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()                                                
        mem_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU memory after cleanup: {mem_mb:.0f} MiB")

if __name__ == '__main__':
    run_training()
