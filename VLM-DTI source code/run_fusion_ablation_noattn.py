import os, sys, csv, gc, argparse, traceback
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings; warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'VLMNET'))
from unified_dataset_v2 import UnifiedDtiDatasetV2, unified_collate_fn_v2
from model import UniQCM2Net

                                                            
DATA_DIR = 'D:/Paper2Data/testData4'
DATASET, SPLIT = 'Patent', 'cold_protein'
ALL_SEEDS = [666, 12345, 2025, 1234, 42]
BS, H, ACCUM = 16, 128, 8                                       
LR, WD, MAX_EPOCHS, WARMUP = 5e-5, 1e-5, 30, 5
LAMBDA_EXT = 1e-4                                                                 
NUM_WORKERS = 0                                          

                                                  

CONFIGS = {
    'Concat':     ('Concat',     'no_attn', None),
    'Cross-Attn': ('Cross-Attn', 'no_attn', None),
    'VLM-TA':     ('VLM-XA',     'no_attn', None),             
    'no_P':       ('VLM-XA',     'no_attn', 'P'),        
    'no_D':       ('VLM-XA',     'no_attn', 'D'),        
}
DEFAULT_CONFIGS = ['VLM-TA', 'Cross-Attn', 'Concat']

BASE_OUT = os.path.join(HERE, 'output', 'batch_results_fusion_noattn')
OUT_CSV = os.path.join(BASE_OUT, 'results.csv')
os.makedirs(BASE_OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', type=str, default=None)
ap.add_argument('--configs', type=str, default=None)
ap.add_argument('--summary', action='store_true')
a = ap.parse_args()
SEEDS = [int(x) for x in a.seeds.split(',')] if a.seeds else ALL_SEEDS
NAMES = a.configs.split(',') if a.configs else DEFAULT_CONFIGS
for n in NAMES:
    if n not in CONFIGS:
        sys.exit(f'未知 config: {n}. 可选: {list(CONFIGS)}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FIELDS = ['Config', 'Seed', 'FusionMode', 'VlmAblation', 'ZeroFreeze',
          'BestEpoch', 'AUC_Test', 'AUPRC_Test', 'F1_Test']

def summarize():
    if not os.path.exists(OUT_CSV):
        print('尚无结果'); return
    d = pd.read_csv(OUT_CSV)
    for c in ['AUC_Test', 'AUPRC_Test', 'F1_Test']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    order = [n for n in CONFIGS if n in set(d.Config)]
    g = d.groupby('Config')[['AUC_Test', 'AUPRC_Test', 'F1_Test']].agg(['mean', 'std', 'size'])
    g = g.reindex(order)
    print(f'\n=== 融合消融（{DATASET}/{SPLIT}, 部署结构 no_attn）===')
    print(g.round(5).to_string())
    if {'VLM-TA', 'Cross-Attn'} <= set(d.Config):
        from scipy import stats
        piv = d.pivot_table(index='Seed', columns='Config', values='AUC_Test')
        print('\n=== 与 VLM-TA 的配对检验（按种子）===')
        for other in [c for c in order if c != 'VLM-TA']:
            if other in piv and 'VLM-TA' in piv:
                dd = piv['VLM-TA'] - piv[other]
                t, p = stats.ttest_rel(piv['VLM-TA'], piv[other])
                print(f'  VLM-TA - {other:11s}: Δ={dd.mean():+.4f} | p={p:.4f} | '
                      f'VLM-TA 更好 {(dd > 0).sum()}/{len(dd)}')
        print('\n参考（因子实验同数据集/切分/超参, 5 种子）: '
              'table_mha(=VLM-TA) 0.7647±0.0122 | no_table(=Cross-Attn) 0.7467±0.0202')

if a.summary:
    summarize(); sys.exit(0)

def set_all_seeds(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    os.environ['PYTHONHASHSEED'] = str(s)

def _nc(df):
    r = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == 'smiles': r[c] = 'SMILES'
        elif cl == 'protein': r[c] = 'Protein'
        elif cl in ('label', 'y'): r[c] = 'Y'
    drop = [c for c in df.columns if c.lower().startswith('unnamed')]
    if r: df = df.rename(columns=r)
    if drop: df = df.drop(columns=drop)
    return df

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); yp, yt = [], []
    for bg_d, bg_p, df_, pf_, lab, dl_, pl_ in loader:
        p, _ = model(bg_d.to(device), bg_p.to(device), df_.to(device),
                     pf_.to(device), dl_.to(device), pl_.to(device))
        yp.extend(p.detach().cpu().numpy().ravel()); yt.extend(np.asarray(lab).ravel())
    yp, yt = np.asarray(yp, float), np.asarray(yt, int)
    pred = (yp >= 0.5).astype(int)
    tp = ((pred == 1) & (yt == 1)).sum(); fp = ((pred == 1) & (yt == 0)).sum()
    fn = ((pred == 0) & (yt == 1)).sum()
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
    return (roc_auc_score(yt, yp), average_precision_score(yt, yp),
            2 * pr * rc / (pr + rc + 1e-9))

def main():
    base = f'{DATA_DIR}/{DATASET}/{SPLIT}'
    for p in [f'{base}/target_train.csv', f'{DATA_DIR}/{DATASET}/Modalities/p_LM.pkl']:
        if not os.path.exists(p):
            sys.exit(f'!! 数据缺失: {p}\n   请检查 DATA_DIR = {DATA_DIR}')

    done = set()
    if os.path.exists(OUT_CSV):
        
        with open(OUT_CSV, 'r', newline='', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                done.add((r['Config'], int(r['Seed'])))
    else:
        with open(OUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(FIELDS)

    tr = [_nc(pd.read_csv(f'{base}/target_train.csv'))]
    if os.path.exists(f'{base}/source_train.csv'):
        tr.append(_nc(pd.read_csv(f'{base}/source_train.csv')))
    tr_df = pd.concat(tr, ignore_index=True)
    vl_df = _nc(pd.read_csv(f'{base}/target_valid.csv'))
    ts_df = _nc(pd.read_csv(f'{base}/target_test.csv'))
    m = f'{DATA_DIR}/{DATASET}/Modalities'
    ds_args = (f'{m}/smiles_all.npy', f'{m}/p_LM.pkl', f'{m}/d_LM.pkl',
               f'{DATA_DIR}/{DATASET}/sequence_to_pdb.json',
               f'{DATA_DIR}/{DATASET}/CompoundGraph',
               f'{DATA_DIR}/{DATASET}/ProteinGraph/package_bin')
    print(f'载入数据 {DATASET}/{SPLIT}: Train={len(tr_df)} Val={len(vl_df)} Test={len(ts_df)}',
          flush=True)
    train_ds = UnifiedDtiDatasetV2(tr_df, *ds_args, load_llm=True)
    val_ds   = UnifiedDtiDatasetV2(vl_df, *ds_args, load_llm=True)
    test_ds  = UnifiedDtiDatasetV2(ts_df, *ds_args, load_llm=True)
    LK = dict(collate_fn=unified_collate_fn_v2, num_workers=NUM_WORKERS, pin_memory=True)
    print(f'{len(NAMES)} 臂 x {len(SEEDS)} 种子 = {len(NAMES)*len(SEEDS)} run | '
          f'已完成 {len(done)} | 臂: {NAMES}', flush=True)

    for seed in SEEDS:
        for name in NAMES:
            fmode, vabl, zf = CONFIGS[name]
            if (name, seed) in done:
                print(f'SKIP {name}/seed={seed}'); continue
            print(f'\n{"="*66}\n[{name}] fusion={fmode} vlm_ablation={vabl} '
                  f'zero_freeze={zf} seed={seed}\n{"="*66}', flush=True)
            model = opt = None
            try:
                set_all_seeds(seed)
                tl  = DataLoader(train_ds, batch_size=BS, shuffle=True,  **LK)
                vl2 = DataLoader(val_ds,   batch_size=BS, shuffle=False, **LK)
                tsl = DataLoader(test_ds,  batch_size=BS, shuffle=False, **LK)

                model = UniQCM2Net(gnn_hidden=H, fusion_mode=fmode,
                                   vlm_ablation=vabl).to(device)
                                                  
                if zf is not None:
                    tbl = model.vlm.P_virt if zf == 'P' else model.vlm.D_virt
                    with torch.no_grad():
                        tbl.zero_()
                    tbl.requires_grad_(False)

                opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                       lr=LR, weight_decay=WD)
                sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode='max', factor=0.5, patience=10, min_lr=1e-6)
                bce = torch.nn.BCELoss()
                best_val, best_ep, best = 0.0, -1, None
                wdir = os.path.join(BASE_OUT, 'models', name); os.makedirs(wdir, exist_ok=True)
                wpath = os.path.join(wdir, f'seed_{seed}.pth')
                opt.zero_grad(set_to_none=True)

                for ep in range(1, MAX_EPOCHS + 1):
                    if ep <= WARMUP:
                        for g_ in opt.param_groups: g_['lr'] = LR * ep / WARMUP
                    model.train(); tot = 0.0
                    for i, (bg_d, bg_p, df_, pf_, lab, dl_, pl_) in enumerate(tl):
                        preds, cl = model(bg_d.to(device), bg_p.to(device), df_.to(device),
                                          pf_.to(device), dl_.to(device), pl_.to(device))
                        preds = torch.nan_to_num(torch.clamp(preds, 1e-7, 1 - 1e-7))
                        loss = (bce(preds, lab.to(device).float()) + LAMBDA_EXT * cl) / ACCUM
                        loss.backward(); tot += loss.item() * ACCUM
                        if (i + 1) % ACCUM == 0 or (i + 1) == len(tl):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                            opt.step(); opt.zero_grad(set_to_none=True)
                    v_auc, _, _ = evaluate(model, vl2)
                    if ep > WARMUP: sch.step(v_auc)
                    t = evaluate(model, tsl)
                    if v_auc > best_val:
                        best_val, best_ep, best = v_auc, ep, t
                        torch.save(model.state_dict(), wpath)
                    print(f'E{ep:2d} Loss={tot/len(tl):.4f} Val={v_auc:.4f} Test={t[0]:.4f}',
                          flush=True)

                with open(OUT_CSV, 'a', newline='', encoding='utf-8-sig') as f:
                    csv.writer(f).writerow([name, seed, fmode, vabl, zf or '', best_ep,
                                            f'{best[0]:.5f}', f'{best[1]:.5f}', f'{best[2]:.5f}'])
                print(f'DONE [{name}] seed={seed}: E{best_ep} AUC={best[0]:.5f} '
                      f'AUPRC={best[1]:.5f} F1={best[2]:.5f}', flush=True)
            except Exception:
                traceback.print_exc()
            finally:
                try: del model, opt, tl, vl2, tsl
                except Exception: pass
                gc.collect(); torch.cuda.empty_cache()

    print('\n=== COMPLETE ===')
    summarize()

if __name__ == '__main__':
    main()
