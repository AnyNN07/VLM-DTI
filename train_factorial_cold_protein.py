from __future__ import annotations

import argparse
import csv
import copy
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)
from torch.utils.data import DataLoader

SEEDS = ( 42,666,1234,2025,12345)
DATASETS = ("human", "Patent", "celegans", "Article")
VARIANTS = ("no_table_no_mha","no_mha","no_table","table_mha" )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--code-root", type=Path, default=Path(r"E:\LINUX_code\solve_OOD_to24G_LocalFinal"))
    p.add_argument("--data-root", type=Path, default=Path(r"D:\Paper2Data\testData4"))
    p.add_argument("--output-dir", type=Path, default=Path(r"E:\LINUX_code\solve_OOD_to24G_LocalFinal\output\factorial_shard"))
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum-steps", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--test-each-epoch", action="store_true",
                   help="Diagnostic only; formal protocol evaluates the best checkpoint once.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force", action="store_true", help="Rerun completed cells; creates a unique suffixed run directory.")
    return p.parse_args()

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "smiles":
            rename[c] = "SMILES"
        elif cl == "protein":
            rename[c] = "Protein"
        elif cl in ("label", "y"):
            rename[c] = "Y"
    df = df.rename(columns=rename)
    return df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")

def load_frames(data_root: Path, dataset: str):
    base = data_root / dataset / "cold_protein"
    train_files = [base / "target_train.csv"]
    if (base / "source_train.csv").exists():
        train_files.append(base / "source_train.csv")
    train = pd.concat([normalize_columns(pd.read_csv(p)) for p in train_files], ignore_index=True)
    valid = normalize_columns(pd.read_csv(base / "target_valid.csv"))
    test = normalize_columns(pd.read_csv(base / "target_test.csv"))
    return train, valid, test

def dataset_kwargs(data_root: Path, dataset: str) -> dict:
    root = data_root / dataset
    return dict(
        smiles_npy=str(root / "Modalities" / "smiles_all.npy"),
        p_lm_pkl=str(root / "Modalities" / "p_LM.pkl"),
        d_lm_pkl=str(root / "Modalities" / "d_LM.pkl"),
        seq_to_pdb_json=str(root / "sequence_to_pdb.json"),
        comp_graph_dir=str(root / "CompoundGraph"),
        prot_graph_dir=str(root / "ProteinGraph" / "package_bin"),
        load_llm=True,
    )

def dataset_view(base, frame: pd.DataFrame):
                                                                                             
    view = copy.copy(base)
    f = frame.reset_index(drop=True)
    view.smiles_col = [str(x).strip() for x in f["SMILES"].tolist()]
    view.protein_col = [str(x).strip() for x in f["Protein"].tolist()]
    view.label_col = f["Y"].tolist()
    view.length = len(f)
    return view

def model_spec(variant: str):
    if variant == "table_mha":
        return "VLM-XA", False
    if variant == "no_table":
        return "Cross-Attn", False
    if variant == "no_mha":
        return "VLM", False
    if variant == "no_table_no_mha":
        return "VLM", True
    raise ValueError(variant)

def tensor_sha256(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(t.dtype).encode("ascii"))
        h.update(np.asarray(t.shape, dtype=np.int64).tobytes())
        h.update(t.numpy().tobytes())
    return h.hexdigest()

def build_model(UniQCM2Net, variant: str, canonical_state: dict, seed: int):
    fusion_mode, zero_tables = model_spec(variant)
    set_seeds(seed)
    model = UniQCM2Net(gnn_hidden=128, fusion_mode=fusion_mode, vlm_ablation="no_attn")
    own = model.state_dict()
    copied = {}
    for name, value in canonical_state.items():
        if name in own and own[name].shape == value.shape:
            copied[name] = value
    incompat = model.load_state_dict(copied, strict=False)
    if zero_tables:
        with torch.no_grad():
            model.vlm.P_virt.zero_()
            model.vlm.D_virt.zero_()
        model.vlm.P_virt.requires_grad_(False)
        model.vlm.D_virt.requires_grad_(False)
    meta = {
        "fusion_mode": fusion_mode,
        "vlm_ablation": "no_attn",
        "zero_and_freeze_tables": zero_tables,
        "common_tensors_copied": len(copied),
        "missing_after_common_copy": list(incompat.missing_keys),
        "unexpected_after_common_copy": list(incompat.unexpected_keys),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "initial_state_sha256": tensor_sha256(model.state_dict()),
    }
    return model, meta

def metrics(labels, scores) -> dict:
    y = np.asarray(labels, dtype=int).reshape(-1)
    p = np.asarray(scores, dtype=float).reshape(-1)
    hard = np.round(p).astype(int)
    pr, rc, _ = precision_recall_curve(y, p)
    precision = precision_score(y, hard, zero_division=0)
    recall = recall_score(y, hard, zero_division=0)
    return {
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else 0.5,
        "prauc": float(auc(rc, pr)),
        "auprc": float(average_precision_score(y, p)),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy_score(y, hard)),
        "f1": float(2 * precision * recall / (precision + recall + 1e-5)) if precision + recall else 0.0,
    }

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    ys, ps = [], []
    for bg_d, bg_p, d_feat, p_feat, labels, d_len, p_len in loader:
        pred, _ = model(
            bg_d.to(device), bg_p.to(device), d_feat.to(device), p_feat.to(device),
            d_len.to(device), p_len.to(device)
        )
        ys.extend(labels.numpy().reshape(-1).tolist())
        ps.extend(pred.detach().cpu().numpy().reshape(-1).tolist())
    return metrics(ys, ps)

def write_json_new(path: Path, obj) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    i = 2
    while base.with_name(base.name + f"__rerun{i}").exists():
        i += 1
    return base.with_name(base.name + f"__rerun{i}")

def run_cell(args, dataset, seed, variant, UnifiedDtiDatasetV2, collate, UniQCM2Net, device):
    cell_base = args.output_dir / "runs" / dataset / "cold_protein" / f"seed_{seed}" / variant
    completed = cell_base / "completed.json"
    if completed.exists() and not args.force:
        print(f"SKIP completed {dataset}/seed={seed}/{variant}", flush=True)
        return
                                                                                           
    cell = unique_dir(cell_base) if (args.force or cell_base.exists()) else cell_base
    cell.mkdir(parents=True, exist_ok=True)

    set_seeds(seed)
    canonical = UniQCM2Net(gnn_hidden=128, fusion_mode="VLM-XA", vlm_ablation="no_attn")
    canonical_state = {k: v.detach().cpu().clone() for k, v in canonical.state_dict().items()}
    canonical_sha = tensor_sha256(canonical_state)
    del canonical
    model, model_meta = build_model(UniQCM2Net, variant, canonical_state, seed)
    model_meta["canonical_initial_state_sha256"] = canonical_sha
    model_meta["dataset"] = dataset
    model_meta["seed"] = seed
    model_meta["variant"] = variant
    write_json_new(cell / "model_initialization.json", model_meta)
    model.to(device)

    train_df, val_df, test_df = load_frames(args.data_root, dataset)
    kw = dataset_kwargs(args.data_root, dataset)
    train_ds = UnifiedDtiDatasetV2(train_df, **kw)
    val_ds = dataset_view(train_ds, val_df)
    test_ds = dataset_view(train_ds, test_df)

                                                                                     
    set_seeds(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    common_loader = dict(batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers,
                         pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **common_loader)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader)
    test_loader = DataLoader(test_ds, shuffle=False, **common_loader)

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=5e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10, min_lr=1e-6)
    bce = torch.nn.BCELoss()
    epoch_csv = cell / "epoch_metrics.csv"
    with epoch_csv.open("x", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "lr", "val_auc", "val_prauc", "val_auprc", "val_precision",
            "val_recall", "val_accuracy", "val_f1", "test_auc", "test_prauc", "test_auprc",
            "test_precision", "test_recall", "test_accuracy", "test_f1", "elapsed_seconds"
        ])

    best_val, best_epoch, best_test = -float("inf"), -1, None
    best_path = cell / "best_state.pth"
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if epoch <= 5:
            for group in opt.param_groups:
                group["lr"] = 5e-5 * epoch / 5
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for i, (bg_d, bg_p, d_feat, p_feat, labels, d_len, p_len) in enumerate(train_loader):
            pred, cl = model(
                bg_d.to(device), bg_p.to(device), d_feat.to(device), p_feat.to(device),
                d_len.to(device), p_len.to(device)
            )
            pred = torch.nan_to_num(torch.clamp(pred, 1e-7, 1 - 1e-7))
            loss = (bce(pred, labels.to(device).float()) + 1e-4 * cl) / args.accum_steps
            loss.backward()
            running += float(loss.detach().cpu()) * args.accum_steps
            if (i + 1) % args.accum_steps == 0 or i + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

        val = evaluate(model, val_loader, device)
        if epoch > 5:
            scheduler.step(val["auc"])
        test = (evaluate(model, test_loader, device) if args.test_each_epoch else
                {k: "" for k in ("auc", "prauc", "auprc", "precision", "recall", "accuracy", "f1")})
        elapsed = time.time() - started
        row = [epoch, running / max(1, len(train_loader)), opt.param_groups[0]["lr"]]
        row += [val[k] for k in ("auc", "prauc", "auprc", "precision", "recall", "accuracy", "f1")]
        row += [test[k] for k in ("auc", "prauc", "auprc", "precision", "recall", "accuracy", "f1")]
        row += [elapsed]
        with epoch_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        print(f"{dataset} seed={seed} {variant} epoch={epoch}/{args.epochs} "
              f"val_auc={val['auc']:.6f} test_auc={test['auc']} elapsed={elapsed:.1f}s", flush=True)
        if val["auc"] > best_val:
            best_val, best_epoch = val["auc"], epoch
            best_test = test if args.test_each_epoch else None
            torch.save(model.state_dict(), best_path)

    if not args.test_each_epoch:
        model.load_state_dict(torch.load(best_path, map_location=device), strict=True)
        best_test = evaluate(model, test_loader, device)

    result = {
        "dataset": dataset, "split": "cold_protein", "seed": seed, "variant": variant,
        "best_epoch": best_epoch, "best_validation_auc": best_val, "test_at_best_validation": best_test,
        "epochs": args.epochs, "batch_size": args.batch_size, "accum_steps": args.accum_steps,
        "optimizer": "Adam", "learning_rate": 5e-5, "weight_decay": 1e-5,
        "checkpoint_selection": "maximum validation AUROC", "elapsed_seconds": time.time() - started,
        "checkpoint_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
    }
    write_json_new(cell / "completed.json", result)
    print(f"DONE {dataset}/seed={seed}/{variant}: {json.dumps(result, ensure_ascii=False)}", flush=True)

    del model, opt, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.code_root))
    sys.path.insert(0, str(args.code_root / "VLMNET"))
    from unified_dataset_v2 import UnifiedDtiDatasetV2, unified_collate_fn_v2
    from model import UniQCM2Net

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    protocol_path = args.output_dir / "protocol.json"
    if not protocol_path.exists():
        protocol = {
            "created_unix": time.time(), "code_root": str(args.code_root), "data_root": str(args.data_root),
            "formal_expected_datasets": list(DATASETS), "formal_expected_seeds": list(SEEDS),
            "formal_expected_variants": list(VARIANTS),
            "invocation_datasets": args.datasets, "invocation_seeds": args.seeds, "invocation_variants": args.variants,
            "epochs": args.epochs, "batch_size": args.batch_size, "accum_steps": args.accum_steps,
            "effective_batch_size": args.batch_size * args.accum_steps,
            "test_evaluation": "each epoch" if args.test_each_epoch else "once after selecting best validation checkpoint",
            "num_workers": args.num_workers, "device": str(device), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "variant_definition": {
                "table_mha": "VLM-XA/no_attn: type residual + within-molecule graph->LM MHA",
                "no_table": "Cross-Attn: no type residual; same within-molecule graph->LM MHA",
                "no_mha": "VLM/no_attn: type residual; no within-molecule graph->LM MHA",
                "no_table_no_mha": "VLM/no_attn with P_virt,D_virt zeroed and frozen; no MHA",
            },
        }
        try:
            write_json_new(protocol_path, protocol)
        except FileExistsError:
                                                                       
            pass
    for dataset in args.datasets:
        for seed in args.seeds:
            for variant in args.variants:
                run_cell(args, dataset, seed, variant, UnifiedDtiDatasetV2, unified_collate_fn_v2,
                         UniQCM2Net, device)

if __name__ == "__main__":
    main()
