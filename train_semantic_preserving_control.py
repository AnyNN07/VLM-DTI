from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

DATASETS = ("Patent", "celegans")
SEEDS = (42, 1234, 2025, 12345, 666)
                            
VARIANTS = (
    "table_resid_mha", "no_table_resid_mha",
    "table_resid_uniform", "no_table_resid_uniform",
)
EXPECTED_CELLS = len(DATASETS) * len(SEEDS) * len(VARIANTS)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--code-root", type=Path,
                   default=Path(r"E:\LINUX_code\solve_OOD_to24G_LocalFinal"))
    p.add_argument("--data-root", type=Path, default=Path(r"D:\Paper2Data\testData4"))
    p.add_argument("--output-dir", type=Path,
                   default=Path(r"E:\LINUX_code\solve_OOD_to24G_LocalFinal\output\semantic_control"))
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum-steps", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--test-each-epoch", action="store_true",
                   help="Diagnostic only; formal protocol tests the best-validation checkpoint once.")
    p.add_argument("--force", action="store_true",
                   help="Preserve an existing cell and write a uniquely suffixed rerun directory.")
    p.add_argument("--validate-only", action="store_true",
                   help="Run one real batch through every selected dataset/variant and do not train.")
    return p.parse_args()

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def tensor_sha256(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(t.dtype).encode("ascii"))
        h.update(np.asarray(t.shape, dtype=np.int64).tobytes())
        h.update(t.numpy().tobytes())
    return h.hexdigest()

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
    return df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")],
                           errors="ignore")

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

def variant_spec(variant: str) -> dict:
    specs = {
        "table_resid_mha": {"fusion_mode": "VLM-XA", "table": True,
                            "semantic_fusion": "residualized_mha"},
        "no_table_resid_mha": {"fusion_mode": "VLM-XA", "table": False,
                               "semantic_fusion": "residualized_mha"},
        "table_resid_uniform": {"fusion_mode": "VLM-XA", "table": True,
                                "semantic_fusion": "residualized_uniform"},
        "no_table_resid_uniform": {"fusion_mode": "VLM-XA", "table": False,
                                   "semantic_fusion": "residualized_uniform"},
    }
    return specs[variant]

def _uniform_semantic_residual_forward(
    module,
    query,
    key,
    value,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    average_attn_weights=True,
    is_causal=False,
):
                                                                   

                                                                           
                                                                           
                                                                               
                                                                                
       
    del key, need_weights, attn_mask, average_attn_weights, is_causal
    if not module.batch_first:
        raise RuntimeError("Semantic control requires batch_first=True")
    if query.ndim != 3 or value.ndim != 3:
        raise RuntimeError("Expected batched [B,L,E] tensors")
    embed_dim = module.embed_dim
    if module.in_proj_weight is not None:
        w_v = module.in_proj_weight[2 * embed_dim: 3 * embed_dim]
        b_v = (module.in_proj_bias[2 * embed_dim: 3 * embed_dim]
               if module.in_proj_bias is not None else None)
    else:
        w_v = module.v_proj_weight
        b_v = (module.in_proj_bias[2 * embed_dim: 3 * embed_dim]
               if module.in_proj_bias is not None else None)
    projected_value = F.linear(value, w_v, b_v)
    if key_padding_mask is None:
        valid = torch.ones(value.shape[:2], dtype=torch.bool, device=value.device)
    else:
        valid = ~key_padding_mask.to(dtype=torch.bool)
    weight = valid.to(projected_value.dtype).unsqueeze(-1)
    denom = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = (projected_value * weight).sum(dim=1, keepdim=True) / denom
    pooled = pooled.expand(-1, query.shape[1], -1)
    semantic = F.linear(pooled, module.out_proj.weight, module.out_proj.bias)
    return query + semantic, None

def _residualized_mha_forward(
    module,
    query,
    key,
    value,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    average_attn_weights=True,
    is_causal=False,
):
    out, weights = module.semantic_base_forward(
        query, key, value, key_padding_mask=key_padding_mask,
        need_weights=need_weights, attn_mask=attn_mask,
        average_attn_weights=average_attn_weights, is_causal=is_causal,
    )
    return query + out, weights

def patch_semantic_fusion(model, semantic_fusion: str) -> None:
    for name in ("cross_attn_d", "cross_attn_p"):
        module = getattr(model, name)
        if semantic_fusion == "residualized_uniform":
            module.forward = types.MethodType(_uniform_semantic_residual_forward, module)
            module.semantic_control = "masked_uniform_VO_plus_query_residual"
        elif semantic_fusion == "residualized_mha":
            module.semantic_base_forward = module.forward
            module.forward = types.MethodType(_residualized_mha_forward, module)
            module.semantic_control = "query_plus_query_dependent_mha"
        else:
            raise ValueError(semantic_fusion)

def build_model(UniQCM2Net, variant: str, canonical_state: dict, seed: int):
    spec = variant_spec(variant)
    set_seeds(seed)
    model = UniQCM2Net(gnn_hidden=128, fusion_mode="VLM-XA", vlm_ablation="no_attn")
    own = model.state_dict()
    copied = {name: value for name, value in canonical_state.items()
              if name in own and own[name].shape == value.shape}
    incompat = model.load_state_dict(copied, strict=False)
    if not spec["table"]:
        with torch.no_grad():
            model.vlm.P_virt.zero_()
            model.vlm.D_virt.zero_()
        model.vlm.P_virt.requires_grad_(False)
        model.vlm.D_virt.requires_grad_(False)
    state_before_patch = tensor_sha256(model.state_dict())
    patch_semantic_fusion(model, spec["semantic_fusion"])
    meta = {
        **spec,
        "variant": variant,
        "vlm_ablation": "no_attn",
        "common_tensors_copied": len(copied),
        "missing_after_common_copy": list(incompat.missing_keys),
        "unexpected_after_common_copy": list(incompat.unexpected_keys),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "initial_state_sha256": tensor_sha256(model.state_dict()),
        "state_hash_unchanged_by_forward_patch": state_before_patch == tensor_sha256(model.state_dict()),
        "uniform_control_definition": (
            "masked mean of canonical MHA V projection; canonical MHA out projection; "
            "broadcast over graph queries; output=query+semantic; outer model LayerNorm unchanged"
            if spec["semantic_fusion"] == "residualized_uniform" else None
        ),
        "qk_forward_usage": (
            "unused_in_uniform_control" if spec["semantic_fusion"] == "residualized_uniform"
            else "used_by_residualized_mha"
        ),
        "causal_boundary": (
            "Both arms use an explicit query residual and are not the deployed no-residual architecture. "
            "The comparison estimates query-dependent versus uniform token weighting within this controlled model."
        ),
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
        "f1": float(2 * precision * recall / (precision + recall + 1e-5))
              if precision + recall else 0.0,
    }

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    ys, ps = [], []
    for bg_d, bg_p, d_feat, p_feat, labels, d_len, p_len in loader:
        pred, _ = model(
            bg_d.to(device), bg_p.to(device), d_feat.to(device), p_feat.to(device),
            d_len.to(device), p_len.to(device),
        )
        ys.extend(labels.numpy().reshape(-1).tolist())
        ps.extend(pred.detach().cpu().numpy().reshape(-1).tolist())
    return metrics(ys, ps)

def make_datasets_and_loaders(args, dataset, UnifiedDtiDatasetV2, collate, seed):
    train_df, val_df, test_df = load_frames(args.data_root, dataset)
    train_ds = UnifiedDtiDatasetV2(train_df, **dataset_kwargs(args.data_root, dataset))
    val_ds = dataset_view(train_ds, val_df)
    test_ds = dataset_view(train_ds, test_df)
    set_seeds(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = dict(batch_size=args.batch_size, collate_fn=collate,
                  num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader

def run_cell(args, dataset, seed, variant, UnifiedDtiDatasetV2, collate, UniQCM2Net, device):
    cell_base = args.output_dir / "runs" / dataset / "cold_protein" / f"seed_{seed}" / variant
    if (cell_base / "completed.json").exists() and not args.force:
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
    model_meta.update({"canonical_initial_state_sha256": canonical_sha,
                       "dataset": dataset, "seed": seed})
    write_json_new(cell / "model_initialization.json", model_meta)
    model.to(device)

    data = make_datasets_and_loaders(args, dataset, UnifiedDtiDatasetV2, collate, seed)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = data
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                           lr=5e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=10, min_lr=1e-6,
    )
    bce = torch.nn.BCELoss()
    epoch_csv = cell / "epoch_metrics.csv"
    with epoch_csv.open("x", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "lr", "val_auc", "val_prauc", "val_auprc",
            "val_precision", "val_recall", "val_accuracy", "val_f1",
            "test_auc", "test_prauc", "test_auprc", "test_precision", "test_recall",
            "test_accuracy", "test_f1", "elapsed_seconds",
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
                d_len.to(device), p_len.to(device),
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
        "variant_spec": variant_spec(variant), "best_epoch": best_epoch,
        "best_validation_auc": best_val, "test_at_best_validation": best_test,
        "epochs": args.epochs, "batch_size": args.batch_size, "accum_steps": args.accum_steps,
        "optimizer": "Adam", "learning_rate": 5e-5, "weight_decay": 1e-5,
        "checkpoint_selection": "maximum validation AUROC",
        "test_policy": "once after best-validation selection" if not args.test_each_epoch else "diagnostic_each_epoch",
        "elapsed_seconds": time.time() - started,
        "checkpoint_sha256": sha256(best_path),
    }
    write_json_new(cell / "completed.json", result)
    print(f"DONE {dataset}/seed={seed}/{variant}: {json.dumps(result, ensure_ascii=False)}", flush=True)

    del model, opt, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def grad_l1(param) -> float | None:
    if param is None or param.grad is None:
        return None
    return float(param.grad.detach().abs().sum().cpu())

def validate_only(args, UnifiedDtiDatasetV2, collate, UniQCM2Net, device) -> None:
    rows = []
    seed = args.seeds[0]
    for dataset in args.datasets:
                                                                                  
        train_df, _, _ = load_frames(args.data_root, dataset)
        ds = UnifiedDtiDatasetV2(train_df, **dataset_kwargs(args.data_root, dataset))
        loader = DataLoader(ds, batch_size=min(args.batch_size, 2), shuffle=False,
                            collate_fn=collate, num_workers=0)
        batch = next(iter(loader))
        bg_d, bg_p, d_feat, p_feat, labels, d_len, p_len = batch
        for variant in args.variants:
            set_seeds(seed)
            canonical = UniQCM2Net(gnn_hidden=128, fusion_mode="VLM-XA", vlm_ablation="no_attn")
            canonical_state = {k: v.detach().cpu().clone() for k, v in canonical.state_dict().items()}
            canonical_sha = tensor_sha256(canonical_state)
            del canonical
            model, meta = build_model(UniQCM2Net, variant, canonical_state, seed)
            model.to(device).train()
            model.zero_grad(set_to_none=True)
            pred, cl = model(
                bg_d.to(device), bg_p.to(device), d_feat.to(device), p_feat.to(device),
                d_len.to(device), p_len.to(device),
            )
            loss = torch.nn.functional.binary_cross_entropy(
                torch.clamp(pred, 1e-7, 1 - 1e-7), labels.to(device).float(),
            ) + 1e-4 * cl
            loss.backward()
            uniform = variant_spec(variant)["semantic_fusion"] == "residualized_uniform"
            qk_l1, v_l1, out_l1 = [], [], []
            for name in ("cross_attn_d", "cross_attn_p"):
                module = getattr(model, name)
                g = module.in_proj_weight.grad
                if g is None:
                    qk_l1.append(None)
                    v_l1.append(None)
                else:
                    e = module.embed_dim
                    qk_l1.append(float(g[:2 * e].abs().sum().detach().cpu()))
                    v_l1.append(float(g[2 * e:].abs().sum().detach().cpu()))
                out_l1.append(grad_l1(module.out_proj.weight))
            lm_grad = grad_l1(model.drug_llm_encoder.proj.weight)
            graph_grad = grad_l1(model.drug_jk.weight)
            p_table_grad = grad_l1(model.vlm.P_virt)
            d_table_grad = grad_l1(model.vlm.D_virt)
            table_on = variant_spec(variant)["table"]
            row = {
                "dataset": dataset, "seed": seed, "variant": variant,
                "semantic_fusion": variant_spec(variant)["semantic_fusion"],
                "canonical_initial_state_sha256": canonical_sha,
                "initial_state_sha256": meta["initial_state_sha256"],
                "prediction_shape": list(pred.shape),
                "prediction_finite": bool(torch.isfinite(pred).all()),
                "loss_finite": bool(torch.isfinite(loss)),
                "drug_lm_encoder_grad_l1": lm_grad,
                "drug_graph_encoder_grad_l1": graph_grad,
                "protein_table_grad_l1": p_table_grad,
                "drug_table_grad_l1": d_table_grad,
                "mha_qk_grad_l1": qk_l1,
                "mha_v_grad_l1": v_l1,
                "mha_out_grad_l1": out_l1,
                "uniform_qk_exact_zero": (
                    all(x == 0.0 for x in qk_l1) if uniform else None
                ),
                "semantic_path_gradient_nonzero": bool(lm_grad is not None and lm_grad > 0),
                "graph_path_gradient_nonzero": bool(graph_grad is not None and graph_grad > 0),
                "validation_ok": bool(
                    torch.isfinite(pred).all() and torch.isfinite(loss)
                    and lm_grad is not None and lm_grad > 0
                    and graph_grad is not None and graph_grad > 0
                    and (not uniform or all(x == 0.0 for x in qk_l1))
                    and (uniform or all(x is not None and x > 0 for x in qk_l1))
                    and all(x is not None and x > 0 for x in v_l1)
                    and all(x is not None and x > 0 for x in out_l1)
                    and ((p_table_grad is not None and p_table_grad > 0
                          and d_table_grad is not None and d_table_grad > 0) if table_on
                         else (p_table_grad is None and d_table_grad is None))
                ),
            }
            rows.append(row)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del ds, loader, batch
                                                                           
    for dataset in args.datasets:
        idx = {(r["variant"]): r for r in rows if r["dataset"] == dataset}
        pairs = (("table_resid_mha", "table_resid_uniform"),
                 ("no_table_resid_mha", "no_table_resid_uniform"))
        for va, vb in pairs:
            if va in idx and vb in idx:
                same = idx[va]["initial_state_sha256"] == idx[vb]["initial_state_sha256"]
                idx[va][f"state_equal_to_{vb}"] = same
                idx[vb][f"state_equal_to_{va}"] = same
                if not same:
                    idx[va]["validation_ok"] = False
                    idx[vb]["validation_ok"] = False
    report = {
        "mode": "validate_only_no_training", "device": str(device), "rows": rows,
        "all_valid": all(r["validation_ok"] for r in rows),
    }
    write_json_new(args.output_dir / "validation_only.json", report)
    if not report["all_valid"]:
        raise RuntimeError("Validation-only checks failed; inspect validation_only.json")
    print(json.dumps(report, ensure_ascii=False), flush=True)

def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    forbidden = (args.code_root / "output" / "factorial_shard").resolve()
    if args.output_dir == forbidden or forbidden in args.output_dir.parents:
        raise ValueError("Refusing to write inside the existing factorial_shard")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.code_root))
    sys.path.insert(0, str(args.code_root / "VLMNET"))
    from unified_dataset_v2 import UnifiedDtiDatasetV2, unified_collate_fn_v2
    from model import UniQCM2Net

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    protocol_path = args.output_dir / "protocol.json"
    if not protocol_path.exists():
        model_path = args.code_root / "VLMNET" / "model.py"
        vlm_path = args.code_root / "VLMNET" / "VLM.py"
        protocol = {
            "created_unix": time.time(), "code_root": str(args.code_root.resolve()),
            "data_root": str(args.data_root.resolve()),
            "formal_expected_datasets": list(DATASETS),
            "formal_expected_seeds": list(SEEDS),
            "formal_expected_variants": list(VARIANTS),
            "formal_expected_cells": EXPECTED_CELLS,
            "invocation_datasets": args.datasets, "invocation_seeds": args.seeds,
            "invocation_variants": args.variants,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "accum_steps": args.accum_steps,
            "effective_batch_size": args.batch_size * args.accum_steps,
            "test_evaluation": "each epoch diagnostic" if args.test_each_epoch
                               else "once after selecting best validation checkpoint",
            "validate_only": args.validate_only, "num_workers": args.num_workers,
            "device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "variant_definition": {v: variant_spec(v) for v in VARIANTS},
            "uniform_control": (
                "same LM encoder and masks; canonical MHA V projection; masked mean over valid LM tokens; "
                "canonical MHA output projection; broadcast; add graph query residual; unchanged outer LayerNorm/BAN"
            ),
            "residualized_mha_control": (
                "canonical query-dependent MHA output plus the same explicit graph-query residual; "
                "unchanged outer LayerNorm/BAN"
            ),
            "interpretation_boundary": (
                "Both arms are residualized controlled architectures, not the deployed no-residual MHA. "
                "The contrast estimates query-dependent versus uniform LM-token weighting in this modified architecture."
            ),
            "source_sha256": {
                "training_script": sha256(Path(__file__).resolve()),
                "model.py": sha256(model_path), "VLM.py": sha256(vlm_path),
            },
        }
        try:
            write_json_new(protocol_path, protocol)
        except FileExistsError:
            pass
    if args.validate_only:
        validate_only(args, UnifiedDtiDatasetV2, unified_collate_fn_v2, UniQCM2Net, device)
        return
    for dataset in args.datasets:
        for seed in args.seeds:
            for variant in args.variants:
                run_cell(args, dataset, seed, variant, UnifiedDtiDatasetV2,
                         unified_collate_fn_v2, UniQCM2Net, device)

if __name__ == "__main__":
    main()
