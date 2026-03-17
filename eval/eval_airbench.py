from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.rescoring import core_rescore, default_predict
from core.utils import ensure_dir, load_yaml, stable_mean, stable_std


def build_parser() -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Evaluate CoRE on AIR-Bench SoundQA-style multiple-choice AQA.")
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--dataset_path", type=str, required=False)
    parser.add_argument("--audio_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=False, default="outputs/airbench")

    parser.add_argument("--model_type", type=str, choices=["qwen2audio", "kimiaudio"], required=False)
    parser.add_argument("--model_name_or_path", type=str, required=False)

    parser.add_argument("--method", type=str, choices=["default", "core", "core_silence"], default="core")
    parser.add_argument("--num_permutations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=None)

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--block_ms", type=float, default=40.0)
    parser.add_argument("--reverse_prob", type=float, default=0.5)
    parser.add_argument("--crossfade_ms", type=float, default=3.0)

    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--run_name", type=str, default=None)

    if pre_args.config:
        cfg = load_yaml(pre_args.config)
        if isinstance(cfg, dict):
            parser.set_defaults(**cfg)

    args = parser.parse_args()

    required_fields = ["dataset_path", "model_type", "model_name_or_path"]
    missing = [k for k in required_fields if getattr(args, k, None) in (None, "")]
    if missing:
        raise ValueError(f"Missing required arguments: {missing}")

    return parser


def build_model(model_type: str, model_name_or_path: str):
    if model_type == "qwen2audio":
        from models.qwen2_audio_adapter import Qwen2AudioAdapter

        model = Qwen2AudioAdapter(model_name_or_path=model_name_or_path)
        sample_rate = getattr(model, "sampling_rate", None)
        return model, sample_rate

    if model_type == "kimiaudio":
        from models.kimi_audio_adapter import KimiAudioAdapter

        model = KimiAudioAdapter(model_name_or_path=model_name_or_path)
        return model, None

    raise ValueError(f"Unsupported model_type: {model_type}")


def load_examples(dataset_path: str) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if path.suffix == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ["examples", "data", "items", "records"]:
                if key in obj and isinstance(obj[key], list):
                    return obj[key]
        raise ValueError(f"Unsupported JSON format in {dataset_path}")

    raise ValueError(f"Unsupported dataset file format: {dataset_path}")


def resolve_audio_path(example: dict[str, Any], audio_root: str | None = None) -> str:
    for key in ["audio", "audio_path", "wav", "wav_path", "path", "file"]:
        if key in example:
            p = str(example[key])
            if audio_root is not None and not Path(p).is_absolute():
                return str(Path(audio_root) / p)
            return p
    raise KeyError("Missing audio path field. Expected one of: audio, audio_path, wav, wav_path, path, file")


def resolve_question(example: dict[str, Any]) -> str:
    for key in ["question", "query", "instruction", "prompt"]:
        if key in example:
            return str(example[key])
    raise KeyError("Missing question field. Expected one of: question, query, instruction, prompt")


def resolve_options(example: dict[str, Any]) -> list[str]:
    for key in ["options", "choices", "candidates", "answers"]:
        if key in example:
            options = example[key]
            if not isinstance(options, list) or len(options) == 0:
                raise ValueError(f"Options must be a non-empty list, got: {options}")
            return [str(x) for x in options]
    raise KeyError("Missing options field. Expected one of: options, choices, candidates, answers")


def resolve_category(example: dict[str, Any]) -> str:
    for key in ["category", "subset", "task", "split_name"]:
        if key in example:
            return str(example[key])
    return "ALL"


def resolve_example_id(example: dict[str, Any], fallback_index: int) -> str:
    for key in ["id", "uid", "example_id", "qid", "question_id"]:
        if key in example:
            return str(example[key])
    return str(fallback_index)


def resolve_label_index(example: dict[str, Any], options: list[str]) -> int:
    for key in ["label", "label_idx", "answer_idx", "target", "target_idx", "gt_idx"]:
        if key in example:
            value = example[key]
            if isinstance(value, int):
                if 0 <= value < len(options):
                    return value
            if isinstance(value, str):
                v = value.strip()
                if v.isdigit():
                    idx = int(v)
                    if 0 <= idx < len(options):
                        return idx
                if len(v) == 1 and v.upper() in [chr(65 + i) for i in range(len(options))]:
                    return ord(v.upper()) - 65
                if v in options:
                    return options.index(v)

    for key in ["answer", "gt_answer", "label_text"]:
        if key in example:
            value = str(example[key]).strip()
            if value in options:
                return options.index(value)
            if len(value) == 1 and value.upper() in [chr(65 + i) for i in range(len(options))]:
                return ord(value.upper()) - 65

    raise KeyError("Could not resolve label index from example.")


def stable_hash_int(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def permute_options(
    options: list[str],
    label_idx: int,
    perm_seed: int,
) -> tuple[list[str], int, list[int]]:
    indices = list(range(len(options)))
    rng = random.Random(perm_seed)
    rng.shuffle(indices)

    permuted_options = [options[i] for i in indices]
    permuted_label_idx = indices.index(label_idx)
    return permuted_options, permuted_label_idx, indices


def summarize_by_permutation(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_perm: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_perm[int(r["permutation_index"])].append(r)

    permutation_metrics = []
    overall_accs = []

    category_names = sorted({str(r["category"]) for r in records})
    category_accs_by_perm: dict[str, list[float]] = defaultdict(list)

    for perm_idx in sorted(by_perm.keys()):
        items = by_perm[perm_idx]
        correct = sum(int(x["correct"]) for x in items)
        total = len(items)
        acc = correct / total if total > 0 else 0.0
        overall_accs.append(acc)

        category_metrics = {}
        for category in category_names:
            category_items = [x for x in items if str(x["category"]) == category]
            category_correct = sum(int(x["correct"]) for x in category_items)
            category_total = len(category_items)
            category_acc = category_correct / category_total if category_total > 0 else 0.0
            category_metrics[category] = {
                "accuracy": category_acc,
                "correct": category_correct,
                "total": category_total,
            }
            category_accs_by_perm[category].append(category_acc)

        permutation_metrics.append(
            {
                "permutation_index": perm_idx,
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "categories": category_metrics,
            }
        )

    category_summary = {}
    for category, vals in category_accs_by_perm.items():
        category_summary[category] = {
            "accuracy_mean": stable_mean(vals),
            "accuracy_std": stable_std(vals),
        }

    overall_summary = {
        "accuracy_mean": stable_mean(overall_accs),
        "accuracy_std": stable_std(overall_accs),
        "categories": category_summary,
    }

    return permutation_metrics, overall_summary


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    examples = load_examples(args.dataset_path)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    output_dir = ensure_dir(Path(args.output_dir))
    run_name = args.run_name or f"airbench_{args.model_type}_{args.method}"
    run_dir = ensure_dir(output_dir / run_name)

    model, adapter_sr = build_model(args.model_type, args.model_name_or_path)
    sample_rate = adapter_sr or args.sample_rate

    all_records: list[dict[str, Any]] = []

    for perm_idx in range(args.num_permutations):
        desc = f"[AIR-Bench] permutation {perm_idx + 1}/{args.num_permutations}"
        for ex_idx, ex in enumerate(tqdm(examples, desc=desc)):
            ex_id = resolve_example_id(ex, ex_idx)
            category = resolve_category(ex)
            audio_path = resolve_audio_path(ex, args.audio_root)
            question = resolve_question(ex)
            options = resolve_options(ex)
            label_idx = resolve_label_index(ex, options)

            perm_seed = args.seed + perm_idx * 100000 + ex_idx
            permuted_options, permuted_label_idx, perm_indices = permute_options(
                options=options,
                label_idx=label_idx,
                perm_seed=perm_seed,
            )

            cf_seed = args.seed + stable_hash_int(ex_id)

            if args.method == "default":
                out = default_predict(
                    model_adapter=model,
                    audio=audio_path,
                    question=question,
                    options=permuted_options,
                    normalize=args.normalize,
                )
                pred_index = int(out["pred_index"])
                pred_option = str(out["pred_option"])
                beta = None
                u_j = None
                u_h = None
                final_logits = out["logits"].tolist()
            else:
                out = core_rescore(
                    model_adapter=model,
                    audio=audio_path,
                    question=question,
                    options=permuted_options,
                    sample_rate=sample_rate,
                    normalize=args.normalize,
                    block_ms=args.block_ms,
                    reverse_prob=args.reverse_prob,
                    crossfade_ms=args.crossfade_ms,
                    seed=cf_seed,
                    counterfactual_mode="core" if args.method == "core" else "silence",
                )
                pred_index = int(out.pred_index)
                pred_option = str(out.pred_option)
                beta = float(out.beta)
                u_j = float(out.u_j)
                u_h = float(out.u_h)
                final_logits = out.final_logits.tolist()

            gt_option = permuted_options[permuted_label_idx]
            correct = int(pred_index == permuted_label_idx)

            record = {
                "benchmark": "airbench",
                "model_type": args.model_type,
                "model_name_or_path": args.model_name_or_path,
                "method": args.method,
                "permutation_index": perm_idx,
                "example_index": ex_idx,
                "example_id": ex_id,
                "category": category,
                "audio_path": audio_path,
                "question": question,
                "options": permuted_options,
                "original_options": options,
                "permutation": perm_indices,
                "ground_truth_index": permuted_label_idx,
                "ground_truth_option": gt_option,
                "pred_index": pred_index,
                "pred_option": pred_option,
                "correct": correct,
                "beta": beta,
                "u_j": u_j,
                "u_h": u_h,
                "final_logits": final_logits,
            }
            all_records.append(record)

    predictions_path = run_dir / "predictions.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    permutation_metrics, overall_summary = summarize_by_permutation(all_records)

    summary = {
        "benchmark": "airbench",
        "model_type": args.model_type,
        "model_name_or_path": args.model_name_or_path,
        "method": args.method,
        "dataset_path": args.dataset_path,
        "num_examples": len(examples),
        "num_permutations": args.num_permutations,
        "sample_rate": sample_rate,
        "block_ms": args.block_ms,
        "reverse_prob": args.reverse_prob,
        "crossfade_ms": args.crossfade_ms,
        "seed": args.seed,
        "normalize": bool(args.normalize),
        "permutation_metrics": permutation_metrics,
        "overall": overall_summary,
        "prediction_file": str(predictions_path),
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved predictions to: {predictions_path}")
    print(f"Saved summary to:     {summary_path}")
    print(
        f"\n[AIR-Bench] {args.model_type} + {args.method}: "
        f"{overall_summary['accuracy_mean'] * 100:.2f} ± {overall_summary['accuracy_std'] * 100:.2f}"
    )
    for category, stats in overall_summary["categories"].items():
        print(f"  - {category}: {stats['accuracy_mean'] * 100:.2f} ± {stats['accuracy_std'] * 100:.2f}")

    return summary


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args)
