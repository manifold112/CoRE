from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


def aggregate_from_summaries(summary_files: list[str]) -> list[dict[str, Any]]:
    rows = []
    for path in summary_files:
        data = load_json(path)
        benchmark = data.get("benchmark", "unknown")
        model_type = data.get("model_type", "unknown")
        method = data.get("method", "unknown")
        overall = data.get("overall", {})
        row = {
            "source_file": str(path),
            "benchmark": benchmark,
            "model_type": model_type,
            "method": method,
            "accuracy_mean": float(overall.get("accuracy_mean", 0.0)),
            "accuracy_std": float(overall.get("accuracy_std", 0.0)),
        }

        if benchmark == "dcase":
            subsets = overall.get("subsets", {})
            for key, val in subsets.items():
                row[f"{key}_mean"] = float(val.get("accuracy_mean", 0.0))
                row[f"{key}_std"] = float(val.get("accuracy_std", 0.0))
        else:
            categories = overall.get("categories", {})
            for key, val in categories.items():
                row[f"{key}_mean"] = float(val.get("accuracy_mean", 0.0))
                row[f"{key}_std"] = float(val.get("accuracy_std", 0.0))

        rows.append(row)
    return rows


def aggregate_from_predictions(prediction_files: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for path in prediction_files:
        records = load_jsonl(path)
        for r in records:
            key = (
                str(r.get("benchmark", "unknown")),
                str(r.get("model_type", "unknown")),
                str(r.get("method", "unknown")),
            )
            grouped[key].append(r)

    rows = []

    for (benchmark, model_type, method), records in grouped.items():
        by_perm: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in records:
            by_perm[int(r["permutation_index"])].append(r)

        accs = []
        for perm_idx in sorted(by_perm.keys()):
            items = by_perm[perm_idx]
            correct = sum(int(x["correct"]) for x in items)
            total = len(items)
            acc = correct / total if total > 0 else 0.0
            accs.append(acc)

        row = {
            "source_file": "MULTI",
            "benchmark": benchmark,
            "model_type": model_type,
            "method": method,
            "accuracy_mean": mean(accs),
            "accuracy_std": std(accs),
        }

        if benchmark == "dcase":
            subset_names = sorted({str(x.get("subset", "ALL")) for x in records})
            for subset in subset_names:
                subset_accs = []
                for perm_idx in sorted(by_perm.keys()):
                    items = [x for x in by_perm[perm_idx] if str(x.get("subset", "ALL")) == subset]
                    correct = sum(int(x["correct"]) for x in items)
                    total = len(items)
                    subset_accs.append(correct / total if total > 0 else 0.0)
                row[f"{subset}_mean"] = mean(subset_accs)
                row[f"{subset}_std"] = std(subset_accs)
        else:
            category_names = sorted({str(x.get("category", "ALL")) for x in records})
            for category in category_names:
                category_accs = []
                for perm_idx in sorted(by_perm.keys()):
                    items = [x for x in by_perm[perm_idx] if str(x.get("category", "ALL")) == category]
                    correct = sum(int(x["correct"]) for x in items)
                    total = len(items)
                    category_accs.append(correct / total if total > 0 else 0.0)
                row[f"{category}_mean"] = mean(category_accs)
                row[f"{category}_std"] = std(category_accs)

        rows.append(row)

    return rows


def print_markdown_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No rows to display.")
        return

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    preferred = [
        "benchmark",
        "model_type",
        "method",
        "accuracy_mean",
        "accuracy_std",
        "BQA_mean",
        "BQA_std",
        "TSQA_mean",
        "TSQA_std",
        "CQA_mean",
        "CQA_std",
        "source_file",
    ]
    remaining = [k for k in sorted(all_keys) if k not in preferred]
    columns = [k for k in preferred if k in all_keys] + remaining

    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join(["---"] * len(columns)) + "|")

    for r in rows:
        vals = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                if "mean" in c or "std" in c:
                    vals.append(f"{v * 100:.2f}")
                else:
                    vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        print("| " + " | ".join(vals) + " |")


def save_json(rows: list[dict[str, Any]], output_path: str | None) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nSaved aggregated results to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate evaluation results.")
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Input files. Can be one or more summary.json or predictions.jsonl files.",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["summary", "prediction", "auto"],
        default="auto",
        help="How to interpret input files.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save aggregated rows as JSON.",
    )
    args = parser.parse_args()

    paths = [str(Path(p)) for p in args.inputs]

    if args.input_type == "summary":
        rows = aggregate_from_summaries(paths)
    elif args.input_type == "prediction":
        rows = aggregate_from_predictions(paths)
    else:
        if all(str(p).endswith(".json") for p in paths):
            rows = aggregate_from_summaries(paths)
        elif all(str(p).endswith(".jsonl") for p in paths):
            rows = aggregate_from_predictions(paths)
        else:
            raise ValueError(
                "Could not infer input type automatically. "
                "Please set --input_type summary or --input_type prediction."
            )

    print_markdown_table(rows)
    save_json(rows, args.output_json)


if __name__ == "__main__":
    main()
