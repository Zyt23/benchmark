import argparse
import csv
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_NAMES = ["mae", "mse", "rmse", "mape", "mspe"]
MODEL_ORDER = ["Transformer", "TimesNet", "PatchTST", "DLinear", "iTransformer"]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect QAR long-term forecasting metrics into tables.")
    parser.add_argument("--run_tags", nargs="+", required=True, help="Run tag(s) under logs/long_term_forecast.")
    parser.add_argument("--output_dir", required=True, help="Artifact directory to create.")
    parser.add_argument("--log_root", default="logs/long_term_forecast", help="Log root containing run tag dirs.")
    parser.add_argument("--remote_project", default="", help="Remote project path recorded in README.")
    parser.add_argument("--compact_root", default="datasetall_tsfile_compact", help="Compact cache root recorded in README.")
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--label_len", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=20)
    parser.add_argument("--copy_result_arrays", action="store_true",
                        help="Copy full result directories including pred.npy/true.npy. Default copies metrics.npy only.")
    return parser.parse_args()


def read_summary(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def parse_metrics(result_dir):
    result_dir = Path(result_dir)
    metrics_path = result_dir / "metrics.npy"
    if not metrics_path.exists():
        return {name: np.nan for name in METRIC_NAMES}
    values = np.load(metrics_path, allow_pickle=False).astype(float).tolist()
    return dict(zip(METRIC_NAMES, values))


def safe_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_code_snapshot(output_dir):
    code_dir = output_dir / "code"
    files = [
        "run.py",
        "data_provider/data_loader.py",
        "data_provider/data_factory.py",
        "exp/exp_long_term_forecasting.py",
        "scripts/long_term_forecast/run_QAR_tsfile_forecast_shiftN80.sh",
        "collect_qar_forecast_metrics_tables.py",
    ]
    for rel in files:
        src = Path(rel)
        if src.exists():
            safe_copy(src, code_dir / rel)


def make_markdown_table(df):
    lines = ["| model | mae | mse | rmse | mape | mspe |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for _, row in df.iterrows():
        lines.append(
            "| {model} | {mae:.6f} | {mse:.6f} | {rmse:.6f} | {mape:.6f} | {mspe:.6f} |".format(
                model=row["model"],
                mae=row["mae"],
                mse=row["mse"],
                rmse=row["rmse"],
                mape=row["mape"],
                mspe=row["mspe"],
            )
        )
    return "\n".join(lines)


def model_sort_key(model):
    if model in MODEL_ORDER:
        return MODEL_ORDER.index(model)
    return len(MODEL_ORDER), model


def natural_dataset_key(name):
    parts = re.findall(r"\d+|\D+", str(name))
    return "".join(part.zfill(8) if part.isdigit() else part for part in parts)


def write_readme(output_dir, all_df, args):
    readme = output_dir / "README.md"
    lines = []
    lines.append("# QAR tsfile compact long-term forecasting")
    lines.append("")
    lines.append("- Task: `long_term_forecast`")
    lines.append(f"- Run tags: `{', '.join(args.run_tags)}`")
    if args.remote_project:
        lines.append(f"- Remote project: `{args.remote_project}`")
    lines.append(f"- Compact cache root: `{args.compact_root}`")
    lines.append(f"- Window: `seq_len={args.seq_len}`, `label_len={args.label_len}`, `pred_len={args.pred_len}`")
    lines.append("- Metrics: lower is better for all columns.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    best_rows = []
    for dataset in sorted(all_df["dataset"].unique(), key=natural_dataset_key):
        g = all_df[all_df["dataset"] == dataset]
        valid = g[g["status"] == 0].copy()
        if len(valid) == 0:
            continue
        best = valid.sort_values(["mse", "mae"], ascending=[True, True]).iloc[0]
        best_rows.append(best)
    if best_rows:
        best_df = pd.DataFrame(best_rows)
        best_df["_dataset_order"] = best_df["dataset"].map(natural_dataset_key)
        best_df = best_df.sort_values("_dataset_order").drop(columns=["_dataset_order"])
        lines.append("| dataset | best_model | mae | mse | rmse | mape | mspe |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for _, row in best_df.iterrows():
            lines.append(
                "| {dataset} | {model} | {mae:.6f} | {mse:.6f} | {rmse:.6f} | {mape:.6f} | {mspe:.6f} |".format(
                    dataset=row["dataset"],
                    model=row["model"],
                    mae=row["mae"],
                    mse=row["mse"],
                    rmse=row["rmse"],
                    mape=row["mape"],
                    mspe=row["mspe"],
                )
            )
    else:
        lines.append("No successful runs found.")

    lines.append("")
    lines.append("## Per-dataset metric tables")
    lines.append("")
    for dataset in sorted(all_df["dataset"].unique(), key=natural_dataset_key):
        g = all_df[all_df["dataset"] == dataset]
        lines.append(f"### {dataset}")
        lines.append("")
        g = g.copy()
        g["_order"] = g["model"].map(lambda m: model_sort_key(m))
        g = g.sort_values("_order")
        lines.append(make_markdown_table(g))
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- 这版不是分类任务；入口是 `run.py --task_name long_term_forecast --data QAR_forecast`。")
    lines.append("- 每个航班/窗口作为一个独立样本切分，默认用前 60 个时间点预测后 20 个时间点。")
    lines.append("- 指标是在 compact cache 的归一化数值空间上计算的，主要用于比较模型和数据集之间的预测难度。")
    lines.append("- `mape/mspe` 对这套 compact 数据不太适合：真实值里有 0，百分比误差会出现 `inf/nan`；判断结果时主要看 `mae/mse/rmse`。")
    lines.append("- 如果某个数据集本身 compact 特征全零或字段未适配，预测指标可能会异常好看，需要结合数据诊断一起解释。")

    readme.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    all_rows = []
    for run_tag in args.run_tags:
        log_dir = Path(args.log_root) / run_tag
        summary_rows = read_summary(log_dir / "summary.tsv")
        safe_copy(log_dir, output_dir / "logs" / run_tag)

        for row in summary_rows:
            dataset = row.get("dataset", "")
            model = row.get("model", "")
            status = int(row.get("status", "1") or 1)
            result_dir = row.get("result_dir", "")
            metrics = parse_metrics(result_dir) if status == 0 and result_dir else {name: np.nan for name in METRIC_NAMES}

            if result_dir and args.copy_result_arrays:
                result_name = Path(result_dir).name
                safe_copy(result_dir, output_dir / "results" / dataset / model / result_name)
            elif result_dir:
                metrics_src = Path(result_dir) / "metrics.npy"
                metrics_dst = output_dir / "results" / dataset / "{}_metrics.npy".format(model)
                safe_copy(metrics_src, metrics_dst)

            out = {
                "run_tag": run_tag,
                "dataset": dataset,
                "model": model,
                "status": status,
                "log": row.get("log", ""),
                "result_dir": result_dir,
            }
            out.update(metrics)
            all_rows.append(out)

    if not all_rows:
        raise SystemExit("No rows collected. Check --run_tags and summary.tsv.")

    all_df = pd.DataFrame(all_rows)
    all_df["_dataset_num"] = all_df["dataset"].map(natural_dataset_key)
    all_df["_model_order"] = all_df["model"].map(model_sort_key)
    all_df = all_df.sort_values(["_dataset_num", "_model_order"]).drop(columns=["_dataset_num", "_model_order"])
    all_df.to_csv(output_dir / "all_metrics.csv", index=False)

    for dataset, g in all_df.groupby("dataset"):
        g.to_csv(output_dir / "tables" / f"{dataset}_forecast_metrics.csv", index=False)

    copy_code_snapshot(output_dir)
    write_readme(output_dir, all_df, args)


if __name__ == "__main__":
    main()
