"""跨提供方运行的补充质量摘要。

默认递归读取 ``runs/**/*.jsonl``，并与 export.py 一样排除任何位于
``_smoke`` 目录下的文件。也可以在命令行显式传入一个或多个文件、目录
或通配符。

产出：
  out/extra_arm_summary.csv       每个实验臂的调用、失败、token 与延迟摘要
  out/extra_condition_summary.csv 每个实验臂 x 条件的风险摘要
"""
import argparse
import os

import numpy as np
import pandas as pd

from export import discover_jsonl, load_calls


def summarize_arms(calls):
    """按有效 arm（旧数据已由 load_calls 回退到 model）汇总运行质量。"""
    d = calls.copy()
    for column, default in {
        "RID": pd.NA,
        "cond": pd.NA,
        "latency_ms": np.nan,
        "thought_tokens": np.nan,
        "visible_output_tokens": np.nan,
    }.items():
        if column not in d:
            d[column] = default

    def one(group):
        latency = pd.to_numeric(group["latency_ms"], errors="coerce")
        return pd.Series({
            "calls": len(group),
            "successful_calls": int(group["ok"].sum()),
            "failed_calls": int((~group["ok"]).sum()),
            "failure_rate": float((~group["ok"]).mean()),
            "patients": group["RID"].nunique(dropna=True),
            "conditions": group["cond"].nunique(dropna=True),
            "prompt_tokens": pd.to_numeric(group["prompt_tokens"], errors="coerce").sum(min_count=1),
            "completion_tokens": pd.to_numeric(group["completion_tokens"], errors="coerce").sum(min_count=1),
            "thought_tokens": pd.to_numeric(group["thought_tokens"], errors="coerce").sum(min_count=1),
            "visible_output_tokens": pd.to_numeric(
                group["visible_output_tokens"], errors="coerce").sum(min_count=1),
            "latency_p50_ms": latency.median(),
            "latency_p95_ms": latency.quantile(0.95),
        })

    if d.empty:
        return pd.DataFrame(columns=["arm"])
    rows = []
    for arm, group in d.groupby("arm", dropna=False, sort=True):
        rows.append({"arm": arm, **one(group).to_dict()})
    return pd.DataFrame(rows)


def summarize_conditions(calls):
    """正式、成功且非哨兵的调用按 arm x cond 汇总。"""
    d = calls.copy()
    sentinel = (d["is_sentinel"].fillna(False).astype(bool) if "is_sentinel" in d
                else pd.Series(False, index=d.index))
    d = d[d["ok"] & ~sentinel]
    if d.empty:
        return pd.DataFrame(columns=["arm", "cond", "n", "patients", "risk_mean",
                                     "risk_sd", "risk_min", "risk_max"])
    return (d.groupby(["arm", "cond"], dropna=False, sort=True)
             .agg(n=("risk", "size"), patients=("RID", "nunique"),
                  risk_mean=("risk", "mean"), risk_sd=("risk", "std"),
                  risk_min=("risk", "min"), risk_max=("risk", "max"))
             .reset_index())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="*",
                        help="显式 JSONL 文件、目录或通配符；默认递归读取 runs/**/*.jsonl")
    parser.add_argument("--outdir", default="out")
    args = parser.parse_args(argv)

    paths = discover_jsonl(args.jsonl)
    if not paths:
        parser.error("未找到可分析的 JSONL（runs/_smoke/ 会被自动排除）")
    calls = load_calls(paths)

    os.makedirs(args.outdir, exist_ok=True)
    arm_summary = summarize_arms(calls)
    condition_summary = summarize_conditions(calls)
    arm_summary.to_csv(os.path.join(args.outdir, "extra_arm_summary.csv"), index=False)
    condition_summary.to_csv(os.path.join(args.outdir, "extra_condition_summary.csv"), index=False)
    print(f"读取 {len(paths)} 个文件、{len(calls)} 条去重调用、{calls.arm.nunique()} 个 arm")
    print(f"补充摘要 -> {args.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
