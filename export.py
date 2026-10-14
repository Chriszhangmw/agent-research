"""
把一个或多个 run_full.py 产出的 JSONL 合并成分析/可视化用的整洁表。
产出：
  out/calls.parquet      一行一次调用（长表，含全部元数据）—— 画任何图都从这张表出发
  out/patient_cond.csv   一行 = 患者 x 条件 x 实验臂（重复已取均值，并带 SD/n）
  out/wide.csv           一行 = 患者 x 实验臂，条件铺成列 —— 直接做配对对比
  out/patients.csv       患者分层变量（人口学、基线、生物标志物、结局、不一致度）
  out/sentinel.csv       哨兵漂移监测
  out/qc.txt             质量与可复现性报告
用法：
  python export.py                         # 递归读取 runs/**/*.jsonl
  python export.py runs/a.jsonl other/*.jsonl
"""
import os, json, glob, argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RUN_GLOB = os.path.join("runs", "**", "*.jsonl")
CALL_KEY = ["run_id", "provider", "arm", "temperature", "month",
            "RID", "cond", "rep", "prompt_sha1"]
MISSING_DX_LABELS = {"", "nan", "none", "null", "na", "n/a", "not available"}


def _is_smoke_path(path):
    """只按目录段排除 ``_smoke``，避免误伤名称中偶含 smoke 的正式文件。"""
    return any(part.casefold() == "_smoke" for part in Path(path).parent.parts)


def discover_jsonl(inputs=None):
    """展开显式文件/通配符；未指定时递归发现正式运行结果。

    目录参数也会被递归展开。所有入口都排除位于 ``_smoke`` 目录下的文件，
    并做稳定排序和去重，以保证重复运行得到相同的读取顺序。
    """
    specs = list(inputs) if inputs else [DEFAULT_RUN_GLOB]
    found = []
    for spec in specs:
        spec = os.fspath(spec)
        if os.path.isdir(spec):
            matches = glob.glob(os.path.join(spec, "**", "*.jsonl"), recursive=True)
        elif os.path.isfile(spec):
            matches = [spec]
        else:
            matches = glob.glob(spec, recursive=True)
        for match in matches:
            if (os.path.isfile(match) and match.casefold().endswith(".jsonl")
                    and not _is_smoke_path(match)):
                found.append(os.path.normpath(match))
    return sorted(dict.fromkeys(found), key=lambda p: os.path.normcase(p))


def with_effective_arm(frame):
    """补齐统一分组键：优先使用 arm，旧记录回退到 model。"""
    d = frame.copy()
    if "arm" not in d:
        d["arm"] = pd.NA
    missing = d["arm"].isna() | d["arm"].astype("string").str.strip().eq("").fillna(True)
    if "model" in d:
        d.loc[missing, "arm"] = d.loc[missing, "model"]
    return d


def _add_compatible_model(frame):
    """让尚未迁移到 arm 的 analyze_full.py 仍能逐臂分析。"""
    d = frame.copy()
    d["model"] = d["arm"]
    columns = [c for c in d.columns if c != "model"]
    columns.insert(columns.index("arm") + 1, "model")
    return d.loc[:, columns]

def clean_dx36(values):
    """保留真实诊断标签，同时把常见的文本缺失值还原为缺失。"""
    s = pd.Series(values, copy=True).astype("string").str.strip()
    return s.mask(s.str.casefold().isin(MISSING_DX_LABELS))

def classify_dx36(values):
    """Dementia=1，MCI/CN=0；未观测或其他标签保持为可空值。"""
    labels = clean_dx36(values).str.casefold()
    y = pd.Series(pd.NA, index=labels.index, dtype="Int64")
    y.loc[labels.eq("dementia").fillna(False)] = 1
    y.loc[labels.isin(["mci", "cn"])] = 0
    return y

def load_calls(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                try: rows.append(json.loads(line))
                except: pass
    d = with_effective_arm(pd.DataFrame(rows))
    for c in ["prompt_tokens", "completion_tokens", "visible_output_tokens", "thought_tokens"]:
        nested = (d["usage"].apply(lambda u: u.get(c) if isinstance(u, dict) else np.nan)
                  if "usage" in d else pd.Series(np.nan, index=d.index))
        if c in d:
            d[c] = d[c].where(d[c].notna(), nested)
        else:
            d[c] = nested
    d["ok"] = d.risk.notna()
    # 同一个 run 内，相同调用键只能代表同一个计划任务。优先保留最近一次成功记录；
    # run_id 和 arm 都在键中，因此不同运行/实验臂绝不会被误合并。
    duplicate_groups = duplicate_rows_removed = 0
    dedup_key_available = set(CALL_KEY) <= set(d.columns)
    if dedup_key_available:
        eligible = d[CALL_KEY].notna().all(axis=1)
        duplicated = eligible & d.duplicated(CALL_KEY, keep=False)
        if duplicated.any():
            duplicate_groups = d.loc[duplicated].groupby(CALL_KEY, dropna=False).ngroups
            ranked = d.loc[eligible].assign(
                _ts_sort=pd.to_datetime(d.loc[eligible, "ts"], errors="coerce", utc=True)
                if "ts" in d else pd.NaT
            ).sort_values(CALL_KEY + ["ok", "_ts_sort"],
                          kind="stable", na_position="first")
            keep_idx = ranked.drop_duplicates(CALL_KEY, keep="last").index
            drop_idx = d.index[eligible].difference(keep_idx)
            duplicate_rows_removed = len(drop_idx)
            d = d.drop(index=drop_idx).copy()
    d["order"] = d.groupby(["run_id"], sort=False).cumcount()  # 在该 run 内的实际落盘序位
    d.attrs["duplicate_groups"] = duplicate_groups
    d.attrs["duplicate_rows_removed"] = duplicate_rows_removed
    d.attrs["dedup_key_available"] = dedup_key_available
    return d

def patient_meta(adni, cards_path="data/cards.json"):
    cards = json.load(open(cards_path))
    base = pd.DataFrame([{"RID": c["RID"], "dx24": c.get("dx24"), "dx36": c.get("dx36")}
                         for c in cards])
    base["dx36"] = clean_dx36(base["dx36"])
    base["converter36"] = classify_dx36(base["dx36"])
    if not adni or not os.path.exists(adni):
        return base
    A = pd.read_csv(adni, low_memory=False)
    b = A[A.M == 0].set_index("RID")
    def amy(r):
        if pd.notna(r.get("AV45")): return int(r.AV45 > 1.11)
        v = r.get("ABETA")
        if pd.notna(v):
            try: return int(float(str(v).replace(">","").replace("<","")) < 980)
            except: return np.nan
        return np.nan
    m = pd.DataFrame({
        "AGE": b.AGE, "SEX": b.PTGENDER, "EDU": b.PTEDUCAT, "APOE4": b.APOE4,
        "MMSE_bl": b.MMSE, "CDRSB_bl": b.CDRSB, "ADAS13_bl": b.ADAS13, "FAQ_bl": b.FAQ,
        "HIPPO_bl": b.Hippocampus, "ICV": b.ICV,
        "AMYLOID_POS": b.apply(amy, axis=1),
    })
    m["HIPPO_norm"] = 1000 * m.HIPPO_bl / m.ICV
    out = base.merge(m.reset_index(), on="RID", how="left")
    # 「生物学高危但临床稳定」= 锚定最危险的那群人
    out["discordant_stable"] = ((out.AMYLOID_POS == 1) & (out.APOE4 >= 1) &
                                (out.dx24 == "MCI") &
                                (out.dx36 == "MCI")).fillna(False).astype(int)
    return out

def order_sentinel_blocks(s):
    """按负 rep 所编码的真实轮次排序；时间戳用于同轮次的稳定排序。"""
    s = with_effective_arm(s)
    s["_rep_order"] = pd.to_numeric(s["rep"], errors="coerce").abs()
    s["_ts_order"] = pd.to_datetime(s["ts"], errors="coerce", utc=True)
    group = ["run_id", "arm", "cond"]
    s = s.sort_values(group + ["_rep_order", "_ts_order"], kind="stable")
    s["block"] = s.groupby(group, sort=False).cumcount()
    return _add_compatible_model(s.drop(columns=["_rep_order", "_ts_order"]))

def summarize_sentinels(sent):
    sent = with_effective_arm(sent)
    s = (sent.groupby(["run_id", "arm", "cond", "rep"], dropna=False)
             .agg(risk_mean=("risk", "mean"), ts=("ts", "min"), n=("risk", "size"))
             .reset_index())
    return order_sentinel_blocks(s)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="*",
                    help="显式 JSONL 文件、目录或通配符；默认递归读取 runs/**/*.jsonl")
    ap.add_argument("--adnimerge", default="")
    ap.add_argument("--cards", default="data/cards.json")
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()
    paths = discover_jsonl(a.jsonl)
    if not paths:
        ap.error("未找到可分析的 JSONL（runs/_smoke/ 会被自动排除）")
    os.makedirs(a.outdir, exist_ok=True)

    d = load_calls(paths)
    main = d[(~d.is_sentinel) & d.ok]
    sent = d[d.is_sentinel & d.ok]

    # 1) 长表
    try: main.to_parquet(f"{a.outdir}/calls.parquet", index=False)
    except Exception: main.to_csv(f"{a.outdir}/calls.csv", index=False)

    # 2) 患者 x 条件 x 实验臂（重复聚合）
    g = (main.groupby(["arm","temperature","RID","cond","cond_family","belief_level",
                       "source","firewall","dx24","dx36"], dropna=False)
              .agg(risk_mean=("risk","mean"), risk_sd=("risk","std"), n=("risk","size"),
                   risk_min=("risk","min"), risk_max=("risk","max"),
                   traj_mode=("traj", lambda s: s.mode().iat[0] if len(s.mode()) else None),
                   conf_mode=("conf", lambda s: s.mode().iat[0] if len(s.mode()) else None))
              .reset_index())
    g = _add_compatible_model(g)
    g.to_csv(f"{a.outdir}/patient_cond.csv", index=False)

    # 3) 宽表：条件铺成列，便于任意配对对比
    # set_index/unstack 会保留未发送 temperature 的模型（temperature 为 NaN）；
    # pivot_table 默认会丢掉这类索引行。
    w = (g.set_index(["arm","temperature","RID","cond"])["risk_mean"]
          .unstack("cond").reset_index())
    w.columns.name = None
    w = _add_compatible_model(w)
    meta = patient_meta(a.adnimerge, a.cards)
    w = w.merge(meta, on="RID", how="left")
    # 常用派生量，省得每次重算
    if {"src_self","clean_slate"} <= set(w.columns):
        w["eff_self"]       = w.src_self - w.clean_slate
    for c in ["src_physician","src_prior_ai","src_patient","src_patient_nat","src_none",
              "ctrl_irrelevant","fw_seed_high","fw_clean"]:
        if c in w.columns and "clean_slate" in w.columns:
            w["eff_" + c.replace("src_","").replace("fw_","fw_")] = w[c] - w.clean_slate
    if {"fw_seed_high","fw_clean"} <= set(w.columns):
        w["eff_fw_net"] = w.fw_seed_high - w.fw_clean          # 防火墙下的净污染
    if {"src_self","src_self_low"} <= set(w.columns):
        w["eff_highlow"] = w.src_self - w.src_self_low         # 因果对比
    w.to_csv(f"{a.outdir}/wide.csv", index=False)
    meta.to_csv(f"{a.outdir}/patients.csv", index=False)

    # 4) 哨兵漂移
    if len(sent):
        s = summarize_sentinels(sent)
        s.to_csv(f"{a.outdir}/sentinel.csv", index=False)

    # 5) QC
    L = []
    L.append(f"文件 {len(paths)}  调用 {len(d)}  主任务 {len(main)}  哨兵 {len(sent)}")
    if d.attrs.get("dedup_key_available"):
        L.append(f"重复调用键 {d.attrs.get('duplicate_groups', 0)} 组，安全去重 "
                 f"{d.attrs.get('duplicate_rows_removed', 0)} 行"
                 "（run/provider/arm/month/prompt 分隔）")
    else:
        L.append("重复调用键未检查：输入缺少完整的 run/provider/arm/month/prompt 键")
    L.append(f"失败率 {(1-d.ok.mean())*100:.2f}%   重试>1 的比例 {(d.attempts>1).mean()*100:.2f}%")
    if "prompt_tokens" in d:
        L.append(f"token 合计 输入{d.prompt_tokens.sum()/1e6:.1f}M 输出{d.completion_tokens.sum()/1e6:.1f}M")
    L.append(f"延迟 中位{d.latency_ms.median():.0f}ms P95 {d.latency_ms.quantile(.95):.0f}ms")
    L.append("")
    L.append("== 运行间可复现性（同患者同条件跨重复）==")
    r = (main.groupby(["arm","cond"]).risk
             .agg(pooled_sd=lambda s: s.std()).reset_index())
    within = (main.groupby(["arm","cond","RID"]).risk.std()
                  .groupby(level=[0,1]).mean().rename("within_sd").reset_index())
    L.append(within.round(2).to_string(index=False))
    L.append("")
    L.append("== 时间位置与结果是否相关（应≈0，否则存在run内漂移）==")
    for (mo, c), gg in main.groupby(["arm","cond"]):
        if len(gg) > 50:
            L.append(f"  {mo} {c:18s} r(order, risk) = {np.corrcoef(gg['order'], gg.risk)[0,1]:+.3f}")
    open(f"{a.outdir}/qc.txt","w",encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:8])); print(f"\n已写出 -> {a.outdir}/")
