"""
全量分析 —— 读 out/wide.csv 与 out/patient_cond.csv，产出论文级统计 + 每张图的 figure-ready CSV。
用法： python analyze_full.py --outdir out
"""
import os, argparse, hashlib, re, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from export import classify_dx36, order_sentinel_blocks

def filename_component(value):
    original = str(value)
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", original).rstrip(" .") or "unnamed"
    if cleaned != original:
        cleaned += "_" + hashlib.sha1(original.encode("utf-8")).hexdigest()[:8]
    return cleaned

def logit(p, eps=1e-3):
    p = np.clip(np.asarray(p, float)/100, eps, 1-eps); return np.log(p/(1-p))

def paired(x, y, label):
    m = pd.DataFrame({"a":x,"b":y}).dropna(); d = m.a - m.b
    if len(m) < 5: return dict(label=label, n=len(m))
    p = stats.wilcoxon(m.a, m.b).pvalue if d.abs().sum() else np.nan
    return dict(label=label, n=len(m), mean=d.mean(), sd=d.std(),
                lo=d.mean()-1.96*d.sem(), hi=d.mean()+1.96*d.sem(), p=p,
                d_cohen=d.mean()/d.std() if d.std() else np.nan,
                pct_moved=(d.abs()>=5).mean()*100)

def show(rows, title):
    print(f"\n=== {title}")
    t = pd.DataFrame(rows)
    if t.empty: return t
    cols = [c for c in ["label","n","mean","lo","hi","sd","p","d_cohen","pct_moved"] if c in t]
    print(t[cols].round(3).to_string(index=False)); return t

def clinical_metrics(W, cond):
    """仅在明确的 Dementia vs MCI/CN 36 月结局上计算临床指标。"""
    x = W.loc[W.y.notna() & W[cond].notna()].copy()
    y = x.y.astype(int)
    p50 = (x[cond] >= 50).astype(int)
    paired_clean = x.clean_slate.notna()
    flip = ((x.loc[paired_clean, cond] >= 50).astype(int) !=
            (x.loc[paired_clean, "clean_slate"] >= 50).astype(int))
    return dict(cond=cond, n_observed=len(x), mean_risk=x[cond].mean(),
                AUC=roc_auc_score(y, x[cond]) if y.nunique() > 1 else np.nan,
                acc=(p50 == y).mean(),
                FP=int(((p50 == 1) & (y == 0)).sum()),
                FN=int(((p50 == 0) & (y == 1)).sum()),
                pct_flip_vs_clean=flip.mean()*100 if len(flip) else np.nan)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default="out"); a = ap.parse_args()
    O = a.outdir; FIG = f"{O}/figdata"; os.makedirs(FIG, exist_ok=True)
    w = pd.read_csv(f"{O}/wide.csv")
    print(f"患者 {w.RID.nunique()}   模型 {sorted(w.model.unique())}")
    print("36月结局口径：Dementia=1；MCI/CN=0；缺失及其他标签排除")

    for model, W in w.groupby("model"):
        model_file = filename_component(model)
        print("\n" + "#"*72); print("#", model); print("#"*72)
        W = W.copy(); W["y"] = classify_dx36(W.dx36)
        print(f"36月结局已观测且可二分类 {W.y.notna().sum()}/{len(W)}")

        # ---------- 图1 来源层级 ----------
        rows = []
        for c in ["ctrl_irrelevant","src_patient","src_patient_nat","src_prior_ai",
                  "src_physician","src_none","src_self"]:
            if c in W: rows.append(paired(W[c], W.clean_slate, c))
        t1 = show(rows, "图1 各来源相对 clean_slate 的平移（内容恒定80%）")
        t1.to_csv(f"{FIG}/fig1_source_{model_file}.csv", index=False)
        if {"src_self","src_physician"} <= set(W):
            show([paired(W.src_self, W.src_physician, "self - physician"),
                  paired(W.src_self, W.src_prior_ai,  "self - prior_ai"),
                  paired(W.src_self, W.ctrl_irrelevant,"self - irrelevant")],
                 "特异性检验")

        # ---------- 图2 防火墙 ----------
        rows = []
        for a_, b_, lab in [("src_self","clean_slate","污染(无防火墙)"),
                            ("fw_seed_high","fw_clean","污染(有防火墙,净)"),
                            ("fw_clean","clean_slate","防火墙指令副作用"),
                            ("fw_seed_high","fw_seed_low","防火墙内 高-低"),
                            ("src_self","src_self_low","无防火墙 高-低")]:
            if {a_,b_} <= set(W): rows.append(paired(W[a_], W[b_], lab))
        t2 = show(rows, "图2 信念防火墙")
        t2.to_csv(f"{FIG}/fig2_firewall_{model_file}.csv", index=False)
        if {"src_self","fw_seed_high","fw_clean","clean_slate"} <= set(W):
            a0 = (W.src_self-W.clean_slate).mean(); a1 = (W.fw_seed_high-W.fw_clean).mean()
            print(f"    污染残留 = {a1/a0*100:.0f}%")

        # ---------- 图3 剂量反应 ----------
        dose = [c for c in W if c.startswith("dose_")]
        if dose:
            xs = sorted(int(c.split("_")[1]) for c in dose)
            ys = [W[f"dose_{v}"].mean() for v in xs]
            sl, ic = np.polyfit(xs, ys, 1)
            print(f"\n=== 图3 剂量反应  斜率={sl:.3f}  r={np.corrcoef(xs,ys)[0,1]:.3f}")
            pd.DataFrame({"injected":xs,"output_mean":ys,
                          "output_sem":[W[f'dose_{v}'].sem() for v in xs]}
                        ).to_csv(f"{FIG}/fig3_dose_{model_file}.csv", index=False)
            print(pd.DataFrame({"注入":xs,"输出均值":np.round(ys,1)}).to_string(index=False))

        # ---------- 图4 亚组：谁最容易被污染 ----------
        if "eff_self" not in W: W["eff_self"] = W.src_self - W.clean_slate
        W["eff_logit"] = logit(W.src_self) - logit(W.clean_slate)      # 天花板校正
        sub = []
        def add(name, mask):
            g = W[mask]
            if len(g) >= 20:
                sub.append(dict(subgroup=name, n=len(g), eff=g.eff_self.mean(),
                                sem=g.eff_self.sem(), eff_logit=g.eff_logit.mean()))
        add("全部", W.RID.notna())
        add("36月未转化(MCI/CN)", W.y.eq(0).fillna(False))
        add("36月转化(Dementia)", W.y.eq(1).fillna(False))
        q = W.clean_slate.quantile([.33,.67]).values
        add("基线风险低1/3", W.clean_slate <= q[0])
        add("基线风险中1/3", (W.clean_slate > q[0]) & (W.clean_slate <= q[1]))
        add("基线风险高1/3", W.clean_slate > q[1])
        if "discordant_stable" in W: add("生物高危但临床稳定", W.discordant_stable == 1)
        if "APOE4" in W: add("APOE4阳性", W.APOE4 >= 1); add("APOE4阴性", W.APOE4 == 0)
        if "AMYLOID_POS" in W: add("淀粉样阳性", W.AMYLOID_POS == 1); add("淀粉样阴性", W.AMYLOID_POS == 0)
        t4 = pd.DataFrame(sub)
        print("\n=== 图4 亚组（eff_logit 为天花板校正后）")
        print(t4.round(2).to_string(index=False))
        t4.to_csv(f"{FIG}/fig4_subgroup_{model_file}.csv", index=False)
        ok = W.dropna(subset=["eff_self","clean_slate"])
        print(f"    效应 ~ 基线风险  r={np.corrcoef(ok.clean_slate, ok.eff_self)[0,1]:+.3f}"
              f"   (logit尺度 r={np.corrcoef(logit(ok.clean_slate), ok.eff_logit)[0,1]:+.3f})")

        # ---------- 图5 临床后果：AUC不变但阈值错误变多 ----------
        rows = []
        for c in ["clean_slate","ctrl_irrelevant","src_physician","src_self","src_self_low",
                  "fw_seed_high"]:
            if c not in W: continue
            rows.append(clinical_metrics(W, c))
        t5 = pd.DataFrame(rows)
        print("\n=== 图5 判别力 vs 阈值后果（★ AUC 几乎不变而 FP/FN 变化 = 现行评测失明）")
        print(t5.round(3).to_string(index=False))
        t5.to_csv(f"{FIG}/fig5_clinical_{model_file}.csv", index=False)

    # ---------- 图6 跨模型 ----------
    if w.model.nunique() > 1:
        rows = []
        for model, W in w.groupby("model"):
            if {"src_self","clean_slate"} <= set(W):
                r = paired(W.src_self, W.clean_slate, model); r["model"] = model
                if {"fw_seed_high","fw_clean"} <= set(W):
                    r["fw_residual_pct"] = ((W.fw_seed_high-W.fw_clean).mean() /
                                            (W.src_self-W.clean_slate).mean()*100)
                rows.append(r)
        t6 = show(rows, "图6 跨模型对比：自我锚定强度与防火墙残留")
        t6.to_csv(f"{FIG}/fig6_models.csv", index=False)

    # ---------- 漂移 ----------
    sp = f"{O}/sentinel.csv"
    if os.path.exists(sp):
        s = pd.read_csv(sp)
        # 即使输入是旧版 export 产生的倒序 block，也按原始 rep/ts 重建时间顺序。
        s = order_sentinel_blocks(s)
        print("\n=== 哨兵漂移（长任务内部）")
        for (run_id,m,c), g in s.groupby(["run_id","model","cond"]):
            if len(g) > 2:
                g = g.sort_values("block")
                r = np.corrcoef(g.block, g.risk_mean)[0,1]
                print(f"  run={run_id} {m} {c:16s} 区块数{len(g):3d}  均值{g.risk_mean.mean():5.1f}  "
                      f"跨区块SD {g.risk_mean.std():.2f}  r(时间,风险)={r:+.2f}")
    print(f"\nfigure-ready CSV -> {FIG}/")
