"""
牵引越小，准确率越高？—— 对已有数据的再分析，不需要任何新的模型调用。

    python pull_vs_accuracy.py            # 两部分都跑
    python pull_vs_accuracy.py main       # 只跑主实验臂
    python pull_vs_accuracy.py pipeline   # 只跑管线臂

第一部分（主实验臂，figdata/patient_long.csv + perpatient_pull.csv）
    注入的 80%/20% 先验与真实结局无关，所以任何 β>0 在期望上必然损害准确率。
    这一部分不是在"检验"这个命题，而是在"标定"它：每单位 β 值多少个百分点。

第二部分（管线臂，figdata/pipeline8_cond.csv + pipeline8_summary.csv）
    Agent A 的输出是真有信息量的，所以最优 β 并不是 0。
    这一部分才是真正的检验：在同样能看到 A 的输出的前提下，
    牵引被压下去的条件（firewall / source-labelled）错得更少吗？
"""
import sys
import numpy as np, pandas as pd
from scipy import stats

D, TH = "figdata", 50


def fe_fit(df, xcols, ycol, group="model", weights=None):
    """y ~ xcols + group 固定效应；返回 {xcol: (系数, SE, P)}"""
    X = pd.get_dummies(df[group], drop_first=True).astype(float).reset_index(drop=True)
    for c in xcols[::-1]:
        X.insert(0, c, pd.to_numeric(df[c], errors="coerce").values)
    X.insert(0, "const", 1.0)
    Xv = X.values.astype(float); yv = pd.to_numeric(df[ycol], errors="coerce").values.astype(float)
    ok = np.isfinite(Xv).all(1) & np.isfinite(yv)
    Xv, yv = Xv[ok], yv[ok]
    if weights is not None:
        w = np.sqrt(np.asarray(weights, float)[ok]); Xv = Xv * w[:, None]; yv = yv * w
    b, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    r = yv - Xv @ b; dof = len(yv) - np.linalg.matrix_rank(Xv)
    if dof < 5: raise RuntimeError(f"残差自由度只有 {dof}，不估计")
    s2 = r @ r / dof; XtXi = np.linalg.pinv(Xv.T @ Xv)
    out = {}
    for c in xcols:
        i = list(X.columns).index(c); se = np.sqrt(s2 * XtXi[i, i])
        out[c] = (b[i], se, 2 * stats.norm.sf(abs(b[i] / se)))
    return out, len(yv), dof


# ====================================================== 第一部分：主实验臂
def main_arm():
    L  = pd.read_csv(f"{D}/patient_long.csv")
    PP = pd.read_csv(f"{D}/perpatient_pull.csv")
    W  = L.pivot_table(index=["model", "RID"], columns="cond", values="risk").reset_index()
    W  = (W.merge(L.groupby(["model", "RID"]).dx36.first().reset_index(), on=["model", "RID"])
            .merge(PP, on=["model", "RID"], how="left"))
    W  = W[W.dx36.isin(["CN", "MCI", "Dementia"])].copy()
    W["y"] = (W.dx36 == "Dementia").astype(int)
    hit = lambda c: ((W[c] >= TH).astype(int) == W.y).astype(int)
    for a, c in [("ok0", "clean_slate"), ("ok80", "src_self"), ("ok20", "src_self_low"),
                 ("okfw0", "fw_clean"), ("okfw80", "fw_seed_high"), ("okfw20", "fw_seed_low")]:
        W[a] = hit(c)
    # 80% 与 20% 取平均 = "随机塞进来一条自我归属的先验"的期望效果
    W["d_acc"]   = (W.ok80 + W.ok20) / 2 - W.ok0
    W["d_accfw"] = (W.okfw80 + W.okfw20) / 2 - W.okfw0
    W["dist"]    = (W.clean_slate - TH).abs()          # 离决策线多远 = 这个病人本来有多好判
    Wb = W[W.beta_i.notna()].copy()

    print("=" * 74)
    print("第一部分  主实验臂：注入的先验与结局无关，所以 β 只会伤害准确率")
    print("=" * 74)

    print("\n[1.1] 模型层面  平均 βᵢ  vs  注入先验造成的准确率变化")
    g = (W.groupby("model").agg(n=("y", "size"), beta=("beta_i", "mean"),
                                acc0=("ok0", "mean"), d=("d_acc", "mean"))
           .reset_index().sort_values("beta", ascending=False))
    g["基线准确率%"] = (g.acc0 * 100).round(1); g["准确率变化pp"] = (g.d * 100).round(2)
    print(g[["model", "n", "beta", "基线准确率%", "准确率变化pp"]].round(3).to_string(index=False))
    rs, ps = stats.spearmanr(g.beta, g.d)
    print(f"  Spearman rho = {rs:+.3f}   P = {ps:.4f}   (n = {len(g)} 个模型)")

    print("\n[1.2] 病人层面  固定效应回归：准确率变化 ~ βᵢ + 模型FE")
    for cols, lab in [(["beta_i"], "仅 βᵢ"), (["beta_i", "dist"], "βᵢ + 到决策线的距离")]:
        o, n, dof = fe_fit(Wb, cols, "d_acc")
        b, se, p = o["beta_i"]
        print(f"  {lab:<22s} {b*100:+7.2f} pp / 单位βᵢ   SE {se*100:.2f}   P = {p:.1e}"
              f"   (n={n})   → βᵢ 0→0.3 时 {b*0.3*100:+.2f} pp")
    print("  第二行扣除了『这个病人本来离 50% 线多远』这个混杂——高 βᵢ 的病人本来就更靠近阈值。")

    print("\n[1.3] 病人层面  按 βᵢ 三分位（每个模型内部分位）")
    Wb["tert"] = Wb.groupby("model").beta_i.transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["低", "中", "高"]))
    t = (Wb.groupby("tert", observed=True)
           .agg(n=("y", "size"), beta=("beta_i", "mean"), acc0=("ok0", "mean"), d=("d_acc", "mean"))
           .reset_index())
    t["基线准确率%"] = (t.acc0 * 100).round(1); t["准确率变化pp"] = (t.d * 100).round(2)
    print(t[["tert", "n", "beta", "基线准确率%", "准确率变化pp"]].round(3).to_string(index=False))
    lo, hi = Wb[Wb.tert == "低"].d_acc, Wb[Wb.tert == "高"].d_acc
    print(f"  高 − 低 = {(hi.mean()-lo.mean())*100:+.2f} pp   Welch P = "
          f"{stats.ttest_ind(hi, lo, equal_var=False).pvalue:.1e}")

    print("\n[1.4] firewall 把牵引压下去之后，准确率损失收回了多少")
    f = W.groupby("model").agg(d=("d_acc", "mean"), dfw=("d_accfw", "mean")).reset_index()
    f["无firewall pp"] = (f.d * 100).round(2); f["有firewall pp"] = (f.dfw * 100).round(2)
    f["改善 pp"] = ((f.dfw - f.d) * 100).round(2)
    print(f[["model", "无firewall pp", "有firewall pp", "改善 pp"]].to_string(index=False))
    pt = stats.ttest_rel(f.dfw, f.d).pvalue
    print(f"  改善的模型 {(f['改善 pp'] > 0).sum()}/{len(f)}   平均 {f['改善 pp'].mean():+.2f} pp"
          f"   配对 t 检验 P = {pt:.4f}")
    print("  ⚠ 这一格大概率不显著：firewall 可靠地压低了 β，但在这一臂里压低 β 并不等于准确率明显回升，")
    print("    因为注入的先验一半推高一半压低，平均下来对准确率的净影响本就不大。如实报告，不要强说。")


# ====================================================== 第二部分：管线臂
def pipeline_arm():
    try:
        C = pd.read_csv(f"{D}/pipeline8_cond.csv")
        S = pd.read_csv(f"{D}/pipeline8_summary.csv").set_index("model")
    except FileNotFoundError:
        print("\n找不到 figdata/pipeline8_cond.csv —— 先跑 python extract_pipeline8.py")
        return
    C = C.merge(S[["n36"]], left_on="model", right_index=True)
    C["errors"]    = C.err_retained + C.err_introduced
    C["err_rate"]  = C.errors / C.n36 * 100
    C["acc"]       = 100 - C.err_rate

    print("\n" + "=" * 74)
    print("第二部分  管线臂：A 的输出是真有信息量的，所以这才是真正的检验")
    print("=" * 74)

    print("\n[2.1] 各条件的平均表现（8 个模型）")
    piv = (C.groupby("cond").agg(beta=("beta", "mean"), 错误率=("err_rate", "mean"),
                                 保留A的错=("err_retained", "sum"), 引入新错=("err_introduced", "sum"))
             .reindex(["B_independent", "B_serial", "B_sourced", "B_firewall", "B_irrelevant"]))
    print(piv.round(3).to_string())
    print("  B_independent 的 β 没有意义（它看不到 A），列在这里只作为『完全不参考上游』的准确率基准。")

    ref = piv.loc["B_independent", "错误率"]
    print(f"\n  相对『完全不参考上游』({ref:.2f}% 错误率）：")
    for c in ["B_serial", "B_sourced", "B_firewall", "B_irrelevant"]:
        print(f"    {c:<15s} {piv.loc[c,'错误率'] - ref:+.2f} pp   (β = {piv.loc[c,'beta']:.3f})")
    best = piv.drop("B_independent").错误率.idxmin()
    print(f"  错误最少的条件：{best}")

    print("\n[2.2] 关键回归：错误率 ~ β + 模型FE（只用能看到 A 的四个条件）")
    sub = C[C.cond != "B_independent"].copy()
    o, n, dof = fe_fit(sub, ["beta"], "err_rate")
    b, se, p = o["beta"]
    print(f"  β 系数 = {b:+.2f} pp / 单位β   SE {se:.2f}   P = {p:.4f}   (n = {n} 个 模型×条件, dof={dof})")
    print(f"  → β 由 0.12（朴素串接）降到 0.03（firewall），错误率变化 {b*(0.03-0.12):+.2f} pp")
    if p < 0.05 and b > 0:
        print("  ✓ 牵引越小错误越少，在 A 的输出真有信息量的条件下成立。")
    else:
        print("  ✗ 没达到显著。这说明『压低牵引 → 准确率提升』在你的数据里只是方向性的，")
        print("    论文里就写成『牵引不带来准确率收益』而不是『牵引有害』——后者你证不了。")

    print("\n[2.3] 为什么：把错误拆成两种")
    d = (C.groupby("cond")[["err_retained", "err_introduced"]].sum()
           .reindex(["B_independent", "B_serial", "B_sourced", "B_firewall", "B_irrelevant"]))
    d["合计"] = d.sum(1)
    d.columns = ["A错时B跟着错", "A对时B反而错", "合计"]
    print(d.to_string())
    print("  『A错时B跟着错』= 牵引在传播上游错误；『A对时B反而错』= B 没用上 A 的正确信息。")
    print("  一个好的交接方式应当同时压低两列。对比 B_serial 与 B_firewall 这两行即可。")

    print("\n[2.4] 按 |gap| 分层：牵引是否随上游可信度调整")
    try:
        T = pd.read_csv(f"{D}/pipeline8_strata.csv")
        print(T.groupby("stratum").agg(n=("n", "sum"), beta=("beta", "mean"),
                                       平均绝对平移=("mean_abs", "mean")).round(3).to_string())
        print("  理想情况下 |gap| 越大（A 与 B 独立判断差得越多，A 越可疑）β 应当越小。")
    except FileNotFoundError:
        print("  （没有 pipeline8_strata.csv，跳过）")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "main"): main_arm()
    if which in ("both", "pipeline"): pipeline_arm()
