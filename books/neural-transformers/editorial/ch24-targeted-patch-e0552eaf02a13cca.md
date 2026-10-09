<<<PATCH 24>>>
<<<OLD>>>
對 $N_b$ 取全期望時，定義空箱的貢獻為零；非空箱的貢獻為 $\frac{N_b}{N}|X_b|$。不必計算空箱中未定義的 $X_b$，下式只對 $n=1,\ldots,N$ 求和：
$$ E\left[ \frac{N_b}{N} |X_b| \right] = \sum_{n=1}^{N} \frac{n}{N} P(N_b=n) E[ |X_b| \mid N_b=n ] $$
$$ \geq \sum_{n=1}^{N} \frac{n}{N} P(N_b=n) | \delta_b | $$
$$ = | \delta_b | \cdot \frac{1}{N} E[N_b] $$
由於 $E[N_b] = N q_b$，所以：
$$ E\left[ \frac{N_b}{N} |X_b| \right] \geq | \delta_b | \cdot q_b $$

對所有箱求和：
$$ E[\text{ECE}_{\text{empirical}}] = \sum_{b=1}^{B} E\left[ \frac{N_b}{N} |X_b| \right] \geq \sum_{b=1}^{B} q_b | \delta_b | = \text{ECE}_{\text{true, binned}} $$
<<<NEW>>>
令箱貢獻 $Y_b$ 在 $N_b=0$ 時為 $0$，在 $N_b>0$ 時為 $\frac{N_b}{N}|X_b|$。這使 $Y_b$ 在整個樣本空間均有定義，不必計算空箱中未定義的 $X_b$。只對 $n=1,\ldots,N$ 求和：
$$ E[Y_b] = \sum_{n=1}^{N} \frac{n}{N} P(N_b=n) E[ |X_b| \mid N_b=n ] $$
$$ \geq \sum_{n=1}^{N} \frac{n}{N} P(N_b=n) | \delta_b | $$
$$ = | \delta_b | \cdot \frac{1}{N} E[N_b] $$
由於 $E[N_b] = N q_b$，所以 $E[Y_b]\geq q_b|\delta_b|$。對所有箱求和，且 $\text{ECE}_{\text{empirical}}=\sum_bY_b$，得到：
$$ E[\text{ECE}_{\text{empirical}}] = \sum_{b=1}^{B} E[Y_b] \geq \sum_{b=1}^{B} q_b | \delta_b | = \text{ECE}_{\text{true, binned}} $$
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
def synthetic_eval():
    """確定性、未訓練的示意評估；每個 source_id 僅屬一個 split。"""
    records = [
        ("id-a", 1, "id", "routine", [2., 0.], 0),
        ("id-b", 2, "id", "alert", [0., 2.], 1),
        ("ood-a", 3, "ood", "routine", [1., 0.], 0),
        ("ood-b", 4, "ood", "alert", [2., 0.], 1),
    ]
    assert len({r[0] for r in records}) == len(records)
    result = {}
    for split in ("id", "ood"):
        for group in ("routine", "alert"):
            part = [r for r in records if r[2] == split and r[3] == group]
            logits = np.array([r[4] for r in part])[:, None, :]
            labels = np.array([r[5] for r in part], dtype=int)[:, None]
            mask = np.ones(labels.shape, dtype=bool)
            ppl, mean_nll = compute_global_ppl(logits, labels, mask)
            lp = compute_stable_log_softmax(logits)[:, 0, :]
            confidence = np.exp(np.max(lp, axis=-1))
            correct = (np.argmax(lp, axis=-1) == labels[:, 0]).astype(int)
            ece, _ = compute_ece(confidence, correct, [0, 0.5, 1])
            result[split, group] = {
                "tokens": int(np.sum(mask)), "total_nll": mean_nll * mask.sum(),
                "ppl": ppl, "events": len(correct), "ece": ece,
                "decisions": len(correct),
                "curve": risk_coverage(confidence, correct, [0, 0.8, 1]),
            }
    return result
<<<NEW>>>
def check_sources(records):
    """同一來源可有多個時間點，但不得跨 split。"""
    owners = {}
    for source, time, split, group, logits, label in records:
        if source in owners and owners[source] != split:
            raise ValueError("Source crosses splits.")
        owners[source] = split

def synthetic_eval():
    """固定小資料；valid 只選閾值，id/ood 只評估。"""
    records = [
        ("val-a", 1, "valid", "routine", [2., 0.], 0),
        ("val-b", 2, "valid", "alert", [1., 0.], 1),
        ("id-a", 1, "id", "routine", [2., 0.], 0),
        ("id-a", 2, "id", "routine", [1., 0.], 0),
        ("id-b", 1, "id", "alert", [0., 2.], 1),
        ("ood-a", 1, "ood", "routine", [1., 0.], 0),
        ("ood-a", 2, "ood", "routine", [1., 0.], 0),
        ("ood-b", 1, "ood", "alert", [2., 0.], 1),
    ]
    check_sources(records)
    result = {}

    def evaluate(part):
        if not part:
            raise ValueError("Empty evaluation group.")
        logits = np.array([r[4] for r in part], dtype=float)[:, None, :]
        labels = np.array([r[5] for r in part], dtype=int)[:, None]
        mask = np.ones(labels.shape, dtype=bool)
        ppl, mean_nll = compute_global_ppl(logits, labels, mask)
        lp = compute_stable_log_softmax(logits)[:, 0, :]
        confidence = np.exp(np.max(lp, axis=-1))
        correct = (np.argmax(lp, axis=-1) == labels[:, 0]).astype(int)
        ece, _ = compute_ece(confidence, correct, [0, .75, 1])
        return {
            "tokens": int(mask.sum()), "total_nll": float(mean_nll * mask.sum()),
            "ppl": float(ppl), "events": len(correct), "ece": float(ece),
            "decisions": len(correct), "confidence": confidence,
            "correctness": correct,
        }

    validation = evaluate([r for r in records if r[2] == "valid"])
    tau = select_threshold(validation["confidence"],
                           validation["correctness"], [0, .8, 1], .5)
    for split in ("id", "ood"):
        for group in ("routine", "alert", "all"):
            part = [r for r in records if r[2] == split
                    and (group == "all" or r[3] == group)]
            row = evaluate(part)  # ECE 不平均各群；整體重新聚合事件
            row["selected_tau"] = tau
            row["selected_point"] = risk_coverage(
                row["confidence"], row["correctness"], [tau])[0]
            row["curve"] = risk_coverage(
                row["confidence"], row["correctness"], [0, .8, 1])
            result[split, group] = row
    return result
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
    result = synthetic_eval()
    assert result["ood", "alert"]["total_nll"] > result["ood", "routine"]["total_nll"]
    assert sum(result["ood", g]["tokens"] for g in ("routine", "alert")) == 2
    assert select_threshold(np.array([.8, .6]), np.array([1, 0]),
                            [0, .8, 1], .5) == .8
<<<NEW>>>
    result = synthetic_eval()
    assert result["ood", "alert"]["total_nll"] > result["ood", "routine"]["total_nll"]
    for split in ("id", "ood"):
        groups = [result[split, g] for g in ("routine", "alert")]
        whole = result[split, "all"]
        assert whole["tokens"] == sum(g["tokens"] for g in groups)
        assert np.isclose(whole["total_nll"], sum(g["total_nll"] for g in groups))
        assert np.isclose(whole["ppl"], np.exp(
            sum(g["total_nll"] for g in groups) / whole["tokens"]))
        assert whole["selected_tau"] == .8
    assert result["ood", "all"]["ppl"] > result["id", "all"]["ppl"]
    assert not np.isclose(result["ood", "all"]["ppl"],
                          np.mean([result["ood", g]["ppl"]
                                   for g in ("routine", "alert")]))
    assert select_threshold(np.array([.8, .6]), np.array([1, 0]),
                            [0, .8, 1], .5) == .8
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
    expect_value_error(compute_ece, [.5], [1], [0, .5, .5, 1])
<<<NEW>>>
    expect_value_error(compute_ece, [.5], [1], [0, .5, .5, 1])
    expect_value_error(check_sources, [
        ("same", 1, "id", "routine", [1., 0.], 0),
        ("same", 2, "ood", "routine", [1., 0.], 0),
    ])
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
2.  **ECE** 受分箱策略影響，有限樣本下存在正偏差，僅作為相對指標。
<<<NEW>>>
2.  **ECE** 依賴資料、樣本量與分箱規則；在固定分箱及獨立同分布條件下，經驗 ECE 的期望不低於母體固定分箱 ECE。它可描述單一模型並輔助比較，不能單獨作為品質或安全保證。
<<<END>>>