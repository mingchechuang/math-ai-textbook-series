<<<PATCH 01>>>
<<<OLD>>>
在以下條件下，經驗 ECE 的期望值大於或等於真實的分箱校準誤差：
<<<NEW>>>
在以下條件下，經驗 ECE 的期望值不低於母體固定分箱 ECE：
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
因此，ECE 僅作為相對指標，用於比較同條件下的不同模型，不可作為絕對品質保證。
<<<NEW>>>
ECE 是依賴資料、樣本量與分箱規則的描述性估計量；可在相同評估程序下輔助比較，但不可單獨作為絕對品質或安全保證。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
**解釋**：
*   PPL 上升表示 OOD 數據的語言模式與訓練集差異大（如不同術語、異常模式）。
*   ECE 上升表示模型在 OOD 上的機率分佈更失準。
*   **分群結果**：若將 OOD 數據分為「常規操作」（Token 數 800）與「異常告警」（Token 數 200）兩群。
    *   常規操作 Mean NLL = 1.5 $\\rightarrow$ PPL $\\approx 4.48$.
    *   異常告警 Mean NLL = 5.5 $\\rightarrow$ PPL $\\approx 244.7$.
    *   整體 PPL 並非兩群 PPL 的平均，而是基於總 NLL 加權。
    *   整體 Mean NLL = $(800 \\times 1.5 + 200 \\times 5.5) / 1000 = (1200 + 1100) / 1000 = 2.3$.
    *   整體 PPL = $\\exp(2.3) \\approx 9.97$.
    *   這表明模型對新異常模式缺乏泛化能力。
<<<NEW>>>
**解釋**：
*   PPL 上升表示模型對評估集真實 token 指派的平均概率較低；僅憑此數值不能確定差異原因。
*   ECE 上升表示依本例的分箱與預測事件計算出的校準差距較大。
*   **分群結果**：若將 OOD 數據分為「常規操作」（有效 token 數 800）與「異常告警」（有效 token 數 200）兩群。
    *   常規操作 Mean NLL = 1.5 $\\rightarrow$ PPL $\\approx 4.48$.
    *   異常告警 Mean NLL = 5.5 $\\rightarrow$ PPL $\\approx 244.7$.
    *   整體 PPL 並非兩群 PPL 的平均，而是基於總 NLL 加權。
    *   整體 Mean NLL = $(800 \\times 1.5 + 200 \\times 5.5) / 1000 = (1200 + 1100) / 1000 = 2.3$.
    *   整體 PPL = $\\exp(2.3) \\approx 9.97$.
    *   這表示模型與告警群的 token 分布匹配較差；是否屬於語義泛化失敗，仍須控制詞彙、模板、長度、tokenizer 與標註規則後另行判定。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
    """
    Compute stable log-softmax.
    Input: logits (N, L, V)
    Output: log_probs (N, L, V)
    """
<<<NEW>>>
    """
    Compute stable log-softmax.
    Input: logits (N, L, V); all entries, including masked positions, must
    be finite.
    Output: log_probs (N, L, V)
    """
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
    validation = evaluate([r for r in records if r[2] == "valid"])
    tau = select_threshold(validation["confidence"],
                           validation["correctness"], [0, .8, 1], .5)
    for split in ("id", "ood"):
<<<NEW>>>
    validation = evaluate([r for r in records if r[2] == "valid"])
    thresholds = [0, .8, 1]
    target_coverage = .5
    tau = select_threshold(validation["confidence"],
                           validation["correctness"], thresholds, target_coverage)
    result["validation", "all"] = {
        **validation,
        "curve": risk_coverage(validation["confidence"],
                               validation["correctness"], thresholds),
    }
    result["selection"] = {
        "target_coverage": target_coverage,
        "thresholds": thresholds,
        "selected_tau": tau,
    }
    for split in ("id", "ood"):
<<<END>>>