<<<PATCH 24>>>
<<<OLD>>>
定義母體固定分箱 ECE 為：
$$ \text{ECE}_{\text{true, binned}} = \sum_{b=1}^{B} q_b \left| E[A - S \mid S \in I_b] \right| $$
其中 $q_b = P(S \in I_b)$ 是樣本落入箱 $b$ 的母體機率。
<<<NEW>>>
令 $q_b=P(S\in I_b)$。只對 $q_b>0$ 的箱定義 $\delta_b=E[A-S\mid S\in I_b]$，並定義母體固定分箱 ECE 為：
$$ \text{ECE}_{\text{true, binned}} = \sum_{b:q_b>0} q_b|\delta_b| $$
若 $q_b=0$，條件期望無須定義，該箱的母體貢獻為零；此時 $N_b=0$ 幾乎處處，故下述樣本箱貢獻 $Y_b$ 也幾乎處處為零。以下條件抽樣與 Jensen 推導只針對 $q_b>0$ 的箱。
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
令 $\delta_b = E[ A - S \mid S \in I_b ]$，則：
$$ E[ |X_b| \mid N_b = n ] \geq | \delta_b | $$
<<<NEW>>>
因此，對 $q_b>0$ 且 $n>0$ 的箱：
$$ E[ |X_b| \mid N_b = n ] \geq | \delta_b | $$
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
由於 $E[N_b] = N q_b$，所以 $E[Y_b]\geq q_b|\delta_b|$。對所有箱求和，且 $\text{ECE}_{\text{empirical}}=\sum_bY_b$，得到：
$$ E[\text{ECE}_{\text{empirical}}] = \sum_{b=1}^{B} E[Y_b] \geq \sum_{b=1}^{B} q_b | \delta_b | = \text{ECE}_{\text{true, binned}} $$
<<<NEW>>>
對 $q_b>0$，由 $E[N_b]=Nq_b$ 得 $E[Y_b]\geq q_b|\delta_b|$；對 $q_b=0$，$E[Y_b]=0$。因此 $\text{ECE}_{\text{empirical}}=\sum_bY_b$，而
$$ E[\text{ECE}_{\text{empirical}}]=\sum_{b=1}^{B}E[Y_b]\geq\sum_{b:q_b>0}q_b|\delta_b|=\text{ECE}_{\text{true, binned}} $$
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
    validation = evaluate([r for r in records if r[2] == "valid"])
    tau = select_threshold(validation["confidence"],
                           validation["correctness"], [0, .8, 1], .5)
<<<NEW>>>
    validation = evaluate([r for r in records if r[2] == "valid"])
    thresholds = [0, .8, 1]
    target_coverage = .5
    validation["curve"] = risk_coverage(
        validation["confidence"], validation["correctness"], thresholds)
    tau = select_threshold(validation["confidence"],
                           validation["correctness"], thresholds,
                           target_coverage)
    result["validation", "all"] = validation
    result["selection"] = {
        "thresholds": thresholds, "target_coverage": target_coverage,
        "selected_tau": tau,
        "validation_point": risk_coverage(
            validation["confidence"], validation["correctness"], [tau])[0],
    }
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
*(註：以上引用為標準學術參考，具體版本與細節請查閱原始論文。本章未使用任何未公開的實測數據。)*
<<<NEW>>>
*(註：N1 僅核對摘要；N6、Guo et al. 與 Niculescu-Mizil & Caruana 尚未逐條核對，目前只列作延伸閱讀，不作已查證證據。本章未使用未公開實測資料。)*
<<<END>>>