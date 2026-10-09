<<<PATCH 24>>>
<<<OLD>>>
**限制**：此證明依賴於分箱邊界固定且樣本 i.i.d.。若分箱邊界依資料估計（自適應分箱），或樣本非獨立，則不等式方向可能改變或無法直接成立。因此，ECE 僅作為相對指標，用於比較同條件下的不同模型，不可作為絕對品質保證。
<<<NEW>>>
**限制**：此證明依賴於分箱邊界固定且樣本 i.i.d.。若分箱邊界依資料估計（自適應分箱），或樣本非獨立，則不等式方向可能改變或無法直接成立。ECE 是依賴資料、樣本量與分箱規則的描述性估計量，可描述單一模型並在相同評估程序下輔助比較，但不能單獨作為絕對品質或安全保證。
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
    # 1. Validation
    if logits.ndim != 3:
<<<NEW>>>
    # 1. Validation；mask 只控制 reduction，不豁免無效位置的 NaN/Inf。
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels)
    valid_mask = np.asarray(valid_mask)
    if logits.ndim != 3:
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
    perplexity = np.exp(mean_nll)
    
    if not np.isfinite(perplexity):
        print(f"Warning: PPL overflow. Mean NLL is {mean_nll}.")
        
    return perplexity, mean_nll
<<<NEW>>>
    # 明確拒絕不可表示的 PPL；mean_nll 仍可供呼叫端另行記錄。
    if not np.isfinite(mean_nll) or mean_nll > np.log(np.finfo(float).max):
        raise OverflowError("PPL not representable; record mean NLL instead.")
    perplexity = np.exp(mean_nll)
    return perplexity, mean_nll
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
        N, C = probs.shape
        if labels.shape != (N,):
<<<NEW>>>
        N, C = probs.shape
        if N == 0 or C == 0:
            raise ValueError("N and C must be positive.")
        if labels.shape != (N,):
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
4.  **解答**：
    *   繪製 R-Curve：x 軸為 Coverage $C(\tau)$，y 軸為 Risk $R(\tau)$。
    *   OOD 集通常具有更高的基礎錯誤率和更大的信心分布偏差。
    *   因此，為了達到相同的 Risk（例如 5%），OOD 集可能需要更高的 $\tau$（更激进的拒答），從而導致更低的 Coverage。這反映了模型在未知分布上需要更保守的策略以控制風險。
<<<NEW>>>
4.  **解答**：
    *   先按來源文件與時間切分，再建立窗口；詞表只由訓練資料擬合。另保留驗證集、ID 測試集及 OOD 測試集，記錄每個單步決策的信心、正誤及來源。
    *   預先固定閾值候選、信心定義與最低 Coverage；只在驗證集呼叫 `risk_coverage`，以 Coverage 為橫軸、Risk 為縱軸。接受數為零的點標示 Risk 未定義，不拿它與其他點比較。
    *   用 `select_threshold` 在驗證集的可行候選中選 $\tau^\star$；若沒有候選達到約束，報告不可達。固定此閾值，再分別對 ID 與 OOD 測試集報告決策數、接受數、Coverage 及 Risk，不用測試結果重選閾值。
    *   OOD 可能須以更高閾值換取較低 Coverage，也可能因高信心錯誤而完全無法靠提高閾值達到目標 Risk；只有實際曲線能區分兩者。單步結果不得當成整段回答的正確性或安全性。
<<<END>>>