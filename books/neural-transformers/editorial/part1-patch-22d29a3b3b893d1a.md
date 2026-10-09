<<<PATCH 01>>>
<<<OLD>>>
### 命題 1.4（有限假設類的均勻偏差界）

設 $\mathcal{H}=\{f_1,\dots,f_M\}$ 是**有限**假設類，$|M|=\mathcal{H}$ 為其基數。
<<<NEW>>>
### 命題 1.4（有限假設類的均勻偏差界）

設 $\mathcal{H}=\{f_1,\dots,f_M\}$ 是**有限**假設類，其基數為 $M$。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
    if np.any(valid_nll < 0):
        raise ValueError("NLL for valid tokens must be non-negative.")
    if np.any(~np.isfinite(valid_nll)):
        if np.any(valid_nll == np.inf):
            return np.inf
        raise ValueError("Non-finite NLL in valid tokens.")
<<<NEW>>>
    if np.any(valid_nll < 0):
        raise ValueError("NLL for valid tokens must be non-negative.")
    if np.any(np.isnan(valid_nll)):
        raise ValueError("NaN NLL in valid tokens.")
    if np.any(np.isinf(valid_nll)):
        return np.inf
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
6.  **故障（驗證）**：驗證負概率、非布林 mask 等非法輸入被拒絕。
<<<NEW>>>
6.  **故障（驗證）**：驗證負概率、非布林 mask 等非法輸入被拒絕。
7.  **邊界（NaN 與 inf）**：有效 NLL 同時含 `inf` 與 `nan` 時預期拋出 `ValueError`（NaN 優先拒絕），不靜默回傳 `inf`。
<<<END>>>
<<<PATCH 06>>>
<<<OLD>>>
**邊界測試 2：極端但有限 logits。** $z=(1000,1001,999)$，target=1。預期 loss 約 $0.407606$，不出現 inf 或 nan。若把 $z$ 改成三個相等巨大值，例如 $(10^{300},10^{300},10^{300})$，數學上 loss 應為 $\log 3$，但 float64 在 $10^{300}+0.477$ 時可能把 $0.477$ 捨入掉，得到 0。
<<<NEW>>>
**邊界測試 2：極端但有限 logits。** $z=(1000,1001,999)$，target=1。預期 loss 約 $0.407606$，不出現 inf 或 nan。若把 $z$ 改成三個相等巨大值，例如 $(10^{300},10^{300},10^{300})$，數學上 loss 應為 $\ln 3\approx 1.099$，但 float64 在 $10^{300}+1.099$ 時可能把 $1.099$ 捨入掉，得到 0。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
3.  **邊界（零機率）**：驗證 `inf` 返回，且僅影響特定樣本。
<<<NEW>>>
3.  **邊界（零機率）**：驗證 `inf` 返回，且僅影響特定樣本。純機率域可直接觀察到 $+\infty$；由有限 logits 出發的 CE（見第 6 章 `cross_entropy_loss_and_grad`）於本卷核心實作拒絕非有限輸入，不會由有限 logits 產生 `inf`。
<<<END>>>