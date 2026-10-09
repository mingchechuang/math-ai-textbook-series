作為 Volume V 的獨立審稿者，我將針對第7至第12章的數學定義、實作細節、梯度推導及測試契約進行逐項核對。重點在於確認自足性、shape 一致性、梯度累加規則、數值穩定性策略以及資料洩漏防範機制。

---

### 1. 數學與實作一致性核對

#### 第07章：矩陣微分、VJP與反向傳播
*   **Shape 與廣播核對**：
    *   定義 $X \in \mathbb{R}^{B \times D_{in}}$, $W \in \mathbb{R}^{D_{in} \times D_{out}}$。
    *   程式碼中 `W` 初始化為 `(Din, 1)`，`b` 為 `(1,)`。
    *   `forward_backward` 中 `dW = X.T @ dlogits`。
        *   $X.T: (D_{in}, B)$, `dlogits`: $(B, 1)$ $\rightarrow dW: (D_{in}, 1)$。符合 $W$ 的 shape。
    *   `db = np.sum(dlogits, axis=0)`。
        *   `dlogits`: $(B, 1)$ $\rightarrow db: (1,)`。符合 $b$ 的 shape。
    *   **結論**：梯度 shape 與參數 shape 一致，廣播反向求和軸正確。
*   **損失函數核對**：
    *   穩定 BCE 公式：$\ell(z,y) = \max(z,0) - zy + \log(1+e^{-|z|})$。
    *   程式碼 `per_item = np.maximum(logits, 0.0) - logits * y + np.log1p(np.exp(-np.abs(logits)))`。
    *   導數 $\frac{\partial \ell}{\partial z} = \sigma(z) - y$。
    *   程式碼 `dlogits = (sigmoid(logits) - y) / logits.shape[0]`。
    *   因為是 batch mean loss，除以 $B$ 正確。
*   **有限差分**：
    *   使用中央差分，容差檢查邏輯完整。
    *   **潛在問題**：`gradient_checks` 中 `direction` 向量是手動定義的固定值，而非隨機生成。雖然對於特定參數點驗證是可以的，但作為通用檢查，隨機方向更具代表性。不過，章節中明確指出這是「預期行為」，且用於驗證特定線性組合的正確性，可接受。
*   **測試契約**：
    *   包含正常、邊界（B=1, 極端 logit）、故障（非法標籤、空 batch、非有限值）。
    *   `expect_value_error` 邏輯正確。

#### 第08章：批次線性層與broadcast的梯度
*   **通用前導軸處理**：
    *   定義 $X$ 形狀為 `(*S, Din)`。
    *   `affine_forward` 檢查 `x.ndim < 2`，允許任意前導軸。
    *   `affine_backward` 中 `xf = x.reshape(-1, w.shape[0])`。
        *   這裡 `x` 是 `(*S, Din)`，reshape 為 `(N, Din)`，其中 $N = \prod S$。
        *   `dw = xf.T @ gf`。$xf.T: (Din, N)$, `gf`: $(N, Dout)$ $\rightarrow dw: (Din, Dout)$。正確。
        *   `db = gf.sum(axis=0)`。`gf`: $(N, Dout)$ $\rightarrow db: (Dout,)$。正確。
        *   `dx = (g @ w.T).reshape(x.shape)`。$g: (*S, Dout)$, `w.T`: $(Dout, Din)$ $\rightarrow$ 廣播乘法後 shape 為 `(*S, Din)`。正確。
*   **共享參數梯度累加**：
    *   `two_use_grads` 中 `dw1 + dw2`。符合分支梯度累加規則。
*   **測試**：
    *   手算例一與程式斷言一致。
    *   邊界測試 $B=1$ 的 3D 輸入處理正確。
    *   故障測試：偏置 shape 錯誤被拒絕。

#### 第09章：NumPy兩層網路：逐項梯度驗證
*   **模型結構**：
    *   $Z_1 = X W_1 + b_1$, $H = \tanh(Z_1)$, $Z_2 = H W_2 + b_2$。
    *   Shape 檢查：
        *   $W_1: (D_{in}, D_{hid})$, $b_1: (D_{hid},)$.
        *   $W_2: (D_{hid}, 1)$, $b_2: (1,)$.
    *   梯度推導：
        *   $dZ_2 = (Y_{pred} - Y_{true}) / B$.
        *   $dW_2 = H^T dZ_2$. $(D_{hid}, B) \times (B, 1) \rightarrow (D_{hid}, 1)$. 正確。
        *   $db_2 = \sum dZ_2$. $(1,)$. 正確。
        *   $dH = dZ_2 W_2^T$. $(B, 1) \times (1, D_{hid}) \rightarrow (B, D_{hid})$. 正確。
        *   $dZ_1 = dH \odot (1 - H^2)$. 正確。
        *   $dW_1 = X^T dZ_1$. $(D_{in}, B) \times (B, D_{hid}) \rightarrow (D_{in}, D_{hid})$. 正確。
        *   $db_1 = \sum dZ_1$. $(D_{hid},)$. 正確。
*   **損失函數**：
    *   使用 `np.logaddexp(0.0, self.Z2) - self.Y_true * self.Z2`。
    *   這是穩定的 BCE 形式。
*   **有限差分**：
    *   `numerical_gradient` 恢復狀態邏輯正確（try-finally）。
    *   `check_gradients` 使用 `np.nditer` 遍歷多維陣列索引，正確。
*   **邊界與故障**：
    *   處理了非有限值、shape 錯誤。
    *   極大 logit 測試：$Z_2=100, Y=0$。Loss $\approx 100$。$dZ_2 \approx (1-0)/1 = 1$。正確。
*   **小資料過擬合檢查**：
    *   使用獨立模型和固定訓練樣本，斷言 loss 下降和準確率。這是合理的除錯測試。

#### 第10章：初始化、梯度尺度與正規化
*   **LayerNorm**：
    *   前向：沿最後一軸計算 mean/var。
    *   反向：
        *   $dx = \frac{\text{inv\_std}}{n} (n dxhat - \sum dxhat - xhat \sum dxhat \odot xhat)$.
        *   這是標準的 LayerNorm 反向公式。
        *   $d\gamma = \sum dy \odot xhat$ (沿前導軸).
        *   $d\beta = \sum dy$ (沿前導軸).
    *   程式碼中 `reduce_axes = tuple(range(dy.ndim - 1))`。正確。
*   **BatchNorm**：
    *   訓練模式：沿 batch 軸 (axis=0) 計算 mean/var。
    *   Running stats 更新：`new_mean = momentum * running_mean + (1-momentum) * mean`.
    *   推論模式：只讀 running stats，不更新。
    *   **注意**：簡化版 BatchNorm 沒有 $\gamma, \beta$，這在章節中已明確說明。
*   **初始化**：
    *   He: $\sqrt{2/fan\_in}$.
    *   Xavier: $\sqrt{6/(fan\_in+fan\_out)}$.
    *   符合常用公式。
*   **測試**：
    *   有限差分核對 LayerNorm 梯度。
    *   邊界測試：常數輸入，方差為0，$\epsilon$ 防止除零。
    *   故障測試：非法 shape, 非有限值。

#### 第11章：SGD、動量、Adam與AdamW
*   **Adam 偏差修正**：
    *   $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$.
    *   $\hat{m}_t = m_t / (1-\beta_1^t)$.
    *   程式碼中 `self.t` 從 0 開始，`step` 中先 `self.t += 1`。
    *   `bc1 = 1.0 - self.b1 ** self.t`.
    *   第一次 step, $t=1$, $bc1 = 1-\beta_1$. 正確。
*   **AdamW vs Adam+L2**：
    *   Adam+L2: `grads = [g + self.wd * p ...]` 在計算 $m, v$ 之前。
    *   AdamW: `upd = upd + self.wd * p` 在計算更新量之後。
    *   這正確實現了兩者不等價的特性。
*   **全域梯度裁剪**：
    *   `global_grad_norm` 計算所有參數梯度的 L2 範數總和開根號。
    *   `clip_global_norm` 使用單一縮放因子。
    *   符合全域裁剪定義。
*   **測試**：
    *   手算例題 11.1 和 11.2 與程式邏輯一致。
    *   故障測試：負學習率、$\beta_1=1$、NaN 梯度。

#### 第12章：訓練流程、資料洩漏與除錯
*   **資料切分**：
    *   按群組切分，確保 train/val/test 群組互斥。
    *   `assert_disjoint_groups` 檢查群組集合交集為空。
*   **標準化**：
    *   只用訓練集計算 mean/std。
    *   應用於所有集合。
    *   避免洩漏。
*   **模型選擇**：
    *   使用驗證集 loss 進行早停。
    *   測試集只在最後評估一次。
*   **Checkpoint**：
    *   保存模型參數、優化器狀態（此處為 SGD 無狀態，但保存了 RNG state 和配置）。
    *   恢復時檢查配置和群組一致性。
*   **損失與梯度**：
    *   多類別 Softmax CE。
    *   穩定計算：`shifted = logits - logits.max(axis=1, keepdims=True)`.
    *   `nll = -shifted[arange(B), y] + log(sum(exp(shifted), axis=1))`.
    *   梯度：`dlogits = probs; dlogits[arange(B), y] -= 1; dlogits /= B`.
    *   正確。
*   **測試**：
    *   檢查 shape、切分、標準化統計。
    *   Checkpoint 讀回測試。
    *   中止後恢復測試。
    *   故障測試：群組洩漏、零方差、NaN、錯誤標籤。

### 2. 跨章一致性與依賴檢查

*   **Shape 約定**：
    *   全卷統一 $X$ 為 row-major (batch, features)。
    *   $W$ 為 (Din, Dout)。
    *   梯度 shape 與參數一致。
    *   各章實作均遵守此約定。
*   **廣播反向**：
    *   偏置梯度均沿 batch (及 time) 軸求和。
    *   各章實作一致。
*   **損失平均**：
    *   均明確使用 batch mean，且在反向傳播中只除以一次 B。
    *   各章實作一致。
*   **數值穩定性**：
    *   BCE 使用 `logaddexp` 或 `max(z,0)-zy+log1p(exp(-|z|))`。
    *   Softmax 使用減 max。
    *   Sigmoid 使用分支避免溢出。
    *   各章實作一致且正確。
*   **資料洩漏防範**：
    *   第12章明確強調切分先於預處理。
    *   其他章節的實驗若涉及資料，均使用合成資料且明確聲明不參與真實決策。
*   **依賴性**：
    *   各章程式碼自足，僅依賴 NumPy。
    *   沒有未定義的模組或外部檔案依賴（除 checkpoint 測試中的 npz/json，但這是測試的一部分）。

### 3. 具體問題與最小修法

經過詳細核對，未發現數學錯誤、shape 不一致、梯度計算錯誤或資料洩漏漏洞。程式碼邏輯與數學推導高度一致，測試覆蓋了正常、邊界和故障情況，且明確區分了預期結果與實測結果。

**潛在微調建議（非錯誤，僅為最佳化）**：
1.  **第07章**：`gradient_checks` 中的 `direction` 向量可以改為隨機生成，以增強測試的泛化性。但當前固定向量對於驗證特定線性關係是有效的，且章節中已說明這是預期行為。
2.  **第09章**：`TwoLayerNet` 的 `__init__` 中，如果 `dim_in` 或 `dim_hid` 不是正整數，會拋出 `ValueError`。這與第10章的 `initialize` 函數行為一致。
3.  **第12章**：`train` 函數中，如果 `resume=True`，會重新生成資料並檢查群組一致性。這確保了資料切分的重現性。如果資料生成規則改變，checkpoint 將無法恢復，這是正確的安全行為。

### 4. 結論

所有章節的數學推導、實作細節、梯度計算、測試設計及資料處理均符合 Volume V 的約定和要求。沒有發現需要修正的實質問題。程式碼自足、可驗證，且明確遵守了不虛構執行結果、不載入不可信資料等安全規範。

VERDICT: APPROVE