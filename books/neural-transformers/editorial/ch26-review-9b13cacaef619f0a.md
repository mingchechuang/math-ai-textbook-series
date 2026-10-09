我將以獨立審稿者的身分，針對第26章「量化、蒸餾與效率的證據」進行嚴格審閱。

### 1. 獨立重算與核對

**1. 定義與形狀核對 (Definition & Shape)**
*   **量化定義**：$s = \max |x| / q_{max}$。
    *   若 $x=[1.2, 0.4, ...]$，$b=3 \implies q_{max}=3$。$s=1.2/3=0.4$。
    *   $x/s = [3, 1, 2.25, ...]$。
    *   Round: $[3, 1, 2, ...]$。
    *   Clip: $[3, 1, 2, ...]$ (在 $[-3,3]$ 內)。
    *   Dequant: $[1.2, 0.4, 0.8, ...]$。
    *   誤差界 $s/2 = 0.2$。實際最大誤差 $0.1$。符合。
*   **逐通道量化 (Per-channel)**：
    *   程式碼中 `quantize_per_column` 對 `W (Din, Dout)` 沿 `axis=0` 取 max。
    *   這意味著對每一**列** (Column, 即輸出通道) 計算一個 scale。
    *   形狀：`max_abs` 形狀 `(1, Dout)`，`scale` 形狀 `(1, Dout)`。
    *   廣播時，`W / scale` 是 `(Din, Dout) / (1, Dout)`，正確。
    *   注意：通常 "Per-Channel" 在 PyTorch/ML 語境中常指沿 Channel (Output) 軸，這裡的實現是正確的。

**2. 蒸餾梯度核對 (Distillation Gradient)**
*   **Loss**: $\mathcal{L} = \frac{1}{N} \sum_n \mathrm{KL}(p_n || q_n)$。
*   **Proposition 26.2**: $\frac{\partial \mathrm{KL}}{\partial z_{s,j}} = \frac{q_j - p_j}{T}$。
    *   推導檢查：
        $\mathrm{KL}(p||q) = \sum p \log p - \sum p \log q$。
        $\frac{\partial}{\partial z_j} (\sum p \log q) = \sum_k p_k \frac{\partial \log q_k}{\partial z_j}$。
        $\log q_k = \frac{z_k}{T} - \log \sum e^{z/T}$。
        $\frac{\partial \log q_k}{\partial z_j} = \frac{\delta_{kj}}{T} - \frac{q_j}{T}$。
        Sum over $k$: $\frac{1}{T} (1 - q_j \sum p_k) = \frac{1}{T}(1 - q_j)$? **等等，這裡有誤區。**
        
        讓我重新仔細推導 $\frac{\partial}{\partial z_j} \sum_k p_k \log q_k$。
        項 $\sum_k p_k \log q_k$。
        $\frac{\partial}{\partial z_j} (p_k \log q_k) = p_k \frac{1}{q_k} \frac{\partial q_k}{\partial z_j}$。
        已知 softmax 梯度 $\frac{\partial q_k}{\partial z_j} = \frac{1}{T} q_k (\delta_{kj} - q_j)$。
        所以 $\frac{\partial}{\partial z_j} \sum_k p_k \log q_k = \sum_k p_k \frac{1}{q_k} \frac{1}{T} q_k (\delta_{kj} - q_j) = \frac{1}{T} \sum_k p_k (\delta_{kj} - q_j) = \frac{1}{T} (p_j - q_j \sum_k p_k) = \frac{1}{T} (p_j - q_j)$。
        因此 $\frac{\partial}{\partial z_j} (-\sum p \log q) = -\frac{1}{T}(p_j - q_j) = \frac{1}{T}(q_j - p_j)$。
        **推導正確。**

*   **程式碼梯度核對**:
    *   `distill_kl_and_grad` 回傳 `grad = (q - p) / (T * z_s.shape[0])`。
    *   這裡 `z_s.shape[0]` 是 $N$。
    *   所以 `grad` 是 $\frac{1}{N} \frac{q-p}{T}$。
    *   這是對**平均損失** $\mathcal{L} = \frac{1}{N} \sum \mathrm{KL}$ 的梯度。
    *   在 `manual_grads` 中：
        `dW = X.T @ dz`。
        若 $dz$ 是對平均損失的梯度，則 $dW = X^T \frac{\partial \mathcal{L}}{\partial z}$。
        這符合鏈式法則。
    *   **手算例 26.2 核對**:
        $C=3, T=2$。
        $z_t = [2, 1, 0] \implies z_t/T = [1, 0.5, 0]$。
        $z_s = [0, 1, 2] \implies z_s/T = [0, 0.5, 1]$。
        $p = \text{softmax}([1, 0.5, 0]) = \frac{[e^1, e^{0.5}, 1]}{Z}$。
        $q = \text{softmax}([0, 0.5, 1]) = \frac{[1, e^{0.5}, e^1]}{Z}$。
        注意 $q$ 是 $p$ 的反轉。
        $p_0 = e^1/Z, p_1 = e^{0.5}/Z, p_2 = 1/Z$。
        $q_0 = 1/Z, q_1 = e^{0.5}/Z, q_2 = e^1/Z$。
        $\frac{q-p}{T}$:
        $j=0$: $(1/Z - e^1/Z)/2 = (1-e)/2Z$。
        $j=1$: $(e^{0.5}/Z - e^{0.5}/Z)/2 = 0$。
        $j=2$: $(e^1/Z - 1/Z)/2 = (e-1)/2Z$。
        
        文中計算：
        $p \approx [0.506, 0.307, 0.186]$。
        $q \approx [0.186, 0.307, 0.506]$。
        $q-p \approx [-0.320, 0, 0.320]$。
        $(q-p)/2 \approx [-0.160, 0, 0.160]$。
        文中寫的是 $[-0.16008, 0, 0.16008]$。
        核對數值：
        $Z = 2.718+1.649+1 = 5.367$。
        $p_0 = 2.718/5.367 \approx 0.5064$。
        $q_0 = 1/5.367 \approx 0.1863$。
        $q_0 - p_0 \approx -0.3201$。
        除以 $T=2$ 得 $-0.16005$。
        文中 $-0.16008$ 略有微小差異，可能是中間步驟精度的問題，但在誤差範圍內可接受。
        **關鍵檢查**：文中說「梯度為負（往負梯度方向走即增大）」。
        $z_{s,2}$ 對應 $j=2$。梯度為 $+0.16008$。
        負梯度方向是 $-0.16008$。
        更新 $z_{s,2} \leftarrow z_{s,2} - \eta (0.16008)$。
        因為梯度是正的，減去正數會讓 $z_{s,2}$ **變小**。
        但文中直觀檢查說：「學生的第 3 類機率太低... 損失要下降就得把 $z_{s,3}$ 推大，所以梯度為負... 符合。」
        **這裡有邏輯錯誤！**
        若梯度為正，下降方向（負梯度）是减小 $z$。
        若要增大 $z$，梯度必須為負。
        在此例中，$q_2 (0.506) > p_2 (0.186)$。學生對第2類的預估**高於**教師。
        KL 散度要求學生接近教師。既然學生太高了，應該**降低** $z_{s,2}$ 以減少 $q_2$。
        所以梯度為正（表示增加 $z$ 會增加損失，因此要往負梯度方向走，即减小 $z$）。
        文中的直觀解釋「把 $z_{s,3}$ 推大」是**錯誤**的。應該是小 $z_{s,3}$（索引2）。
        而且文中說「梯度為負」，但計算結果第3項（索引2）是正的 $+0.16008$。
        文中寫：$\frac{[-0.32016, 0, 0.32016]}{2} = [-0.16008, 0, 0.16008]$。
        第3個分量（索引2）是 $+0.16008$。
        文中直觀檢查：「所以梯度為負」。這與計算結果（正）矛盾。
        **這是嚴重的數學/直覺錯誤。**

**3. 量化誤差界證明**
*   命題 26.1 證明邏輯完整。
*   Round 的定義：NumPy 的 `np.round` 是 Banker's Rounding (四捨六入五成雙)。
*   證明中假設 `round` 是取最近整數。若 $t$ 恰好是 $k+0.5$，NumPy 會取偶數。
*   誤差界 $|t - \text{round}(t)| \le 0.5$ 仍然成立。
*   證明無誤。

**4. 程式碼與測試**
*   `quantize_symmetric`:
    *   `np.clip(np.round(x / scale), -qmax, qmax)`。
    *   若 $x$ 中有值超過 $q_{max} \cdot s$？
    *   定義 $s = \max |x| / q_{max}$。
    *   則 $\max |x|/s = q_{max}$。
    *   所以 $x/s$ 最大絕對值為 $q_{max}$。
    *   Round 後不會超過 $q_{max}$（除非 $x/s$ 剛好是 $q_{max} + 0.5$？不可能，因為 max 是 $q_{max}$）。
    *   所以 Clip 通常是 redundant 的，但為安全保留是好的。
*   `distill_kl_and_grad`:
    *   使用了 `logsumexp` 來計算 log-softmax 嗎？
    *   程式碼中：
        ```python
        st = z_t / T
        ss = z_s / T
        logp = st - logsumexp(st, axis=-1)
        logq = ss - logsumexp(ss, axis=-1)
        kl_per_sample = np.sum(p * (logp - logq), axis=-1)
        ```
        這裡 $p$ 是 softmax 概率。
        $\mathrm{KL} = \sum p (\log p - \log q)$。
        這是正確的 KL 定義。
        梯度部分直接用了 `(q - p) / (T * N)`。
        這與 KL 的解析解一致。
        **注意**：這裡計算 KL 用的是 log-softmax，但梯度用的是解析式。這是標準做法，避免數值不穩定。
*   `fd_grad`: 中心差分實現正確。

**5. 資料洩漏與評估**
*   陷阱二提到「用測試集校準尺度」是洩漏。
*   D1 題目要求明確切分。
*   符合規範。

### 2. 問題列表與最小修法

**問題 1：手算例 26.2 中的直觀解釋錯誤**
*   **原句**：「直觀檢查：學生的第 3 類機率太低（$0.186$ 對教師的 $0.506$），損失要下降就得把 $z_{s,3}$ 推大，所以梯度為負（往負梯度方向走即增大），符合。」
*   **原因**：
    1.  在此例中，學生 $z_s=[0,1,2]$，教師 $z_t=[2,1,0]$。
    2.  $q$ (學生) 在索引 2 的機率是 $0.506$，$p$ (教師) 在索引 2 的機率是 $0.186$。
    3.  學生對第 3 類（索引 2）的預估**高於**教師，不是「太低」。
    4.  計算出的梯度第 3 分量（索引 2）是 $+0.16008$（正數）。
    5.  若梯度為正，下降方向（負梯度）是**減小** $z_{s,3}$。
    6.  原文說「梯度為負」與計算結果 $+0.16008$ 矛盾，且「推大」與「梯度為負（若指代方向）」混淆。正確邏輯：因為學生預估過高，需要降低 logit，故梯度為正（增加 logit 會增加損失），更新時減去該梯度導致 logit 下降。
*   **最小修法**：
    將該段落改為：
    「直觀檢查：在此例中，學生對第 3 類（索引 2）的機率 $q_2=0.506$ 高於教師的 $p_2=0.186$。為了讓學生接近教師，應降低 $z_{s,2}$。計算出的梯度第 3 分量為 $+0.16008$（正），表示增加 $z_{s,2}$ 會增加損失。因此，沿負梯度方向（減去正梯度）更新時，$z_{s,2}$ 會減小，這符合「降低過高預估」的直覺。」

**問題 2：A1 習題解答中的 Rounding 約定描述不精確**
*   **原句**：「$3.5$ 的四捨五入取決於約定（本卷採用 NumPy 的 banker's rounding，即取到最近偶數 $4$）——這裡正是要提醒的陷阱：若規定用「四捨五入到偶」則得 $4$，若用「遠離零」則得 $4$...」
*   **原因**：
    *   NumPy `np.round(3.5)` 確實是 `4.0` (Banker's rounding, 3.5 取最近的偶數 4)。
    *   「遠離零」的 rounding (Round half away from zero) 對於 $3.5$ 也會得到 $4$。
    *   對於 $2.5$，Banker's 得 $2$，Round half away 得 $3$。
    *   原文說「若用『遠離零』則得 4」，這與 Banker's 結果相同，無法區分兩者差異，且未舉出能區分兩者的例子（如 2.5）。
    *   更重要的是，量化標準中通常明確定義 rounding mode。說「取決於約定」是好的，但例子選擇不佳。
*   **最小修法**：
    將 A1 解答中關於 rounding 的說明簡化並修正例子：
    「$3.5$ 的取整取決於約定。NumPy 的 `np.round` 採用 Banker's rounding（五成雙），故 $3.5 \to 4$。若採用 Round-half-up（四捨五入），亦為 $4$。但對於 $2.5$，前者得 $2$，後者得 $3$。本卷程式碼使用 NumPy 預設行為，故以 Banker's rounding 為準。實作中應明確文件化此行為。」

**問題 3：程式碼中 `distill_kl_and_grad` 的 KL 計算可能數值不穩定（邊緣情況）**
*   **原因**：雖然使用了 `logsumexp`，但 `kl_per_sample = np.sum(p * (logp - logq), axis=-1)`。
    *   若 $p_k$ 非常小（接近 0），`logp` 可能接近 $-\infty$（若 $p_k$ 下溢為 0，`log(0)` 是 $-\infty$，但 `p * -inf` 是 `nan` 或 `0` 取決於實現）。
    *   在 `softmax_temp` 中，如果 $z/T$ 非常小，`exp` 可能下溢為 0。
    *   若 $p_k = 0$，則 $p_k \log p_k$ 定義為 0。
    *   若 `logp` 計算出 $-\infty$，而 $p_k=0$，`0 * -inf` 是 `NaN`。
    *   NumPy 中 `0 * -inf` 是 `nan`。這會導致 `kl_per_sample` 出現 `nan`。
    *   正確的實現應避免直接計算 `p * log p` 若 `p` 可能為 0，或使用 `np.where(p > 0, p * (logp - logq), 0)`。
    *   或者，更穩健的是直接使用交叉熵形式：$-\sum p \log q + \text{const}$。但 KL 需要兩項。
    *   鑑於這是「效率與證據」章，且要求邊界測試，應處理此數值陷阱。
*   **最小修法**：
    在程式碼 `distill_kl_and_grad` 中，修改 KL 計算行：
    ```python
    # 避免 0 * -inf 產生 NaN
    with np.errstate(divide='ignore'):
        log_ratio = logp - logq
        # 若 p 為 0，該項貢獻為 0
        term = np.where(p > 0, p * log_ratio, 0.0)
    kl_per_sample = np.sum(term, axis=-1)
    ```

**問題 4：B2 習題解答中梯度公式未考慮 T 的一致性**
*   **原句**：「...合成梯度是兩項相加：$(q-p)/(TN)+\alpha\,(q-y)/N$...」
*   **原因**：
    *   蒸餾項 $(q-p)/(TN)$ 中的 $q, p$ 是溫度 $T$ 下的 softmax。
    *   硬標籤項 $\alpha \mathrm{CE}$ 通常使用 $T=1$ 的 softmax，或者也使用溫度 $T$？
    *   若硬標籤項也使用溫度 $T$ 的 softmax $q_T$，則梯度是 $\alpha (q_T - y)/N$。
    *   若硬標籤項使用 $T=1$ 的 softmax $q_1$，則梯度是 $\alpha (q_1 - y)/N$。
    *   原文寫 $\alpha (q-y)/N$，這裡的 $q$ 指代不明。前文定義 $q$ 為溫度 $T$ 下的分布。
    *   若 $q$ 是 $T$ 下的，則公式正確（假設硬標籤項也用 $T$）。
    *   但通常硬標籤交叉熵不縮放溫度。若題目要求「加入硬標籤交叉熵項」，通常指標準 CE。
    *   解答中註解說「注意硬標籤項不除 T」，這暗示硬標籤項的梯度分母是 $N$，分子是 $(q_{T=1} - y)$？或者分子是 $(q_T - y)$？
    *   如果 $q$ 在公式中定義為 $T$ 下的，那麼寫 $(q-y)/N$ 意味著硬標籤項也用了 $T$。
    *   為了避免混淆，應明確 $q_{hard} = \text{softmax}(z_s)$ (即 $T=1$) 和 $q_{soft} = \text{softmax}(z_s/T)$。
*   **最小修法**：
    修改 B2 解答：
    「加入 $\alpha \mathrm{CE}$ 後，總損失對 $z_s$ 的梯度是兩項相加。設 $q_{soft} = \text{softmax}(z_s/T)$，$q_{hard} = \text{softmax}(z_s)$。若硬標籤項使用標準 CE（$T=1$），梯度為 $\frac{q_{soft}-p}{TN} + \alpha \frac{q_{hard}-y}{N}$。若硬標籤項也使用溫度 $T$，則第二項為 $\alpha \frac{q_{soft}-y}{N}$。程式實作時需明確選擇。」

### 3. 其他檢查

*   **篇幅**：4233 字，符合 3000-4500 目標。
*   **依賴**：NumPy，自足。
*   **GPU/Benchmark**：明確聲明不捏造 benchmark，只討論理論與設計事實。
*   **資料洩漏**：強調切分順序。
*   **測試**：提供了正常/邊界/故障測試。

### 4. 結論

章節核心數學（量化誤差界、蒸餾梯度）正確，但**手算例 26.2 的直觀解釋存在嚴重邏輯與符號錯誤**（方向反了），這會嚴重誤導讀者對梯度意義的理解。此外，程式碼中 KL 計算潛在的 `0 * -inf` 數值不穩定問題需修復。A1 和 B2 的解答細節需精確化。

這些問題屬於「必須修復」等級，否則章節在教學正確性和程式數值穩定性上存在缺陷。

VERDICT: REVISE