<<<PATCH 24.01>>>
<<<OLD>>>
| Mean NLL | 1.0 | 2.5 |
| PPL | 2.718 | 12.182 |
<<<NEW>>>
| Mean NLL | 1.0 | 2.3 |
| PPL | 2.718 | 9.97 |
<<<END>>>
<<<PATCH 24.02>>>
<<<OLD>>>
此機制能有效降低「已回答樣本」中的風險（Risk），但代價是降低了「覆蓋率」（Coverage，即回答的比例）。
<<<NEW>>>
若信心分數能有效排序錯誤風險，選擇性拒答可能降低「已回答樣本」中的風險（Risk），但代價是降低了「覆蓋率」（Coverage，即回答的比例）；是否成立必須由驗證集 R-Curve 檢查。
<<<END>>>
<<<PATCH 24.03>>>
<<<OLD>>>
*   t=2: logits $[-100, 10, -100, -100]$，真實標籤 $y=1$ (B)。
    *   $m = 10$.
    *   $\sum e^{z-m} \approx e^{-110} + e^{0} + e^{-110} + e^{-110} \approx 1$.
    *   $\text{LSE} = 10 + \ln(1) = 10$.
    *   $\text{NLL}_2 = 10 - 10 = 0$.
*   t=3: logits $[-100, -100, 1, -100]$，真實標籤 $y=2$ (C)。
    *   同理，$\text{NLL}_3 \approx 0$.
*   Total NLL$_1 = 3.4402 + 0 + 0 = 3.4402$.
<<<NEW>>>
*   t=2: logits $[-100, 10, -100, -100]$，真實標籤 $y=1$ (B)。
    *   $m = 10$.
    *   $\sum e^{z-m} = e^{-110} + e^{0} + e^{-110} + e^{-110} = 1 + 3e^{-110}$.
    *   $\text{LSE} = 10 + \ln(1 + 3e^{-110}) \approx 10$.
    *   $\text{NLL}_2 = \ln(1 + 3e^{-110}) \approx 0$（極小但非精確零）。
*   t=3: logits $[-100, -100, 1, -100]$，真實標籤 $y=2$ (C)。
    *   同理，$\text{NLL}_3 = \ln(1 + 3e^{-101}) \approx 0$。
*   Total NLL$_1 = 3.4402 + \text{NLL}_2 + \text{NLL}_3 \approx 3.4402$.
<<<END>>>
<<<PATCH 24.04>>>
<<<OLD>>>
當 $n=0$ 時，$X_b$ 未定義，但在 ECE 公式中 $\frac{N_b}{N} |X_b|$ 項為 0（若定義 $0 \cdot \infty = 0$ 或明確跳過）。對於 $n>0$：
<<<NEW>>>
當 $n=0$ 時，直接定義該箱貢獻為 $0$；不需要引入 $0 \cdot \infty$ 的約定。對於 $n>0$：
<<<END>>>
<<<PATCH 24.05>>>
<<<OLD>>>
### 6.1 正常測試
*   **輸入**：見例題 4.1 和 4.2 的數據。
*   **預期輸出**：
    *   `PPL`: 3.342 (約)
    *   `Mean NLL`: 1.2066 (約)
    *   `ECE` (bins [0, 0.5, 1.0]): 0.27 (約)
*   **驗證**：若實作正確，預期與手算結果一致。
<<<NEW>>>
### 6.1 自足測試程式

以下程式直接構造例題 4.1 與 4.2 的資料，並以斷言檢查函式輸出。因尚未實際執行，僅陳述預期行為：若實作正確，斷言不觸發。

```python
import numpy as np

# 構造例題 4.1 的 logits、labels、valid_mask
logits = np.array([
    [[0., 1., 2., 3.],
     [-100., 10., -100., -100.],
     [-100., -100., 1., -100.]],
    [[0., 0., 0., 0.],
     [0., 0., 0., 0.],
     [0., 0., 0., 0.]]
], dtype=np.float64)
labels = np.array([
    [0, 1, 2],
    [3, 0, 0]
], dtype=np.int64)
valid_mask = np.array([
    [True, True, True],
    [True, False, False]
], dtype=bool)

ppl, mean_nll = compute_global_ppl(logits, labels, valid_mask)
assert np.isclose(mean_nll, 1.2066, atol=1e-3)
assert np.isclose(ppl, 3.342, atol=1e-3)

# 構造例題 4.2 的 ECE 資料
probs = np.array([0.9, 0.9, 0.8, 0.7, 0.7, 0.5, 0.5, 0.3, 0.3, 0.1])
correctness = np.array([1, 0, 1, 1, 0, 1, 0, 0, 0, 0])
bin_edges = [0.0, 0.5, 1.0]
ece, stats = compute_ece(probs, correctness, bin_edges)
assert np.isclose(ece, 0.27, atol=1e-3)

# 邊界測試：全 False mask 應拋出 ValueError
try:
    compute_global_ppl(logits, labels, np.zeros_like(valid_mask, dtype=bool))
    assert False, "Expected ValueError for no valid tokens"
except ValueError as e:
    assert "No valid tokens" in str(e)

# 故障測試：label 越界
bad_labels = labels.copy()
bad_labels[0, 0] = 4
try:
    compute_global_ppl(logits, bad_labels, valid_mask)
    assert False, "Expected ValueError for label out of bounds"
except ValueError as e:
    assert "Label out of bounds" in str(e)

# 故障測試：NaN logits
bad_logits = logits.copy()
bad_logits[0, 0, 0] = np.nan
try:
    compute_global_ppl(bad_logits, labels, valid_mask)
    assert False, "Expected ValueError for non-finite logits"
except ValueError as e:
    assert "non-finite" in str(e)

# 故障測試：空 ECE 輸入
try:
    compute_ece(np.array([]), np.array([]), bin_edges)
    assert False, "Expected ValueError for empty ECE input"
except ValueError as e:
    assert "Empty input" in str(e)

# 故障測試：非遞增 bins
try:
    compute_ece(probs, correctness, [0.0, 0.6, 0.5, 1.0])
    assert False, "Expected ValueError for non-increasing bins"
except ValueError as e:
    assert "strictly increasing" in str(e)

print("All assertion-based tests passed (pre-execution expectation).")
```
<<<END>>>