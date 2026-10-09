# 第24章獨立審稿

## 一、重新核算

本次未執行程式，以下均為靜態推導與 NumPy 語義核對。

### 1. PPL 例題

第一位置：

$$
\operatorname{LSE}([0,1,2,3])
=3+\ln(1+e^{-1}+e^{-2}+e^{-3})
\approx3.44019.
$$

第二、三位置的 NLL 是極小正數；第四位置為 $\ln4\approx1.38629$。所以：

$$
\mathrm{MeanNLL}\approx1.20662,
\qquad
\mathrm{PPL}\approx3.3422.
$$

正文與 `test_normal` 的預期數值正確。

### 2. ECE 例題

低信心箱貢獻為：

$$
\frac3{10}\left|0-\frac{0.7}{3}\right|=0.07,
$$

高信心箱貢獻為：

$$
\frac7{10}\left|\frac37-\frac57\right|=0.20.
$$

總 ECE 為 $0.27$，正確。

### 3. validation 與固定閾值

validation 兩筆資料的 confidence 約為 $0.8808$、$0.7311$，correctness 為 $[1,0]$。候選閾值 $[0,.8,1]$ 下：

- $\tau=0$：coverage $1$、risk $0.5$；
- $\tau=.8$：coverage $0.5$、risk $0$；
- $\tau=1$：接受數為零、risk 未定義。

故 `select_threshold` 預期選出 `.8`。

固定 `.8` 後：

- ID all 接受兩筆，兩筆都正確，coverage $2/3$、risk $0$；
- OOD all 接受一筆，該筆錯誤，coverage $1/3$、risk $1$。

此資料確實能展示同一拒答閾值在 OOD 下可能失效。

### 4. ECE 命題

本版已正確將母體分箱 ECE 改為：

$$
\sum_{b:q_b>0}q_b|\delta_b|,
$$

並對 $q_b=0$ 的箱另行處理，因此上一版零機率條件事件的缺口已修復。這部分不應再以舊問題退稿。

---

## 二、已達標部分

現稿以下內容可以保留：

1. PPL 採有效 token 總 NLL／有效 token 總數。
2. 沒有平均樣本 PPL 或群組 PPL。
3. NLL 從 logits 與穩定 log-sum-exp 計算。
4. 極端有限 logits 的高信心 NLL 被正確描述為極小正數。
5. `safe_labels` 避免 padding 的 `-1` 被當成最後一類。
6. mask 只控制 loss reduction，不豁免非有限 logits。
7. 空有效 token、非法 label、非法 mask dtype 均有拒絕政策。
8. ECE 的 bin 邊界與 $0.5$、$1.0$ 歸箱規則清楚。
9. ECE 空輸入與非法 bins 有故障政策。
10. ECE 命題已處理樣本空箱及母體零機率箱。
11. coverage 定義為回答比例。
12. 空接受集 risk 回傳 NaN，而非錯誤當成零。
13. validation 選閾值，ID/OOD test 不參與調參。
14. validation curve、target、selected threshold 與 selected point 已保存。
15. ID/OOD 各自有 routine、alert、all。
16. 整體 PPL 由總 NLL 與總 token 重算。
17. 整體 ECE 由原始事件重算，不平均群組 ECE。
18. source 跨 split 有故障測試。
19. OOD risk 非單調有明確反例。
20. 手設千 token 表格與程式小資料沒有混稱。
21. 沒有聲稱程式已執行、模型已訓練或得到實測結果。
22. 引用末註已忠實標示未逐條核對的來源。
23. 正文達最低中文字數要求。

本章是評估章，不需要重複完整 Transformer 模型或訓練 loop；沒有反向傳播，也沒有梯度需要核對。

---

## 三、仍然阻擋核准的問題

### 1. PPL overflow 的能力說明仍與實際 API 矛盾

**逐字原句：**

```python
# 明確拒絕不可表示的 PPL；mean_nll 仍可供呼叫端另行記錄。
if not np.isfinite(mean_nll) or mean_nll > np.log(np.finfo(float).max):
    raise OverflowError("PPL not representable; record mean NLL instead.")
```

**原因：**

一旦拋出 `OverflowError`，函式就不會執行：

```python
return perplexity, mean_nll
```

因此呼叫端無法從此函式取得區域變數 `mean_nll`。目前也沒有：

- 自訂例外屬性保存 `mean_nll`；
- 單獨回傳 total NLL/count 的函式；
- 回傳 `np.inf, mean_nll` 的政策。

所以「mean_nll 仍可供呼叫端另行記錄」及例外訊息中的「record mean NLL instead」不符合實際能力。呼叫端只能重新計算，不能由這個 API 取得。

**最小修法：**

最小可選以下任一方案。

方案 A：允許 PPL 為無限大：

```python
if not np.isfinite(mean_nll):
    raise OverflowError("Mean NLL is not finite.")
if mean_nll > np.log(np.finfo(float).max):
    return np.inf, float(mean_nll)
```

方案 B：保留例外，但移除不實能力敘述：

```python
if not np.isfinite(mean_nll) or mean_nll > limit:
    raise OverflowError("PPL is not representable.")
```

並把註解改成「若需要保留 Mean NLL，應由另一個明確介面回傳」，但若沒有另一介面，就不要聲稱已能做到。

方案 C：自訂例外並保存 `mean_nll`。這較繁複，不是最小修法。

建議方案 A，因為 Mean NLL 仍是有效評估量，回傳 `np.inf` 也能明確表示 PPL 超出浮點範圍。

---

### 2. 新增的 overflow 路徑仍完全沒有測試

**逐字原句：**

```python
z = np.array([[[1000., 0.]]])
assert np.isfinite(compute_global_ppl(
    z, np.array([[0]]), np.array([[True]]))[0])
```

**原因：**

這筆資料的真標籤是最高 logit 類別，所以 NLL 接近零。它只驗證：

- `exp(1000)` 沒有被直接計算；
- 減最大值後的 log-softmax 預期有限。

它不會進入：

```python
raise OverflowError(...)
```

若要觸發該路徑，真標籤應設為類別 1，使 NLL 約為 $1000$。此外，現有 `expect_value_error` 只能捕捉 `ValueError`，不能測 `OverflowError`。

**最小修法：**

增加：

```python
def expect_error(exc_type, fn, *args):
    try:
        fn(*args)
    except exc_type:
        return
    raise AssertionError("Expected exception")
```

若保留 `OverflowError` 政策：

```python
expect_error(
    OverflowError,
    compute_global_ppl,
    np.array([[[1000., 0.]]]),
    np.array([[1]]),
    np.array([[True]])
)
```

若採用 `np.inf` 回傳政策，則改測：

```python
ppl, mean = compute_global_ppl(
    np.array([[[1000., 0.]]]),
    np.array([[1]]),
    np.array([[True]])
)
assert np.isinf(ppl)
assert np.isclose(mean, 1000.0)
```

測試章中的「Logits 溢位」也應改稱「極端有限 logits 的穩定 log-softmax」，因為目前測的是 softmax 穩定性，而非 logits 本身溢位。

---

### 3. classwise ECE 的習題解答仍沒有任何測試

**逐字原句：**

```python
def compute_classwise_ece(probs, labels, bin_edges):
```

本版已正確加入：

```python
if N == 0 or C == 0:
    raise ValueError("N and C must be positive.")
```

**原因：**

函式主體的 shape/reduction 沒有明顯錯誤：

- `probs` 是 $(N,C)$；
- 第 $c$ 類 confidence 是 `probs[:, c]`，shape $(N,)$；
- correctness 是 $\mathbb I(y_i=c)$，shape $(N,)$；
- 每類 ECE 由共同的 `compute_ece` 計算；
- macro ECE 是對 $C$ 個類別作 mean。

但章稿規定程式題須有完整解答。現稿沒有測試：

- 合法 classwise ECE；
- $N=0$；
- $C=0$；
- row sum 非一；
- label 越界；
- 非有限 probabilities。

因此無法判斷新增的防線是否與答案契約一致。

**最小修法：**

在習題解答的程式塊後加入未執行的預期測試，例如：

```python
eces, macro = compute_classwise_ece(
    [[.8, .2], [.3, .7]], [0, 1], [0, .5, 1])
assert len(eces) == 2
assert np.isclose(macro, np.mean(eces))
```

並加入：

```python
expect_value_error(
    compute_classwise_ece,
    np.empty((0, 2)), np.empty((0,), dtype=int), [0, 1])
expect_value_error(
    compute_classwise_ece,
    np.empty((2, 0)), np.array([0, 0]), [0, 1])
expect_value_error(
    compute_classwise_ece,
    [[.8, .3]], [0], [0, 1])
expect_value_error(
    compute_classwise_ece,
    [[.8, .2]], [2], [0, 1])
```

另補 NaN probability。這些測試可列為「預期」，不得宣稱已通過。

---

### 4. 已計算的 ID/OOD selective-risk 核心結果仍未被測試

**逐字原句：**

```python
row["selected_point"] = risk_coverage(
    row["confidence"], row["correctness"], [tau])[0]
```

但測試只有：

```python
assert whole["selected_tau"] == .8
```

**原因：**

這只能證明兩個 split 記錄相同閾值，不能證明固定閾值真的被正確套用，也不能證明本 lab 最關鍵的 ID/OOD risk 差異。

依當前確定性資料重算：

- ID：accepted $2$、errors $0$、coverage $2/3$、risk $0$；
- OOD：accepted $1$、errors $1$、coverage $1/3$、risk $1$。

如果 `selected_point` 的欄位次序、mask 或 correctness 計算錯誤，現有測試仍可能通過。

**最小修法：**

加入：

```python
id_point = result["id", "all"]["selected_point"]
ood_point = result["ood", "all"]["selected_point"]

assert id_point[1] == 2
assert id_point[2] == 0
assert np.isclose(id_point[3], 2 / 3)
assert np.isclose(id_point[4], 0.0)

assert ood_point[1] == 1
assert ood_point[2] == 1
assert np.isclose(ood_point[3], 1 / 3)
assert np.isclose(ood_point[4], 1.0)
```

並核對 validation：

```python
vp = result["selection"]["validation_point"]
assert vp[1] == 1
assert vp[2] == 0
assert np.isclose(vp[3], .5)
assert np.isclose(vp[4], 0.0)
```

這些是依資料直接可重算的預期值，不是虛構實測。

---

## 四、應同步修正但不宜單獨阻擋的問題

### 5. $Y_b$ 最好正式作分段定義

**逐字原句：**

> 「令箱貢獻 $Y_b$ 在 $N_b=0$ 時為 $0$，在 $N_b>0$ 時為 $\frac{N_b}{N}|X_b|$。」

**原因：**

本版證明實質上已正確。只是正文先說：

> 「定義隨機變數 $X_b=\mathrm{Acc}_b-\bar p_b$」

而空箱中 $\mathrm{Acc}_b$、$\bar p_b$ 未定義。後文才說不必計算空箱 $X_b$。為讓證明完全無歧義，應先定義全樣本空間上的 $Y_b$，再只在 $N_b>0$ 時引入 $X_b$。

**最小修法：**

$$
Y_b=
\begin{cases}
\frac{N_b}{N}|\mathrm{Acc}_b-\bar p_b|,&N_b>0,\\
0,&N_b=0.
\end{cases}
$$

這是形式化改善，不必推翻本版 Jensen 推導。

---

### 6. Selective Risk 公式仍重複限制索引

**逐字原句：**

$$
R(\tau)=
\frac{\sum_{i:s(x_i)\ge\tau}\mathbb I(\hat y_i\ne y_i)}
{\sum_{i:s(x_i)\ge\tau}\mathbb I(s(x_i)\ge\tau)}.
$$

**原因：**

在限制後的求和範圍內，分母 indicator 恆為一。公式數值正確，但未與程式的 `accepted` mask 形式一致。

**最小修法：**

定義：

$$
a_i(\tau)=\mathbb I(s(x_i)\ge\tau),
$$

再寫：

$$
C(\tau)=\frac1N\sum_i a_i(\tau),
$$

$$
R(\tau)=
\frac{\sum_i a_i(\tau)\mathbb I(\hat y_i\ne y_i)}
{\sum_i a_i(\tau)}.
$$

---

### 7. `check_sources` 仍可能重複計入同一事件

**逐字原句：**

```python
for source, time, split, group, logits, label in records:
    if source in owners and owners[source] != split:
        raise ValueError("Source crosses splits.")
```

**原因：**

它已正確防止 source 跨 split，但沒有拒絕重複 `(source,time)`。同一事件若重複列入，PPL、ECE 與 risk 都會被重複計權。未知 split/group 也未拒絕。

**最小修法：**

增加 `(source,time)` 唯一性與 allowlist。這是資料完整性防線，不是新的模型功能。

---

### 8. 手設表格仍沒有 ECE event count

**逐字原句：**

| Total Tokens | 1000 | 1000 |
| ECE (5 bins) | 0.05 | 0.20 |

**原因：**

PPL 分母是 valid token count；ECE 分母是 calibration event count；risk 還涉及 decision count 與 accepted count。表格只有 token 數，不能說明 ECE 的樣本規模。

**最小修法：**

增加：

- Valid token count；
- Calibration event count；
- Decision count；
- Accepted count。

若 token 與事件一一對應，也要明確寫出。

---

### 9. OOD PPL 的結論仍超出指標支持範圍

**逐字原句：**

> 「這表明模型對新異常模式缺乏泛化能力。」

**原因：**

PPL 上升直接表示模型給該群真實 token 較低概率，或與該 token 分布匹配較差。它不能單獨排除詞彙、模板、長度、tokenizer、上下文或標註規則的影響。

**最小修法：**

改為：

> 「這表示模型與告警群的 token 分布匹配較差；是否屬於語義泛化失敗，仍須控制詞彙、模板、長度、tokenizer、上下文與標註規則。」

小結的「揭示模型在分布偏移下的脆弱性」也宜改成「量化模型在指定合成偏移下的表現差異」。

---

### 10. `compute_stable_log_softmax` 的 shape 契約仍不完整

**逐字原句：**

```python
Input: logits (N, L, V)
```

**原因：**

函式本身沒有 `np.asarray`、沒有三維檢查，也沒有拒絕空末軸。它實際上沿最後一軸操作，介面與 docstring 不完全一致。

**最小修法：**

要嘛正式接受任意至少一維 array-like：

```python
logits = np.asarray(logits, dtype=float)
if logits.ndim < 1 or logits.shape[-1] == 0:
    raise ValueError(...)
```

要嘛要求 `ndim == 3`。兩種都可以，但需一致。

---

## 五、資料洩漏、mask、shape、梯度及能力核對

- `logits` 為 $(N,L,V)$，labels/mask 為 $(N,L)$，gather 後為 $(N,L)$，shape 正確。
- reduction 是有效 token 求總和後只除一次，正確。
- ECE confidence/correctness 均為 $(N,)$，無錯誤 broadcast。
- `risk_coverage` 的 accepted mask 與 correctness shape 一致。
- validation、ID、OOD 的來源沒有交叉，測試集沒有參與閾值選擇。
- 本章無反向傳播，沒有梯度問題。
- 本章無 cache、position 或 attention mask，無須強行加入不相關測試。
- 章稿沒有虛構外部工具執行、下載、訓練時間、硬體或收斂結果。
- 未核對來源已明確降級為延伸閱讀，符合證據狀態。

---

## 六、結論

本版已修正 ECE 母體零機率箱，並保存 validation 選閾值證據，主要數學與 OOD 聚合流程已接近完成。剩餘真正阻擋項集中在自足程式與故障測試：overflow API 的能力敘述不成立且沒有測試；classwise ECE 沒有任何測試；固定閾值的 ID/OOD `selected_point` 已算出卻未驗證。完成這三組小修後，其他公式形式、表格分母與 OOD 措辭可同輪收尾，屆時可望核准。

VERDICT: REVISE