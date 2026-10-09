## 獨立審稿結論

本輪大部分內容仍維持上一輪可批准狀態，且 `perplexity` 對 NaN 的優先拒絕已改善。不過新加入的「測試與預期結果」文字產生兩個新的證據契約問題：

1. 文字宣稱測試函數涵蓋「有效 NLL 同時含 `inf` 與 `nan`」的場景，但程式中並沒有這個測試；
2. 文字引用尚未提供的第 6 章函數 `cross_entropy_loss_and_grad`，並斷言有限 logits 不會產生 `inf`。這不只無法由當前章稿自足核對，在有限精度浮點運算中也不具有無條件保證。

上一輪的批准不能遮蔽新版本新增的問題。這兩項都能局部修正，但目前仍需修改。我沒有執行程式，以下僅為手算與靜態審查。

---

# 一、重新計算結果

## 1. 例題 4.1

$$
-\ln0.9\approx0.1053605,
\qquad
-\ln0.6\approx0.5108256.
$$

所以：

$$
\text{Total NLL}
=2(0.1053605)+0.5108256
\approx0.7215466,
$$

$$
\text{Mean NLL}
\approx0.2405155\text{ nats}.
$$

轉換為 bits：

$$
0.2405155/\ln2\approx0.34699\text{ bits}.
$$

稿中數值正確。

## 2. KL 習題

$$
\begin{aligned}
D_{\mathrm{KL}}(P\parallel Q)
&=0.1\log_2(0.5)
 +0.2\log_2(2/3)
 +0.7\log_2(1.4)\\
&\approx0.122806\text{ bits}.
\end{aligned}
$$

稿中 $0.12281$ bits 正確。

## 3. 困惑度例題

有效 NLL 是 $2.0,0.5,0.5$：

$$
\overline L=\frac{3}{3}=1,
\qquad
PP=e\approx2.71828.
$$

錯誤 batch PP 平均：

$$
\frac{e^2+e^{0.5}}2\approx4.51889.
$$

稿中數值正確。

## 4. 整合題

$$
\text{Total NLL}=3.5,
\qquad
\text{Mean NLL}=3.5/3\approx1.16667,
$$

$$
\text{bits/token}\approx1.68315,
\qquad
PP\approx3.21127.
$$

答案正確。

---

# 二、本輪 `perplexity` 修改核對

**原句：**

```python
if np.any(valid_nll < 0):
    raise ValueError("NLL for valid tokens must be non-negative.")
if np.any(np.isnan(valid_nll)):
    raise ValueError("NaN NLL in valid tokens.")
if np.any(np.isinf(valid_nll)):
    return np.inf
```

## 行為分析

### 有限負數

若有效 NLL 含有限負數，第一個條件拒絕。正確。

### `-inf`

在 NumPy 比較語義下，`-np.inf < 0` 為 True，因此 `-inf` 會在第一個條件被拒絕，而不會進入第三個條件。這符合 NLL 不得為負的契約。

### NaN

`nan < 0` 為 False，但下一個條件會拒絕 NaN。正確。

### `+inf`

不小於零、也不是 NaN，因此第三個條件返回 `np.inf`。這符合零正確類別概率導致無限 NLL、無限 perplexity 的數學定義。

### 同時含 NaN 與 `+inf`

NaN 檢查在 `isinf` 前，因此會拋出 `ValueError`，不會因其中有 `+inf` 而靜默回傳 `inf`。這個策略是明確且合理的。

因此函數本身的這次修改正確。

---

# 三、新增的測試覆蓋宣稱與實際程式不一致

**原句：**

> `上述 run_expected_tests() 函數涵蓋了正常、邊界與故障場景：`

接著新增：

> `7. 邊界（NaN 與 inf）：有效 NLL 同時含 inf 與 nan 時預期拋出 ValueError（NaN 優先拒絕），不靜默回傳 inf。`

## 問題

`run_expected_tests()` 的第 7 個實際測試是：

```python
try:
    perplexity(np.array([1.0, 2.0]), np.array([1, 1]))
    assert False, "Should raise error for non-bool mask."
except ValueError:
    pass
```

它只測試非布林 mask，沒有呼叫：

```python
perplexity(
    np.array([np.inf, np.nan]),
    np.array([True, True])
)
```

所以「測試函數涵蓋」與程式內容不一致。這不是執行與否的問題，而是靜態可見的測試案例根本不存在。

本卷要求不能虛構測試成功或測試覆蓋。雖然文字只寫「預期」，沒有宣稱實際跑過，但仍不能把未寫入測試函數的案例描述為該函數已涵蓋。

## 最小修法

二選一。

### 修法 A：加入實際測試

```python
try:
    perplexity(
        np.array([np.inf, np.nan]),
        np.array([True, True])
    )
    assert False, "NaN should be rejected before returning inf."
except ValueError:
    pass
print("Test 8 (NaN priority) passed.")
```

這樣文字中的測試覆蓋描述才與程式相符。

### 修法 B：刪除覆蓋宣稱

把第 7 點改成：

> 額外建議測試：有效 NLL 同時含 `inf` 與 `nan` 時應拋出 `ValueError`。

此時它只是未來測試建議，不再聲稱已存在於 `run_expected_tests()`。

---

# 四、不可驗證的第 6 章函數與過強的有限 logits 保證

**原句：**

> `由有限 logits 出發的 CE（見第 6 章 cross_entropy_loss_and_grad）於本卷核心實作拒絕非有限輸入，不會由有限 logits 產生 inf。`

這句有兩層問題。

## 1. 當前章稿不自足

本次只審第 4 章，沒有提供第 6 章程式，也沒有提供可匯入模組。因而無法核對：

- `cross_entropy_loss_and_grad` 是否真的存在；
- 其函數名稱是否如此；
- 它是否拒絕非有限輸入；
- 它使用何種 dtype；
- 它如何處理極端但有限的 logits；
- 它是否直接用 log-sum-exp；
- 它是否可能在中間差值溢位。

章節綱要只說第 6 章會實作穩定 logits 版 CE，不能據此斷言某個具名函數及其行為已存在。

這與本卷「不得虛構已有可匯入模組」及「核對作者是否虛構能力」直接相關。

## 2. 「有限 logits 不會產生 inf」在浮點運算中不是無條件真命題

在精確實數數學中，有限 logits 的 softmax 各類概率嚴格為正，CE 有限。但 NumPy `float64` 的「有限」範圍非常大。例如：

```python
z = np.array([1e308, -1e308])
```

兩個輸入元素都是有限值，但差值：

```python
1e308 - (-1e308)
```

可能超出 `float64` 最大有限值而成為 `inf`。即使採穩定 log-sum-exp 形式，某些中間減法或最終 `logsumexp(z) - z_target` 仍可能溢位為 `inf`。

所以只能在附加條件下保證輸出有限，例如：

- 精確實數算術；
- 或 logits 差值落在所用 dtype 的有限表示範圍；
- 或實作明確定義極端有限輸入的溢位策略。

僅僅「輸入元素有限」不足以保證所有中間運算和輸出有限。

## 最小修法

最小且最安全的方式是刪除整個跨章函數宣稱，只保留：

> 純機率介面允許正確類別概率精確為零，因此 CE 可為 $+\infty$。logits 介面的數值穩定策略與極端有限輸入行為留待第 6 章定義。

這既不預告不存在的具名函數，也不作過強的浮點保證。

若一定要保留函數名，必須等第 6 章實際提供後再交叉核對，而且應把「不會產生 `inf`」改成有 dtype 與幅度條件的敘述。

---

# 五、核心概率與 loss 邏輯

## 1. 交叉熵

`cross_entropy` 的輸入及輸出 shape 正確：

- `y_pred`：`(N,C)`；
- index label：`(N,)`；
- soft／one-hot label：`(N,C)`；
- 輸出：`(N,)`。

reduction 明確沿最後類別軸：

```python
np.sum(ce_terms, axis=-1)
```

沒有跨 batch 平均。

零機率安全處理：

```python
mask = y_onehot > 0
ce_terms[mask] = y_onehot[mask] * log_q[mask]
```

避免 $0\cdot(-\infty)$。若 $P>0,Q=0$，相應樣本設為 `inf`。正確。

## 2. KL

一維函數先檢查支撐集，再做除法，沒有除零。

批次答案使用：

```python
safe = p_support & (q > 0)
```

因此只在 $P>0,Q>0$ 處取對數。`bad` 沿最後軸求 `any`，只污染違規樣本。正確。

## 3. 值域

正文現已正確說明：

- 最終值域為 $[0,+\infty]$；
- $\ln0=-\infty$ 是中間約定；
- $0\ln a=0$；
- $P>0,Q=0$ 時 CE 與 KL 為 $+\infty$；
- $H(P)=+\infty$ 時不使用差值形式。

這些先前問題均已修正，不應重複拒稿。

---

# 六、KL 證明與 Fisher 局部結果

KL 非負性證明的主體正確：

$$
D_{\mathrm{KL}}(P\parallel Q)
\geq-\ln\sum_{x\in S}Q(x)
\geq0.
$$

等號條件也正確導出 $P=Q$。

對可數空間，出版時仍宜補「使用擴展期望版 Jensen」；但這是局部嚴謹性補充，有限分類主體沒有錯。

Fisher 展開現在有有限空間、嚴格正概率、二次連續可微及 $\delta\to0$ 等條件，並提供推導途徑。數學方向正確。可將 `\delta^T` 改成 `\delta^\top`，但這只是排版。

---

# 七、其餘非阻斷建議

## 1. Proper scoring rule 的模型族條件

**原句：**

> `在真分布期望下，NLL 最小化對應於學習真條件分布。`

更精確地說，若允許在所有合法分布中優化，或模型族包含真分布，最小值在真分布達成；受限模型族只能得到族內最佳近似。

最小補上「若模型族能表示真條件分布」。

## 2. ECE

**原句：**

> `校準評估應使用 ECE 等專門指標。`

ECE 受分箱策略影響。建議改成「可搭配可靠度圖及 ECE，但 ECE 不能單獨證明校準」。

## 3. 養殖案例

**原句：**

> `這是一個高風險信號`

高 NLL 不等於現場高風險。建議改成「高 NLL 的待查信號」。稿件已說明是合成示例、非實際安全閾值，沒有設備控制問題。

## 4. 來源註記

**原句：**

> `本章未執行外部程式驗證其內容`

是否執行程式與是否查閱文獻不同。建議改成「未提供逐章頁碼或逐條查閱紀錄，不聲稱已逐項核對」。

---

# 八、篇幅與章節範圍

正文 3304 字，超過 3000 字下限。已有：

- 完整小命題證明；
- 三個逐步手算；
- 自足 NumPy 程式；
- 正常、邊界、故障測試；
- 四類習題及解答；
- 有效 token perplexity；
- 零機率與支撐集策略。

本章不是完整 Transformer 章，不要求加入模型、訓練 loop、KV cache、position 或 attention mask。

---

# 最終判定

數學與核心程式仍然基本正確。本輪阻斷來自新加文字，而不是原有概率實作：

1. 宣稱 `run_expected_tests()` 涵蓋 NaN／`inf` 混合測試，但程式沒有該測試；
2. 引用未提供的第 6 章具名函數並無條件宣稱有限 logits 不會產生 `inf`，既非當前章自足證據，也不符合極端浮點輸入的實際可能行為。

加入一個 NaN 優先測試，並刪除或限定跨章 logits 斷言，即可恢復批准條件。

VERDICT: REVISE