## 獨立審稿結論

本輪再有一項明確改善：困惑度的文字直覺已改為「正確 token 機率幾何平均的倒數」，並把「100 個候選隨機猜測」限制在每步均勻分布的特殊情形。上一輪對該句的批評已失效，不應再沿用。

目前主要數學手算、nats／bits、似然分解、one-hot CE、有效 token reduction、perplexity overflow 閾值及 shape 契約均正確。正文 3156 字也符合最低篇幅。

但批次 KL 的程式解答仍在 $P>0,Q=0$ 的位置先做除以零；這和本章反覆強調的「以 mask 避免非法中間運算」直接矛盾。另有 CE 值域沒有完整列出、可數空間的證明條件沒有說清、未證明的 Fisher 展開缺少可核對來源，以及測試宣稱覆蓋不足等問題。這些均可局部修正，不需要重寫全章，但現稿仍不宜批准。

我未執行稿中程式。以下結果均由手算與逐行程式分析得到。

---

# 一、重新計算

## 1. 分類 NLL

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
=0.7215466/3
\approx0.2405155\text{ nats}.
$$

轉為 bits：

$$
0.2405155/\ln2\approx0.34699\text{ bits}.
$$

稿中 $0.72155$、$0.24052$、$0.3470$ 正確。

## 2. KL 習題

$$
\begin{aligned}
D_{\mathrm{KL}}(P\parallel Q)
&=0.1\log_2(0.5)
 +0.2\log_2(2/3)
 +0.7\log_2(1.4)\\
&\approx-0.1-0.116993+0.339799\\
&\approx0.122806\text{ bits}.
\end{aligned}
$$

稿中 $0.12281$ bits 正確。

## 3. PAD 困惑度

有效 NLL 為 $2.0,0.5,0.5$：

$$
\text{Mean NLL}=\frac{3.0}{3}=1,
\qquad
PP=e\approx2.71828.
$$

錯誤 batch PP 平均為：

$$
\frac{e^2+e^{0.5}}2\approx4.51889.
$$

稿中結果正確。

## 4. 整合題

$$
\text{Total NLL}=3.5,
\qquad
\text{Mean NLL}=3.5/3\approx1.16667,
$$

$$
\text{bits/token}=1.16667/\ln2\approx1.68315,
$$

$$
PP=e^{1.16667}\approx3.21127.
$$

稿中答案正確。

---

# 二、已通過項目

## 1. i.i.d. 與自回歸條件

**原句：**

> `若 $x_1,\dots,x_N$ 在給定參數 $\theta$ 下為獨立同分布（i.i.d.）……`

乘積似然所需條件已列出。

**原句：**

> `此分解來自機率鏈式法則，不要求各 token 無條件獨立，也不額外假設固定階數的馬可夫性`

此句正確。自回歸鏈式分解不要求固定階數馬可夫假設。

## 2. Total NLL 與 Mean NLL

序列 Total NLL 的負號正確；平均值只在固定正分母下與總和有相同最優點的條件也已寫明。

## 3. one-hot CE

$$
-\sum_cy_{ic}\ln q_{ic}
=-\ln q_{i,y_i}
$$

推導正確。正確類別機率為零時，該筆 NLL／CE 為 $+\infty$。

## 4. shape 與 reduction

- `cross_entropy`：`(N,C)` 到 `(N,)`；
- `kl_divergence`：`(C,)` 到純量；
- 批次 KL：預期 `(N,C)` 到 `(N,)`；
- `perplexity`：沿有效 token 集合求和，除以有效 token 數一次。

沒有發現 batch 軸被誤聚合、錯誤 broadcast 或平均兩次。

## 5. Perplexity overflow 閾值

**原句：**

```python
if mean_nll > np.log(np.finfo(np.float64).max):
    return np.inf
```

這已修正先前硬編碼 700 的錯誤。輸入也固定轉成 `float64`，所以閾值 dtype 與實際輸出一致。

## 6. Perplexity 直覺

**原句：**

> `一般情況下，它是平均對數損失的指數，也等於正確 token 機率幾何平均的倒數。`

以及：

> `只有每一步都均勻分布在相同數量候選上的特殊情形，才等同於在 100 個等可能候選間猜測。`

這次表述正確。

---

# 三、主要程式問題：批次 KL 仍先除以零

**原句：**

```python
p_support = p > 0
bad = np.any(p_support & (q == 0), axis=-1)
terms = np.zeros_like(p, dtype=np.float64)
terms[p_support] = p[p_support] * np.log(p[p_support] / q[p_support])
```

## 原因

`bad` 已找出 $P>0,Q=0$ 的位置，但 `terms` 仍用全部 `p_support`。測試第二列為：

$$
P=[0.5,0.5,0],
\qquad
Q=[0.5,0,0.5].
$$

因此第二個類別會實際執行：

$$
0.5/0,
$$

再計算 $\ln(\infty)$。雖然最後：

```python
out[bad] = np.inf
```

使輸出值符合數學定義，但中間仍產生 divide-by-zero。這與正文所說：

> `必須使用掩蔽（masking）`

以及：

> `推薦使用索引賦值`

的安全契約不一致。

這不是 shape 錯誤，也沒有把一筆違規擴散到其他 batch；但它確實沒有避免故障運算，且範例本身必然走到該路徑。

## 最小修法

```python
p_support = p > 0
bad = np.any(p_support & (q == 0), axis=-1)
safe = p_support & (q > 0)

terms = np.zeros_like(p, dtype=np.float64)
terms[safe] = p[safe] * (
    np.log(p[safe]) - np.log(q[safe])
)

out = np.sum(terms, axis=-1)
out[bad] = np.inf
return out
```

如此：

- $P=0,Q=0$ 不計算；
- $P=0,Q>0$ 不計算；
- $P>0,Q>0$ 正常計算；
- $P>0,Q=0$ 直接由 `bad` 設為 `inf`；
- 不會先做除零；
- 每筆 batch 的結果保持獨立。

一維 `kl_divergence` 已在發現支撐集違規後立即返回 `np.inf`，所以一維版沒有此問題。

---

# 四、交叉熵的值域仍沒有完整說明

**原句：**

> `若 $P(x)>0$ 且 $Q(x)=0$，則 $D_{KL}(P\|Q)=+\infty$。`

## 原因

同一條件也直接使：

$$
-P(x)\ln Q(x)=+\infty,
$$

因此：

$$
H(P,Q)=+\infty.
$$

章綱明確要求「交叉熵、KL 及熵的域」，所以不能只列 KL 的無限條件。程式 `cross_entropy` 已正確把該筆設成 `inf`，但數學定義段沒有同步完整陳述。

## 最小修法

改成：

> 若存在 $x$ 使 $P(x)>0$ 且 $Q(x)=0$，則該交叉熵項為 $+\infty$，故 $H(P,Q)=+\infty$，且 $D_{\mathrm{KL}}(P\parallel Q)=+\infty$。

## 擴展值域措辭

**原句：**

> `採用擴展實數系 $\mathbb R\cup\{+\infty,-\infty\}$。`

對合法離散概率，熵、CE、KL、NLL 的最終值皆在 $[0,+\infty]$。$-\infty$ 只在中間量 $\ln0$ 出現。

**最小修法：**

> 本章的熵、CE、KL 與 NLL 允許取擴展非負實數 $[0,+\infty]$；中間對數約定 $\ln0=-\infty$。

## 零權重約定

**原句：**

> `$0\cdot\ln0$ 定義為 $0$。`
>
> `$P(x)\ln(0)$ 定義為 $0$（若 $P(x)=0$）。`

兩句內容重複，第二句也容易脫離括號誤讀。

**最小修法：**

> 零權重項約定為 $0\ln a=0$，包括 $a=0$；若權重為正而被取對數的概率為零，則相應損失項為 $+\infty$。

---

# 五、KL 證明的有限／可數範圍

**原句：**

> `本章定義限於有限或可數離散樣本空間 $\mathcal X$。`

接著命題對「任意兩個概率分布」使用 Jensen。

## 核對

在有限樣本空間中，現有證明完整。可數空間中結論也成立。令：

$$
Z(x)=\frac{Q(x)}{P(x)}
$$

於 $P$ 支撐上，則：

$$
\mathbb E_P[Z]
=\sum_{P(x)>0}Q(x)
\leq1.
$$

使用擴展期望版 Jensen：

$$
\mathbb E_P[-\ln Z]
\geq-\ln\mathbb E_P[Z]
\geq0.
$$

所以命題本身正確，不能以「可數情形結論錯誤」拒稿。但稿件沒有說明此處使用擴展 Jensen，左側也可能是 $+\infty$。依本卷「完整證明」要求，應把證明工具或範圍說清。

## 最小修法

二選一：

1. 把命題限定為有限離散空間，說可數情形需要擴展 Jensen 或截斷極限；
2. 保留可數空間，補上 $\mathbb E_P[Z]\leq1$、左側允許 $+\infty$ 及擴展 Jensen 的一句說明。

習題 2 也應增加「假設 $\mathcal X$ 有限」或「假設相關熵有限」，避免對 $\infty-\infty$ 直接移項。

---

# 六、程式介面與測試

## 1. `safe_log_prob` 的直接呼叫契約

**原句：**

```python
def safe_log_prob(p):
    if np.any(p < 0):
```

在 `cross_entropy` 內，`p` 已由 `_as_float_array` 及 `validate_probability` 處理，因此主流程正常。但此函數本身沒有把 Python list 轉成陣列，也沒有自行拒絕 NaN。

## 最小修法

若只作內部 helper，改名 `_safe_log_prob` 並說明只接受已驗證陣列；若作公開函數，則在開頭呼叫 `_as_float_array` 並檢查 finite。

## 2. 測試覆蓋

現有測試已有正常、邊界與故障三類，不能說完全缺少測試。但仍缺少：

- Python list 標籤；
- soft labels；
- NaN 概率；
- 空有效 mask；
- label batch 長度錯誤；
- $701$ nats 的有限 perplexity；
- 超過 `float64` 指數範圍時的 `inf`；
- 批次 KL 不先除零。

至少應補 NaN、空 mask、perplexity 上下界與批次 KL 安全 mask。

**原句：**

> `執行時應輸出 "All expected tests passed."`

這是預期，不是宣稱已執行。稿件沒有虛構測試成功、設備、耗時或訓練結果。

---

# 七、資訊幾何、校準與來源

## 1. Fisher 展開

**原句：**

> `$$D_{KL}(p_\theta\|p_{\theta+\delta})\approx\frac12\delta^T I(θ)\delta$$`

問題有二：

1. `I(θ)` 應改成 `I(\theta)`；
2. 這是未證明的進階結果，需要可微性、局部固定支撐及可交換微分與期望等條件，並應提供可定位來源。

## 最小修法

標為「未證明的延伸結果」，列出條件與來源；若無已核對來源，則只保留有限制的直覺描述。

## 2. ECE

**原句：**

> `校準評估應使用 ECE 等專門指標。`

ECE 受分箱方式及樣本量影響，不能單獨證明校準。

**最小修法：**

改成：

> 可搭配可靠度圖、ECE 等檢查，但 ECE 受分箱策略影響，不能單獨證明模型已校準。

## 3. 來源聲明

**原句：**

> `本章未執行外部程式驗證其內容，僅引用其定義作為數學依據。`

是否執行程式與是否查閱文獻無關。題目所附來源查閱紀錄沒有 Cover、Goodfellow、Jurafsky–Martin 的逐章核對資訊，也沒有 Fisher 公式的定位來源。

## 最小修法

> 以下為延伸參考；本章稿未提供逐章頁碼或逐條查閱紀錄，不聲稱已逐項核對。

---

# 八、養殖案例與評估範圍

**原句：**

> `這是一個高風險信號，觸發人工審查。`

高 NLL 是模型對觀測的低預測機率，不必然等於實際養殖風險。可能原因包括模型失配、分布偏移、感測異常或罕見正常事件。

## 最小修法

改成：

> 這是一個高 NLL 的待查信號，可提交人工審查。

稿件已正確說明：

- 門檻由驗證集制定；
- 不得查看測試集後調整；
- 案例是合成示例；
- 數值不是實際安全閾值。

沒有資料洩漏或外部設備控制程式。本章也不是完整 Transformer 章，不需加入模型、訓練 loop、KV cache 或 position。

---

# 九、章稿完整性

本章目前已符合：

- 正文超過 3000 字；
- 至少一個小命題證明；
- 三個逐步手算；
- 自足 NumPy 程式；
- 正常／邊界／故障測試；
- 手算／程式／反例／整合習題及答案；
- 明確的有效 token reduction；
- 無虛構執行或訓練紀錄。

因此不需要重寫。剩餘問題集中在批次 KL 的非法中間運算、CE 的值域陳述，以及進階定理與來源條件。

---

# 最終判定

本輪已修正 perplexity 直覺，先前相關疑慮不再成立；主要公式、數值、shape、mask 與篇幅也已合格。

但批次 KL 解答仍必然在支撐集違規案例中先執行除以零，與本章明示的 masking 安全契約衝突；CE 的無限值域及可數空間證明條件也未完整對齊。這些都是局部可修問題，但在修正前仍不宜批准。

VERDICT: REVISE