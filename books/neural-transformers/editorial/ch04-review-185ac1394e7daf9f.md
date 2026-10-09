## 獨立審稿結論

本次稿件在關鍵位置仍與前次相同，前輪指出的三個主要問題尚未修正：

1. `perplexity` 仍以硬編碼 700 把部分有限的 `float64` 結果提前改成 `inf`；
2. 批次 KL 解答仍在已知 $P>0,Q=0$ 時實際執行除以零；
3. KL 命題涵蓋可數無限空間，但證明沒有交代廣義 Jensen／無限和條件。

因此仍不能批准。另一方面，正文已達 3134 字、序列 NLL 負號正確、鏈式分解正確、四類習題齊備；這些舊疑慮不能繼續作為拒稿理由。

我未執行稿中程式，以下程式判斷均由逐行分析所得。

---

# 一、重新計算

## 1. 分類 NLL

三筆正確類別機率為 $0.9,0.9,0.6$：

$$
-\ln0.9\approx0.1053605,\qquad-\ln0.6\approx0.5108256.
$$

所以

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

轉成 bits：

$$
0.2405155/\ln2\approx0.34699\text{ bits}.
$$

稿中的 $0.72155$、$0.24052$、$0.3470$ 正確。

## 2. KL 手算

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

有效 token NLL 為 $2.0,0.5,0.5$，因此

$$
\overline L=\frac{3.0}{3}=1,
\qquad PP=e\approx2.71828.
$$

錯誤的 batch perplexity 平均是

$$
\frac{e^2+e^{0.5}}2\approx4.51889.
$$

稿中結果正確。

## 4. 整合題

$$
\text{Total NLL}=1.0+0.5+2.0=3.5,
$$

$$
\text{Mean NLL}=3.5/3\approx1.16667\text{ nats},
$$

$$
\text{bits/token}=1.16667/\ln2\approx1.68315,
$$

$$
PP=e^{1.16667}\approx3.21127.
$$

答案正確。

---

# 二、已通過的定義、shape 與 reduction

以下部分已正確，不應沿用早期審稿結論：

## 1. 自回歸分解

**原句：**

> `此分解來自機率鏈式法則，不要求各 token 無條件獨立，也不額外假設固定階數的馬可夫性`

正確。$p(x_{1:T})=\prod_tp(x_t\mid x_{<t})$ 是鏈式法則，不需要固定階數馬可夫假設。

## 2. 序列 NLL

**原句：**

> `$$\text{Total NLL}(\theta)=-\sum_{t=1}^T\ln p_\theta(x_t|x_{<t})$$`

負號正確。

## 3. 總和與平均的等價條件

**原句：**

> `在有效集合預先固定且 $N_{\text{valid}}>0$ 時，Total NLL 與 Mean NLL 只差固定正比例常數`

條件完整。稿件也正確指出分母若依賴 $\theta$，兩者不一定等價。

## 4. shape

- `cross_entropy`：`y_pred` 為 `(N,C)`，輸出 `(N,)`；
- `kl_divergence`：輸入 `(C,)`，輸出純量；
- 批次 KL：輸入 `(N,C)`，沿最後類別軸 reduction，輸出 `(N,)`；
- `perplexity`：只對布林 mask 為 `True` 的 token 求和，除以有效 token 數一次。

沒有錯誤 broadcast、跨 batch reduction 或重複平均。

## 5. 章稿形式

稿件已有三個逐步手算、一個 KL 非負性證明、自足 NumPy 主程式、正常／邊界／故障測試，以及手算／反例／整合／程式四類習題。正文 3134 字已超過最低 3000 字。此章不是完整 Transformer 章，不需要加入 Transformer 模型或訓練 loop。

---

# 三、主要阻斷問題一：Perplexity 的數值域仍錯

## 1. 硬編碼 700 提前把有限值改成無限

**原句：**

```python
if mean_nll > 700: # ln(float_max) approx 709
    return np.inf
```

**原因：**

`float64` 最大有限值的自然對數約為 $709.78$。所以例如：

$$
e^{701}
$$

雖然極大，但數學上有限，而且仍在 `float64` 的可表示範圍內。現在的函數卻直接回傳 `np.inf`。這不是單純精度誤差，而是改變輸出的有限／無限分類，也違反本章反對任意閾值改變精確定義的立場。

**最小修法：**

```python
limit = np.log(np.finfo(np.float64).max)
if mean_nll > limit:
    return np.inf
return float(np.exp(mean_nll))
```

或明確使用：

```python
with np.errstate(over="ignore"):
    return float(np.exp(mean_nll))
```

並說明回傳 `inf` 是 `float64` 表示範圍的結果，不代表數學 perplexity 必然無限。

## 2. 缺少對應邊界測試

現有測試只檢查 $e^1$，無法發現硬編碼 700 的錯誤。

**最小修法：**

增加預期測試：

```python
pp_large = perplexity(
    np.array([701.0]),
    np.array([True])
)
assert np.isfinite(pp_large)
```

再以略大於 `np.log(np.finfo(np.float64).max)` 的 NLL 驗證回傳 `inf`。

## 3. `sum_nll` 非有限沒有獨立處理

**原句：**

```python
sum_nll = np.sum(valid_nll)
mean_nll = sum_nll / count
```

若有限項求和發生 overflow，程式會讓 `mean_nll` 變成 `inf`。在本章的小型案例中影響有限，但既然宣稱數值穩定，至少應顯式檢查：

```python
sum_nll = np.sum(valid_nll, dtype=np.float64)
if not np.isfinite(sum_nll):
    return np.inf
```

這不是要求任意規模下的精確高精度求和，而是要求故障策略明示。

---

# 四、主要阻斷問題二：批次 KL 仍先除以零

**原句：**

```python
p_support = p > 0
bad = np.any(p_support & (q == 0), axis=-1)
terms = np.zeros_like(p, dtype=np.float64)
terms[p_support] = p[p_support] * np.log(p[p_support] / q[p_support])
```

**原因：**

`bad` 已經找出 $P>0,Q=0$ 的位置，但下一行仍以全部 `p_support` 索引。對第二筆測試資料，會實際計算：

$$
0.5/0,
$$

再取 $\log(\infty)$。最後 `out[bad]=np.inf` 雖然令數值答案正確，但中間仍產生 divide-by-zero。這與正文反對先生成非法值再掩蓋的原則矛盾。

此問題不會造成跨 batch 污染，但會產生不必要 warning，也讓「安全處理支撐集」的教學示例不成立。

**最小修法：**

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

主程式的一維 `kl_divergence` 已先檢查支撐集並立即返回，所以一維版沒有這個錯誤；只需修正程式習題答案。

還應增加一個不產生 warning 的故障測試。若不使用 warning 捕捉工具，至少文字上說明安全 mask 的目的，並避免目前必然執行的除零運算。

---

# 五、主要阻斷問題三：KL 證明的域沒有對齊

## 1. 定義涵蓋可數空間，證明卻按有限和書寫

**原句：**

> `本章定義限於有限或可數離散樣本空間 $\mathcal X$。`

以及：

> `命題 4.1：對於任意兩個概率分布 $P,Q$……`

接著直接使用：

$$
-\sum_{x\in S}P(x)\ln\frac{Q(x)}{P(x)}
\geq
-\ln\sum_{x\in S}Q(x).
$$

**原因：**

有限樣本空間中，這個證明正確。可數無限空間中，結論也成立，但需說明使用的是擴展期望版本的 Jensen，且 $D_{\mathrm{KL}}$ 可以是 $+\infty$。目前稿件沒有提供該條件或截斷極限論證，卻把有限證明直接套到「任意」可數分布。

這不是說命題結論錯，而是證明範圍超出已證內容。依本卷契約，完整證明不能靠未聲明的進階版本 Jensen 補洞。

**最小修法：**

把命題限縮為：

> 對有限離散樣本空間上的概率分布 $P,Q$……

並在證明後補一句：

> 可數無限情形可使用廣義 Jensen 或有限截斷再取極限處理，本章不展開。

這是最小且最符合本章有限類別程式的修法。

## 2. 習題 2 也需同樣限域

**原句：**

> `由恆等式 $H(P,Q)=H(P)+D_{KL}(P\|Q)$。`

若 $H(P)=+\infty$，普通代數移項可能遇到 $\infty-\infty$。正文已承認不處理此情形，因此題目也應加上「設 $\mathcal X$ 有限」或「假設相關熵有限」。

---

# 六、CE 與擴展值域仍未完整

## 1. 沒有直接說明 CE 的無限條件

**原句：**

> `若 $P(x)>0$ 且 $Q(x)=0$，則 $D_{KL}(P\|Q)=+\infty$。`

**原因：**

同一條件首先使交叉熵項

$$
-P(x)\ln Q(x)
$$

成為 $+\infty$，所以 $H(P,Q)=+\infty$。本章核心包含 CE 的域，不能只明列 KL。

**最小修法：**

改為：

> 若存在 $x$ 使 $P(x)>0$ 且 $Q(x)=0$，則 $H(P,Q)=+\infty$，且 $D_{\mathrm{KL}}(P\parallel Q)=+\infty$。

## 2. 最終指標不需要 $-\infty$

**原句：**

> `我們採用擴展實數系 $\mathbb R\cup\{+\infty,-\infty\}$。`

**原因：**

$\ln0=-\infty$ 是中間量；合法離散概率下，熵、CE、KL 與 NLL 的最終結果在 $[0,+\infty]$。目前措辭可能使讀者誤以為合法 KL 可以是負無限。

**最小修法：**

改為：

> 本章指標允許取擴展非負實數 $[0,+\infty]$；中間對數約定 $\ln0=-\infty$。

## 3. 零權重規則重複

**原句：**

> `$0\cdot\ln0$ 定義為 $0$。`
>
> `$P(x)\ln(0)$ 定義為 $0$（若 $P(x)=0$）。`

兩句可合併為：

> 零權重項約定為 $0\ln a=0$，包括 $a=0$；若權重為正而被取對數的概率為零，則相應損失項為 $+\infty$。

---

# 七、困惑度直覺仍過度一般化

**原句：**

> `若 $PP=100$，表示模型在平均意義上的預測困難度相當於在 100 個候選中隨機猜測。`

**原因：**

一般情況下：

$$
PP
=\left(
\prod_{t\in\mathrm{valid}}
p(x_t\mid x_{<t})
\right)^{-1/N_{\mathrm{valid}}},
$$

也就是正確 token 機率幾何平均的倒數。模型不一定在每一步對 100 個候選給予均勻機率。稿件前一句雖有限制到均勻分布，這一句仍容易恢復成無條件解釋。

**最小修法：**

改為：

> $PP=100$ 表示正確 token 機率的幾何平均為 $1/100$；只有均勻候選的特殊情形，才等同於在 100 個等可能候選間猜測。

---

# 八、測試與故障範圍

現有 `run_expected_tests()` 已包含正常、邊界及故障測試，不能說完全缺少測試。但尚未覆蓋數個正文明確承諾的行為：

1. Python list index labels；
2. soft labels；
3. NaN 概率拒絕；
4. 空有效 token 拒絕；
5. label batch 長度錯誤；
6. 一維或三維 `y_pred`；
7. 複數輸入拒絕；
8. 701 nats 仍應得到有限 `float64` perplexity；
9. 批次 KL 支撐集違規時不執行除零。

至少須補第 3、4、8、9 項，因它們直接涉及本章的非有限值與數值穩定性契約。

**原句：**

> `執行時應輸出 "All expected tests passed."`

這是預期結果，不是聲稱已執行。稿件沒有虛構執行成功、訓練結果、設備或耗時。

---

# 九、AI、幾何、養殖與來源

## 1. ECE 限制

**原句：**

> `校準評估應使用 ECE 等專門指標。`

ECE 受分箱方式、樣本量及聚合策略影響，不應暗示單一 ECE 足以證明校準。

**最小修法：**

改成「可搭配可靠度圖、ECE 等檢查，但 ECE 受分箱限制，不能單獨證明校準」。

## 2. Fisher 公式

**原句：**

> `$$D_{KL}(p_\theta\|p_{\theta+\delta})\approx\frac12\delta^T I(θ)\delta$$`

`I(θ)` 應改成 `I(\theta)`。此外，這是未證明的進階局部結果，需要可微、固定支撐及交換微分與期望等正則條件，或可定位來源。最小修法是標成「未證明的延伸結果」，不能讓一句「適當正則條件」取代來源與條件。

## 3. 養殖能力界線

**原句：**

> `這是一個高風險信號，觸發人工審查。`

高 NLL 是模型驚訝度，不等同現場風險。雖然後文已說合成示例、非安全閾值，仍宜改成「高 NLL 的待查信號，可提交人工審查」。不得讓模型直接控制設備，本稿目前沒有這類行為。

## 4. 來源註記

**原句：**

> `本章未執行外部程式驗證其內容，僅引用其定義作為數學依據。`

是否執行程式與是否查閱來源無關。提供的來源核對紀錄沒有說明已查閱 Cover、Goodfellow、Jurafsky–Martin 的具體章頁，也沒有支持 Fisher 局部公式的定位資訊。

**最小修法：**

改成：

> 以下為延伸參考；本章稿未提供逐章頁碼或逐條查閱紀錄，不聲稱已逐項核對。

---

# 十、總評

本章已滿足篇幅、手算、證明、程式、測試和四類習題等大部分結構要求；數學主線、nats／bits、one-hot CE、token 加權 perplexity、PAD mask 與主要 shape 均正確。完整 Transformer、cache、position 與訓練 loop 不屬本章範圍，不應要求補入。

但稿中仍有一個會直接把有限結果改成無限的程式錯誤，以及一個會在示例中實際除以零的程式錯誤；KL 證明的聲稱範圍也仍超出已交代的證明條件。這些是實質問題，不是風格偏好。均可局部修正，不需重寫全章。

VERDICT: REVISE