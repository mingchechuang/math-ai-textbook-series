## 獨立審稿結論

修訂稿已修正上一版的大部分核心錯誤：KL 非負性證明現在正確處理 $P$ 與 $Q$ 的支撐集差異；nats／bits 轉換方向已統一；CE 不再把整個批次沿所有軸歸一化；零機率不再任意裁切成 $10^{-15}$；NaN 驗證、逐樣本 CE、有效 token mask、perplexity 加權反例與離散養殖案例也都有明顯改善。作者仍只寫「預期」，沒有虛構程式已執行或測試已通過。

然而，目前仍未達可批准程度。最直接的硬性問題是正文僅標示 1918 字，低於每章最低 3000 中文字；習題缺少規定的「整合」類；測試仍只是敘述而不是自足測試程式。程式方面還有 dtype、rank、batch shape、布林 mask、負概率容差及 `np.where` 先計算非法乘積等問題。理論方面則有似然乘積缺少 i.i.d. 條件、熵／CE 的零項約定不完整、資訊幾何結論仍然過強，以及養殖例中把 $P$ 與模型預測 $Q$ 混淆。

---

# 一、先行重算與核對

## 1. KL 非負性證明

修訂後寫成

$$
\mathbb E_P\left[\frac{Q(X)}{P(X)}\right]
=\sum_{x\in S}Q(x)\leq1,
$$

再由 Jensen 得

$$
D_{\mathrm{KL}}(P\parallel Q)
\geq-\ln\sum_{x\in S}Q(x)\geq0.
$$

這次推導正確。等號分析也補足了兩個條件：

1. $Q/P$ 在 $S$ 上為常數；
2. $Q$ 在 $S$ 外沒有質量。

因此可推出 $P=Q$。上一版的支撐集漏項已修復，不應再次以舊審稿理由拒絕。

## 2. nats／bits 轉換

修訂後

$$
L_{\rm bits}=\frac{L_{\rm nats}}{\ln2}
=L_{\rm nats}\log_2e
$$

正確。例題中的

$$
0.24052/\ln2\approx0.3470\text{ bits}
$$

也正確。

## 3. KL 習題

$$
0.1\log_2(0.5)+0.2\log_2(2/3)+0.7\log_2(1.4)
$$

約為

$$
-0.1-0.116993+0.339799=0.122806\text{ bits}.
$$

稿中 $0.12281$ bits 可接受。

## 4. Perplexity 反例

Batch A：1 token、總 NLL 10；Batch B：99 tokens、總 NLL 0。

錯誤平均為

$$
\frac{e^{10}+1}{2}\approx11013.7,
$$

正確 token 加權結果為

$$
\exp\left(\frac{10}{100}\right)=e^{0.1}\approx1.10517.
$$

稿中數值合理，且成功展示不能平均不同 batch 的 perplexity。

---

# 二、仍須修正的理論問題

## 1. 似然乘積缺少獨立同分布條件

**原句：**

> `設樣本集 $\mathcal{D} = \{x_1, \dots, x_N\}$ 來自分布 $p_\theta(x)$。似然函數為：`
>
> `$$L(\theta)=\prod_{i=1}^N p_\theta(x_i)$$`

**問題：**

僅說「來自分布」不足以推出聯合機率可分解為乘積。這需要樣本在給定 $\theta$ 時獨立；若還要稱為同一個 $p_\theta$，則通常假設 i.i.d.。序列模型則不是無條件獨立，而是依鏈式法則分解為條件機率。

**最小修法：**

改成「若 $x_1,\ldots,x_N$ 在給定 $\theta$ 下為 i.i.d.，則似然為……」。另加一句：

$$
p_\theta(x_{1:T})=\prod_{t=1}^T p_\theta(x_t\mid x_{<t})
$$

是序列模型的條件分解，不是假設 token 彼此獨立。

## 2. 「取 log 避免下溢」須明確禁止先算乘積

**原句：**

> `取對數後變為求和，便於處理長序列並避免數值下溢。`

**問題：**

若程式先算 $L=\prod_i p_i$，乘積可能已下溢為零，再算 `log(L)` 無法挽救。必須直接累加每項 log probability。

**最小修法：**

補一句：「實作時直接計算 $\sum_i\ln p_i$，不可先形成機率乘積再取對數。」

## 3. 熵與交叉熵的域及零項約定仍不完整

**原句：**

> `香農熵 $H(P)$ 衡量分布 $P$ 的內在不確定性：`
>
> `$$H(P)=-\sum_xP(x)\ln P(x)$$`

以及：

> `$$H(P,Q)=-\sum_xP(x)\ln Q(x)$$`

**問題：**

稿中只對 KL 列出零機率規則，沒有明確說明：

- 熵中的 $0\ln0$ 定義為 0；
- CE 中 $P(x)=0$ 時該項為 0，即使 $Q(x)=0$；
- 若 $P(x)>0,Q(x)=0$，CE 也是 $+\infty$；
- 公式目前針對有限或可數離散樣本空間；
- 在可數無限空間中，熵也可能為 $+\infty$。

本章核心明列「熵的域」，這些不是可省略細節。

**最小修法：**

在支撐集規則中同時列出熵、CE、KL 的擴展實數約定，並說明本章程式限於有限類別軸。

## 4. CE、NLL 與 one-hot 標籤的關係沒有正式推導

本章標題與核心同時涉及 NLL 和 CE，但目前主要靠敘述連接兩者。應至少補一個簡短推導。對樣本 $i$ 的 one-hot 標籤 $y_{ic}$：

$$
H(y_i,q_i)
=-\sum_{c=1}^Cy_{ic}\ln q_{ic}
=-\ln q_{i,y_i}.
$$

因此分類資料上的平均 one-hot CE 正是正確類別的平均 NLL。若 $y_i$ 是 soft label，CE 不再只是 gather 一個正確類別機率。這是適合補足正文篇幅的必要內容，而不是灌水。

## 5. 資訊幾何敘述仍然過強

**原句：**

> `這表明 NLL 最小化是在 Fisher 計量下尋找最接近經驗分布的模型參數。`

**問題：**

前面的局部展開只描述鄰近參數分布間 KL 的二階項，且需要可微、支撐集適當、可交換微分與積分等正則條件。它不能直接推出一般的 NLL 最小化是「在 Fisher 計量下尋找最接近經驗分布」；這混合了局部參數幾何與經驗分布到模型分布的全域投影。

**最小修法：**

保留局部公式，但改成：

> 在適當正則條件及小 $\delta$ 下，KL 的二階項由 Fisher 資訊矩陣決定。這只是局部近似，不表示 KL 是全域對稱距離，也不保證 NLL 優化等同於最短 Fisher geodesic。

如果不提供來源與條件，宜把此段標為延伸直覺而非已完整證明的結論。

## 6. 養殖案例混淆參考分布 $P$ 與模型預測 $Q$

**原句：**

> `若實際落入 Bin 5（30-35度），而 $P_5=0$，則該事件 NLL 為 $\infty$。`

**問題：**

觀測 token 的模型 NLL 是 $-\ln Q_5$，是否無限取決於模型預測 $Q_5$ 是否為零，不是參考分布 $P_5$ 是否為零。此例恰巧同時設定 $P_5=Q_5=0$，所以數值結論碰巧成立，但理由寫錯。

**最小修法：**

改成：

> 因模型預測 $Q_5=0$，若觀測落入 Bin 5，模型 NLL 為 $-\ln Q_5=\infty$。$P_5=0$ 則表示所設定的參考分布也把該箱排除，但它不是這筆模型 NLL 無限的直接原因。

另應明示這是合成示例，分箱與數值不是實際養殖安全閾值。

## 7. 「當且僅當」習題實質上只是代數恆等式

**原句：**

> `證明 $H(P,Q)\geq H(P)$ 當且僅當 $D_{KL}(P\|Q)\geq0$。`

**問題：**

在 $D_{\mathrm{KL}}=H(P,Q)-H(P)$ 已成立的域內，這只是移項等價，並不是 KL 非負性的另一個證明。當交叉熵或 KL 為無限時，也應說明使用擴展實數且 $H(P)$ 有限，避免不定式。

**最小修法：**

題目改為「利用命題 4.1 與恆等式證明 $H(P,Q)\geq H(P)$，並分析等號條件」。不要把代數改寫描述成獨立非負性證明。

---

# 三、程式的 shape、dtype 與邊界問題

## 1. `validate_probability` 接受負概率，與數學契約及後續函數矛盾

**原句：**

```python
if np.any(p < -1e-12):
    raise ValueError(...)
```

**問題：**

例如 $p=[-10^{-13},1+10^{-13}]$ 會通過驗證，但概率不能為負。若它是 `y_pred`，後續 `safe_log_prob` 又使用 `p < 0` 拒絕，導致兩個驗證函數契約不一致；若它是 soft label，負項可能被 `y_true_onehot > 0` mask 靜默忽略。

**最小修法：**

非負性使用精確條件：

```python
if np.any(p < 0):
    raise ValueError(...)
```

容差只用於「總和是否接近 1」，不要用容差把負概率視為合法。

## 2. 公開函數沒有統一轉成 NumPy 浮點陣列

**原句：**

```python
if not np.all(np.isfinite(p)):
```

以及多處直接使用 `.shape`、`.ndim`、`.dtype.kind`。

**問題：**

傳入 Python list 時沒有 `.shape` 或 `.ndim`；傳入整數概率陣列時，`zeros_like(y_pred)` 會得到整數 dtype。CE 中：

```python
ce_terms = np.zeros_like(y_pred)
```

若 `y_pred` 是整數陣列，浮點乘積或無限值寫入整數陣列可能截斷或報錯。

**最小修法：**

每個公開入口先做：

```python
p = np.asarray(p, dtype=np.float64)
```

標籤索引另用 `np.asarray(y_true)`，驗證整數 dtype 後再轉成明確整數型。`ce_terms` 使用：

```python
ce_terms = np.zeros_like(y_pred, dtype=np.float64)
```

## 3. `cross_entropy` 沒有驗證 `y_pred` 必須是二維 `(N,C)`

**原句：**

```python
if np.any(y_true < 0) or np.any(y_true >= y_pred.shape[1]):
```

**問題：**

若 `y_pred` 是合法的一維概率向量 `(C,)`，`validate_probability` 會接受，但 `shape[1]` 直接失敗。若是三維陣列，函數可能沿最後軸驗證，卻用二維索引操作，介面不一致。

**最小修法：**

在開頭加入：

```python
if y_pred.ndim != 2:
    raise ValueError("y_pred must have shape (N, C).")
```

並要求 $N>0,C>0$。

## 4. index 標籤沒有驗證 batch 長度

**原句：**

```python
y_true_onehot[np.arange(y_true.shape[0]), y_true] = 1.0
```

**問題：**

若 `len(y_true) < y_pred.shape[0]`，剩餘列保持全零，之後被當作零損失樣本；若 `len(y_true) > N`，索引越界。這是嚴重的靜默 batch shape 錯誤。

**最小修法：**

加入：

```python
if y_true.shape != (y_pred.shape[0],):
    raise ValueError("Index labels must have shape (N,).")
```

## 5. `kl_divergence` 宣稱只接受 `(C,)`，實際接受高 rank

**原句：**

> `p_true, q_model: (C,) 1D probability vectors.`

**問題：**

程式只檢查 shape 相等。若輸入 `(N,C)`，`validate_probability` 會逐列接受，最後把所有 batch 的 KL 全部求和並回傳單一 float，與介面不符。

**最小修法：**

明確加入：

```python
if p_true.ndim != 1 or q_model.ndim != 1:
    raise ValueError(...)
```

若想支援 batch，另寫明回傳 `(N,)` 並沿 `axis=-1` reduction，不可兩種語義混用。

## 6. `valid_mask` 沒有驗證為布林陣列

**原句：**

```python
count = np.sum(valid_mask)
...
nll_values[valid_mask]
```

**問題：**

若傳入整數 mask `[1,1,0]`，NumPy 會把它當整數索引而不是布林 mask，選取第 1、1、0 項，造成重複與錯誤 token 計數。shape 相同不能防止此錯誤。

**最小修法：**

加入：

```python
if valid_mask.dtype != np.bool_:
    raise ValueError("valid_mask must be boolean.")
```

並先用 `np.asarray`，要求 `nll_values.ndim == valid_mask.ndim`；若介面承諾 `(N,)`，兩者都應是一維。

## 7. Perplexity 對負 NLL 與指數 overflow 沒有策略

**原句：**

```python
return np.exp(mean_nll)
```

**問題：**

由合法離散概率產生的 NLL 應非負。函數目前接受負 NLL，可能得到 $PP<1$，卻沒有說明這是否允許。非常大的有限 mean NLL 也可能在 `np.exp` 時溢位成 `inf`，並產生 warning。

**最小修法：**

對本章離散 NLL 介面拒絕有效位置上的負值。對 overflow 可明確規定「若 mean NLL 超出浮點可表示範圍則回傳 `inf`」，並使用 `np.errstate(over="ignore")`；不得把它稱為精確有限結果。

## 8. soft-label 解答中的 `np.where` 仍會先計算 `0 * -inf`

**原句：**

```python
terms = np.where(y_true_soft > 0, y_true_soft * log_q, 0)
```

**問題：**

NumPy 通常會先計算 `y_true_soft * log_q`，再由 `np.where` 選取，因此 $0\times-\infty$ 的非法運算仍可能產生 runtime warning。最終輸出可能被選成 0，但這不等於避免了中間 NaN。

**最小修法：**

與主函數相同，使用真正的索引賦值：

```python
terms = np.zeros_like(y_pred, dtype=np.float64)
mask = y_true_soft > 0
terms[mask] = y_true_soft[mask] * log_q[mask]
```

## 9. 主 CE 中的 bad branch 可以簡化，但必須保證浮點 dtype

**原句：**

```python
ce_terms[mask] = y_true_onehot[mask] * log_q[mask]
...
ce_per_sample = -np.sum(ce_terms, axis=-1)
ce_per_sample[sample_bad] = np.inf
```

**核對：**

若所有陣列為浮點，這套逐樣本邏輯能給出正確的 `inf`，shape 也已修成 `(N,)`。此處不是數學錯誤。真正問題是前述整數 dtype 可能令 `ce_terms` 無法容納 `-inf`。因此不需重寫演算法，只需固定輸入及工作陣列 dtype。

---

# 四、測試仍不符合自足故障測試要求

**原句：**

> `以下測試驗證程式的正確性。`

後面只有自然語言輸入與預期值，沒有 `assert`、例外檢查或可執行測試函數。

**問題：**

「測試策略」可以用文字，但本卷契約要求自足 CPU 程式及正常／邊界／故障測試。現在讀者仍不能直接執行一段自足測試來核對 shape、數值和例外。更不應使用「驗證程式的正確性」這種像是已完成驗證的語氣；在沒有執行紀錄時只能稱為「預期測試」。

**最小修法：**

在同一程式區塊補一個不依賴外部測試框架的 `run_expected_tests()`，至少包含：

- 正常 CE 數值與 `(3,)` shape；
- index 與 one-hot CE 一致；
- soft label；
- 一筆 $Q_y=0$ 只令該筆為 `inf`；
- KL 支撐集不含時 `inf`；
- $P=Q$ 時 KL 為 0；
- NaN、負概率、未歸一化概率；
- 一維／三維錯誤 `y_pred`；
- label batch 長度錯誤；
- 非布林 mask；
- 空有效 token；
- PAD 位置可含非有限值但不參與 reduction，或明確規定全部位置均須有限；
- token 加權 perplexity 反例。

只能寫「預期通過」或「執行時應通過」，不能寫「已通過」。

---

# 五、習題契約仍缺一類

本卷明定每章要有「手算／程式／反例／整合」四類習題及完整解答。目前題型是：

1. 手算；
2. 證明；
3. 反例；
4. 程式。

沒有整合題。

**最小修法：**

新增整合題，例如給定兩個包含 PAD 的 token batch，要求：

1. 由正確類別機率算每 token NLL；
2. 使用 loss mask 排除 PAD；
3. 合計全資料有效 token 總 NLL；
4. 算 nats/token、bits/token 與 perplexity；
5. 比較錯誤的 batch PP 平均。

並提供完整逐步答案。這也能合理補足目前不足的正文篇幅。

---

# 六、資料洩漏與評估範圍

本章沒有訓練 loop，也不是「完整 Transformer」章，因此不需要加入模型或訓練迴圈；不能因全卷有 Transformer 要求而錯誤要求本章實作完整 Transformer。

養殖案例若要提到「觸發人工審查」，應補充異常門檻只能由訓練／驗證資料制定，不能查看測試集後調整。重疊時間窗口也必須先按來源或時間切分，再建窗口。此章可只用一句範圍聲明，不必展開完整資料管線。

「高 NLL」只能是模型不意外性分數，不單獨證明感測器故障、真實危害或 OOD。稿中已寫「而非自動判定故障」，方向正確；再補上合成案例與非操作門檻即可。

---

# 七、來源與能力宣稱

參考書目中的 Cover、Goodfellow、Jurafsky–Martin 不在題目提供的 N1–N6 已核對來源清單內。稿末已誠實寫：

> `本章未執行外部程式驗證其內容`

但「未執行外部程式」和「是否實際查閱來源」是兩件不同的事。若沒有核對這些書的相應章節，不能聲稱它們已支持本文特定定義或 Fisher 幾何結論。

**最小修法：**

把未查閱來源標成「延伸參考，未逐條核對」，或提供實際核對過的章節資訊。Fisher 局部展開尤其需要可定位來源及正則條件。稿件沒有聲稱已訓練模型、取得設備 benchmark 或通過測試，這方面沒有虛構能力。

---

# 八、篇幅硬性不合格

輸入標示：

> `"measured_characters": 1918`

本卷規定每章正文最低 3000、目標 4500，而且公式、程式、英文與參考來源不充字數。即使修訂稿視覺上較長，可計入的中文正文仍未達下限。這是明確的形式契約，不可批准。

最有效的補充方向是：

1. 推導 one-hot CE 等於 NLL；
2. 定義有限／可數離散域與擴展實數零項規則；
3. 區分 i.i.d. 似然與自回歸條件似然；
4. 增加有 PAD mask 的整合手算；
5. 解釋總 NLL、每樣本平均、每 token 平均不能混用；
6. 補足自足測試碼及其預期；
7. 限縮 Fisher 幾何與異常檢測的能力宣稱。

這些都是本章核心所需內容，不是為湊字數重複敘述。

---

# 最終判定

本輪已實質修復上一版最嚴重的 KL 證明、單位轉換、批次 reduction、零概率和 perplexity 加權問題，不能沿用舊理由否定。但現稿仍有可觸發錯誤結果的 shape／dtype／mask 問題，soft-label 解答仍會計算 $0\cdot-\infty$，似然條件和分布域未完整交代，資訊幾何與養殖案例各有一處過強或錯置敘述；此外缺少自足測試碼、整合習題，正文也遠低於 3000 字硬下限。需再修訂後方可批准。

VERDICT: REVISE