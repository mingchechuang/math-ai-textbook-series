# 第24章獨立審稿

## 一、重算結果

本次未執行程式，只作數學與 NumPy 語義核對。

### 1. PPL

第一個有效位置：

$$
\operatorname{LSE}([0,1,2,3])
=3+\ln(1+e^{-1}+e^{-2}+e^{-3})
\approx3.44019.
$$

其餘兩個高信心正確位置的 NLL 是極小正數，均非精確零；均勻 logits 位置的 NLL 為 $\ln4\approx1.38629$。因此：

$$
\mathrm{MeanNLL}\approx1.20662,
\qquad
\mathrm{PPL}\approx3.3422.
$$

正文、程式 assertion 與預期結果一致。

### 2. ECE

高信心箱的 correctness 為：

$$
[1,0,1,1,0,1,0],
$$

故正確數是四筆：

$$
\bar p_2=\frac57,\qquad
\mathrm{Acc}_2=\frac47.
$$

其加權貢獻為：

$$
\frac7{10}\left|\frac47-\frac57\right|=0.10.
$$

低信心箱貢獻為：

$$
\frac3{10}\left|0-\frac{0.7}{3}\right|=0.07.
$$

所以：

$$
\mathrm{ECE}=0.17.
$$

本版手算、`test_normal` 與「測試與預期結果」均已統一為 `0.17`，上一稿的最後一項數值矛盾已消除。

### 3. 固定閾值

validation 預期選到 $\tau=.8$。固定後：

- ID：accepted $2$、errors $0$、coverage $2/3$、risk $0$；
- OOD：accepted $1$、errors $1$、coverage $1/3$、risk $1$。

新增 assertions 與重算結果一致，且清楚展示拒答策略在 OOD 上可能失效。

---

## 二、定義與證明

### Token 加權 PPL

章稿正確使用：

$$
\mathrm{PPL}
=\exp\left(
\frac{\sum\text{有效 token NLL}}
{\sum\text{有效 token 數}}
\right).
$$

沒有平均樣本 PPL、批次 PPL或群組 PPL。命題 3.1 亦正確區分「重排已計算的 loss 項」與「跨文件串接後重新前向」，後者可能改變條件分布及造成洩漏。

### 固定分箱 ECE

命題 3.2 已正確處理：

- 樣本空箱 $N_b=0$；
- 母體零機率箱 $q_b=0$；
- 固定 bins；
- i.i.d. 條件；
- 自適應 bins 與相依樣本不保證同一結論。

對 $q_b>0$、$N_b=n>0$ 的箱使用 Jensen，再利用 $E[N_b]=Nq_b$ 得到：

$$
E[Y_b]\ge q_b|\delta_b|.
$$

求和後結論成立。散文形式的 $Y_b$ 分段定義雖可改成 cases 公式以提高形式清晰度，但現有定義已足以避免在空箱計算未定義統計量，不構成阻擋。

### Selective Risk

正文公式雖有重複的索引限制與 indicator，但數值定義正確。程式使用 `accepted` mask 實作，分母為 accepted count；空接受集回傳 NaN，沒有誤當作 risk 0。

---

## 三、shape、mask、broadcast 與 reduction

核心 shape 均正確：

- logits：$(N,L,V)$；
- labels：$(N,L)$；
- valid mask：$(N,L)$；
- log-softmax 沿最後的 $V$ 軸；
- gather 結果：$(N,L)$；
- confidence/correctness：$(N,)$；
- classwise probabilities：$(N,C)$。

`safe_labels` 正確避免無效位置的 `-1` 被 NumPy 當成最後一類索引。mask 只控制 loss reduction；所有 logits，包括 mask=False 的位置，仍要求有限。有效 token loss 求和後只除一次。

整體 split 的 PPL 由總 NLL／總 token 重算；整體 ECE 由原始事件重新計算，沒有平均群組 ECE。Risk–coverage 也以原始決策事件計算，沒有拿 token count 代替 accepted count。

本章不涉及反向傳播，無梯度問題；也沒有 cache、position 或 attention mask，不需要加入與本章無關的測試。

---

## 四、程式與測試

### 正常測試

已覆蓋：

- PPL 與 Mean NLL 手算；
- ECE $0.17$；
- ID/OOD 分群與整體聚合；
- validation 選閾值；
- 固定閾值下 ID/OOD accepted count、coverage 與 risk；
- 整體 PPL 不等於群組 PPL 的算術平均。

### 邊界測試

已覆蓋：

- 極端有限 logits；
- PPL 超出 `float64` 表示範圍時回傳 `np.inf` 並保留 Mean NLL；
- ECE 邊界 $0$、$0.5$、$1$；
- 空接受集 risk；
- risk 隨閾值上升的反例。

### 故障測試

已覆蓋：

- 空有效 token；
- label 越界；
- NaN logits；
- 空 ECE；
- 非嚴格遞增 bins；
- source 跨 split。

classwise ECE 習題答案另覆蓋正常、空樣本、空類別、row sum、label 越界與 NaN probabilities。所有測試均標為預期，沒有聲稱已執行。

---

## 五、資料洩漏與評估範圍

資料中同一 source 可含多個時間點，但不跨 validation、ID、OOD。`check_sources` 能拒絕 source 跨 split，符合本章主要洩漏防線。

validation 僅用來：

- 評估候選 thresholds；
- 在 coverage 約束下選 $\tau$；
- 保存 curve 與 selected point。

ID/OOD test 僅套用固定閾值，沒有參與調參。正文也明確要求正式資料先按來源與時間切分，再於 split 內建立窗口。

---

## 六、OOD、安全與證據範圍

本版對高 OOD PPL 的解讀已限制為 token 分布匹配較差，並明言是否屬於語義泛化失敗仍需控制詞彙、模板、長度、tokenizer、上下文與標註規則。

章稿亦清楚區分：

- PPL；
- ECE；
- 單步 selective risk；
- 整段回答的事實正確性；
- 安全性。

手設千 token 表格沒有冒充程式執行結果；程式小資料也沒有冒充已訓練模型。沒有由流暢度推論推理力、疾病辨識或操作安全。

---

## 七、引用與能力核對

末註忠實說明 N1 僅核對摘要，N6、Guo et al. 與 Niculescu-Mizil & Caruana 尚未逐條核對，只作延伸閱讀。章稿沒有聲稱已下載模型、已訓練、已執行測試、取得硬體 benchmark 或使用真實養殖資料。

非阻擋編輯建議仍包括：

1. 將 $Y_b$ 改寫成正式分段公式；
2. 用 $a_i(\tau)$ 統一 Risk 公式與程式 mask；
3. 將「Logits 溢位」改稱「極端有限 logits／PPL 表示範圍」；
4. `compute_stable_log_softmax` 可再明確化 array-like 與 shape 契約；
5. 手設表格可補 calibration event count 與 decision count；
6. 小結的「揭示脆弱性」可弱化為「量化指定偏移下的表現差異」。

以上均不改變核心數學或測試結果，不應因風格偏好阻擋核准。

## 結論

本章已具備完整小命題證明、兩個逐步手算、自足 NumPy 程式、正常／邊界／故障測試、四類習題與解答，以及可稽核的 ID/OOD 評估流程。先前 ECE 的 `0.27` 殘留已統一修正為正確的 `0.17`。未發現仍會改變核心結論、造成資料洩漏或虛構能力的阻擋錯誤。

VERDICT: APPROVE