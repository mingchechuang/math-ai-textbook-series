[thinking-2026-03-25T07:35:42.949Z] <thinking>
Let me analyze the request carefully.

Chapter 19: 風險、限制與人類覆核 (Risk, Limitations, and Human Oversight)
Part: 第五部｜Agent 與水產養殖系統整合
Core math: 殘差、Mahalanobis距離及可逆條件、線性限制與可行集合
Lab: 異常分數不是病害診斷；缺值或過期資料先拒絕執行，不自動加藥投餌

Conventions:
- 繁體中文
- 高中數學與基本Python讀者
- 向量預設列向量 (column vector), x∈R^d, W∈R^(m×d), y=Wx∈R^m
- 批次X∈R^(n×d)每列一筆觀測, Y=XW^T
- 樣本共變異數分母n-1
- 固定seed=42
- 實體量：temperature_c、do_mg_l、ph
- 同一故事：池A
- 每章正文2200～2600個中文字
- 至少一個完整手算數值例
- 至少一個有輸入/預期輸出的Python片段
- 至少3題習題（基本計算、觀念、養殖應用）且逐題附可核對答案
- 推導與直覺都要有
- 一個#章標題及所有指定##小節

Sections:
1. 學習目標與先備知識
2. 從池A提出問題
3. 概念與推導
4. 手算例題
5. Python實驗
6. 連回生成式AI、多模態與Agent
7. 常見錯誤與限制
8. 習題
9. 習題解答
10. 本章小結
11. 參考來源

Available images: ../figures/agent-safety.svg (and others)

Let me write this chapter carefully, following all conventions. The core math is:
- 殘差 (residual)
- Mahalanobis距離及可逆條件
- 線性限制與可行集合

Let me structure this:

1. 學習目標與先備知識 - What students should learn, prerequisites
2. 從池A提出問題 - How pool A's data raises questions about risk
3. 概念與推導 - Residual, Mahalanobis distance, invertibility condition, linear constraints and feasible sets
4. 手算例題 - Hand calculation example with numerical values
5. Python實驗 - Small Python code with input/expected output
6. 連回生成式AI、多模態與Agent - Connect back to the main thread
7. 常見錯誤與限制 - Common mistakes and limitations
8. 習題 - 3 exercises (basic calculation, concept, aquaculture application)
9. 習題解答 - Answers
10. 本章小結 - Summary
11. 參考來源 - References using given source IDs only

Let me write this in 繁體中文, keeping the main text around 2400 Chinese characters (not counting punctuation, English, formulas, and code).

Let me draft the content:

# 第19章 風險、限制與人類覆核

## 學習目標與先備知識

本章目標：
- 用殘差與Mahalanobis距離判斷觀測是否異常
- 理解共變異數矩陣可逆的條件
- 用線性不等式表示操作限制與可行集合
- 認識異常分數的侷限：不是病害診斷

先備：第6章投影與最小平方、第8章SVD與秩、第10章共變異數與PCA、第18章Agent檢索與工具介面。

## 從池A提出問題

池A的agent收到一批感測資料後，要回答：「這筆資料可信嗎？如果可信，建議的下一步是否安全？」

假設池A有兩項感測器：溫度 $temperature_c$ 與溶氧 $do_mg_l$。agent過去累積了訓練時段的樣本均值 $\mu \in \mathbb{R}^2$ 與樣本共變異數矩陣 $\Sigma \in \mathbb{R}^{2\times 2}$。當新觀測 $\mathbf{x} \in \mathbb{R}^2$ 進來時，我們需要一個「距離」來衡量它偏離正常範圍的程度。

歐氏距離忽略了變數間的相關性，也忽略了每個變數的尺度差異。例如溶氧的正常範圍約 $2\sim8\ \text{mg/L}$，溫度約 $20\sim35^\circ\text{C}$，若直接用歐氏距離，溫度偏移 $2$ 度與溶氧偏移 $2\ \text{mg/L}$ 會被當成同等嚴重，這不合理。

此外，agent在提出任何操作建議前，必須檢查：
1. 資料是否缺值或過期？→ 缺值或過期先拒絕執行。
2. 異常分數是否只是「不尋常」而非「有病」？→ 異常分數不是病害診斷。
3. 建議是否落在安全操作的限制範圍內？→ 用線性不等式界定可行集合。

## 概念與推導

### 殘差與異常分數

設 $\mathbf{x}$ 為一筆觀測（列向量的轉置寫法，即 $\mathbf{x} \in \mathbb{R}^2$ 為欄向量），$\mu$ 為訓練集均值。殘差向量为
$$\mathbf{r} = \mathbf{x} - \mu \in \mathbb{R}^2$$

若兩變數不相關且尺度相同，歐氏距離 $\|\mathbf{r}\|$ 就夠用。但池A的溫度與溶氧有負相關（溫度上升、溶氧下降），且尺度不同，我們需要「標準化」的距離。

### Mahalanobis距離

定義 **Mahalanobis距離平方**：
$$d_M^2(\mathbf{x}) = (\mathbf{x}-\mu)^\top \Sigma^{-1}(\mathbf{x}-\mu)$$

其中 $\Sigma$ 必須是 **正定**（positive definite）的，即對任何非零 $\mathbf{v}$ 都有 $\mathbf{v}^\top\Sigma\mathbf{v} > 0$。正定矩陣一定可逆（第7章譜定理：特徵值全正），所以 $\Sigma^{-1}$ 存在。

**可逆條件**：$\Sigma$ 可逆的充要條件是 $\det(\Sigma) \neq 0$，等價於所有特徵值不為零。對 $2\times2$ 實對稱矩陣 $\Sigma = \begin{pmatrix}\sigma_1^2 & \sigma_{12}\\ \sigma_{12} & \sigma_2^2\end{pmatrix}$，可逆條件為 $\sigma_1^2\sigma_2^2 - \sigma_{12}^2 > 0$，即 $|\rho| < 1$（$\rho$ 為相關係數）。若兩感測器完全相關（$\rho = \pm1$），$\Sigma$ 不可逆，Mahalanobis距離不存在——這正是第4章「相依感測器不可識別」的幾何後果。

**直覺**：$\Sigma^{-1}$ 把相關方向「旋轉」回獨立軸，再除以各軸的方差，效果是把橢圓形的正常區域拉回圓形，讓距離有統一意義。

### 線性限制與可行集合

agent提出建議時，操作量（例如調增氧量 $\delta_o$、調整水溫 $\delta_t$）必須落在安全範圍內。用線性不等式表示：
$$A\boldsymbol{\delta} \leq \mathbf{b}, \quad \boldsymbol{\delta}\in\mathbb{R}^m$$

其中 $A \in \mathbb{R}^{k\times m}$ 每列對應一條限制（如「調增量不得超過 $\delta_{\max}$」「調升與調降不能同時發生」），$\mathbf{b}\in\mathbb{R}^k$ 為上限。滿足所有不等式的 $\boldsymbol{\delta}$ 構成 **可行集合**（feasible set），它是多面體（polyhedron）。

若可行集合為空（無解），agent必須拒絕執行並回報人類，而不是隨意放寬限制。

## 手算例題

**情境**：池A訓練集 $n=10$ 筆，算得
$$\mu = \begin{pmatrix}28\\ 5.0\end{pmatrix}, \quad \Sigma = \begin{pmatrix}4 & -1\\ -1 & 0.25\end{pmatrix}$$

**步驟1：檢查 $\Sigma$ 可逆**
$$\det(\Sigma) = 4\times0.25 - (-1)^2 = 1 - 1 = 0$$

等等，$\det = 0$，$\Sigma$ 不可逆！相關係數 $\rho = \frac{-1}{\sqrt{4\times0.25}} = \frac{-1}{1} = -1$，兩變數完全負相關。這代表兩支感測器量的是同一件事，Mahalanobis距離無定義。

**修正**：假設實際 $\Sigma = \begin{pmatrix}4 & -0.6\\ -0.6 & 0.25\end{pmatrix}$，則
$$\det(\Sigma) = 4\times0.25 - 0.36 = 1 - 0.36 = 0.64 > 0$$
$\rho = \frac{-0.6}{1} = -0.6$，可逆。

**步驟2：求 $\Sigma^{-1}$**
$$\Sigma^{-1} = \frac{1}{0.64}\begin{pmatrix}0.25 & 0.6\\ 0.6 & 4\end{pmatrix} = \begin{pmatrix}0.3906 & 0.9375\\ 0.9375 & 6.25\end{pmatrix}$$

**步驟3：新觀測 $\mathbf{x} = \begin{pmatrix}33\\ 2.0\end{pmatrix}$，算殘差**
$$\mathbf{r} = \begin{pmatrix}33-28\\ 2.0-5.0\end{pmatrix} = \begin{pmatrix}5\\ -3\end{pmatrix}$$

**步驟4：Mahalanobis距離平方**
$$d_M^2 = \begin{pmatrix}5 & -3\end{pmatrix}\begin{pmatrix}0.3906 & 0.9375\\ 0.9375 & 6.25\end{pmatrix}\begin{pmatrix}5\\ -3\end{pmatrix}$$

先算中間：
$$\Sigma^{-1}\mathbf{r} = \begin{pmatrix}0.3906\times5 + 0.9375\times(-3)\\ 0.9375\times5 + 6.25\times(-3)\end{pmatrix} = \begin{pmatrix}1.953-2.8125\\ 4.6875-18.75\end{pmatrix} = \begin{pmatrix}-0.8594\\ -14.0625\end{pmatrix}$$

再內積：
$$d_M^2 = 5\times(-0.8594) + (-3)\times(-14.0625) = -4.297 + 42.1875 = 37.89$$

此值遠大於 $\chi^2_2$ 的 $95\%$ 分位數 $5.99$（僅供直覺，非嚴格統計判定），agent應標記此筆為 **高異常**，但 **不得** 直接結論「魚有病」，應回報人類覆核。

**步驟5：線性限制檢查**
假設agent建議調增溶氧 $\delta_o = 3.0\ \text{mg/L}$，限制為 $0 \leq \delta_o \leq 2.0$。因 $3.0 > 2.0$，建議超出可行集合，agent應拒絕該建議並提示人類調整。

## Python實驗

以下程式碼用NumPy重算上述數值，請在隔離環境執行：

```python
import numpy as np

mu = np.array([28.0, 5.0])
Sigma = np.array([[4.0, -0.6],
                  [-0.6, 0.25]])

# 1. 檢查可逆
det = np.linalg.det(Sigma)
print(f"det(Sigma) = {det:.4f}")  # 預期: 0.6400

# 2. 求逆
Sigma_inv = np.linalg.inv(Sigma)
print(f"Sigma_inv =\n{Sigma_inv}")
# 預期:
# [[ 0.3906  0.9375]
#  [ 0.9375  6.25  ]]

# 3. 新觀測
x = np.array([33.0, 2.0])
r = x - mu
print(f"residual r = {r}")  # 預期: [5. -3.]

# 4. Mahalanobis 距離平方
d2 = r @ Sigma_inv @ r
print(f"d_M^2 = {d2:.2f}")  # 預期: 37.89

# 5. 線性限制：delta_o 是否在 [0, 2] 內
delta_o = 3.0
feasible = (delta_o >= 0) and (delta_o <= 2.0)
print(f"feasible = {feasible}")  # 預期: False
```

## 連回生成式AI、多模態與Agent

Mahalanobis距離在本章的角色是 **資料有效性檢查**：agent在檢索SOP（第18章）與生成建議之前，先用 $\Sigma^{-1}$ 判斷輸入觀測是否可信。若 $\Sigma$ 不可逆或觀測缺值，agent應 **拒絕執行** 而非硬算。

在生成式AI管線中，異常分數只是 **觸發人類覆核的訊號**，不是診斷。LLM（或任何語言模型）可能把高 $d_M^2$ 解讀為「魚病了」，但線性代數只告訴我們「這筆資料偏離訓練分布」，原因可能是感測器漂移、季節變化、或真正的病害——區分需要現場專業人員，模型不能替代。

多模態融合（第16章）中，若某模態缺值，該模態的embedding無法可靠計算，agent應在融合前就拒絕，而非用零向量填充（零向量會被誤認為「正常」）。

![agent安全檢查流程](../figures/agent-safety.svg)

## 常見錯誤與限制

1. **把異常分數當診斷**：$d_M^2$ 大只表示「不尋常」，不表示「有病」。病害判斷需要水產專業。
2. **忽略 $\Sigma$ 可逆條件**：若感測器高度相關或樣本數 $n$ 小於變數維度 $d$，$\Sigma$ 奇異，$\Sigma^{-1}$ 不存在，Mahalanobis距離無定義。此時應降維或加正則化，但不能假裝逆矩陣存在。
3. **用測試集統計量算Mahalanobis**：$\mu$ 和 $\Sigma$ 必須只用訓練時段估計，測試/實際日只套用（第9章資料洩漏原則）。
4. **可行集合為空時自動放寬**：若所有建議都超出限制，agent應停止並回報人類，不能自行修改 $A$ 或 $\mathbf{b}$。
5. **缺值自動填零**：缺值表示「未知」，填零會把缺失偽造成「量到零」，造成錯誤的異常判斷。正確做法是標記缺值並拒絕該筆。

## 習題

**習題1（基本計算）**：池A兩維感測 $\mathbf{x}=(t, o)$，訓練集得 $\mu=(30, 4.0)$，$\Sigma=\begin{pmatrix}1 & 0\\ 0 & 0.16\end{pmatrix}$。新觀測 $\mathbf{x}=(32, 2.0)$。
(a) 求 $\Sigma^{-1}$。
(b) 算 $d_M^2$。
(c) 若限制為 $|t-30|\leq 3$ 且 $|o-4.0|\leq 1.5$，此觀測是否在可行集合內？

**習題2（觀念）**：為什麼 $\Sigma$ 必須正定才能定義Mahalanobis距離？若 $\Sigma$ 半正定但奇異（如兩感測器完全同向），該怎麼辦？請用本章的線性代數語言回答。

**習題3（養殖應用）**：池A有溫度與溶氧兩支感測器，過去7天每天一筆，$n=7$。agent建議「若 $d_M^2 > 10$ 則自動增加溶氧至 $6\ \text{mg/L}$」。請指出此建議在安全性與統計上的至少兩處問題，並說明應如何修正。

## 習題解答

**習題1**
(a) $\Sigma$ 已對角，$\Sigma^{-1}=\begin{pmatrix}1 & 0\\ 0 & 6.25\end{pmatrix}$。
(b) $\mathbf{r}=(2, -2.0)$，$d_M^2 = 2^2\times1 + (-2.0)^2\times6.25 = 4 + 25 = 29$。
(c) $|t-30|=2\leq3$ ✓；$|o-4.0|=2.0 > 1.5$ ✗。不在可行集合內。

**習題2**
Mahalanobis距離含 $\Sigma^{-1}$，若 $\Sigma$ 奇異（$\det=0$），逆不存在，距離無定義。正定保證所有特徵值 $>0$，逆存在且 $\mathbf{v}^\top\Sigma^{-1}\mathbf{v}>0$（距離非負且只在 $\mathbf{x}=\mu$ 時為零）。若 $\Sigma$ 半正定奇異，可用SVD（第8章）把 $\Sigma$ 分解，丟棄零奇異值方向，改用 **偽逆** $\Sigma^{+}$ 計算加權距離，但這已不是嚴格Mahalanobis距離，且零奇異值方向上的偏差完全不被懲罰，應視為資料冗餘並考慮合併感測器或降維。

**習題3**
(1) **安全問題**：$d_M^2>10$ 不等於「需要增氧」，可能是感測器漂移或季節變化；自動增氧到固定 $6\ \text{mg/L}$ 忽略當前溶氧值與水體容積，可能過量。修正：agent只建議、由人類批准，且增氧量應在可行集合 $0\leq\delta_o\leq\delta_{\max}$ 內。
(2) **統計問題**：$n=7$ 且 $d=2$，$\Sigma$ 自由度僅 $n-1=6$，估计不穩定；$d_M^2$ 的閾值 $10$ 未說明對應的誤報率，不能直接當作判定標準。修正：增加訓練樣本、用訓練集分佈擬合閾值，並標明「此為異常訊號，非診斷」。

## 本章小結

本章建立三個工具：(1) 殘差與Mahalanobis距離，用於判斷觀測是否偏離訓練分布，前提是 $\Sigma$ 正定可逆；(2) 線性不等式與可行集合，用於界定agent建議的安全範圍；(3) 人類覆核作為最終關卡——缺值拒絕、異常回報、建議須批准。線性代數提供的是「結構上的安全性檢查」，不是「生物上的診斷」。在生成式AI與多模態agent管線中，這些檢查是資料進入檢索與生成步驟前的必要閘門。

## 參考來源

- R1: NumPy linear algebra reference.
- R2: Deep Learning, Chapter 2: Linear Algebra.
- R7: ReAct: Synergizing Reasoning and Acting in Language Models.
- R8: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.

</thinking>

Let me count the Chinese characters in the main text (excluding punctuation, English, formulas, and code). Let me be more careful and write the final version.

Let me recount. The convention says "每章正文2200～2600個中文字；中文標點、英文、公式與程式碼不計。以腳本計數為準。"

Let me write the final version now, making sure it's well-structured and complete.

Actually, let me re-read the hand calculation. I had a nice "gotcha" where det=0 first, then I corrected it. Let me make sure the numbers work out.

$\Sigma = \begin{pmatrix}4 & -0.6\\ -0.6 & 0.25\end{pmatrix}$