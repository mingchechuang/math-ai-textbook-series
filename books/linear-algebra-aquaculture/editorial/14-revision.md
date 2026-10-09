# 第14章 低秩更新與LoRA

## 學習目標與先備知識

學完本章，你應能：

1. 寫出權重更新 $\Delta W = \frac{\alpha}{r}BA$ 的形狀：$A \in \mathbb{R}^{r \times d}$、$B \in \mathbb{R}^{m \times r}$、$\Delta W \in \mathbb{R}^{m \times d}$，並說明 $\text{rank}(\Delta W) \le r$ 的來源。
2. 在固定 $W \in \mathbb{R}^{m \times d}$、低秩 $r$ 與縮放因子 $\alpha$ 下，計算 LoRA 可訓練參數量 $r(m+d)$ 與原矩陣參數量 $md$ 的比值。
3. 舉出一個「低秩更新改變了輸出，但知識仍可能錯誤」的例子，並說明需要哪些額外機制（檢索、資料有效性檢查、人工批准）來降低風險。

先備知識：矩陣乘法與轉置（第3章）、秩與零空間（第4、5章）、SVD 截斷與近似誤差（第8章）、線性層與 softmax（第12章）。本章不要求會訓練大型模型，只需要會數形狀、做參數計算、判斷輸入方向是否落在零空間。

## 從池A提出問題

池A 的系統（第1章資料字典）有一個「水質→風險詞」的線性層：輸入 $x \in \mathbb{R}^{3}$ 是 $(\text{temperature\_c},\ \text{do\_mg\_l},\ \text{ph})$ 標準化後的列向量，$W \in \mathbb{R}^{4\times 3}$ 把 3 維特徵映到 4 個風險詞上的 logit，再經 softmax 出分數（第12章）。四個詞假設依序為「正常」、「檢查增氧機」、「停投餌」、「人工覆核」。

現在我們想讓模型「多懂一點」：例如在夏季夜間、溶氧偏低時，把「檢查增氧機」這個詞的 logit 推高。直覺做法是把 $W$ 整體重訓一次。問題是：

- 本例 $W$ 只有 $4\times 3=12$ 個參數，重訓尚可；但真實 LLM 的投影矩陣常有上億參數，整體重訓昂貴，且可能改變已學到的其他能力（通常稱「遺忘」，這是經驗動機，不是本章要證明的命題）。
- 我們想加的調整，在教學合成資料中常集中在少數有效方向上。這裡把「調整只活在低維方向」當作 LoRA 的方法假設或經驗動機來使用，而非線性代數定理：若情境需要的更新方向很多，低秩參數化就會有系統性漏掉，這正是後面的限制。

LoRA 的核心想法（R5）：把更新寫成 $\Delta W = \frac{\alpha}{r}BA$，其中 $A \in \mathbb{R}^{r \times d}$、$B \in \mathbb{R}^{m \times r}$、$r \ll \min(m,d)$。只訓練 $A,B$，凍結 $W$。這裡 $A$ 是「輸入投影」，把 $d$ 維輸入映到 $r$ 維瓶頸；$B$ 是「輸出展開」，把 $r$ 維瓶頸映回 $m$ 維輸出。

## 概念與推導

**形狀與乘法鏈。** 設 $W \in \mathbb{R}^{m \times d}$，輸入 $x \in \mathbb{R}^{d}$（列向量）。LoRA 把輸出改成

$$y = (W + \Delta W)x = Wx + \frac{\alpha}{r}BAx,$$

其中 $A \in \mathbb{R}^{r \times d}$、$B \in \mathbb{R}^{m \times r}$。逐段看形狀：$Ax \in \mathbb{R}^{r}$（$r\times d$ 乘 $d\times 1$），$B(Ax) \in \mathbb{R}^{m}$（$m\times r$ 乘 $r\times 1$）。所以 $\Delta W = \frac{\alpha}{r}BA \in \mathbb{R}^{m \times d}$，與 $W$ 同形狀，才能相加。注意乘法順序是 $BA$，不是 $AB$；若寫成 $A \in \mathbb{R}^{d\times r}$、$B \in \mathbb{R}^{m\times r}$，則 $BA$ 的內維 $r$ 與 $r$ 仍可相乘，但得到的形狀是 $m\times r$，不是 $m\times d$，不能當 $\Delta W$。標準相容寫法就是把 $A$ 放在 $r\times d$。

**秩上界。** 由第4章「乘積的秩不超過任一因子的秩」：

$$\text{rank}(\Delta W) = \text{rank}(BA) \le \min(\text{rank}(B),\ \text{rank}(A)) \le r.$$

（縮放 $\frac{\alpha}{r}$ 是非零常數，不改變秩。）所以 $\Delta W$ 的列空間——所有可能輸出方向——最多佔 $\mathbb{R}^{m}$ 中 $r$ 維子空間；由第5章秩零度定理，$\dim(\ker \Delta W) = d - \text{rank}(\Delta W) \ge d - r$。直覺：落在 $\ker(\Delta W)$ 的輸入方向不產生任何更新輸出，LoRA 對這些方向「沒有作用」。這不等同「所有非更新方向都被凍結」；更準確地說，更新只在 $A$ 能讀到、$B$ 能展開的那個低維結構上產生偏移。若真實需要的調整落在高維子空間，低秩會漏掉。

**縮放因子。** LoRA 論文（R5）常用 $\Delta W = \frac{\alpha}{r}BA$。$\alpha$ 是超參數；除以 $r$ 讓改變瓶頸寬度時，$B,A$ 的平均貢獻尺度較穩定，避免 $r$ 變大就讓偏移爆炸。本章數值例固定 $\alpha = 2$、$r=1$，故縮放係數 $\alpha/r = 2$。

**參數量。** 原矩陣有 $md$ 個參數；LoRA 只訓練 $A$ 的 $rd$ 個與 $B$ 的 $mr$ 個，合計 $r(m+d)$ 個（$W$ 凍結、不計入可訓練參數）。比值 $\frac{r(m+d)}{md}$。注意：推理時仍要用 $W + \Delta W$ 輸出（或事先合併成 $W' = W + \Delta W$），記憶體佔用不一定減少；LoRA 的收益主要在「只訓練小參數、只存小更新」，不是推理記憶體自動下降。

**為什麼「低秩」不等於「知識正確」。** $\Delta W$ 是對 logit 的線性偏移，只能做 $BAx$ 這種線性加總，不能自行產生特徵間的乘積交互。若 $A,B$ 在少量情境資料上擬合，它可能把某詞 logit 推高，但沒有「理解」何時適用、何時不適用。例如本例把「檢查增氧機」logit 推高可能合理，但同一更新可能同時把「停投餌」拉低——而後者在夜間低溶氧時未必正確。正確知識需要：檢索有來源的 SOP（第18章，R8）、資料有效性檢查（第19章）、人工批准。線性代數只保證「更新是低秩、形狀相容」，不保證「更新是對的」。

## 手算例題

固定 $W \in \mathbb{R}^{4\times 3}$（凍結），$r=1$，$\alpha=2$。令

$$A = \begin{bmatrix} 1 & 0 & 0 \end{bmatrix} \in \mathbb{R}^{1\times 3},\quad
B = \begin{bmatrix} 1 \\ -1 \\ 0 \\ 2 \end{bmatrix} \in \mathbb{R}^{4\times 1}.$$

$A$ 只讀 temperature 成分（第一維）；$B$ 對四個詞的偏移係數依序為 $[1,-1,0,2]$，即「檢查增氧機」推高、「停投餌」拉低、「人工覆核」大幅推高。

**Step 1（形狀）。** $BA \in \mathbb{R}^{4\times 3}$（$4\times1$ 乘 $1\times3$），$\Delta W = \frac{\alpha}{r}BA = 2\,BA \in \mathbb{R}^{4\times 3}$，與 $W$ 同形狀。

**Step 2（計算 $BA$）。** $BA = B A = \begin{bmatrix}1\cdot1 & 1\cdot0 & 1\cdot0 \\ -1\cdot1 & -1\cdot0 & -1\cdot0 \\ 0\cdot1 & 0\cdot0 & 0\cdot0 \\ 2\cdot1 & 2\cdot0 & 2\cdot0\end{bmatrix} = \begin{bmatrix}1&0&0\\-1&0&0\\0&0&0\\2&0&0\end{bmatrix}$。

**Step 3（$\Delta W$）。** $\Delta W = 2\,BA = \begin{bmatrix}2&0&0\\-2&0&0\\0&0&0\\4&0&0\end{bmatrix}$。

**Step 4（秩與零空間）。** $\Delta W$ 只有第 1 欄（column）非零，其餘三欄皆為 0，故 $\text{rank}(\Delta W)=1 \le r=1$。零空間：解 $\Delta W\,x = 0$，即 $2x_1 = 0 \Rightarrow x_1 = 0$，$x_2, x_3$ 自由，維度 $2 = d - r$。這表示：只要輸入的 temperature 成分不變，LoRA 完全不改變 logit——更新無法感知 do 或 ph 方向的變化。

**Step 5（參數量）。** 原 $W$：$4\times3 = 12$。LoRA：$r(m+d) = 1\times(4+3) = 7$。比值 $7/12 \approx 0.583$。本例 $W$ 太小所以比例高；若 $m=d=4096$、$r=8$，比值 $= \frac{8\times8192}{4096^2} = \frac{65536}{16777216} \approx 0.0039$，約 0.39%。當 $r \ge \min(m,d)$ 時，$BA$ 可表達任意 $m\times d$ 矩陣更新（滿秩），但參數量 $r(m+d)$ 可能已超過 $md$，仍是因子化訓練，不等於直接訓練 $W$。

**Step 6（輸出偏移，純手算，與 Python 案例分離）。** 取 $x = [1,\ -0.5,\ 0]^T$（temperature 高、do 偏低、ph 中性）。$Ax = [1\cdot1 + 0\cdot(-0.5) + 0\cdot 0] = [1]$。$B(Ax) = B \cdot 1 = [1,-1,0,2]^T$。乘 $\alpha/r = 2$ 得偏移 $[2,-2,0,4]^T$。假設凍結 $W$ 給出 $Wx = [0.1, 0.2, -0.3, 0.0]^T$（此為手算假設值，不是 Python 中 $W$ 的實算值），則新 logit $= [2.1,\ -1.8,\ -0.3,\ 4.0]^T$：「停投餌」(詞2) 被拉低、「檢查增氧機」(詞2 若依前述順序應為詞2) 與「人工覆核」被推高。注意「停投餌」被拉低在本情境未必正確——這是「低秩不保證知識正確」的具體表現：$A$ 只看 temperature，無法區分「夜間低溶氧」與「白天高溫低溶氧」。

## Python實驗

```python
import numpy as np
np.random.seed(42)

# 固定 W: 4x3 (凍結)
W = np.array([
    [ 0.3, -0.2,  0.1],
    [ 0.1,  0.4, -0.1],
    [-0.2,  0.1,  0.3],
    [ 0.0,  0.2, -0.2],
])
m, d = W.shape
r, alpha = 1, 2

A = np.array([[1.0, 0.0, 0.0]])      # r x d = 1 x 3
B = np.array([[1.0], [-1.0], [0.0], [2.0]])  # m x r = 4 x 1

# ΔW = (alpha/r) B A
dW = (alpha / r) * (B @ A)
print("dW shape:", dW.shape)          # (4, 3)
print("rank(ΔW):", np.linalg.matrix_rank(dW))  # 1
print("原參數:", m*d, "LoRA參數:", r*(m+d),
      "比值: {:.4f}".format(r*(m+d)/(m*d)))

# 推理：y = (W + ΔW) x
x = np.array([1.0, -0.5, 0.0])
y_orig = W @ x
y_lora = (W + dW) @ x
print("y_orig:", np.round(y_orig, 3))
print("y_lora:", np.round(y_lora, 3))
print("offset (should equal 2*B*A@x):", np.round(y_lora - y_orig, 3))
```

**預期輸出（人工核對，非程式已執行）：**

```
dW shape: (4, 3)
rank(ΔW): 1
原參數: 12 LoRA參數: 7 比值: 0.5833
y_orig: [ 0.4  -0.1 -0.25 -0.1 ]
y_lora: [ 2.4 -2.1 -0.25  3.9 ]
offset (should equal 2*B*A@x): [ 2. -2.  0.  4. ]
```

人工核對：$Wx = [0.3(1) - 0.2(-0.5) + 0.1(0),\ 0.1(1) + 0.4(-0.5) - 0.1(0),\ -0.2(1) + 0.1(-0.5) + 0.3(0),\ 0(1) + 0.2(-0.5) - 0.2(0)]^T = [0.4,\ -0.1,\ -0.25,\ -0.1]^T$。偏移 $2\,BAx = 2\,[1,-1,0,2]^T = [2,-2,0,4]^T$。兩者相加得 $[2.4, -2.1, -0.25, 3.9]^T$。核對重點：`rank` 應為 1（$\le r$）；`offset` 只依賴 $A,B,x$，與 $W$ 具體數值無關。請讀者在隔離環境用 NumPy 執行；此程式不連接任何真實設備、不下載模型、不下達任何養殖指令。

## 連回生成式AI、多模態與Agent

- **生成式AI**：LLM 的每個 Transformer 層都有多個線性投影（Q/K/V/Output，第13章）。LoRA 可只訓練這些投影的 $A,B$，凍結其餘權重（R5）。多模態模型（CLIP，R4）的投影層同理可加 LoRA 做情境微調。
- **多模態**：池A 的水面影像 embedding 與文字 embedding 對齊後（第16章），若要讓「低溶氧 + 水面異常」同時提升某風險詞，只需在文字塔與影像塔的各線性層加低秩適配，參數量遠小於兩塔總和。但影像與文字的「異常」語義仍靠前層非線性表示與配對資料，LoRA 本身不產生跨模態交互。
- **Agent**：第18章的 agent 用檢索（R8）拉出有來源的 SOP，再用 LoRA 微調過的模型生成建議。關鍵分離：LoRA 控制「模型傾向」，檢索控制「事實依據」，資料有效性與人工批准（第19章）控制「能不能執行」。三者缺一，低秩適配都可能產生看似合理但無來源的建議。

## 常見錯誤與限制

1. **以為 $\Delta W$ 能表達任意更新。** 不能：$\text{rank}(\Delta W)\le r$。若需要的更新秩 $> r$，低秩適配會系統性漏掉部分方向（與第8章截斷 SVD 誤差同一機制）。
2. **把 $r$ 調很大當作「無損」。** $r$ 增大後參數比 $\frac{r(m+d)}{md}$ 接近或超過 1，失去 LoRA 的節約意義；$r \ge \min(m,d)$ 時可表達任意更新，但仍需額外情境資料才不會過擬合，且不等於直接訓練 $W$。
3. **忽略縮放 $\alpha/r$。** 若不除 $r$，$r$ 改變時 $B,A$ 的學習尺度不同，跨 $r$ 比較或遷移會不穩定。
4. **把「logit 偏移符合預期」當「知識正確」。** Step 6 顯示同一更新可能把「停投餌」錯誤拉低。正確性依賴資料來源、情境條件（含交互特徵）與人工覆核，不是低秩結構能保證的。
5. **推理記憶體誤解。** 推理仍需 $W+\Delta W$（或合併後 $W'$），記憶體不減；省的是訓練參數與梯度記憶體。
6. **形狀寫錯。** 若把 $A$ 寫成 $\mathbb{R}^{d\times r}$ 又要求 $\Delta W = BA \in \mathbb{R}^{m\times d}$，需確認乘法順序；本節採 $A \in \mathbb{R}^{r\times d}$、$B \in \mathbb{R}^{m\times r}$、$\Delta W = \frac{\alpha}{r}BA$。

## 習題

**1.（基本計算）** 設 $W \in \mathbb{R}^{8\times 6}$，LoRA $r=2$，$\alpha=4$。
(a) 寫出 $A,B$ 的形狀與 $\Delta W$ 的形狀；(b) 計算 LoRA 可訓練參數量與原 $W$ 參數量的比值；(c) 若 $\text{rank}(A)=1$、$\text{rank}(B)=2$，$\text{rank}(\Delta W)$ 最大為幾？為何？

**2.（觀念）** 為什麼 $\text{rank}(\Delta W) \le r$ 與「零空間維度至少 $d-r$」相關？若 $d=3,\ r=1$，零空間最小維度是幾？舉一個零空間方向，說明該方向上 LoRA 為何「完全沒作用」。

**3.（養殖應用）** 池A 想加 LoRA 讓「夏季夜間 + do_mg_l 低」提升「檢查增氧機」logit。若輸入只有 $x = [\text{temperature},\ \text{do},\ \text{ph}]^T$ 且 $A$ 只使用 $[1,0,0]$（temperature 成分），請說明：(a) 此 $A$ 為何無法區分「夜間低溶氧」與「白天高溫低溶氧」；(b) 建議如何改輸入特徵或 $A,B$，或引入哪些其他機制（檢索/資料有效性/人工批准）來降低錯誤建議風險。

## 習題解答

**1.** (a) $A \in \mathbb{R}^{2\times 6}$，$B \in \mathbb{R}^{8\times 2}$，$\Delta W = \frac{\alpha}{r}BA \in \mathbb{R}^{8\times 6}$。
(b) LoRA 參數 $= r(m+d) = 2(8+6) = 28$；原 $W = 8\times6 = 48$；比值 $= 28/48 \approx 0.583$。
(c) $\text{rank}(BA) \le \min(\text{rank}(B),\text{rank}(A)) = \min(2,1) = 1$，故 $\text{rank}(\Delta W) \le 1$（縮放 $\alpha/r = 2$ 是非零常數，不改變秩）。限制來自 $A$ 的秩只有 1：瓶頸只有一個有效方向。

**2.** 由秩零度定理（第5章），$\dim(\ker \Delta W) = d - \text{rank}(\Delta W) \ge d - r$。$d=3,\ r=1$ 時零空間最小維度 $= 2$。零空間方向例：$x = [0,1,0]^T$（do 方向）。若 $A = [1,0,0]$，則 $Ax = 1\cdot0 + 0\cdot1 + 0\cdot0 = 0$，故 $BAx = 0$，LoRA 對該輸入輸出零偏移——沿 do 方向的輸入變化完全不被此 LoRA 感知，更新「沒作用」。

**3.** (a) $A = [1,0,0]$ 只讀 temperature，$Ax = x_1$ 與 do（$x_2$）及「夜間」無關。若夜間低溶氧與白天高溫低溶氧的 temperature 相同、do 相同，$Ax$ 相同，LoRA 給出相同偏移，無法用「夜間」條件區分；而「夜間 × 低溶氧」是交互作用，固定線性映射 $BAx$ 只能做線性加總，不能自行產生乘積交互。本質：需要的更新方向落在包含 do 與夜間指標的子空間，但 $A$ 的列空間未涵蓋，秩不足。
(b) 改法分層：(i) 輸入端加入「夜間」指標 $x_4 \in \{0,1\}$ 與交互特徵 $x_5 = x_4 \cdot \mathbb{1}[\text{do 低}]$（由資料管線預計算，非由 $BA$ 產生），讓 $A$ 能讀到 $x_4, x_5$；(ii) 提高 $r$ 並用含夜/日標籤的情境資料擬合 $A,B$；(iii) 檢索（R8）拉出「夜間低溶氧」SOP 條目作為事實依據，LoRA 只調整生成傾向、事實由檢索提供；(iv) 資料有效性檢查（第19章）：若 do 感測器斷線或過期，agent 先拒絕執行；(v) 人工批准：增氧操作由現場專業人員依 SOP 制定與執行，模型不直接下指令。這些機制可降低部分風險，但風險是否可接受須由專業人員依實際評估驗證，本章不替現場決策負責。

## 本章小結

LoRA 把權重更新寫成 $\Delta W = \frac{\alpha}{r}BA$，其中 $A \in \mathbb{R}^{r\times d}$、$B \in \mathbb{R}^{m\times r}$，用 $r(m+d)$ 個可訓練參數表達秩 $\le r$ 的更新，參數比 $\frac{r(m+d)}{md}$ 隨 $m,d$ 增大而急降。形狀鏈 $x \xrightarrow{A} \mathbb{R}^{r} \xrightarrow{B} \mathbb{R}^{m}$ 讓 $\dim(\ker \Delta W) \ge d - r$，落在零空間的輸入方向不產生更新輸出。低秩結構只保證「參數少、更新方向受限、形狀相容」，不保證「知識正確」：$BAx$ 是線性加總，不能自行產生交互，同一偏移可能在不同情境下對錯混雜。在池A 的系統中，LoRA 控制模型傾向，必須與輸入端情境/交互特徵、檢索（R8）、資料有效性檢查與人工批准（第19章）組合，才能降低部分風險；風險是否可接受須由專業人員驗證。線性代數是生成式AI 的骨架；非線性、統計、因果與控制仍不可省略或假裝已證明。

## 參考來源

- R5. LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685
- R8. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. https://arxiv.org/abs/2005.11401
- R4. Learning Transferable Visual Models From Natural Language Supervision. https://arxiv.org/abs/2103.00020
- R1. NumPy linear algebra reference. https://numpy.org/doc/stable/reference/routines.linalg.html
- R2. Deep Learning, Chapter 2: Linear Algebra. https://www.deeplearningbook.org/contents/linear_algebra.html