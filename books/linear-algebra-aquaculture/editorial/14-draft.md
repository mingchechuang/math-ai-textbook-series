```markdown
# 第14章 低秩更新與LoRA

## 學習目標與先備知識

學完本章，你應能：

1. 寫出一個權重矩陣更新 $\Delta W = BA$ 的形狀，並說明 $\text{rank}(\Delta W) \le r$ 的來源。
2. 在固定 $W \in \mathbb{R}^{m \times d}$、低秩 $r$ 與縮放因子 $\alpha$ 下，計算 LoRA 的參數量與原矩陣參數量的比值。
3. 舉出一個「低秩更新改變了輸出，但知識仍可能錯誤」的例子，並說明需要哪些額外機制（檢索、資料有效性檢查、人工批准）才能進入養殖建議。

先備知識：矩陣乘法與轉置（第3章）、秩與零空間（第4、5章）、SVD 截斷與近似誤差（第8章）、線性層與 softmax（第12章）。本章不要求會訓練大型模型，只需要會數形狀與做參數計算。

## 從池A提出問題

池A 的系統（第1章資料字典）有一個「水質→風險詞」的線性層：輸入 $x \in \mathbb{R}^{3}$ 是 $(\text{temperature\_c},\ \text{do\_mg\_l},\ \text{ph})$ 標準化後的向量，$W \in \mathbb{R}^{4\times 3}$ 把 3 維特徵映到 4 個風險詞上的 logit，再經 softmax 出分數（第12章）。

現在我們想讓模型「多懂一點」：例如在夏季夜間、溶氧偏低時，把「檢查增氧機」這個詞的 logit 推高。直覺做法是把 $W$ 整體重訓一次。問題是：

- $W$ 只有 $4\times 3=12$ 個參數時還好；但真實 LLM 的 $W$ 常有上億參數，整體重訓昂貴且可能破壞已學到的能力（「遺忘」）。
- 我們真正想加的調整，常常只活在一個低維方向上——「夏季夜間 + 低溶氧」這個情境，在本例中大致是 1～2 個有效方向。

LoRA 的核心想法（R5）：把更新寫成 $\Delta W = BA$，其中 $A \in \mathbb{R}^{d \times r}$、$B \in \mathbb{R}^{m \times r}$、$r \ll \min(m,d)$。只訓練 $A,B$，凍結 $W$。

## 概念與推導

**形狀與秩。** 設 $W \in \mathbb{R}^{m \times d}$，輸入 $x \in \mathbb{R}^{d}$。LoRA 把輸出改成

$$y = (W + \Delta W)x = Wx + BAx,$$

其中 $A \in \mathbb{R}^{d \times r}$、$B \in \mathbb{R}^{m \times r}$。乘法鏈 $x \xrightarrow{A} \mathbb{R}^{r} \xrightarrow{B} \mathbb{R}^{m}$：先把 $x$ 投到 $r$ 維「瓶頸」空間，再展開回 $m$ 維。由第4章「秩不超过乘積中任一因子的秩」：

$$\text{rank}(\Delta W) = \text{rank}(BA) \le \min(\text{rank}(B),\ \text{rank}(A)) \le r.$$

所以 $\Delta W$ 的列空間（所有可能輸出方向）最多佔 $\mathbb{R}^{m}$ 中 $r$ 維子空間；零空間維度至少 $d-r$（第5章秩零度定理）。**直覺**：更新只能沿 $r$ 個方向改行為，其餘方向被凍結。若真實需要的調整落在一個高維子空間，低秩就會漏掉。

**縮放因子。** LoRA 論文（R5）常用 $\Delta W = \dfrac{\alpha}{r} BA$。$\alpha$ 是超參數；當 $r$ 增大時，$\dfrac{\alpha}{r}$ 讓 $B,A$ 的「平均貢獻」不因瓶頸變寬而爆炸。本章數值例固定 $\alpha = 2,\ r=1$，故縮放係數 $=2$。

**參數量。** 原矩陣 $md$ 個參數；LoRA 只有 $dr + mr = r(m+d)$ 個可訓練參數（$W$ 凍結、不計入訓練參數）。比值 $\dfrac{r(m+d)}{md}$。注意：推理時仍要用 $W+BA$ 輸出，記憶體佔用不一定減少；LoRA 的收益主要在「只訓練小參數」。

**為什麼「低秩」不等於「知識正確」。** $\Delta W$ 只是對 logit 的線性偏移。若 $A,B$ 是在少量情境資料上擬合的，它可能把某詞 logit 推高，但沒有理解「為何」或「何時不適用」。例如本例把「檢查增氧機」logit 推高是合理的，但同一更新可能同時把「停投餌」也推高——而後者在夜間低溶氧時未必正確。正確知識需要：檢索有來源的 SOP（第18章，R8）、資料有效性檢查（第19章）、人工批准。線性代數只保證「更新是低秩的」，不保證「更新是對的」。

## 手算例題

固定 $W \in \mathbb{R}^{4\times 3}$（凍結），$r=1$，$\alpha=2$。令

$$A = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} \in \mathbb{R}^{3\times 1},\quad
B = \begin{bmatrix} 1 \\ -1 \\ 0 \\ 2 \end{bmatrix} \in \mathbb{R}^{4\times 1}.$$

（$A$ 只取 temperature 成分；$B$ 把「檢查增氧機」(第1詞)與「停投餌」(第2詞) 都調整。）

**Step 1（形狀）。** $BA \in \mathbb{R}^{4\times 3}$，$\Delta W = \frac{\alpha}{r}BA = 2\,BA$。

**Step 2（計算 $BA$）。** $BA = B \cdot A = \begin{bmatrix}1\cdot1 & 1\cdot0 & 1\cdot0 \\ -1\cdot1 & -1\cdot0 & -1\cdot0 \\ 0\cdot1 & 0\cdot0 & 0\cdot0 \\ 2\cdot1 & 2\cdot0 & 2\cdot0\end{bmatrix} = \begin{bmatrix}1&0&0\\-1&0&0\\0&0&0\\2&0&0\end{bmatrix}$。

**Step 3（$\Delta W$）。** $\Delta W = 2\,BA = \begin{bmatrix}2&0&0\\-2&0&0\\0&0&0\\4&0&0\end{bmatrix}$。

**Step 4（秩與零空間）。** $\Delta W$ 只有第1行（列）非零，故 $\text{rank}(\Delta W)=1 \le r=1$。零空間：解 $\Delta W\,x=0$，即 $2x_1=0 \Rightarrow x_1=0$，$x_2,x_3$ 自由，維度 $2 = d-r$。這表示：只要 temperature 成分不變，LoRA 完全不改變 logit——更新無法感知 ph 或 do 的變化方向。

**Step 5（參數量）。** 原 $W$：$4\times3=12$。LoRA：$r(m+d)=1\times(4+3)=7$。比值 $7/12 \approx 58\%$。在本例 $W$ 太小所以比例高；真實 $m=d=4096,\ r=8$ 時比值 $= \frac{8\times8192}{4096^2} = \frac{65536}{16777216} \approx 0.39\%$。

**Step 6（輸出偏移）。** 取 $x = [1, -0.5, 0]^T$（temperature 高、do 偏低、ph 中性）。$Wx$ 假設已算出為 $[0.1, 0.2, -0.3, 0.0]^T$（凍結值）。LoRA 偏移 $BAx = B(Ax) = B \cdot (1\cdot1 + 0\cdot(-0.5) + 0\cdot 0) = B \cdot 1 = [1,-1,0,2]^T$；乘 $\alpha/r = 2$ 得 $[2,-2,0,4]^T$。新 logit $= [2.1, -1.8, -0.3, 4.0]^T$：「停投餌」(詞2) logit 被大幅拉低、增氧機與第4詞被推高。注意詞2被拉低在本情境未必正確——這正是「低秩不保證知識正確」的具體表現：$A$ 只看 temperature，無法區分「夜間低溶氧」與「白天高溫」。

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

A = np.array([[1.0], [0.0], [0.0]])   # d x r
B = np.array([[1.0], [-1.0], [0.0], [2.0]])  # m x r

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

**預期輸出（人工核對）：**

```
dW shape: (4, 3)
rank(ΔW): 1
原參數: 12 LoRA參數: 7 比值: 0.5833
y_orig: [ 0.55  0.1  -0.05 -0.1 ]
y_lora: [ 2.55 -1.9  -0.05  3.9 ]
offset (should equal 2*B*A@x): [ 2. -2.  0.  4. ]
```

核對重點：`rank` 應為 1（$\le r$）；`offset` 等於手算 Step 6 的 $[2,-2,0,4]^T$，與 $W$ 的具體數值無關（只依賴 $A,B,x$）。在隔離環境用 NumPy 執行；此程式不連接任何真實設備。

## 連回生成式AI、多模態與Agent

- **生成式AI**：LLM 的每個 Transformer 層都有多個線性投影（Q/K/V/Output，第13章）。LoRA 可只訓練這些投影的 $A,B$，凍結其餘權重（R5）。多模態模型（CLIP，R4）的投影層同理可加 LoRA 做情境微調。
- **多模態**：池A 的水面影像 embedding 與文字 embedding 對齊後（第16章），若要讓「低溶氧 + 水面異常」同時提升某風險詞，只需在文字塔與影像塔的各線性層加低秩適配，參數量遠小於兩塔總和。
- **Agent**：第18章的 agent 用檢索（R8）拉出有來源的 SOP，再用 LoRA 微調過的模型生成建議。關鍵分離：LoRA 控制「模型傾向」，檢索控制「事實依據」，資料有效性與人工批准（第19章）控制「能不能執行」。三者缺一，低秩適配都可能產生看似合理但無來源的建議。

## 常見錯誤與限制

1. **以為 $\Delta W$ 能表達任意更新。** 不能：$\text{rank}(\Delta W)\le r$。若需要的更新秩 $>r$，低秩適配會系統性漏掉部分方向（第8章截斷 SVD 誤差的同一機制）。
2. **把 $r$ 調很大當作「無損」。** $r$ 增大使參數比 $\frac{r(m+d)}{md}$ 接近 1，退化成整體重訓，失去 LoRA 的節約意義；且 $r$ 過大易過擬合少量情境資料。
3. **忽略縮放 $\alpha/r$。** 若不除 $r$，$r$ 改變時 $B,A$ 的學習尺度不同，跨 $r$ 比較或遷移會不穩定。
4. **把「logit 偏移正確」當「知識正確」。** 本例 Step 6 顯示同一更新可能把「停投餌」錯誤拉低。正確性依賴資料來源、情境條件與人工覆核，不是低秩結構能保證的。
5. **推理記憶體誤解。** 推理仍需 $W+BA$（或合併後 $W'$），記憶體不減；省的是訓練參數與梯度記憶體。

## 習題

**1.（基本計算）** 設 $W \in \mathbb{R}^{8\times 6}$，LoRA $r=2$，$\alpha=4$。
(a) 寫出 $A,B$ 的形狀；(b) 計算 LoRA 可訓練參數量與原 $W$ 參數量的比值；(c) 若 $\text{rank}(A)=1$、$\text{rank}(B)=2$，$\text{rank}(\Delta W)$ 最大為幾？為何？

**2.（觀念）** 為什麼 $\text{rank}(\Delta W) \le r$ 與「零空間維度至少 $d-r$」相關？若 $d=3,\ r=1$，零空間最小維度是幾？舉一個零空間方向，說明該方向上 LoRA 為何「完全沒作用」。

**3.（養殖應用）** 池A 想加 LoRA 讓「夏季夜間 + do_mg_l 低」提升「檢查增氧機」logit。若 $A$ 只使用 $[1,0,0]^T$（temperature 成分），請說明：(a) 此 $A$ 為何無法區分「夜間低溶氧」與「白天高溫低溶氧」；(b) 建議如何改 $A,B$ 或引入其他機制（檢索/資料有效性/人工批准）來降低錯誤建議風險。

## 習題解答

**1.** (a) $A \in \mathbb{R}^{6\times 2}$，$B \in \mathbb{R}^{8\times 2}$。
(b) LoRA 參數 $= r(m+d) = 2(8+6)=28$；原 $W = 8\times 6=48$；比值 $= 28/48 \approx 0.583$。
(c) $\text{rank}(BA)\le \min(\text{rank}(B),\text{rank}(A)) = \min(2,1)=1$，故 $\text{rank}(\Delta W)\le 1$（縮放 $\alpha/r$ 是非零常數，不改變秩）。限制來自 $A$ 的秩只有 1。

**2.** 由秩零度定理（第5章），$\dim(\text{null}(\Delta W)) = d - \text{rank}(\Delta W) \ge d - r$。$d=3,r=1$ 時零空間最小維度 $=2$。零空間方向例：$x = [0,1,0]^T$（do 方向）。若 $A x = 0$（如 $A=[1,0,0]^T$ 時 $Ax=0$），則 $BAx=0$，LoRA 對該輸入輸出零偏移——沿 do 方向的輸入變化完全不被此 LoRA 感知，更新「沒作用」。

**3.** (a) $A=[1,0,0]^T$ 只讀 temperature，$Ax = x_1$ 與 do_mg_l（$x_2$）無關。夜間低溶氧與白天高溫低溶氧若 temperature 相同、do 相同，$Ax$ 相同，LoRA 給出相同偏移，無法用「夜間」這個條件區分——而正確行為可能不同（夜間低溶氧更緊急）。本質：需要的更新方向落在包含 do 成分的子空間，但 $A$ 的列空間（1維，沿 temperature）未涵蓋，秩不足。
(b) 改法：(i) 讓 $A$ 包含 do 成分，例如 $r=2$，$A=\begin{bmatrix}1&0\\0&1\\0&0\end{bmatrix}$，$B$ 的兩列分別對應 temperature 與 do 的調整，並用含「夜間/白天」標籤的情境資料擬合，使 $B$ 能學習交互；但此仍只捕捉線性組合，若「夜間×低溶氧」是交互作用，純線性 $BA$ 仍不足，需在輸入端加入「夜間」特徵（第1章資料字典）。(ii) 更穩健：檢索（R8）拉出「夜間低溶氧」SOP 條目作為依據，LoRA 只生成格式、事實由檢索提供；(iii) 資料有效性檢查（第19章）：若 do 感測器斷線或過期，agent 先拒絕執行；(iv) 人工批准：增氧操作由現場人員依 SOP 執行，模型不直接下指令。四層疊加才把「低秩適配不保證知識正確」的風險壓到可接受。

## 本章小結

LoRA 把權重更新寫成 $\Delta W = \frac{\alpha}{r}BA$，用 $r(m+d)$ 個參數表達秩 $\le r$ 的更新，參數比 $\frac{r(m+d)}{md}$ 隨 $m,d$ 增大而急降。形狀鏈 $x\xrightarrow{A}\mathbb{R}^r\xrightarrow{B}\mathbb{R}^m$ 讓零空間至少 $d-r$ 維，更新只能沿 $r$ 個方向改行為。低秩結構只保證「參數少、更新方向受限」，不保證「知識正確」：同一偏移可能在不同情境下對錯混雜。在池A 的系統中，LoRA 控制模型傾向，必須與檢索（R8）、資料有效性檢查與人工批准（第19章）組合，才能形成可稽核的養殖建議輔助。線性代數是骨架；非線性、統計、因果與控制仍不可省略。

## 參考來源

- R5. LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685
- R8. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. https://arxiv.org/abs/2005.11401
- R4. Learning Transferable Visual Models From Natural Language Supervision. https://arxiv.org/abs/2103.00020
- R1. NumPy linear algebra reference. https://numpy.org/doc/stable/reference/routines.linalg.html
- R2. Deep Learning, Chapter 2: Linear Algebra. https://www.deeplearningbook.org/contents/linear_algebra.html
```