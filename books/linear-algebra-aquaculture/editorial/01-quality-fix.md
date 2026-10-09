# 第01章 從生成式AI問題看見線性代數

## 學習目標與先備知識

讀完本章，你應該能夠：用標量、向量、矩陣、張量描述一筆養殖資料的形狀（shape）與單位；把「模型輸出」和「真實現況」分開看待；並寫出第一個 $y=Wx$ 的最小數值例子。先備知識只要高中數學的座標與乘法，以及看得懂 `for` 迴圈與串列（list）。本章不預設你懂微積分、機率或深度學習，這些會在後面章節逐步補上。

全書用同一個虛構案例貫穿：虛構池A的水質感測器、水面影像與工作日誌。這套資料全是教學合成，不含任何真實物種的適用閾值，也不能拿去操作真實設備。閱讀時請把「這是用來練習符號的假資料」放在心上，不要把書中任何數字當成現場標準。

## 從池A提出問題

假設池A週邊有三種量測：水溫 `temperature_c`（攝氏）、溶氧 `do_mg_l`（mg/L）、酸鹼值 `ph`（無量綱）。某日感測器每十分鐘回報一次，於是「一個時間點」是一個長度為3的向量，「一整天」是一疊向量。管理員又上傳了一張水面影像，並在日誌寫下「池水看起來偏濁」。最後他們希望一個 agent 能讀懂這些材料，查詢標準作業程序（SOP），提出唯讀的檢查計畫。

要讓電腦處理這串問題，第一步不是選模型，而是寫**資料字典**：每一欄叫什麼、是什麼單位、形狀多大、缺值怎麼標。以下是池A的最小字典草稿。

| 名稱 | 型別 | 形狀 | 單位 | 備註 |
|---|---|---|---|---|
| `temperature_c` | 標量 | $1$ | 攝氏 | 單一感測器讀值 |
| `sensor_vec` | 向量 | $3$ | 混合 | 一筆時間點觀測 $[t,\ do,\ ph]$ |
| `day_matrix` | 矩陣 | $n\times 3$ | 混合 | 每列一筆觀測 |
| `image_patch` | 張量 | $H\times W\times C$ | 灰階或RGB | 影像區塊 |
| `log_token` | 離散 | 序列 | — | 文字切詞 |
| `tool_call` | 結構化 | dict | — | 欄位依 SOP 而定，唯讀查詢參數 |

這張表本身就是線性代數：我們用「形狀」固定每一種資料可以被哪些運算接受，用「單位」避免把攝氏和 mg/L 直接相加。`tool_call` 那一列則提醒我們，文字與數值之外還有「動作」這種資料，它一樣要被結構化，才能被檢查與記錄。各種形狀如何階層式堆疊，見圖 `../figures/shapes.svg`。

## 概念與推導

**標量（scalar）**是單一數，例如 `do_mg_l = 6.2`。**向量（vector）**是一串數；本書一律約定向量為**行向量（column vector）**，寫成 $\mathbf{x}\in\mathbb{R}^{d}$，$d$ 是維度。若無特別說明，$x$ 就代表直的柱狀，不是橫的列。這個約定看似吹毛求疵，卻是後面所有形狀檢查不出錯的基礎。

**矩陣（matrix）**是把數字排成二維表。$W\in\mathbb{R}^{m\times d}$ 表示 $m$ 列、$d$ 欄。當它作用在 $\mathbf{x}\in\mathbb{R}^{d}$ 上，得到

$$\mathbf{y}=W\mathbf{x},\qquad \mathbf{x}\in\mathbb{R}^{d},\ W\in\mathbb{R}^{m\times d},\ \mathbf{y}\in\mathbb{R}^{m}.$$

形狀檢查是：$(m\times d)(d\times 1)=(m\times 1)$，中間的 $d$ 必須一致。這條規則日後會反覆出現，請當成肌肉記憶。

為什麼是這樣乘？把 $W$ 的第 $i$ 列寫成 $\mathbf{w}_i^{\mathsf T}$，則 $y_i=\mathbf{w}_i^{\mathsf T}\mathbf{x}$，也就是「第 $i$ 個輸出是輸入特徵的一組加權和」。若 $W$ 的第二欄全是0，代表第二個感測特徵完全不影響輸出；若某一列是另一列的兩倍，兩項輸出就完全相關。換句話說，矩陣的列編碼了輸出、欄編碼了輸入，這種「列對輸出、欄對輸入」的讀法，是往後看懂注意力與線性層的關鍵。

**張量（tensor）**就是更高維的數字陣列。影像 $H\times W\times C$ 是三維張量；一個批次（batch）的感測資料是三維：$\text{batch}\times\text{time}\times\text{feature}$。以池A為例，若一天有 $T=144$ 個十分鐘時間點、一批 $B=7$ 天，則整批形狀是 $7\times 144\times 3$，索引 `X[b, t, k]` 表示第 $b$ 天第 $t$ 時的第 $k$ 個量測。名稱嚇人，本質只是「多一層索引」。這裡要分清兩件事：張量的**維數**（有幾層索引，例如三維張量）與矩陣的**秩（rank）**（線性獨立的方向數，後面章節才定義）是不同概念，初學時最容易把兩者混為一談。

**批次寫法**：把 $n$ 筆觀測疊成 $X\in\mathbb{R}^{n\times d}$，每列一筆。同一組權重對整批做映射時，慣例寫成

$$Y=XW^{\mathsf T},\qquad X\in\mathbb{R}^{n\times d},\ W\in\mathbb{R}^{m\times d},\ Y\in\mathbb{R}^{n\times m}.$$

這裡用轉置 $W^{\mathsf T}$ 把 $W$ 從 $m\times d$ 變成 $d\times m$，使 $(n\times d)(d\times m)=(n\times m)$。這和逐筆算 $W\mathbf{x}$ 再疊起來的結果相同，原因是矩陣乘法對每一列獨立作用：輸出矩陣的第 $i$ 列恰好等於 $W\mathbf{x}_i$ 的轉置。

最後要強調一句：**模型不是現實**。$y=Wx$ 只是我們對世界的線性描述。真實池水的溶氧受光合作用、耗氧、溫度、對流等非線性與延遲影響；線性代數是骨架，不是完整真相。任何後續的 agent 建議都必須經人工覆核。

## 手算例題

令一個 $2\times 3$ 的權重與一筆三維感測特徵為

$$W=\begin{bmatrix}2&-1&0\\1&1&3\end{bmatrix},\qquad \mathbf{x}=\begin{bmatrix}1\\2\\3\end{bmatrix}.$$

要算 $\mathbf{y}=W\mathbf{x}$。先確認形狀：$(2\times 3)(3\times 1)=(2\times 1)$，合法。

第一列與 $\mathbf{x}$ 做內積：$2\cdot 1+(-1)\cdot 2+0\cdot 3=2-2+0=0$。
第二列與 $\mathbf{x}$ 做內積：$1\cdot 1+1\cdot 2+3\cdot 3=1+2+9=12$。

所以

$$\mathbf{y}=\begin{bmatrix}0\\12\end{bmatrix}.$$

若換成一整批兩筆，$X=\begin{bmatrix}1&2&3\\0&1&1\end{bmatrix}$，則 $Y=XW^{\mathsf T}$。先寫 $W^{\mathsf T}=\begin{bmatrix}2&1\\-1&1\\0&3\end{bmatrix}$，再逐列乘：

第一筆：$[1,2,3]\cdot[2,-1,0]^{\mathsf T}=0$，與 $[1,2,3]\cdot[1,1,3]^{\mathsf T}=12$；第二筆：$[0,1,1]\cdot[2,-1,0]^{\mathsf T}=-1$，$[0,1,1]\cdot[1,1,3]^{\mathsf T}=4$。故 $Y=\begin{bmatrix}0&12\\-1&4\end{bmatrix}$，形狀 $2\times 2$，與 $XW^{\mathsf T}$ 的預期一致。

這裡刻意每一步都寫出中間值，因為後面章節的梯度推導會需要你回頭核對這些小數字。

## Python實驗

以下用 `numpy` 重現上面的計算。程式由讀者在自己的隔離環境執行；本書不代為執行，也不連任何真實設備。

```python
import numpy as np
np.random.seed(42)

# 單筆：y = W @ x
W = np.array([[2, -1, 0],
              [1,  1, 3]], dtype=float)      # shape (2, 3)
x = np.array([[1], [2], [3]], dtype=float)   # shape (3, 1) 行向量

y = W @ x
print("W.shape =", W.shape, " x.shape =", x.shape, " y.shape =", y.shape)
print("y =\n", y)

# 批次：Y = X @ W.T
X = np.array([[1, 2, 3],
              [0, 1, 1]], dtype=float)       # shape (2, 3)，每列一筆
Y = X @ W.T
print("Y.shape =", Y.shape)
print("Y =\n", Y)
```

預期輸出（數值可人工核對）：

```
W.shape = (2, 3)  x.shape = (3, 1)  y.shape = (2, 1)
y =
 [[ 0.]
 [12.]]
Y.shape = (2, 2)
Y =
 [[ 0. 12.]
 [-1.  4.]]
```

注意 `np.random.seed(42)` 只讓亂數可重現，**不代表**統計結果可靠；真正評估仍要靠資料切分與不確定性分析。

## 連回生成式AI、多模態與Agent

為什麼標量到矩陣值得先學？因為生成式AI的每個零件都能用這套符號寫出來。詞嵌入是查表得到向量；線性層就是 $W\mathbf{x}$；注意力的 Q、K、V 都是矩陣；影像與文字要對齊，靠的是把兩種特徵投影到同一空間後比相似度（對比學習的直覺）。到最後，一個 agent 要「檢索 SOP、呼叫工具、提出計畫」，中間仍是不斷的向量比較與矩陣乘法，只是外面包了一層決策流程（檢索增強與推理行動的架構見 R8、R7）。多模態融合的形狀約定見圖 `../figures/fusion.svg`，全書地圖見 `../figures/roadmap.svg`。

但要牢記分工：模型負責產生表示與候選建議，安全控制、劑量與投餌決策**不能**外包給模型。線性代數只是骨架，非線性、統計、因果推論與控制不可被省略，更不能假裝已證明。當 agent 說「建議提高增氧」時，那是一個待檢查的候選，不是已驗證的行動；可信控制鏈與人類覆核的位置見圖 `../figures/agent-safety.svg`。

## 常見錯誤與限制

第一，把矩陣乘法當可交換：$AB$ 通常不等於 $BA$，形狀也常常根本不能相乘。第二，混淆列與欄：中文的「行」容易歧義，本書一律叫**列（row）**與**欄／直行（column）**，向量預設是直的。第三，忽略單位：把 25（攝氏）和 6.2（mg/L）相加沒有物理意義。第四，把 seed 當萬靈丹；固定亂數只保證重現，不保證代表性。第五，把 $\mathbf{y}=W\mathbf{x}$ 的輸出當成事實，忘了它只是線性近似。

還有一類限制與本節主題直接相關：**線性模型無法表達飽和與交互作用**。真實溶氧對溫度的反應不是一條永久直線，模型在訓練分布之外可能嚴重失準。因此後續章節會談殘差、不確定性與可行性檢查，讓「模型輸出」永遠只是流程中的一環，而不是終點。

## 習題

**習題1（基本計算）** 令 $W=\begin{bmatrix}1&0&-2\\3&1&1\end{bmatrix}$，$\mathbf{x}=\begin{bmatrix}2\\1\\4\end{bmatrix}$，求 $W\mathbf{x}$ 並標出形狀。

**習題2（觀念）** 若 $X\in\mathbb{R}^{5\times 3}$、$W\in\mathbb{R}^{4\times 3}$，則 $XW^{\mathsf T}$ 的形狀為何？若改成 $XW$ 可以嗎？為什麼？

**習題3（養殖應用）** 池A一日有三筆感測向量：$[25,6.2,7.4]$、$[25,6.1,7.4]$、$[31,4.0,7.0]$（單位分別為攝氏、mg/L、無量綱）。若只把前兩筆相加再除以2，得到什麼？這個結果能不能直接用來判斷池水健康？請說明理由。

## 習題解答

**習題1** 形狀 $(2\times3)(3\times1)=(2\times1)$。第一列：$1\cdot2+0\cdot1+(-2)\cdot4=2-8=-6$；第二列：$3\cdot2+1\cdot1+1\cdot4=6+1+4=11$。故 $W\mathbf{x}=[-6,\ 11]^{\mathsf T}$。

**習題2** $XW^{\mathsf T}$：$(5\times3)(3\times4)=(5\times4)$，合法。$XW$ 需要 $(5\times3)(4\times3)$，中間 $3\ne4$，故不合法。轉置在此正是為了對齊內維度。

**習題3** 前兩筆平均為 $[(25+25)/2,(6.2+6.1)/2,(7.4+7.4)/2]=[25,6.15,7.4]$。這只是兩筆高度相似觀測的平均，取樣極小、未經標準化、未考慮不確定性，**不能**用來判斷健康；第三筆 $[31,4.0,7.0]$ 與前兩筆差異很大，更提醒我們不能只看平均。此外，本教材不提供任何真實物種的適用閾值，模型或平均數的輸出都不能代替養殖專業人員的現場判斷，也不能自動觸發投餌、加藥或增氧。

## 本章小結

本章建立了四個基本物件：標量、向量、矩陣、張量，以及它們的形狀與單位。核心式子 $y=Wx$ 與批次版 $Y=XW^{\mathsf T}$ 貫穿全書，而「列對輸出、欄對輸入」是讀懂後續章節的鑰匙。最重要的一句話是：模型不是現實。往後我們會依序補上向量與內積、矩陣映射、消去法與秩，慢慢把虛構池A的資料串成一條可稽核的流程。

## 參考來源

- R1：NumPy linear algebra reference
- R2：Deep Learning, Chapter 2: Linear Algebra
- R3：Attention Is All You Need
- R4：Learning Transferable Visual Models From Natural Language Supervision
- R5：LoRA: Low-Rank Adaptation of Large Language Models
- R6：Denoising Diffusion Probabilistic Models
- R7：ReAct: Synergizing Reasoning and Acting in Language Models
- R8：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks