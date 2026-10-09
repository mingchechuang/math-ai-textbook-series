# 第16章 影像、文字與水質的共同表示

![多模態融合示意](../figures/fusion.svg)

## 學習目標與先備知識

讀完本章，你應該能夠：把水面影像、工作日誌文字與水質感測資料，各自整理成「特徵矩陣」；用列向量與內積計算相似度；理解把不同模態投影到同一個共同空間的意義；說出特徵融合與對比學習的直覺，以及它們在資料配對與缺模態時的限制。

先備知識來自前面章節：向量、內積與範數、矩陣乘法與轉置、投影與正交、以及奇異值分解的基本概念。你不需要先懂深度學習，只要記得「向量是欄向量」這條全書共同約定：$x\in\mathbb{R}^d$、$W\in\mathbb{R}^{m\times d}$、$y=Wx\in\mathbb{R}^m$；批次資料 $X\in\mathbb{R}^{n\times d}$ 以每列為一筆觀測，$Y=XW^{\top}$。

## 從池A提出問題

池A同時有：水面影像、養殖人員的工作日誌文字、以及感測器產生的溫度、溶氧、酸鹼值序列。我們想回答一個檢索問題：當感測時段出現異常，有沒有「曾經看過類似影像、也曾經寫下類似描述」的歷史紀錄？但影像、文字、感測三種資料的原始維度不同，不能直接相減或比較。

本章的做法是：每一種模態先經過一個編碼器得到 embedding，再用線性映射「投影」到同一個 $d$ 維共同空間；在那裡，相似度用餘弦（cosine）衡量。這個想法是多模態模型與向量檢索的共同骨架。

## 概念與推導

設感測特徵 $x\in\mathbb{R}^{p}$、影像 embedding $u\in\mathbb{R}^{d_i}$、文字 embedding $v\in\mathbb{R}^{d_t}$。三個維度 $p,d_i,d_t$ 通常不同。選投影矩陣
$$W_i\in\mathbb{R}^{d\times d_i},\qquad W_t\in\mathbb{R}^{d\times d_t},$$
把影像與文字映到同一個 $d$ 維空間：
$$z_i=W_i u\in\mathbb{R}^{d},\qquad z_t=W_t v\in\mathbb{R}^{d}.$$
形狀檢查：$(d\times d_i)(d_i\times 1)=(d\times 1)$，正確。

批次情形：影像的 embedding 矩陣 $X\in\mathbb{R}^{n\times d_i}$，則
$$Z_i=X W_i^{\top}\in\mathbb{R}^{n\times d},\qquad Z_t=X_t W_t^{\top}\in\mathbb{R}^{m\times d}.$$
用全書共同記法 $Y=XW^{\top}$，每列是一筆觀測，每欄是共同空間的一個座標。

投影後先做逐列標準化，避免長度主導相似度。令 $\hat z=z/\lVert z\rVert_2$（先排除零向量）。於是相似度矩陣為
$$S=\hat Z_i\,\hat Z_t^{\top}\in\mathbb{R}^{n\times m},\qquad S_{jk}=\frac{z_i^{(j)}\cdot z_t^{(k)}}{\lVert z_i^{(j)}\rVert\,\lVert z_t^{(k)}\rVert}.$$
$S_{jk}\in[-1,1]$ 是兩模態的餘弦相似度。對比學習的直覺是：希望配對的 $(j,k)$ 對角項高、非配對的低。常用 InfoNCE 損失，對每一列做 softmax：
$$L=-\frac1n\sum_{j=1}^{n}\log\frac{\exp(S_{jj}/\tau)}{\sum_{k=1}^{m}\exp(S_{jk}/\tau)},$$
其中 $\tau>0$ 是溫度，控制分布尖銳程度。這只是目標函數的直覺，不保證學到因果或正確知識。

融合（fusion）可簡單用串接 $h=[z_i;z_t]\in\mathbb{R}^{2d}$，或用凸組合 $h=\alpha z_i+(1-\alpha)z_t$，$0\le\alpha\le1$。若某模態缺失，實務上設該項權重為零，但要注意：單模態表示承載的語義較少，不能假裝等同完整輸入。

## 手算例題

取共同空間 $d=2$，為簡化令 $W_i=W_t=I$，即 embedding 已在共同空間。兩筆影像列向量為
$$a_1=(3,4),\quad a_2=(4,-3);\qquad b_1=(3,4),\quad b_2=(3,-4).$$
先算範數：$\lVert a_1\rVert=\lVert b_1\rVert=5$，$\lVert a_2\rVert=5$，$\lVert b_2\rVert=\sqrt{9+16}=5$。標準化：
$$\hat a_1=(0.6,0.8),\ \hat a_2=(0.8,-0.6),\ \hat b_1=(0.6,0.8),\ \hat b_2=(0.6,-0.8).$$
逐項算內積：
$$S_{11}=0.6\cdot0.6+0.8\cdot0.8=1,\qquad S_{12}=0.6\cdot0.6+0.8\cdot(-0.8)=0.36-0.64=-0.28,$$
$$S_{21}=0.8\cdot0.6+(-0.6)\cdot0.8=0.48-0.48=0,\qquad S_{22}=0.8\cdot0.6+(-0.6)\cdot(-0.8)=0.96.$$
所以
$$S=\begin{pmatrix}1&-0.28\\0&0.96\end{pmatrix}.$$
對角項明顯較高，符合「配對相似」的期待。再用 $\tau=1$ 手算損失的第一列：$\exp(1)=2.718$、$\exp(-0.28)=0.756$，和為 $3.474$，得機率 $0.782$ 與 $0.218$。第二列：$\exp(0)=1$、$\exp(0.96)=2.612$，和為 $3.612$，機率為 $0.277$ 與 $0.723$。故
$$L=-\frac12\bigl(\log 0.782+\log 0.723\bigr)=-\frac12(-0.246-0.324)=0.285.$$

## Python實驗

以下程式只用標準庫與 NumPy，需由讀者在隔離環境自行執行；本文不聲稱已執行。

```python
import numpy as np

# 輸入：2 筆影像、2 筆文字的 2 維 embedding（每列一筆）
A = np.array([[3., 4.],
              [4., -3.]])   # 影像
B = np.array([[3., 4.],
              [3., -4.]])   # 文字

def rownorm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / n                 # 逐列標準化

S = rownorm(A) @ rownorm(B).T    # 相似度矩陣
print(A.shape, B.shape, S.shape)
print(np.round(S, 2))

# 對每一列做 softmax
def softmax_rows(S, tau=1.0):
    E = np.exp(S / tau - np.max(S / tau, axis=1, keepdims=True))
    return E / np.sum(E, axis=1, keepdims=True)

P = softmax_rows(S, tau=1.0)
print(np.round(P, 3))
print(np.round(-np.mean(np.log(np.diag(P))), 3))
```

預期輸出（小數四捨五入）：

```
(2, 2) (2, 2) (2, 2)
[[ 1.   -0.28]
 [ 0.    0.96]]
[[0.782 0.218]
 [0.277 0.723]]
0.285
```

形狀 $(2,2)$ 表示 2 筆影像對 2 筆文字；對角項是配對相似度。若把 `A`、`B` 換成真實 embedding，記得先確認每列非零。

## 連回生成式AI、多模態與Agent

CLIP 這類雙塔模型正是用「影像編碼器 + 文字編碼器 + 共同空間 + 對比損失」把兩種模態對齊（R4）；檢索增強生成則用向量相似度取回文件再交給語言模型（R8）。在池A的 agent 情境，感測、影像、日誌可以被投影到同一表示，agent 再用餘弦檢索最接近的 SOP 片段。但請注意：向量檢索只回傳「相似」的候選，不是因果診斷，也不是安全判斷；任何投餌、加藥、增氧都必須由現場專業人員依規定決定。

## 常見錯誤與限制

第一，忘記逐列標準化，會讓向量長度而非方向決定相似度。第二，零向量沒有方向，必須先排除或另行處理。第三，相似度高不等於健康或正確，S 只是表示層的幾何量測。第四，缺模態時直接補零，會產生誤導性的相似度。第五，融合權重 $\alpha$ 若是人工指定，就不是統計證據。第六，對比學習需要正確的配對資料；配對錯誤會把不相關內容拉近。本章所有資料皆為教學合成，不能用來推論真實物種的閾值。

## 習題

1.（基本計算）設 $a_1=(1,0)$、$a_2=(0,1)$，$b_1=(2,2)$、$b_2=(-2,2)$。計算標準化後的 $\hat a$、$\hat b$，以及 $2\times2$ 相似度矩陣 $S$。
2.（觀念）為什麼投影到共同空間後還要逐列標準化？若某時段影像缺失、以零向量補上，會對相似度造成什麼問題？
3.（養殖應用）池A要把「感測異常時段」與「日誌描述」和「水面影像」配對。請描述一種建立配對資料的方式，以及時間切分時如何避免資料洩漏。

## 習題解答

1. $\lVert a_1\rVert=\lVert a_2\rVert=1$，故 $\hat a_1=(1,0)$、$\hat a_2=(0,1)$。$\lVert b_1\rVert=\lVert b_2\rVert=\sqrt8\approx2.828$，故 $\hat b_1\approx(0.707,0.707)$、$\hat b_2\approx(-0.707,0.707)$。於是
$$S=\begin{pmatrix}0.707&-0.707\\0.707&0.707\end{pmatrix}.$$
對角項 $0.707$ 皆高於非對角項（$0.707$ 與 $-0.707$；注意第二列非對角為正但小於對角）。

2. 標準化把比較限制在方向上，排除「長度」這個與語義未必相關的自由度，使餘弦只反映方向相似。零向量 $\hat z=0/0$ 未定義；補零後與任何向量的相似度都為 0，會讓「缺資料」被誤讀成「與所有描述都不相似」，而不是「未知」，因此缺模態應明確標記，不可當成一般數值。

3. 可以用同一時間窗把三模態綁成一組：感測視窗、該窗內的日誌段落、該窗的代表影像，構成配對樣本 $(x,u,v)$。切分時必須以「時間」為單位，先切訓練、驗證、測試時段，再在各自時段內配對；不可用測試日統計量估計標準化參數或投影，否則造成洩漏。缺模態樣本要分開處理，不能靠補值製造假配對。

## 本章小結

不同模態原始維度不同，但可各自編碼後以 $W_i$、$W_t$ 投影到共同 $d$ 維空間，用 $S=\hat Z_i\hat Z_t^{\top}$ 比較相似度。對比學習以 softmax 拉高配對、壓低非配對；融合可用串接或加權。關鍵限制是：相似度不是因果或安全結論、缺模態不能補零了事、配對品質決定一切，而線性代數只是骨架，統計、因果與控制不可省略。

## 參考來源

R1 NumPy linear algebra reference；R2 Deep Learning, Chapter 2: Linear Algebra；R4 Learning Transferable Visual Models From Natural Language Supervision；R8 Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks。