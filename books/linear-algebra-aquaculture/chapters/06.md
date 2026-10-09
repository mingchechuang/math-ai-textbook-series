# 第06章 正交、投影與最小平方

## 學習目標與先備知識
本章要建立四件事：第一，理解正交、正交基底與投影的幾何意義；第二，會用Gram–Schmidt把一組線性獨立向量變成正交歸一基底；第三，知道QR分解的概念，並用它解最小平方而不必顯式求逆；第四，理解正規方程$A^\top A\hat x=A^\top y$的限制，知道何時該改用QR、SVD或正則化。先備知識只有：向量內積、矩陣乘法、線性組合與轉置。全程預設向量為column vector，$x\in\mathbb R^d$，$W\in\mathbb R^{m\times d}$，$y=Wx\in\mathbb R^m$；批次$X\in\mathbb R^{n\times d}$以每列（row）一筆觀測，$Y=XW^\top$。本章只用到高中數學與基本Python，不預設微積分、機率或深度學習。

## 從池A提出問題
虛構養殖池A在三日同一時段記錄溶氧$do\_mg\_l$與時間$t$（日）。所有資料皆為教學合成，不是真實物種適用閾值，也不代表任何現場安全範圍。資料字典為：$t$是感測器時間，$y$是溶氧量測，$A$是設計矩陣。因為感測器有噪聲，三點不會正好落在一條直線上，所以我們想找一條直線$\hat y=a+bt$，使三個預測值與量測值的殘差平方和最小。

線性代數把這個問題改寫為投影：量測向量$y\in\mathbb R^3$是空間中的一點，而所有可能的直線$a+bt$所形成的集合，是$A$的各個column（欄）所張成的column space$S$。$S$是由$(1,1,1)$與$(0,1,2)$兩方向張成的二維平面。若$y$不在這個平面上，最佳直線就是$y$在$S$的投影。要理解這個「最近點」為何唯一、如何計算，正是本章主題。必須先說清楚：此擬合只描述這三筆模擬資料，不能推論因果，也不能作為投餌、加藥或增氧指令的依據。

## 概念與推導
設$u_1,\dots,u_k\in\mathbb R^n$。若對所有$i\ne j$都有$u_i^\top u_j=0$，稱這組向量正交；若再加上$\|u_i\|=1$，則稱為正交歸一組。正交組必線性獨立：若$\sum c_i u_i=0$，兩邊與$u_j$取內積得$c_j\|u_j\|^2=0$，故$c_j=0$。這個性質是正交基底好用的根源——每個基底方向貢獻互不干擾。

投影的幾何直覺是「垂直落下」。若$q_1,\dots,q_k$是正交歸一組，子空間$S=\operatorname{span}\{q_i\}$，則任意$y\in\mathbb R^n$在$S$的投影為
$$
\operatorname{proj}_S(y)=\sum_{i=1}^k (q_i^\top y)\,q_i .
$$
係數$q_i^\top y$就是$y$沿$q_i$方向的分量。為什麼這是最佳近似？因為對任意$s\in S$，
$$
\|y-s\|^2=\|y-\hat y\|^2+\|\hat y-s\|^2,
$$
其中$\hat y=\operatorname{proj}_S(y)$，且$y-\hat y$與整個$S$正交。右邊第一項固定，第二項只能大於等於零，故$s=\hat y$時距離最小，且此$\hat y$唯一。這說明投影誤差下界只對該子空間內所有向量成立；若真實關係不在子空間中，換更小子空間或更大子空間都是模型選擇問題，本章無法單獨回答。

Gram–Schmidt把線性獨立向量$a_1,\dots,a_k$逐步正交化。先取$u_1=a_1$；對每個$j\ge2$，先減去已經在$u_1,\dots,u_{j-1}$上的分量：
$$
u_j=a_j-\sum_{i=1}^{j-1}\frac{q_i^\top a_j}{q_i^\top q_i}u_i,
$$
再令$q_j=u_j/\|u_j\|$。若分母用$q_i^\top q_i$，即使$q_i$只正交未歸一也正確。把$A=[a_1\ \cdots\ a_d]\in\mathbb R^{n\times d}$逐欄做此過程，就得到$A=QR$，其中$Q\in\mathbb R^{n\times d}$的columns正交歸一，$R\in\mathbb R^{d\times d}$是上三角且對角為正。$R$記錄每一步的歸一與投影係數，因此QR把原始欄轉成「方向清楚」的正交資訊[R1]。

最小平方問題：給$A\in\mathbb R^{n\times d}$與$y\in\mathbb R^n$，求$x\in\mathbb R^d$使$\|Ax-y\|_2^2$最小。假設$y$不在column space中；最佳近似$\hat y=A\hat x$必是$y$在該子空間的投影，因此殘差$r=y-A\hat x$要與$A$的每個column正交，即$A^\top r=0$。展開得正規方程
$$
A^\top A\,\hat x=A^\top y .
$$
若$A$的columns線性獨立，$A^\top A$對稱正定因而可逆，理論上$\hat x=(A^\top A)^{-1}A^\top y$；但這是表示式，不是首選演算法。原因是條件數會被平方：$\kappa(A^\top A)=\kappa(A)^2$，當特徵高度相關或尺度差很大時，直接求逆會大幅放大浮點誤差與輸入擾動。較穩健的做法是先做QR，再解上三角系統$R\hat x=Q^\top y$；$Q$是正交矩陣，不放大二範數。也可用SVD或NumPy的`lstsq`，這些方法不依賴顯式求逆[R1,R2]。若columns線性相依，$\hat x$不唯一，必須移除冗餘特徵、加正則化或改用偽逆，並報告自由度。

## 手算例題
池A三日模擬量測：$t=(0,1,2)$，$y=(1,2,2)$。模型$\hat y=a+bt$，所以設計矩陣與未知數為
$$
A=\begin{bmatrix}1&0\\1&1\\1&2\end{bmatrix},\quad
x=\begin{bmatrix}a\\b\end{bmatrix}.
$$
第一欄是常數項$(1,1,1)$，第二欄是時間$(0,1,2)$。先計算兩個乘積：
$$
A^\top A=\begin{bmatrix}3&3\\3&5\end{bmatrix},\quad
A^\top y=\begin{bmatrix}1+2+2\\0+2+4\end{bmatrix}=\begin{bmatrix}5\\6\end{bmatrix}.
$$
正規方程為
$$
\begin{cases}3a+3b=5,\\3a+5b=6.\end{cases}
$$
兩式相減得$2b=1$，故$b=1/2$；代回第一式$3a+3/2=5$，$3a=7/2$，$a=7/6$。預測為
$$
\hat y=\left(\frac76,\frac{10}{6},\frac{13}{6}\right).
$$
殘差
$$
y-\hat y=\left(-\frac16,\frac26,-\frac16\right),
$$
殘差平方和為$1/36+4/36+1/36=6/36=1/6$。驗算$A^\top r=0$：第一分量$(-1+2-1)/6=0$；第二分量$(0\cdot(-1)+1\cdot2+2\cdot(-1))/6=0$，符合正規方程的要求。

用Gram–Schmidt獨立核對：取$a_1=(1,1,1)$，$a_2=(0,1,2)$。因為$a_1$已是正交方向，令$u_1=a_1$。$a_2$在$u_1$上的投影係數是$(u_1^\top a_2)/(u_1^\top u_1)=3/3=1$，所以$u_2=a_2-u_1=(-1,0,1)$。$y$在$\operatorname{span}\{u_1,u_2\}$的投影為
$$
\frac{y^\top u_1}{u_1^\top u_1}u_1+\frac{y^\top u_2}{u_2^\top u_2}u_2
=\frac53(1,1,1)+\frac12(-1,0,1)
=\left(\frac76,\frac{10}{6},\frac{13}{6}\right),
$$
與正規方程結果一致。這也示範正交的好處：兩個係數$5/3$與$1/2$各自獨立計算，不必解聯立方程。

## Python實驗
以下片段在隔離環境以NumPy執行，不需安裝其他套件；輸入固定為上述三筆資料，輸出可人工核對。程式不以顯式求逆作為主要數值方法。
```python
import numpy as np

t = np.array([0.0, 1.0, 2.0])
y = np.array([1.0, 2.0, 2.0])
A = np.column_stack([np.ones_like(t), t])

coef, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
pred = A @ coef
r = y - pred

print("coef =", np.round(coef, 4))
print("rank =", rank)
print("pred =", np.round(pred, 4))
print("SSE =", np.round(r @ r, 4))
print("norm(r) =", np.round(np.linalg.norm(r), 4))
```
預期輸出：
```
coef = [1.1667 0.5   ]
rank = 2
pred = [1.1667 1.6667 2.8333]
SSE = 0.1667
norm(r) = 0.4082
```
其中$1.1667\approx7/6$，$1.6667\approx10/6$，$2.8333\approx13/6$，SSE$=1/6$，$\|r\|=\sqrt{1/6}\approx0.4082$。`lstsq`回傳最小平方解，內部採用穩定分解。`residuals`只有在特定條件下才給出殘差平方和，故本實驗直接計算$r^\top r$較清楚，也避免對回傳值語意誤解。

## 連回生成式AI、多模態與Agent
正交與投影是生成式AI的骨架之一。詞嵌入或影像特徵都可看成向量，檢索時以內積或餘弦比較方向；注意力機制中的線性投影把輸入映到$Q,K,V$子空間，其中$Q,K\in\mathbb R^{n\times d_k}$、$V\in\mathbb R^{n\times d_v}$，softmax逐列形成加權平均[R3]。多模態對齊常把不同模態的特徵投影到共同空間，再以相似度配對，例如影像與文字的對比學習[R4]。低秩近似、LoRA與PCA都依賴SVD或QR這種由正交概念建立的分解[R5]。

在池A的Agent情境中，可以把當下感測向量投影到訓練時建立的表示，用於唯讀檢索SOP與提出工具計畫，並把檢索來源與信心程度一併呈現給人類[R8]。但要強調三點限制：投影相似不等於因果，也不等於健康或病害診斷；線性探針只是近似；Agent不應自動下投餌、加藥或增氧指令，必須由現場專業人員覆核。線性代數提供骨架，非線性、統計、因果推論與控制不可被省略或假裝已證明。

## 常見錯誤與限制
第一，看到$(A^\top A)^{-1}A^\top y$就以為必須實際求逆；應優先QR或SVD，並檢查秩與條件數。第二，忘記最小平方的關鍵條件是殘差與column space正交，卻用平均絕對誤差代替，導致與投影不一致。第三，特徵單位不同或高度相關時$A^\top A$可能病態，係數對少量擾動極敏感。第四，三筆資料的直線不提供區間估計；本章未引入機率模型，不能說「統計顯著」。第五，投影只是相對於選定子空間與基底的最佳近似，換基底或換特徵都會改變結果。第六，任何擬合都不能代替養殖專業與現場安全程序。第七，把訓練殘差小當成預測好；三點配兩參數本來就容易貼合，這不代表新時段可靠。第八，忽略欄的尺度差異：某特徵數值範圍遠大於另一個時，條件數會更差，宜先標準化，且標準化參數只能由訓練集估計，驗證與測試集只套用，否則會資料洩漏。第九，不要用單一固定seed當成統計可靠性保證。第十，若$A$的columns線性相依，解不唯一，必須明說自由度，不能任意挑一組解就宣稱是唯一模型。

## 習題
1. 基本計算：對$t=(0,1,2)$、$y=(2,2,4)$，求最小平方直線$\hat y=a+bt$，並算殘差平方和。
2. 觀念：給$A\in\mathbb R^{5\times3}$且columns線性獨立，說明$A^\top A$為何可逆，並解釋為何用QR解最小平方通常比直接求$(A^\top A)^{-1}$穩定。
3. 養殖應用：池A量測$t=(0,1,2)$的溶氧$y=(1.0,2.0,2.0)$。若第三筆因校正改為$2.2$，重算$\hat y$的斜率，並說明此變化為何不能直接當成真實池水惡化或改善的證據。

## 習題解答
1. 先算$A^\top A=\begin{bmatrix}3&3\\3&5\end{bmatrix}$，$A^\top y=\begin{bmatrix}2+2+4\\0+2+8\end{bmatrix}=\begin{bmatrix}8\\10\end{bmatrix}$。解$3a+3b=8$、$3a+5b=10$，相減得$2b=2$，所以$b=1$；代回$3a+3=8$，$a=5/3$。預測$\hat y=(5/3,8/3,11/3)$。殘差$=(2-5/3,\ 2-8/3,\ 4-11/3)=(1/3,-2/3,1/3)$，平方和$=1/9+4/9+1/9=6/9=2/3$。可驗算$A^\top r=0$。
2. 若$A$的columns線性獨立，則$Ax=0$只有$x=0$。對任意$z\ne0$，$z^\top A^\top A z=\|Az\|^2>0$，故$A^\top A$對稱正定，因而可逆。直接求逆會把條件數平方，把浮點誤差與輸入擾動放大；QR先做正交變換，$Q$不放大二範數，解上三角系統$R\hat x=Q^\top y$通常更穩健，且可利用$A$是否滿秩來檢查病態。
3. 新資料$y=(1.0,2.0,2.2)$，$A^\top y=(5.2,\ 0+2.0+4.4)=(5.2,6.4)$。方程$3a+3b=5.2$、$3a+5b=6.4$，相減得$2b=1.2$，所以$b=0.6$。原斜率$0.5$，現為$0.6$。這只是三筆合成量測與模型設定的變化：未考慮感測誤差、時序相關、其他水質變數與因果，也不能分辨是校正修正或真實變化，故不能宣稱池水真實惡化或改善，更不能據此自動調整設備。

## 本章小結
正交基底把投影化為係數相加，使每個方向互不干擾；Gram–Schmidt提供由線性獨立向量建立正交基底的步驟，並導出QR分解；最小平方把擬合寫成投影，等價於正規方程$A^\top A\hat x=A^\top y$。但計算時應避免顯式求逆，改用穩定分解，並檢查秩、條件數與殘差是否與column space正交。這些工具往後支撐PCA、正則化、注意力與Agent檢索，但都只是在給定假設下的線性近似，不能取代統計、因果推論、控制與養殖專業判斷。

## 參考來源
[R1] NumPy linear algebra reference。 [R2] Deep Learning, Chapter 2: Linear Algebra。 [R3] Attention Is All You Need。 [R4] Learning Transferable Visual Models From Natural Language Supervision。 [R5] LoRA: Low-Rank Adaptation of Large Language Models。 [R8] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks。