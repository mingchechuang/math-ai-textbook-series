# 第06章 正交、投影與最小平方

## 學習目標與先備知識
本章目標：理解正交、正交基底與投影；用Gram–Schmidt建構正交基底；以QR與最小平方解資料擬合；知道正規方程的數值限制。先備：向量內積、矩陣乘法、線性組合、轉置。全程預設向量為 column vector，$x\in\mathbb R^d$，$W\in\mathbb R^{m\times d}$，$y=Wx\in\mathbb R^m$；批次$X\in\mathbb R^{n\times d}$每列（row）一筆觀測。只用高中數學與基本Python即可。

## 從池A提出問題
虛構池A在三日同一時段記錄溶氧$y$（mg/L）與時間$t$（日）。資料為教學合成，不是真實閾值，也不代表任何物種安全範圍。資料字典：$t$為感測器時間，$y$為溶氧量測，$A$為設計矩陣。因為量測有噪聲，三點不共線，我們想找一條直線$\hat y=a+bt$使殘差平方和最小。線性代數把「找最接近的向量」寫成投影問題：$y\in\mathbb R^3$投影到$A$的各column所張成的column space。此擬合只描述這三筆模擬資料，不能推論因果，也不能作為投餌、加藥或增氧依據。

## 概念與推導
設$u_1,\dots,u_k\in\mathbb R^n$。若$i\ne j$時$u_i^\top u_j=0$，稱為正交組；再要求$\|u_i\|=1$則為正交歸一組。正交組必線性獨立。若$q_1,\dots,q_k$正交歸一，向量$y$在子空間$S=\operatorname{span}\{q_i\}$的投影為
$$
\operatorname{proj}_S(y)=\sum_{i=1}^k (q_i^\top y)q_i,
$$
且$y-\operatorname{proj}_S(y)$與每個$q_i$正交。這是「最近點」的條件：對任意$s\in S$，$\|y-s\|^2=\|y-\hat y\|^2+\|\hat y-s\|^2$，故$\hat y$唯一使距離最小。投影的誤差下界只在該子空間內成立；若真實關係不在子空間中，增加維度可降低訓練殘差，但測試誤差未必下降，這是模型選擇問題，不是本章能單獨回答。

Gram–Schmidt把線性獨立向量$a_1,\dots,a_k$逐步正交化：
$$
u_1=a_1,\quad
u_j=a_j-\sum_{i=1}^{j-1}\frac{q_i^\top a_j}{q_i^\top q_i}u_i,
$$
再令$q_j=u_j/\|u_j\|$。若把$A=[a_1\ \cdots\ a_d]\in\mathbb R^{n\times d}$，此過程給$A=QR$，$Q$的columns正交歸一，$R$為上三角。QR概念讓我們不必顯式形成$A^\top A$，數值上通常更穩健[R1]。若$q_i$只是正交但未歸一，分母為$\|q_i\|^2$；正交的好處是係數互不影響，要加一個新基底向量，不必重解全部係數。

最小平方問題：給$A\in\mathbb R^{n\times d}$、$y\in\mathbb R^n$，求$x\in\mathbb R^d$最小化$\|Ax-y\|_2^2$。當$y$不在$A$的各column所張成的column space中，最佳解是$y$在該子空間的投影$\hat y=A\hat x$，且殘差$r=y-A\hat x$滿足$A^\top r=0$。展開得正規方程
$$
A^\top A\,\hat x=A^\top y.
$$
若$A$的columns線性獨立，$A^\top A$可逆，$\hat x=(A^\top A)^{-1}A^\top y$；但這是理論表示，不是首選演算法。原因：$\kappa(A^\top A)=\kappa(A)^2$，當$A$病態或感測特徵高度相關，直接求逆會放大誤差。實務可用QR解$R\hat x=Q^\top y$，或用SVD；NumPy的`lstsq`即採這類穩定做法[R1,R2]。若columns線性相依，$\hat x$不唯一，需移除冗餘特徵、加正則化或改用偽逆，並報告自由度。

## 手算例題
池A三日模擬量測：$t=(0,1,2)$，$y=(1,2,2)$。模型$\hat y=a+bt$，所以
$$
A=\begin{bmatrix}1&0\\1&1\\1&2\end{bmatrix},\quad
x=\begin{bmatrix}a\\b\end{bmatrix}.
$$
先算
$$
A^\top A=\begin{bmatrix}3&3\\3&5\end{bmatrix},\quad
A^\top y=\begin{bmatrix}5\\6\end{bmatrix}.
$$
正規方程為
$$
\begin{cases}3a+3b=5,\\3a+5b=6.\end{cases}
$$
兩式相減得$2b=1$，故$b=1/2$；代回$3a+3/2=5$，$3a=7/2$，$a=7/6$。因此$\hat y$在$t=0,1,2$的預測為$(7/6,10/6,13/6)$，殘差
$$
y-\hat y=\left(-\frac16,\frac26,-\frac16\right),
$$
殘差平方和$=1/36+4/36+1/36=6/36=1/6$。驗算$A^\top r=0$：$(-1/6+2/6-1/6)=0$，$0\cdot(-1/6)+1\cdot(2/6)+2\cdot(-1/6)=0$。

以Gram–Schmidt核對：$a_1=(1,1,1)$，$a_2=(0,1,2)$。取$u_1=a_1$。$a_2$在$u_1$的投影係數$(u_1^\top a_2)/(u_1^\top u_1)=3/3=1$，故$u_2=a_2-u_1=(-1,0,1)$。$y$在$\operatorname{span}\{u_1,u_2\}$的投影為
$$
\frac{y^\top u_1}{u_1^\top u_1}u_1+\frac{y^\top u_2}{u_2^\top u_2}u_2
=\frac53(1,1,1)+\frac12(-1,0,1)
=\left(\frac76,\frac{10}{6},\frac{13}{6}\right),
$$
與正規方程一致。

## Python實驗
以下片段在隔離環境用NumPy執行；不需要安裝其他套件。輸入固定三筆資料，輸出可人工核對。
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
`lstsq`不以顯式求逆為主要方法；它回傳最小平方解。`residuals`只有在特定條件下才給殘差平方和，故本例直接算$r^\top r$較清楚。

## 連回生成式AI、多模態與Agent
正交與投影是生成式AI的骨架之一：詞嵌入或影像特徵可視為向量，檢索用內積或餘弦比較方向；注意力中的線性投影把$Q,K,V$映到子空間，softmax逐列形成加權平均[R3]。多模態對齊常把不同模態特徵投影到共同空間，再以相似度配對[R4]。最小平方出現在線性探針、校正與低秩近似；QR與SVD則支撐PCA、LoRA與數值穩定[R5]。在池A的Agent情境，可把感測向量投影到訓練時建立的表示，用於唯讀檢索SOP與提出工具計畫；但投影相似不等於因果或健康結論，Agent不應自動下投餌、加藥或增氧指令，必須人工覆核。

## 常見錯誤與限制
第一，把正規方程的解寫成$(A^\top A)^{-1}A^\top y$就以為必須求逆；實際上應優先QR或SVD，並檢查秩與條件數。第二，忘記殘差與column space正交，卻用「平均誤差」代替平方和。第三，特徵單位不同或高度相關時，$A^\top A$可能病態，$a,b$對少量擾動極敏感。第四，三筆資料的直線不提供區間估計；本章未引入機率模型，不能說統計顯著。第五，投影到某子空間只是相對於選定基底的最佳近似，換特徵或換基底會改變結果。第六，任何擬合都不能代替養殖專業與現場安全程序。第七，把訓練集殘差小當成預測好；三點擬合直線幾乎一定通過附近，但這不代表新時段可靠。第八，忽略欄的尺度差異：若一個特徵數值範圍遠大於另一個，$A^\top A$的條件數會更差，宜先標準化，且標準化參數只能由訓練集估計。

## 習題
1. 基本計算：對$t=(0,1,2)$、$y=(2,2,4)$，求最小平方直線$\hat y=a+bt$，並算殘差平方和。
2. 觀念：給$A\in\mathbb R^{5\times3}$且columns線性獨立，說明$A^\top A$為何可逆，並解釋為何用QR解最小平方通常比直接求$(A^\top A)^{-1}$穩定。
3. 養殖應用：池A量測$t=(0,1,2)$的溶氧$y=(1.0,2.0,2.0)$。若第三筆因校正改為$2.2$，重算$\hat y$的斜率，並說明此變化為何不能直接當成真實池水惡化或改善的證據。

## 習題解答
1. $A^\top A=\begin{bmatrix}3&3\\3&5\end{bmatrix}$，$A^\top y=\begin{bmatrix}8\\10\end{bmatrix}$。解$3a+3b=8$、$3a+5b=10$，相減$2b=2$，$b=1$，$a=5/3$。$\hat y=(5/3,8/3,11/3)$。殘差$=(2-5/3,2-8/3,4-11/3)=(1/3,-2/3,1/3)$，平方和$=1/9+4/9+1/9=6/9=2/3$。
2. 若$A$的columns線性獨立，$Ax=0$只有$x=0$。對任意$z\ne0$，$z^\top A^\top A z=\|Az\|^2>0$，故$A^\top A$對稱正定，因而可逆。直接求逆會把條件數平方，放大浮點誤差；QR先做正交變換，$Q$不放大二範數，解$R\hat x=Q^\top y$通常更穩健。
3. 新資料$y=(1.0,2.0,2.2)$，$A^\top y=(5.2,6.4)$。方程$3a+3b=5.2$、$3a+5b=6.4$，相減$2b=1.2$，$b=0.6$。原斜率$0.5$，變為$0.6$。這只是三筆合成量測與模型設定的變化；未考慮感測誤差、時序相關、其他水質變數與因果，不能宣稱池水真實惡化或改善。

## 本章小結
正交基底把投影變成係數相加；Gram–Schmidt給出QR；最小平方把擬合化為$A^\top A x=A^\top y$，但計算時應避免顯式求逆，改用穩定分解並檢查秩、條件數與殘差。這些工具往後支撐PCA、正則化、注意力與Agent檢索，但都只是在給定假設下的線性近似，不能取代統計、因果、控制與養殖專業。

## 參考來源
[R1] NumPy linear algebra reference。 [R2] Deep Learning, Chapter 2: Linear Algebra。 [R3] Attention Is All You Need。 [R4] Learning Transferable Visual Models From Natural Language Supervision。 [R5] LoRA: Low-Rank Adaptation of Large Language Models。