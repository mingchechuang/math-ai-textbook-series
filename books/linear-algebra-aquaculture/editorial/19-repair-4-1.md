## 複審結果

本版章節內容完整，先前問題均已妥善修正。術語依共同約定使用「列向量（column vector，直向）」，且符號與shape一致：
\[
\mathbf x,\boldsymbol\mu,\mathbf r\in\mathbb R^{d\times1},\quad
\boldsymbol\delta\in\mathbb R^{m\times1},\quad
A\in\mathbb R^{k\times m},\quad
\mathbf b\in\mathbb R^{k\times1}.
\]
因此 \(A\boldsymbol\delta-\mathbf b\in\mathbb R^{k\times1}\) 合法。

手工核對數值如下：
\[
\det(\Sigma)=0.64,\qquad
\Sigma^{-1}=
\begin{pmatrix}
0.390625&0.9375\\
0.9375&6.25
\end{pmatrix},
\]
\[
\Sigma^{-1}\mathbf r=(-0.859375,-14.0625)^\top,\qquad
d_M^2=37.890625.
\]
限制例的餘量為 \((1,-5,-2,-4,0)^\top\)，僅第一條越界；習題一的 \(d_M^2=29\) 與第四條限制違反亦正確。

程式未實際執行，但依流程與shape核對，兩次Cholesky求解等價於解 \(\Sigma z=\mathbf r\)，而 `r.T @ Sigma_inv_r` 產生 `(1,1)`，可取得 \(37.8906\)。預期拒絕原因與違反列 `[1]` 亦和程式一致。輸入shape、非有限值、非法時間值、負時效窗口、未來或過期時間都有先行拒絕；可行時也只輸出 `review_required`，不回傳可執行操作。正文對逆平方根、Cholesky白化與 \(\Sigma^{-1}\) 二次型的區分合理。

可逆條件、滿秩必要條件、偽逆限制與離散互斥限制的邊界均交代清楚。三類習題及答案完整，且異常分數未被當成病害診斷。全章不連接設備、不提供真實閾值，人工批准與現場專業責任明確。R1、R2及R7、R8的用途沒有越界；來源清單亦未被宣稱為養殖安全實證。編輯計數2481字，符合至少2200字的現行政策。

VERDICT: APPROVE