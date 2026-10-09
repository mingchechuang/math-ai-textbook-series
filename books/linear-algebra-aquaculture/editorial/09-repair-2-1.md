## 複審結果

本次草稿與上一版實質相同，先前兩項維度問題仍未修正。手算部分則已核對：\(S_c=\frac13\begin{pmatrix}5&4\\4&5\end{pmatrix}\) 的特徵值為 \(3,1/3\)，解釋比例為 \(90\%\)；習題1的 \(S=\begin{pmatrix}1&-1/2\\-1/2&1\end{pmatrix}\)、特徵值 \(3/2,1/2\) 及 \(75\%\) 均正確。

### 1. 中心化公式仍無法相乘

原文定義
\[
x_i,\bar x\in\mathbb R^{1\times d},
\]
卻寫
\[
Z=X-\mathbf1_n\bar x^T.
\]
此時 \(\mathbf1_n\) 若為 \(n\times1\)，而 \(\bar x^T\) 為 \(d\times1\)，兩者不能相乘；同時也違反全書單筆觀測採 column vector 的約定。應統一改為
\[
x_i=X_{i,:}^T\in\mathbb R^d,\qquad
\bar x=\frac1n\sum_i x_i\in\mathbb R^d,
\]
\[
Z=X-\mathbf1_n\bar x^T\in\mathbb R^{n\times d}.
\]
單筆測試式應為
\[
z=\Sigma_{\rm safe}^{-1}(x-\bar x),\qquad t=U_k^Tz.
\]

### 2. 零標準差時 \(\Sigma^{-1}\) 不存在

原文說零標準差以1替代，但仍令
\[
\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_d)
\]
並使用 \(\Sigma^{-1}\) 與 \(1/\sigma_j\)。若任一 \(\sigma_j=0\)，這些式子無定義。應明定
\[
\tilde\sigma_j=\begin{cases}\sigma_j,&\sigma_j>0\\1,&\sigma_j=0,\end{cases}
\quad
\Sigma_{\rm safe}=\operatorname{diag}(\tilde\sigma_1,\ldots,\tilde\sigma_d),
\]
並在全部標準化公式使用 \(\Sigma_{\rm safe}^{-1}\)。

### 3. 統計敘述與文字

「把 \(n\) 當分母，\(S\) 不再是無偏估計」不精確：文中只在指定假設下說明未標準化共變異數 \(S_c\) 無偏；樣本相關矩陣 \(S\) 一般不能據此宣稱無偏。應將此句限縮為 \(S_c\)。

另須把「量纲、必须、注明」改成「量綱、必須、註明」；「如Gaussian」應寫「如聯合Gaussian」。小結稱標準化為必要步驟，與原尺度PCA可合理使用的說明矛盾，應改為依分析目的決定。篇幅2550字合格；本審查未執行Python，實際數值輸出待人工驗證。安全限制與訓練／測試隔離已具備。

VERDICT: REVISE