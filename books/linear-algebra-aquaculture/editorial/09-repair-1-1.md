## 複審結果

本稿的手算與習題數值已核對正確：非退化例的特徵值為 \(3,1/3\)、解釋變異為 \(90\%\)；習題1的特徵值為 \(3/2,1/2\)、第一方向 \((1,-1)/\sqrt2\)、比例 \(75\%\)。資料洩漏、時間切分、程式輸出未宣稱已執行、養殖安全與2550中文字篇幅也符合要求。然而，上輪兩項核心問題仍未修正。

### 1. 中心化公式維度不合法

「概念與推導」定義
\[
x_i,\bar x\in\mathbb R^{1\times d},
\]
因此 \(\bar x^T\in\mathbb R^{d\times1}\)。若 \(\mathbf1_n\in\mathbb R^{n\times1}\)，原式
\[
Z=X-\mathbf1_n\bar x^T
\]
中的乘積是 \((n\times1)(d\times1)\)，不能相乘，且違反全書單筆向量預設為 column vector 的契約。應改為
\[
x_i=X_{i,:}^T\in\mathbb R^d,\qquad
\bar x=\frac1n\sum_i x_i\in\mathbb R^d,
\]
\[
Z=X-\mathbf1_n\bar x^T\in\mathbb R^{n\times d}.
\]
單筆測試資料則直接寫
\[
x\in\mathbb R^d,\quad
z=\Sigma_{\rm safe}^{-1}(x-\bar x),\quad
t=U_k^Tz.
\]
目前把 \(1\times d\) 稱為「列向量」，亦與指定的 column vector 用語衝突。

### 2. 零標準差時仍錯用不存在的逆矩陣

正文雖說 \(\sigma_j=0\) 時以1替代，卻仍定義
\[
\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_d)
\]
並使用 \(\Sigma^{-1}=\operatorname{diag}(1/\sigma_j)\)。只要一欄標準差為零，該逆矩陣便不存在。應像程式一樣先定義
\[
\tilde\sigma_j=\begin{cases}\sigma_j,&\sigma_j>0\\1,&\sigma_j=0,\end{cases}
\quad
\Sigma_{\rm safe}=\operatorname{diag}(\tilde\sigma_j),
\]
其後一律使用 \(\Sigma_{\rm safe}^{-1}\)。

另應將「量纲」「必须」「注明」改為繁體「量綱」「必須」「註明」；「如Gaussian」宜精確寫成「如聯合Gaussian」。小結稱標準化為「必要步驟」也與前文允許原尺度PCA矛盾，應改成依分析目的選擇。本審查未執行Python，實際輸出仍待人工驗證。

VERDICT: REVISE