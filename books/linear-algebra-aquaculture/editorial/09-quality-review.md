## 複審結果

前次所列的尺度誤述、共變異數／相關矩陣PCA區別、手算、零共變異與獨立性、程式輸出及安全限制均已大致修正，但仍有兩項實質問題。

### 1. 向量方向與中心化公式仍互相矛盾

全書契約指定 \(x\in\mathbb R^d\) 為 column vector，但本章將
\[
x_i\in\mathbb R^{1\times d}
\]
稱為「列向量」，實際上這是 row vector，也容易與中文「列／行」混淆。更嚴重的是，既然目前定義 \(\bar x\in\mathbb R^{1\times d}\)，則
\[
\mathbf1_n\bar x^T
\]
是 \((n\times1)(d\times1)\)，乘法不成立。

建議完全遵守全書契約：定義
\[
x_i=X_{i,:}^T\in\mathbb R^d,\qquad
\bar x=\frac1n\sum_i x_i\in\mathbb R^d,
\]
再寫
\[
Z=X-\mathbf1_n\bar x^T\in\mathbb R^{n\times d}.
\]
單筆公式
\[
z=\Sigma_{\rm safe}^{-1}(x-\bar x),\qquad t=U_k^Tz
\]
便可一致成立。文字應直接使用「column vector／row」或「直行向量／橫列資料」，避免把 \(1\times d\) 稱為列向量。

### 2. 零標準差使 \(\Sigma^{-1}\) 未定義

前文先定義
\[
\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_d),
\]
若某個 \(\sigma_j=0\)，此矩陣不可逆；但後文仍直接寫 \(\Sigma^{-1}\) 及 \(1/\sigma_j\)，與「以1代替0」矛盾。應先定義
\[
\tilde\sigma_j=
\begin{cases}
\sigma_j,&\sigma_j>0,\\
1,&\sigma_j=0,
\end{cases}
\qquad
\Sigma_{\rm safe}=\operatorname{diag}(\tilde\sigma_1,\ldots,\tilde\sigma_d),
\]
之後所有標準化公式均使用 \(\Sigma_{\rm safe}^{-1}\)。程式已採此做法，正文需同步。

### 3. 其餘細節

- 習題3的例外應寫「**聯合** Gaussian」；僅各變數邊際為Gaussian仍不足以由零共變異推出獨立。
- 「量纲」「必须」為簡體字，應改為「量綱」「必須」。
- 小結稱標準化為「必要步驟」過強，因正文已說共變異數PCA也可能合理；應改為「當單位與尺度差異不具分析意義時，通常需要標準化」。
- 未提供腳本字數檢查，2200～2600中文字仍待人工驗證；本次亦未執行Python，程式數值輸出仍屬待核對。

VERDICT: REVISE