<<<PATCH 01>>>
<<<OLD>>>
    返回: 字串 "positive definite", "negative definite", "semi-positive definite", "semi-negative definite", "indefinite", "error: ..."
<<<NEW>>>
    返回: 字串 "positive definite", "negative definite", "indefinite (saddle point)"、
    "numerically indeterminate (...)" 或 "error: ..."。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
3.  **故障/退化情況（Degenerate）**：
    *   函數：$f(x, y) = x^4 + y^4$。
    *   駐點：$(0,0)$。
    *   Hessian：$\\begin{pmatrix} 0 & 0 \\\\ 0 & 0 \\end{pmatrix}$。
    *   特徵值：$0, 0$。
    *   預期輸出：`Indeterminate (Hessian semi-positive definite (degenerate)), need higher-order analysis`。
    *   *注意*：雖然此處實際是極小值，但僅憑 Hessian 無法得出此結論，程式應正確標記為不確定。
    *   反例函數：$f(x, y) = x^4 - y^4$。
    *   Hessian：$\\begin{pmatrix} 0 & 0 \\\\ 0 & 0 \\end{pmatrix}$。
    *   預期輸出：同上。
    *   *結論*：相同 Hessian，不同高階項，導致不同極值型態。程式行為一致，符合理論預期。
<<<NEW>>>
3.  **退化邊界情況**：
    *   函數：$f(x, y) = x^4 + y^4$。
    *   駐點：$(0,0)$。
    *   Hessian：$\\begin{pmatrix} 0 & 0 \\\\ 0 & 0 \\end{pmatrix}$。
    *   特徵值：$0, 0$。
    *   預期輸出：`Indeterminate (Hessian numerically indeterminate (eigenvalue near zero)), need higher-order analysis`。
    *   *注意*：雖然此處實際是極小值，但僅憑 Hessian 無法得出此結論，程式應正確標記為不確定。
    *   反例函數：$f(x, y) = x^4 - y^4$。
    *   Hessian：$\\begin{pmatrix} 0 & 0 \\\\ 0 & 0 \\end{pmatrix}$。
    *   預期輸出：同上。
    *   *結論*：相同 Hessian，不同高階項，導致不同極值型態。程式行為一致，符合理論預期。

4.  **故障情況：非對稱Hessian**：
    *   輸入：$\\begin{pmatrix} 1 & 2 \\\\ 0 & 1 \\end{pmatrix}$。
    *   預期輸出：`error: Hessian is not symmetric` 開頭的訊息。
    *   此測試檢查程式拒絕分析非對稱輸入；非對稱本身不是合法的實值 $C^2$ 函數Hessian。
<<<END>>>