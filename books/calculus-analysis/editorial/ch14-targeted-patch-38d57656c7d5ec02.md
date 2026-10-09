<<<PATCH 14>>>
<<<OLD>>>
    返回: 字串 "positive definite", "negative definite", "semi-positive definite", "semi-negative definite", "indefinite", "error: ..."
<<<NEW>>>
    返回: 字串 "positive definite", "negative definite",
          "indefinite (saddle point)",
          "numerically indeterminate (opposite signs near zero)",
          "numerically indeterminate (eigenvalue near zero)"，或 "error: ..."
    近零分類只表示依指定容差無法可靠判別，不是精確半定性的證明。
<<<END>>>
<<<PATCH 14>>>
<<<OLD>>>
我們設計三組測試用例來驗證邏輯與程式行為。
<<<NEW>>>
以下列出正常、鞍點、退化邊界與故障四類預期測試。程式未經執行；列出的結果是依分支邏輯推算，不能代替實際執行紀錄或極值定理的證明。
<<<END>>>
<<<PATCH 14>>>
<<<OLD>>>
3.  **故障/退化情況（Degenerate）**：
    *   函數：$f(x, y) = x^4 + y^4$。
    *   駐點：$(0,0)$。
    *   Hessian：$\begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$。
    *   特徵值：$0, 0$。
    *   預期輸出：`Indeterminate (Hessian semi-positive definite (degenerate)), need higher-order analysis`。
    *   *注意*：雖然此處實際是極小值，但僅憑 Hessian 無法得出此結論，程式應正確標記為不確定。
    *   反例函數：$f(x, y) = x^4 - y^4$。
    *   Hessian：$\begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$。
    *   預期輸出：同上。
    *   *結論*：相同 Hessian，不同高階項，導致不同極值型態。程式行為一致，符合理論預期。
<<<NEW>>>
3.  **退化邊界測試（合法輸入）**：
    *   函數：$f(x, y) = x^4 + y^4$。
    *   駐點：$(0,0)$。
    *   Hessian：$\begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$。
    *   特徵值：$0, 0$。
    *   預期輸出：`Indeterminate (Hessian numerically indeterminate (eigenvalue near zero)), need higher-order analysis`。
    *   *注意*：雖然此處實際是極小值，但僅憑 Hessian 無法得出此結論。程式的「數值不確定」標籤也不表示零矩陣在數學上無法分類。
    *   對照函數：$f(x, y) = x^4 - y^4$。
    *   原點的Hessian同為$\begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$，預期輸出同上；然而沿兩條座標軸分別取得正負值，原點並非極值。相同Hessian無法決定高階項的效果。

4.  **故障測試（非對稱輸入）**：
    *   將 `np.array([[1.0, 2.0], [0.0, 1.0]])` 傳給 `classify_hessian`。它與自身轉置之差的範數大於預設容差，預期回傳以 `error: Hessian is not symmetric` 開頭的字串，而非極值分類；程式中的斷言檢查此前綴。
<<<END>>>