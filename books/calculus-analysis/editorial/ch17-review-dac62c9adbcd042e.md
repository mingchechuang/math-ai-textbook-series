前次指出的程式與測試矛盾仍在。

原句：「取零條約束，並令 $Q=\begin{bmatrix}1&0\\0&0\end{bmatrix}$。受限 Hessian 特徵值為 $0,1$。預期分類為『正半定但不正定』。」

原因：`solve_equality_qp` 先呼叫 `np.linalg.solve(K, rhs)`，之後才分類受限 Hessian。零條約束且 $Q$ 如上時，KKT 矩陣奇異；取文中所述 $q=0$ 也不改變此事，程式會先報錯，無法產生預期分類。

最小修法：將測試改為滿足 LICQ、且 KKT 矩陣非奇異的約束問題，使約化 Hessian具有特徵值 $0,1$；或修改求解流程，明確處理奇異 KKT 系統與駐點相容性。

VERDICT: REVISE