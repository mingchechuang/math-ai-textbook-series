<<<PATCH 01>>>
<<<OLD>>>
預期分類為「正半定但不正定」。
<<<NEW>>>
若直接呼叫 `classify_reduced_hessian(np.array([0.0, 1.0]))`，預期分類為「正半定但不正定」；若呼叫 `solve_equality_qp`，則預期在分類前因奇異 KKT 矩陣拋出 `ValueError`，不能將分類結果當成完整求解的輸出。對滿足 LICQ 的線性等式約束問題，受限 Hessian 有零特徵值即意味 KKT 矩陣奇異，增加約束也無法同時保留該零特徵值及非奇異 KKT 矩陣。
<<<END>>>