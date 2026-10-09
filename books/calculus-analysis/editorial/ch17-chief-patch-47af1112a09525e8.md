<<<PATCH 01>>>
<<<OLD>>>
### 邊界測試：正半定但不正定

取零條約束，並令

$$
Q=
\begin{bmatrix}
1&0\\
0&0
\end{bmatrix}.
$$

受限 Hessian 特徵值為 $0,1$。預期分類為「正半定但不正定」。它與局部最小的二階必要條件相容，也因存在正方向而排除局部最大，但不能保證嚴格局部最小。
<<<NEW>>>
### 邊界測試：正半定但不正定

取零條約束，即 $m=0$，並令

$$
Q=
\begin{bmatrix}
1&0\\
0&0
\end{bmatrix},
\qquad
q=
\begin{bmatrix}
0\\
0
\end{bmatrix}.
$$

此時 $A$ 為 $0\times2$ 矩陣，$b$ 為 $0\times1$ 空向量。KKT 矩陣退化，因此程式應改用最小二乘（或 Moore–Penrose 偽逆）求解，並在結果中標示 KKT 矩陣奇異、解非唯一。取最小範數解時得到 $x=(0,0)^T$。

受限 Hessian 特徵值為 $0,1$。預期分類為「正半定但不正定」。它與局部最小的二階必要條件相容，也因存在正方向而排除局部最大，但不能保證嚴格局部最小。非唯一解不影響約化 Hessian 的特徵值；原點只是其中一個駐點。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    try:
        solution = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("KKT matrix is singular") from exc

    x = solution[:n]
    lam = solution[n:]

    feasibility = A @ x - b
    stationarity = Q @ x + q + A.T @ lam
    reduced = Z.T @ Q @ Z
    eigvals = (
        np.linalg.eigvalsh(reduced)
        if reduced.size
        else np.array([], dtype=float)
    )
    classification = classify_reduced_hessian(eigvals, tol)

    result = {
        "x": x,
        "lambda": lam,
        "rank_A": rank,
        "Z": Z,
        "reduced_hessian": reduced,
        "reduced_eigenvalues": eigvals,
        "feasibility_norm":
            float(np.linalg.norm(feasibility, 2)),
        "stationarity_norm":
            float(np.linalg.norm(stationarity, 2)),
    }
    result.update(classification)
    return result
<<<NEW>>>
    try:
        solution = np.linalg.solve(K, rhs)
        kkt_singular = False
        kkt_rank = n + m
    except np.linalg.LinAlgError:
        solution, _, kkt_rank, _ = np.linalg.lstsq(
            K, rhs, rcond=None
        )
        kkt_singular = True
    kkt_residual = K @ solution - rhs

    x = solution[:n]
    lam = solution[n:]

    feasibility = A @ x - b
    stationarity = Q @ x + q + A.T @ lam
    reduced = Z.T @ Q @ Z
    eigvals = (
        np.linalg.eigvalsh(reduced)
        if reduced.size
        else np.array([], dtype=float)
    )
    classification = classify_reduced_hessian(eigvals, tol)

    result = {
        "x": x,
        "lambda": lam,
        "rank_A": rank,
        "Z": Z,
        "reduced_hessian": reduced,
        "reduced_eigenvalues": eigvals,
        "feasibility_norm":
            float(np.linalg.norm(feasibility, 2)),
        "stationarity_norm":
            float(np.linalg.norm(stationarity, 2)),
        "kkt_singular":
            kkt_singular,
        "kkt_rank":
            kkt_rank,
        "kkt_residual_norm":
            float(np.linalg.norm(kkt_residual, 2)),
    }
    result.update(classification)
    return result
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
因 $\operatorname{rank}A=1<2$，預期程式拋出 `ValueError`，指出 LICQ 失敗。可行集合本身仍可能良好，但乘數不唯一，KKT 矩陣也可能奇異。
<<<NEW>>>
因 $\operatorname{rank}A=1<2$，預期程式拋出 `ValueError`，指出 LICQ 失敗。可行集合本身仍可能良好，但乘數不唯一，KKT 矩陣也可能奇異。若移除秩檢查，最小二乘分支會給出 `kkt_singular=True` 與非唯一的乘數；目前實作改以明確的 LICQ 錯誤先行拒絕，避免誤報。
<<<END>>>