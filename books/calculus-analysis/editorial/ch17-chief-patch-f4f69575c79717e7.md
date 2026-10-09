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

對這個精確二次目標，若 $q=0$，原點確實是非嚴格全域最小；這是利用完整二次模型得到的額外結論，不是一般半正定測試自動提供的嚴格性結論。
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

此時 $A$ 為 $0\times2$ 矩陣，$b$ 為 $0\times1$ 空向量。KKT 矩陣退化，因此程式應改用最小二乘（或 Moore–Penrose 偽逆）求解，並在結果中標示 KKT 矩陣奇異、解非唯一。取最小范數解時得到 $x=(0,0)^T$。

受限 Hessian 特徵值為 $0,1$。預期分類為「正半定但不正定」。它與局部最小的二階必要條件相容，也因存在正方向而排除局部最大，但不能保證嚴格局部最小。非唯一解不影響約化 Hessian 的特徵值；原點只是其中一個駐點。

對這個精確二次目標，若 $q=0$，原點確實是非嚴格全域最小；這是利用完整二次模型得到的額外結論，不是一般半正定測試自動提供的嚴格性結論。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    try:
        solution = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("KKT matrix is singular") from exc

    x = solution[:n]
    lam = solution[n:]
<<<NEW>>>
    try:
        solution = np.linalg.solve(K, rhs)
        kkt_singular = False
        kkt_rank = n + m
        kkt_residual = K @ solution - rhs
    except np.linalg.LinAlgError:
        solution, residuals, kkt_rank, s = np.linalg.lstsq(
            K, rhs, rcond=None
        )
        kkt_singular = True
        kkt_residual = K @ solution - rhs

    x = solution[:n]
    lam = solution[n:]
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
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
<<<NEW>>>
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
<<<END>>>