<<<PATCH 01>>>
<<<OLD>>>
定義母體固定分箱 ECE 為：
$$ \\text{ECE}_{\\text{true, binned}} = \\sum_{b=1}^{B} q_b \\left| E[A - S \\mid S \\in I_b] \\right| $$
其中 $q_b = P(S \\in I_b)$ 是樣本落入箱 $b$ 的母體機率。
<<<NEW>>>
令 $q_b=P(S\\in I_b)$。對 $q_b>0$ 的箱，定義 $\\delta_b=E[A-S\\mid S\\in I_b]$；對 $q_b=0$ 的箱，母體貢獻定為零。母體固定分箱 ECE 定義為：
$$ \\text{ECE}_{\\text{true, binned}} = \\sum_{b:q_b>0} q_b |\\delta_b| $$
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
對於第 $b$ 個箱，考慮條件 $N_b = n > 0$。根據詹森不等式（Jensen's Inequality），對於凸函數 $f(x) = |x|$：
$$ E[ |X_b| \\mid N_b = n ] \\geq | E[ X_b \\mid N_b = n ] | $$

在 i.i.d. 假設下，給定 $N_b=n$，箱內的 $(A_i,S_i)$ 服從以 $S_i\\in I_b$ 為條件的分布。因此：
$$ E[ X_b \\mid N_b = n ] = E[ \\text{Acc}_b - \\bar{p}_b \\mid N_b = n ] = E[ A - S \\mid S \\in I_b ] $$
令 $\\delta_b = E[ A - S \\mid S \\in I_b ]$，則：
$$ E[ |X_b| \\mid N_b = n ] \\geq | \\delta_b | $$

令箱貢獻 $Y_b$ 在 $N_b=0$ 時為 $0$，在 $N_b>0$ 時為 $\\frac{N_b}{N}|X_b|$。這使 $Y_b$ 在整個樣本空間均有定義，不必計算空箱中未定義的 $X_b$。只對 $n=1,\\ldots,N$ 求和：
<<<NEW>>>
若 $q_b=0$，則 $N_b=0$ 幾乎處處，因此 $E[Y_b]=0$。若 $q_b>0$，給定 $N_b=n>0$，箱內樣本對 $(A_i,S_i)$ 服從條件分布 $P((A,S)\\mid S\\in I_b)$。由 Jensen 不等式：
$$ E[|X_b|\\mid N_b=n]\\geq |E[X_b\\mid N_b=n]|. $$
而
$$ E[X_b\\mid N_b=n]=E[\\text{Acc}_b-\\bar p_b\\mid N_b=n]=E[A-S\\mid S\\in I_b]=\\delta_b. $$

為明確處理有限樣本空箱，定義
$$
Y_b=
\\begin{cases}
\\frac{N_b}{N}|\\text{Acc}_b-\\bar p_b|,&N_b>0,\\\\
0,&N_b=0.
\\end{cases}
$$
因此在 $q_b>0$ 時，只對 $n=1,\\ldots,N$ 求和：
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
**Selective Risk $R(\\tau)$**：在同意回答的樣本中，預測錯誤的比例。
    $$ R(\\tau) = \\frac{\\sum_{i: s(x_i) \\ge \\tau} \\mathbb{I}(\\hat{y}_i \\neq y_i)}{\\sum_{i: s(x_i) \\ge \\tau} \\mathbb{I}(s(x_i) \\ge \\tau)} $$
<<<NEW>>>
定義接受指標 $a_i(\\tau)=\\mathbb{I}(s(x_i)\\ge\\tau)$。
*   **Selective Risk $R(\\tau)$**：在同意回答的樣本中，預測錯誤的比例。
    $$ R(\\tau)=\\frac{\\sum_{i=1}^{N}a_i(\\tau)\\mathbb{I}(\\hat{y}_i\\neq y_i)}{\\sum_{i=1}^{N}a_i(\\tau)} $$
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
    def compute_classwise_ece(probs, labels, bin_edges):
        """
        Compute Classwise ECE.
        probs: (N, C)
        labels: (N,)
        """
        probs = np.asarray(probs)
        labels = np.asarray(labels)
        
        if probs.ndim != 2:
            raise ValueError("Probs must be 2D.")
        N, C = probs.shape
        if N == 0 or C == 0:
            raise ValueError("N and C must be positive.")
        if labels.shape != (N,):
            raise ValueError("Label shape mismatch.")
        if not np.all(np.isfinite(probs)):
            raise ValueError("Probs contain non-finite values.")
        if not np.all((probs >= 0) & (probs <= 1)):
            raise ValueError("Probs out of [0, 1].")
        if np.any(np.abs(np.sum(probs, axis=1) - 1.0) > 1e-6):
            raise ValueError("Row sums of probs not equal to 1.")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Labels must be integers.")
        if np.any(labels < 0) or np.any(labels >= C):
            raise ValueError("Labels out of bounds.")
            
        eces = []
        for c in range(C):
            correctness_c = (labels == c).astype(int)
            conf_c = probs[:, c]
            ece_c, _ = compute_ece(conf_c, correctness_c, bin_edges)
            eces.append(ece_c)
            
        macro_ece = np.mean(eces)
        return eces, macro_ece
<<<NEW>>>
    def compute_classwise_ece(probs, labels, bin_edges):
        """
        Compute Classwise ECE.
        probs: (N, C), each row is a probability distribution.
        labels: (N,)
        """
        probs = np.asarray(probs)
        labels = np.asarray(labels)
        if probs.ndim != 2:
            raise ValueError("Probs must be 2D.")
        N, C = probs.shape
        if N == 0 or C == 0:
            raise ValueError("N and C must be positive.")
        if labels.shape != (N,):
            raise ValueError("Label shape mismatch.")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Labels must be integers.")
        if not np.all(np.isfinite(probs)):
            raise ValueError("Probs contain non-finite values.")
        if not np.all((probs >= 0) & (probs <= 1)):
            raise ValueError("Probs out of [0, 1].")
        if not np.all(np.abs(np.sum(probs, axis=1) - 1.0) <= 1e-6):
            raise ValueError("Row sums of probs not equal to 1.")
        if np.any(labels < 0) or np.any(labels >= C):
            raise ValueError("Labels out of bounds.")

        eces = []
        for c in range(C):
            correctness_c = (labels == c).astype(int)
            ece_c, _ = compute_ece(probs[:, c], correctness_c, bin_edges)
            eces.append(ece_c)
        return eces, float(np.mean(eces))

    # 預期測試：正常輸入回傳兩類 ECE，宏平均等於類別平均。
    class_eces, macro = compute_classwise_ece(
        [[.8, .2], [.3, .7]], np.array([0, 1]), [0, .5, 1])
    assert len(class_eces) == 2
    assert np.isclose(macro, np.mean(class_eces))
    # 以下非法輸入應各自拒絕。
    expect_value_error(compute_classwise_ece, np.empty((0, 2)),
                       np.array([], dtype=int), [0, 1])
    expect_value_error(compute_classwise_ece, np.empty((2, 0)),
                       np.array([0, 0]), [0, 1])
    expect_value_error(compute_classwise_ece, [[.8, .3]],
                       np.array([0]), [0, 1])
    expect_value_error(compute_classwise_ece, [[.8, .2]],
                       np.array([2]), [0, 1])
    expect_value_error(compute_classwise_ece, [[np.nan, .2]],
                       np.array([0]), [0, 1])
<<<END>>>