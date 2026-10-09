<<<PATCH 01>>>
<<<OLD>>>
$$ \\text{ECE}_{\\text{true, binned}} = \\sum_{b=1}^{B} q_b \\left| E[A - S \\mid S \\in I_b] \\right| $$
其中 $q_b = P(S \\in I_b)$ 是樣本落入箱 $b$ 的母體機率。
<<<NEW>>>
令 $q_b=P(S\\in I_b)$。對 $q_b>0$ 的箱，定義 $\\delta_b=E[A-S\\mid S\\in I_b]$；對 $q_b=0$ 的箱，母體貢獻定為零。母體固定分箱 ECE 定義為：
$$ \\text{ECE}_{\\text{true, binned}} = \\sum_{b:q_b>0} q_b |\\delta_b| $$
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
$$ E[ |X_b| \\mid N_b = n ] \\geq | E[ X_b \\mid N_b = n ] | $$

在 i.i.d. 假設下，給定 $N_b=n$，箱內的 $(A_i,S_i)$ 服從以 $S_i\\in I_b$ 為條件的分布。因此：
$$ E[ X_b \\mid N_b = n ] = E[ \\text{Acc}_b - \\bar{p}_b \\mid N_b = n ] = E[ A - S \\mid S \\in I_b ] $$
令 $\\delta_b = E[ A - S \\mid S \\in I_b ]$，則：
$$ E[ |X_b| \\mid N_b = n ] \\geq | \\delta_b | $$

令箱貢獻 $Y_b$ 在 $N_b=0$ 時為 $0$，在 $N_b>0$ 時為 $\\frac{N_b}{N}|X_b|$。這使 $Y_b$ 在整個樣本空間均有定義，不必計算空箱中未定義的 $X_b$。只對 $n=1,\\ldots,N$ 求和：
<<<NEW>>>
$$ E[ |X_b| \\mid N_b = n ] \\geq | E[ X_b \\mid N_b = n ] | $$

在 i.i.d. 假設下，給定 $N_b=n$，箱內的 $(A_i,S_i)$ 服從以 $S_i\\in I_b$ 為條件的分布。因此：
$$ E[ X_b \\mid N_b = n ] = E[ \\text{Acc}_b - \\bar{p}_b \\mid N_b = n ] = E[ A - S \\mid S \\in I_b ] $$
令 $\\delta_b = E[ A - S \\mid S \\in I_b ]$，則：
$$ E[ |X_b| \\mid N_b = n ] \\geq | \\delta_b | $$

定義在有限樣本空箱時也有值的箱貢獻：
$$
Y_b=
\\begin{cases}
\\frac{N_b}{N}|\\text{Acc}_b-\\bar p_b|,&N_b>0,\\\\
0,&N_b=0.
\\end{cases}
$$
若 $q_b=0$，則 $N_b=0$ 幾乎處處，故 $E[Y_b]=0$。若 $q_b>0$，只對 $n=1,\\ldots,N$ 求和：
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
*   **Selective Risk $R(\\tau)$**：在同意回答的樣本中，預測錯誤的比例。
    $$ R(\\tau) = \\frac{\\sum_{i: s(x_i) \\ge \\tau} \\mathbb{I}(\\hat{y}_i \\neq y_i)}{\\sum_{i: s(x_i) \\ge \\tau} \\mathbb{I}(s(x_i) \\ge \\tau)} $$
<<<NEW>>>
*   定義接受指標 $a_i(\\tau)=\\mathbb{I}(s(x_i)\\ge\\tau)$。
*   **Selective Risk $R(\\tau)$**：在同意回答的樣本中，預測錯誤的比例。
    $$ R(\\tau)=\\frac{\\sum_{i=1}^{N}a_i(\\tau)\\mathbb{I}(\\hat{y}_i\\neq y_i)}{\\sum_{i=1}^{N}a_i(\\tau)} $$
<<<END>>>