<<<PATCH 01>>>
<<<OLD>>>
**命題 3.2（有限樣本固定分箱 ECE 的非負向上偏差）**
在以下條件下，經驗 ECE 的期望值大於或等於真實的分箱校準誤差：
<<<NEW>>>
**命題 3.2（有限樣本固定分箱 ECE 的非負向上偏差）**
在以下條件下，經驗 ECE 的期望值不低於母體固定分箱 ECE：
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
定義隨機變數 $X_b = \\text{Acc}_b - \\bar{p}_b$。
定義母體固定分箱 ECE 為：
<<<NEW>>>
令 $A_i=\\mathbb{I}(\\hat{y}_i=y_i)$、$S_i=p_{\\max,i}$，並令 $X_b=\\text{Acc}_b-\\bar p_b$（僅在 $N_b>0$ 時定義）。定義母體固定分箱 ECE 為：
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
在 i.i.d. 假設下，給定 $N_b=n$，箱內的 $(A_i,S_i)$ 服從以 $S_i\\in I_b$ 為條件的分布。因此：
$$ E[ X_b \\mid N_b = n ] = E[ \\text{Acc}_b - \\bar{p}_b \\mid N_b = n ] = E[ A - S \\mid S \\in I_b ] $$
令 $\\delta_b = E[ A - S \\mid S \\in I_b ]$，則：
$$ E[ |X_b| \\mid N_b = n ] \\geq | \\delta_b | $$

對 $N_b$ 取全期望時，定義空箱的貢獻為零；非空箱的貢獻為 $\\frac{N_b}{N}|X_b|$。不必計算空箱中未定義的 $X_b$，下式只對 $n=1,\\ldots,N$ 求和：
$$ E\\left[ \\frac{N_b}{N} |X_b| \\right] = \\sum_{n=1}^{N} \\frac{n}{N} P(N_b=n) E[ |X_b| \\mid N_b=n ] $$
$$ \\geq \\sum_{n=1}^{N} \\frac{n}{N} P(N_b=n) | \\delta_b | $$
$$ = | \\delta_b | \\cdot \\frac{1}{N} E[N_b] $$
由於 $E[N_b] = N q_b$，所以：
$$ E\\left[ \\frac{N_b}{N} |X_b| \\right] \\geq | \\delta_b | \\cdot q_b $$

對所有箱求和：
$$ E[\\text{ECE}_{\\text{empirical}}] = \\sum_{b=1}^{B} E\\left[ \\frac{N_b}{N} |X_b| \\right] \\geq \\sum_{b=1}^{B} q_b | \\delta_b | = \\text{ECE}_{\\text{true, binned}} $$

**限制**：此證明依賴於分箱邊界固定且樣本 i.i.d.。若分箱邊界依資料估計（自適應分箱），或樣本非獨立，則不等式方向可能改變或無法直接成立。因此，ECE 僅作為相對指標，用於比較同條件下的不同模型，不可作為絕對品質保證。
<<<NEW>>>
在 i.i.d. 假設下，給定 $N_b=n>0$，箱內樣本對 $(A_i,S_i)$ 服從條件分布 $P((A,S)\\mid S\\in I_b)$。因此：
$$ E[X_b\\mid N_b=n]=E[\\text{Acc}_b-\\bar p_b\\mid N_b=n]=E[A-S\\mid S\\in I_b]. $$
令 $\\delta_b=E[A-S\\mid S\\in I_b]$，由 Jensen 不等式得：
$$ E[|X_b|\\mid N_b=n]\\geq |\\delta_b|. $$

定義在空箱時也有值的箱貢獻隨機變數：
$$
Y_b=
\\begin{cases}
\\frac{N_b}{N}|\\text{Acc}_b-\\bar p_b|,&N_b>0,\\\\
0,&N_b=0.
\\end{cases}
$$
於是：
$$
E[Y_b]=\\sum_{n=1}^{N}\\frac{n}{N}P(N_b=n)E[|X_b|\\mid N_b=n]
\\geq |\\delta_b|\\frac{E[N_b]}{N}
=q_b|\\delta_b|,
$$
其中 $E[N_b]=Nq_b$。由 $\\widehat{\\mathrm{ECE}}=\\sum_bY_b$，對所有箱求和即得：
$$
E[\\widehat{\\mathrm{ECE}}]
\\geq\\sum_{b=1}^{B}q_b|\\delta_b|
=\\mathrm{ECE}_{\\mathrm{true,binned}}.
$$

**限制**：此證明依賴固定分箱邊界及 i.i.d. 樣本。若分箱邊界依資料估計，或樣本非獨立，以上不等式不能直接套用。ECE 是依賴資料、樣本量與分箱規則的描述性估計量；可在相同評估程序下輔助比較，但不可單獨作為絕對品質或安全保證。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
*   **Selective Risk $R(\\tau)$**：在同意回答的樣本中，預測錯誤的比例。
    $$ R(\\tau) = \\frac{\\sum_{i: s(x_i) \\ge \\tau} \\mathbb{I}(\\hat{y}_i \\neq y_i)}{\\sum_{i: s(x_i) \\ge \\tau} \\mathbb{I}(s(x_i) \\ge \\tau)} $$
<<<NEW>>>
*   定義接受指標 $a_i(\\tau)=\\mathbb{I}(s(x_i)\\ge\\tau)$。
*   **Coverage $C(\\tau)$**：模型同意回答的比例。
    $$ C(\\tau)=\\frac{1}{N}\\sum_{i=1}^{N}a_i(\\tau) $$
*   **Selective Risk $R(\\tau)$**：在同意回答的樣本中，預測錯誤的比例。
    $$ R(\\tau)=\\frac{\\sum_{i=1}^{N}a_i(\\tau)\\mathbb{I}(\\hat{y}_i\\neq y_i)}{\\sum_{i=1}^{N}a_i(\\tau)} $$
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
本章建立了語言模型評估的定量基礎。
1.  **PPL** 必須基於 Token-weighted 計算，以反映真實語料規模下的性能。
2.  **ECE** 受分箱策略影響，有限樣本下存在正偏差，僅作為相對指標。
<<<NEW>>>
本章建立了語言模型評估的定量基礎。
1.  **PPL** 必須基於 Token-weighted 計算，以反映真實語料規模下的性能。
2.  **ECE** 受分箱策略影響；固定分箱及 i.i.d. 條件下，經驗 ECE 的期望不低於母體固定分箱 ECE。它是依賴資料與分箱規則的描述性估計量，可輔助比較，但不能單獨作為品質或安全保證。
<<<END>>>