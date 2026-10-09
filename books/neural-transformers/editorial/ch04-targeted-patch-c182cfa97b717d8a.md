<<<PATCH 01>>>
<<<OLD>>>
    # 防止 overflow
    if mean_nll > 700: # ln(float_max) approx 709
        return np.inf
<<<NEW>>>
    # 超出 float64 可表示範圍才回傳 inf；數學值未必無限
    if mean_nll > np.log(np.finfo(np.float64).max):
        return np.inf
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
        terms[p_support] = p[p_support] * np.log(p[p_support] / q[p_support])
<<<NEW>>>
        safe = p_support & (q > 0)
        terms[safe] = p[safe] * (np.log(p[safe]) - np.log(q[safe]))
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
**命題 4.1**：對於任意兩個概率分布 $P, Q$，若 $P(x)>0 \implies Q(x)>0$，則 $D_{KL}(P \| Q) \geq 0$，且等號成立當且僅當 $P(x)=Q(x)$ 對所有 $x$ 成立。
<<<NEW>>>
**命題 4.1**：對於有限離散樣本空間上的概率分布 $P, Q$，若 $P(x)>0 \implies Q(x)>0$，則 $D_{KL}(P \| Q) \geq 0$，且等號成立當且僅當 $P(x)=Q(x)$ 對所有 $x$ 成立。以下證明限於有限情形；可數無限情形需另行處理無限和。
<<<END>>>