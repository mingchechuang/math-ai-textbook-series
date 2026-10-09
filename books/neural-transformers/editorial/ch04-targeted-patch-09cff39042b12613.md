<<<PATCH 01>>>
<<<OLD>>>
**命題 4.1**：對於任意兩個概率分布 $P, Q$，若 $P(x)>0 \implies Q(x)>0$，則 $D_{KL}(P \| Q) \geq 0$，且等號成立當且僅當 $P(x)=Q(x)$ 對所有 $x$ 成立。
<<<NEW>>>
**命題 4.1**：對於有限離散樣本空間上的概率分布 $P, Q$，若 $P(x)>0 \implies Q(x)>0$，則 $D_{KL}(P \| Q) \geq 0$，且等號成立當且僅當 $P(x)=Q(x)$ 對所有 $x$ 成立。以下證明限於有限空間；可數無限情形需要另行處理無限和。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    sum_nll = np.sum(valid_nll)
    mean_nll = sum_nll / count
    
    # 防止 overflow
    if mean_nll > 700: # ln(float_max) approx 709
        return np.inf
        
    return float(np.exp(mean_nll))
<<<NEW>>>
    # 先縮放再沿有效 token 軸求和，避免總和先行溢位
    mean_nll = np.sum(valid_nll / count, dtype=np.float64)
    if mean_nll > np.log(np.finfo(np.float64).max):
        return np.inf  # float64 表示限制，不代表數學上的 PP 無限
    return float(np.exp(mean_nll))
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
        terms[p_support] = p[p_support] * np.log(p[p_support] / q[p_support])
<<<NEW>>>
        safe = p_support & (q > 0)
        terms[safe] = p[safe] * (np.log(p[safe]) - np.log(q[safe]))
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
2.  若 $P(x) > 0$ 且 $Q(x) = 0$，則 $D_{KL}(P \| Q) = +\infty$。
<<<NEW>>>
2.  若存在 $x$ 使 $P(x)>0$ 且 $Q(x)=0$，則 $H(P,Q)=+\infty$ 且 $D_{KL}(P \| Q)=+\infty$。
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
*註：以上來源為標準學術參考，本章未執行外部程式驗證其內容，僅引用其定義作為數學依據。*
<<<NEW>>>
*註：以上列為延伸學術參考；本章稿未提供逐章頁碼或逐條查閱紀錄，不聲稱已逐項核對。*
<<<END>>>