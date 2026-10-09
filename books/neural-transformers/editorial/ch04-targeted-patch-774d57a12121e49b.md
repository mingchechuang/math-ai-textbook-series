<<<PATCH 01>>>
<<<OLD>>>
**擴展實數約定**：
為了處理零機率情況，我們採用擴展實數系 $\mathbb{R} \cup \{+\infty, -\infty\}$。
1.  $0 \cdot \ln 0$ 定義為 $0$。
2.  $P(x) \ln(0)$ 定義為 $0$ （若 $P(x)=0$）。
3.  若 $P(x) > 0$ 且 $Q(x) = 0$，則 $P(x) \ln(P(x)/Q(x)) = +\infty$。
<<<NEW>>>
**零機率約定**：
本章的熵、交叉熵、KL 與 NLL 可取擴展非負實數 $[0,+\infty]$；中間計算約定 $\ln 0=-\infty$。零權重項約定為 $0\ln a=0$（$a\geq0$），特別地 $0\ln0=0$。若 $P(x)>0$ 而 $Q(x)=0$，則 $-P(x)\ln Q(x)=+\infty$，且對應的 KL 項也為 $+\infty$。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
**命題 4.1**：對於任意兩個概率分布 $P, Q$，若 $P(x)>0 \implies Q(x)>0$，則 $D_{KL}(P \| Q) \geq 0$，且等號成立當且僅當 $P(x)=Q(x)$ 對所有 $x$ 成立。
<<<NEW>>>
**命題 4.1**：對於有限離散樣本空間上的概率分布 $P, Q$，若 $P(x)>0 \implies Q(x)>0$，則 $D_{KL}(P \| Q) \geq 0$，且等號成立當且僅當 $P(x)=Q(x)$ 對所有 $x$ 成立。以下證明限於有限空間；可數無限空間的結論需要另行處理無限和。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
**直覺限制**：PP 等於「等效候選詞數量」僅在模型輸出均勻分布的特殊情況下精確成立。一般情況下，它是平均對數損失的指數，反映模型的整體預測困難度。若 $PP=100$，表示模型在平均意義上的預測困難度相當於在 100 個候選中隨機猜測。
<<<NEW>>>
**直覺限制**：一般情況下，PP 是有效位置上正確 token 機率之幾何平均的倒數。$PP=100$ 表示該幾何平均為 $1/100$；只有每步在 100 個候選間均勻預測的特殊情形，才可解讀為在 100 個等可能候選中猜測。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
    sum_nll = np.sum(valid_nll)
    mean_nll = sum_nll / count
    
    # 防止 overflow
    if mean_nll > 700: # ln(float_max) approx 709
        return np.inf
        
    return float(np.exp(mean_nll))
<<<NEW>>>
    # 先除以 count，避免總和先溢位而平均值仍可表示
    mean_nll = np.sum(valid_nll / count, dtype=np.float64)
    if mean_nll > np.log(np.finfo(np.float64).max):
        return np.inf  # float64 表示限制，不代表數學上的 PP 無限
    return float(np.exp(mean_nll))
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
        p_support = p > 0
        bad = np.any(p_support & (q == 0), axis=-1)
        terms = np.zeros_like(p, dtype=np.float64)
        terms[p_support] = p[p_support] * np.log(p[p_support] / q[p_support])
        out = np.sum(terms, axis=-1)
<<<NEW>>>
        p_support = p > 0
        bad = np.any(p_support & (q == 0), axis=-1)
        safe = p_support & (q > 0)
        terms = np.zeros_like(p, dtype=np.float64)
        terms[safe] = p[safe] * (np.log(p[safe]) - np.log(q[safe]))
        out = np.sum(terms, axis=-1)
<<<END>>>