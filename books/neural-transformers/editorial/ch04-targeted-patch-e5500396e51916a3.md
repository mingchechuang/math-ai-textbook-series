<<<PATCH 01>>>
<<<OLD>>>
**擴展實數約定**：
為了處理零機率情況，我們採用擴展實數系 $\mathbb{R} \cup \{+\infty, -\infty\}$。
1.  $0 \cdot \ln 0$ 定義為 $0$。
2.  $P(x) \ln(0)$ 定義為 $0$ （若 $P(x)=0$）。
3.  若 $P(x) > 0$ 且 $Q(x) = 0$，則 $P(x) \ln(P(x)/Q(x)) = +\infty$。

**香農熵** $H(P)$ 衡量分布 $P$ 的內在不確定性：
$$ H(P) = -\sum_{x \in \mathcal{X}} P(x) \ln P(x) $$

**交叉熵** $H(P, Q)$ 衡量用分布 $Q$ 編碼來自 $P$ 的資料所需的平均長度：
$$ H(P, Q) = -\sum_{x \in \mathcal{X}} P(x) \ln Q(x) $$

**KL 散度** $D_{KL}(P \| Q)$ 衡量 $Q$ 與 $P$ 的差異：
$$ D_{KL}(P \| Q) = \sum_{x \in \mathcal{X}} P(x) \ln \frac{P(x)}{Q(x)} = H(P, Q) - H(P) $$

**支撐集規則總結**：
1.  若 $P(x) = 0$，則項 $P(x)\ln(P(x)/Q(x))$ 為 0（即使 $Q(x)=0$）。
2.  若 $P(x) > 0$ 且 $Q(x) = 0$，則 $D_{KL}(P \| Q) = +\infty$。
3.  若 $H(P) = +\infty$（可數無限空間中可能發生），則 KL 散度定義可能涉及 $\infty - \infty$ 的不定式，本章不處理此種病態情況。
<<<NEW>>>
**擴展實數約定**：
本章熵、交叉熵、KL 散度與 NLL 的最終值允許取擴展非負實數 $[0,+\infty]$；中間對數約定 $\ln 0=-\infty$。零權重項約定為 $0\ln a=0$，包括 $a=0$。若權重為正而被取對數的概率為零，則相應損失項為 $+\infty$；特別是若存在 $x$ 使 $P(x)>0$ 且 $Q(x)=0$，則該交叉熵項為 $+\infty$，所以 $H(P,Q)=+\infty$，且 $D_{KL}(P\|Q)=+\infty$。

**香農熵** $H(P)$ 衡量分布 $P$ 的內在不確定性：
$$ H(P) = -\sum_{x \in \mathcal{X}} P(x) \ln P(x) $$

**交叉熵** $H(P, Q)$ 衡量用分布 $Q$ 編碼來自 $P$ 的資料所需的平均長度：
$$ H(P, Q) = -\sum_{x \in \mathcal{X}} P(x) \ln Q(x) $$

**KL 散度** $D_{KL}(P \| Q)$ 衡量 $Q$ 與 $P$ 的差異：
$$ D_{KL}(P \| Q) = \sum_{x \in \mathcal{X}} P(x) \ln \frac{P(x)}{Q(x)} = H(P, Q) - H(P) $$

**支撐集規則總結**：
1.  若 $P(x) = 0$，則項 $P(x)\ln(P(x)/Q(x))$ 為 0（即使 $Q(x)=0$）。
2.  若 $P(x) > 0$ 且 $Q(x) = 0$，則 $H(P,Q)$ 與 $D_{KL}(P \| Q)$ 均為 $+\infty$。
3.  若 $H(P)=+\infty$，差值 $H(P,Q)-H(P)$ 可能涉及 $\infty-\infty$；此時以 KL 的逐項定義為準，不使用該差值恆等式。本章的非負性證明限於有限樣本空間；可數無限空間的推廣需另行處理。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
**命題 4.1**：對於任意兩個概率分布 $P, Q$，若 $P(x)>0 \implies Q(x)>0$，則 $D_{KL}(P \| Q) \geq 0$，且等號成立當且僅當 $P(x)=Q(x)$ 對所有 $x$ 成立。
<<<NEW>>>
**命題 4.1**：對於有限離散樣本空間上的概率分布 $P,Q$，若 $P(x)>0 \implies Q(x)>0$，則 $D_{KL}(P \| Q) \geq 0$，且等號成立當且僅當 $P(x)=Q(x)$ 對所有 $x$ 成立。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
因此 $Q(x) = P(x)$ 對所有 $x \in S$。又在 $S$ 外 $P(x)=0$ 且 $Q(x)=0$，故全域 $P=Q$。證畢。
<<<NEW>>>
因此 $Q(x) = P(x)$ 對所有 $x \in S$。又在 $S$ 外 $P(x)=0$ 且 $Q(x)=0$，故全域 $P=Q$。證畢。本證明針對有限空間；可數無限空間的推廣須使用擴展 Jensen 不等式或有限截斷極限，本章不展開。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
**直覺限制**：PP 等於「等效候選詞數量」僅在模型輸出均勻分布的特殊情況下精確成立。一般情況下，它是平均對數損失的指數，反映模型的整體預測困難度。若 $PP=100$，表示模型在平均意義上的預測困難度相當於在 100 個候選中隨機猜測。
<<<NEW>>>
**直覺限制**：PP 等於「等效候選詞數量」僅在模型輸出均勻分布的特殊情況下精確成立。一般情況下，它是平均對數損失的指數，也等於正確 token 機率幾何平均的倒數。若 $PP=100$，則正確 token 機率的幾何平均為 $1/100$；只有每一步都均勻分布在相同數量候選上的特殊情形，才等同於在 100 個等可能候選間猜測。
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
    p_support = p > 0
        bad = np.any(p_support & (q == 0), axis=-1)
        terms = np.zeros_like(p, dtype=np.float64)
        terms[p_support] = p[p_support] * np.log(p[p_support] / q[p_support])
        out = np.sum(terms, axis=-1)
        out[bad] = np.inf
        return out
<<<NEW>>>
    p_support = p > 0
        bad = np.any(p_support & (q == 0), axis=-1)
        safe = p_support & (q > 0)
        terms = np.zeros_like(p, dtype=np.float64)
        terms[safe] = p[safe] * (np.log(p[safe]) - np.log(q[safe]))
        out = np.sum(terms, axis=-1)
        out[bad] = np.inf
        return out
<<<END>>>