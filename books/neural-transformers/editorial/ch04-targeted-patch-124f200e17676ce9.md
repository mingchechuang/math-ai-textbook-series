<<<PATCH 04>>>
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
本章的熵、交叉熵、KL 與 NLL 的最終值允許取擴展非負實數 $[0,+\infty]$；中間對數約定 $\ln 0=-\infty$。零權重項約定為 $0\ln a=0$，包括 $a=0$。若權重為正而被取對數的概率為零，則相應損失項為 $+\infty$。

**香農熵** $H(P)$ 衡量分布 $P$ 的內在不確定性：
$$ H(P) = -\sum_{x \in \mathcal{X}} P(x) \ln P(x) $$

**交叉熵** $H(P, Q)$ 衡量用分布 $Q$ 編碼來自 $P$ 的資料所需的平均長度：
$$ H(P, Q) = -\sum_{x \in \mathcal{X}} P(x) \ln Q(x) $$

**KL 散度** $D_{KL}(P \| Q)$ 衡量 $Q$ 與 $P$ 的差異：
$$ D_{KL}(P \| Q) = \sum_{x \in \mathcal{X}} P(x) \ln \frac{P(x)}{Q(x)} = H(P, Q) - H(P) $$

**支撐集規則總結**：
1.  若 $P(x) = 0$，則項 $P(x)\ln(P(x)/Q(x))$ 為 0（即使 $Q(x)=0$）。
2.  若存在 $x$ 使 $P(x)>0$ 且 $Q(x)=0$，則該交叉熵項為 $+\infty$，故 $H(P,Q)=+\infty$，且 $D_{KL}(P \| Q)=+\infty$。
3.  若 $H(P) = +\infty$，則 $H(P,Q)-H(P)$ 可能涉及 $\infty-\infty$；此時以 KL 的逐項定義為準，不使用該差值恆等式。
<<<END>>>