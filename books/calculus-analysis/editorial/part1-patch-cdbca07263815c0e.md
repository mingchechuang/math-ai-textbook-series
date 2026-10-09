<<<PATCH 01>>>
<<<OLD>>>
**域與正則性**：$f\in C^2(a,b)$ 是必要條件。若 $f$ 只連續而不可微，本定理的 $f'(x_0)$ 無定義；若 $f''$ 在 $x_0$ 附近無界，定理右端為無窮，界失去意義。$x_0+h$ 必須落在開區間 $(a,b)$ 內。
<<<NEW>>>
**域與正則性**：定理 1.5 採用 $f\in C^2(a,b)$ 作為假設；這是充分條件，本卷不主張它對結論是必要的。若 $f$ 只連續而不可微，本定理的 $f'(x_0)$ 無定義；若 $f''$ 在 $x_0$ 附近無界，定理右端為無窮，界失去意義。$x_0+h$ 必須落在開區間 $(a,b)$ 內。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
由三角不等式，$\|x_k - a\| < 1/k \implies \lim_{k \to \infty} x_k = a$。
<<<NEW>>>
因為 $0\le\|x_k-a\|<1/k$ 且 $1/k\to 0$，故 $x_k\to a$。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
函數 $f: U \subseteq \mathbb{R}^n \to \mathbb{R}^m$ 在點 $a \in U$ 處連續，定義如下：
$$\forall \epsilon > 0, \exists \delta > 0, \forall x \in U, \quad \|x - a\| < \delta \implies \|f(x) - f(a)\| < \epsilon.$$
<<<NEW>>>
函數 $f: U \subseteq \mathbb{R}^n \to \mathbb{R}^m$ 在點 $a \in U$ 處連續，定義如下：
$$\forall \epsilon > 0, \exists \delta > 0, \forall x \in U, \quad \|x - a\| < \delta \implies \|f(x) - f(a)\| < \epsilon.$$
任取 $\mathbb{R}^n$、$\mathbb{R}^m$ 上一組範數；由第3章有限維範數等價，極限與連續性不依範數選擇而變。
<<<END>>>
<<<PATCH 06>>>
<<<OLD>>>
取 $x_k=1/\sqrt{(2k+1)\pi-\pi/2}$，則 $\sin(1/x_k^2)=\sin((2k+1)\pi-\pi/2)=(-1)^k\cdot(-1)^{2k+1}\cdot1$ 的絕對值為 $1$
<<<NEW>>>
取 $x_k=1/\sqrt{(2k+1)\pi-\pi/2}$，則 $\sin(1/x_k^2)=\sin((2k+1)\pi-\pi/2)=\sin(2k\pi+\pi/2)=1$
<<<END>>>
<<<PATCH 01>>>
<<<OLD>>>
本卷後續章節會用小型矩陣把這條等式從分量的角度驗算。
<<<NEW>>>
後續卷章會用小型矩陣把這條等式從分量的角度驗算。
<<<END>>>