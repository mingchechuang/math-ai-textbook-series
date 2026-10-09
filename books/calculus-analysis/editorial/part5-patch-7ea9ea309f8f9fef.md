<<<PATCH 26>>>
<<<OLD>>>
定义
$$
\alpha:=\min\Bigl(a,\ \frac{b}{M}\Bigr)\quad(M>0),\qquad\alpha:=a\quad(M=0).
\tag{26.6}
$$
則在閉區間 $I_\alpha:=[t_0-\alpha,\,t_0+\alpha]$ 上，初值問題 (26.1) 恰有一個 $C^1$ 解 $y:I_\alpha\to\overline{B(y_0,b)}$。
<<<NEW>>>
定义
$$
\alpha:=
\begin{cases}
\min\Bigl(a,\ \frac{b}{M},\ \frac{1}{2L}\Bigr), & M>0,\ L>0,\\
\min\Bigl(a,\ \frac{b}{M}\Bigr), & M>0,\ L=0,\\
a, & M=0.
\end{cases}
\tag{26.6}
$$
則在閉區間 $I_\alpha:=[t_0-\alpha,\,t_0+\alpha]$ 上，初值問題 (26.1) 恰有一個 $C^1$ 解 $y:I_\alpha\to\overline{B(y_0,b)}$。
<<<END>>>
<<<PATCH 26>>>
<<<OLD>>>
**(b) $T$ 為壓縮.** 對 $y,z\in X$，
$$
\lVert(Ty)(t)-(Tz)(t)\rVert
\le\Bigl|\int_{t_0}^{t}\lVert f(s,y(s))-f(s,z(s))\rVert\,ds\Bigr|
\le L\lvert t-t_0\rvert\,\lVert y-z\rVert_\infty
\le L\alpha\lVert y-z\rVert_\infty.
$$
取 $\alpha'=\min(\alpha,\ 1/(2L))$（$L>0$），則 $q=L\alpha'\le 1/2<1$，$T$ 在以 $\alpha'$ 定義的 $X$ 上是壓縮。若 $L=0$，$f$ 關於 $y$ 恆定，$T$ 是常值映射，可直接由 (26.7) 得唯一解 $y(t)=y_0+\int_{t_0}^{t}f(s)\,ds$。以下設 $L>0$，並將記號 $\alpha$ 重設為 $\alpha'$。
<<<NEW>>>
**(b) $T$ 為壓縮.** 對 $y,z\in X$，
$$
\lVert(Ty)(t)-(Tz)(t)\rVert
\le\Bigl|\int_{t_0}^{t}\lVert f(s,y(s))-f(s,z(s))\rVert\,ds\Bigr|
\le L\lvert t-t_0\rvert\,\lVert y-z\rVert_\infty
\le L\alpha\lVert y-z\rVert_\infty.
$$
若 $L=0$，$f$ 關於 $y$ 恆定，$T$ 是常值映射，可直接由 (26.7) 得唯一解 $y(t)=y_0+\int_{t_0}^{t}f(s)\,ds$。以下設 $L>0$。若 $M=0$，則 $f$ 在 $R$ 上恆為零，$T$ 仍為常值映射，壓縮比可取為 $0$；若 $M>0$，則由 (26.6) 的定義 $\alpha\le 1/(2L)$，所以 $q=L\alpha\le 1/2<1$。因此 $T$ 在以 $\alpha$ 定義的 $X$ 上是壓縮。
<<<END>>>
<<<PATCH 26>>>
<<<OLD>>>
2. **充分而非必要**：Lipschitz 條件對存在唯一是**充分條件**。Osgood 型例子 $y'=\lvert y\rvert^{1/2}$ 展示非 Lipschitz 仍可唯一；故不能倒過來宣稱「唯一必 Lipschitz」。
<<<NEW>>>
2. **充分而非必要**：Lipschitz 條件對存在唯一是**充分條件**。例如 $y'=-\sqrt{\lvert y\rvert},\ y(0)=0$ 在 $y=0$ 附近非 Lipschitz，但原點初值問題仍唯一（可用分離變數與比較論證）；故不能倒過來宣稱「唯一必 Lipschitz」。
<<<END>>>
<<<PATCH 26>>>
<<<OLD>>>
7. **忽略 $t$ 方向連續性。** 定理 26.4 的證明用 $f$ 連續與緊緻性得到 $M<\infty$，也隱含 $f$ 在 $R$ 上一致連續以便積分極限的連續性。若只給 $y$-Lipschitz、$f$ 隨 $t$ 有跳斷，則 $T$ 未必映到連續函數空間，$X$ 完備性假設失效，定理不成立。以 $f(t,y)=\mathrm{sgn}(t)$（在 $t=0$ 跳斷）與 $y_0=0$ 舉例，即使 $f$ 對 $y$ 是 $0$-Lipschitz，$Ty$ 在 $t=0$ 也不連續，$Tx\notin X$。
<<<NEW>>>
7. **忽略 $t$ 方向連續性。** 定理 26.4 的證明用 $f$ 連續與緊緻性得到 $M<\infty$，也隱含 $f$ 在 $R$ 上一致連續以便積分極限的連續性。若只給 $y$-Lipschitz、$f$ 隨 $t$ 有跳斷，則 $T$ 未必映到連續函數空間，$X$ 完備性假設失效，定理不成立。以 $f(t,y)=\mathrm{sgn}(t)$（在 $t=0$ 跳斷）與 $y_0=0$ 舉例，即使 $f$ 對 $y$ 是 $0$-Lipschitz，$(Ty)(t)=\int_0^t\mathrm{sgn}(s)\,ds=|t|$ 雖為連續函數，但在 $t=0$ 不可微，故不能由積分方程得到本節所要求的 $C^1$ 經典解。
<<<END>>>
<<<PATCH 30>>>
<<<OLD>>>
$$
s(T)=
\begin{pmatrix}
s_1(T)\\
s_2(T)
\end{pmatrix}
=
\begin{pmatrix}
1\\
0.8
\end{pmatrix}
+

\begin{pmatrix}
1+0.1T & 0.2\\
0.3 & 0.9-0.05T
\end{pmatrix}
\begin{pmatrix}
\theta_1\\
\theta_2
\end{pmatrix},
\qquad T\text{ 的單位為 }{}^\circ\mathrm C .
$$
<<<NEW>>>
$$
s(T)=
\begin{pmatrix}
s_1(T)\\
s_2(T)
\end{pmatrix}
=
\begin{pmatrix}
1\\
0.8
\end{pmatrix}
+

\begin{pmatrix}
1+0.1T & 0.2\\
0.3 & 0.9-0.05T
\end{pmatrix}
\begin{pmatrix}
\theta_1\\
\theta_2
\end{pmatrix},
\qquad T\text{ 的單位為 }{}^\circ\mathrm C .
$$
其中矩陣元素皆已按輸出單位 $\mathrm{mg/L}$ 解讀；$0.1$ 與 $0.05$ 的單位為 $\mathrm{mg/(L\cdot{}^\circ C)}$，或等價地先把 $T$ 以 $1{}^\circ C$ 無因次化。參數 $\theta_1,\theta_2$ 無因次。
<<<END>>>