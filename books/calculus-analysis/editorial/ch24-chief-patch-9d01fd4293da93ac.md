<<<PATCH 24-01>>>
<<<OLD>>>
例如，定义 $f_n: [0,1] \to \mathbb{R}$ 为：
$$ f_n(x) = \begin{cases} 1 - n x, & 0 \le x \le \frac{1}{n} \\ 0, & \frac{1}{n} < x \le 1 \end{cases} $$
对任何固定 $x \in (0,1]$，当 $n > 1/x$ 时 $f_n(x)=0$，故逐点极限为 $f(x)=0$（连续）。但 $\sup_{x \in [0,1]} |f_n(x)| = f_n(0) = 1$，故不一致收敛。
<<<NEW>>>
例如，定义 $f_n: [0,1] \to \mathbb{R}$ 为连续三角尖峰：对 $n \ge 2$，
$$ f_n(x) = \begin{cases} n x, & 0 \le x \le \frac{1}{n}, \\ 2 - n x, & \frac{1}{n} < x \le \frac{2}{n}, \\ 0, & \frac{2}{n} < x \le 1, \end{cases} $$
并对 $n=1$ 可任意定义（例如 $f_1(x)=0$）。则对任何固定 $x \in (0,1]$，当 $n > 2/x$ 时 $f_n(x)=0$，故逐点极限为 $f(x)=0$（连续）。但 $\sup_{x \in [0,1]} |f_n(x)| = 1$（在 $x=1/n$ 处取得），故不一致收敛。
<<<END>>>

<<<PATCH 24-02>>>
<<<OLD>>>
3.  **[反例/证明]** 证明：若 $\{h_n\}$ 在 $[a,b]$ 上可微，$h_n'$ 一致收敛于 $g$，且 $h_n$ 在 $[a,b]$ 上逐点收敛，则 $h_n$ 必一致收敛。
    *提示*：利用微积分基本定理将 $h_n(x)$ 表示为 $h_n(a) + \int_a^x h_n'(t) dt$。
<<<NEW>>>
3.  **[反例/证明]** 证明：若 $\{h_n\} \subset C^1([a,b])$，$h_n'$ 一致收敛于 $g$，且 $h_n$ 在 $[a,b]$ 上逐点收敛，则 $h_n$ 必一致收敛。
    *提示*：利用微积分基本定理将 $h_n(x)$ 表示为 $h_n(a) + \int_a^x h_n'(t) dt$。
<<<END>>>

<<<PATCH 24-03>>>
<<<OLD>>>
    -   取 sup norm：
        $$ \sup_x |h_n(x) - h_m(x)| \le |h_n(a) - h_m(a)| + \int_a^x |h_n'(t) - h_m'(t)| dt $$
        $$ \le |h_n(a) - h_m(a)| + (b-a) \sup_t |h_n'(t) - h_m'(t)| $$
<<<NEW>>>
    -   取 sup norm：
        $$ \sup_x |h_n(x) - h_m(x)| \le |h_n(a) - h_m(a)| + \sup_x \left| \int_a^x (h_n'(t) - h_m'(t)) dt \right| $$
        $$ \le |h_n(a) - h_m(a)| + (b-a) \sup_t |h_n'(t) - h_m'(t)| $$
<<<END>>>

<<<PATCH 24-04>>>
<<<OLD>>>
故 $F'(t) = \int_a^b \frac{\partial f}{\partial t}(x,t) \, dx$。$\blacksquare$

## 逐步手算例题
<<<NEW>>>
故 $F'(t) = \int_a^b \frac{\partial f}{\partial t}(x,t) \, dx$。$\blacksquare$

**定理 24.8 (函数级数逐项微分)** 设 $f_n \in C^1([a,b])$。若数值级数 $\sum_{n=1}^\infty f_n(c)$ 在某点 $c \in [a,b]$ 收敛，且导数级数 $\sum_{n=1}^\infty f_n'$ 在 $[a,b]$ 上一致收敛，则函数级数 $\sum_{n=1}^\infty f_n$ 在 $[a,b]$ 上一致收敛，其和函数 $S(x) = \sum_{n=1}^\infty f_n(x)$ 在 $[a,b]$ 上可微，且
$$ S'(x) = \sum_{n=1}^\infty f_n'(x). $$
**证明：** 令 $S_N = \sum_{n=1}^N f_n$。则 $S_N \in C^1([a,b])$，$S_N' = \sum_{n=1}^N f_n'$。由假设 $\sum f_n(c)$ 收敛，故 $S_N(c)$ 收敛。由假设 $\sum f_n'$ 一致收敛，故 $S_N'$ 一致收敛于 $g = \sum f_n'$。由定理 24.5，$S_N$ 一致收敛于某可微函数 $S$，且 $S' = g$。即得结论。$\blacksquare$

**注意**：仅有原级数 $\sum f_n$ 一致收敛不足以逐项微分。例如 $f_n(x) = \frac{\sin(nx)}{n}$ 在 $[0,\pi]$ 上，$\sum f_n$ 一致收敛（由 M-判准，$|f_n| \le 1/n$），但 $\sum f_n' = \sum \cos(nx)$ 不一致收敛，故不能逐项微分。

## 逐步手算例题
<<<END>>>