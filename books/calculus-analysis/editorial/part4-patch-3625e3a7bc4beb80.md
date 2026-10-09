<<<PATCH 21>>>
<<<OLD>>>
**充分條件**：$C^1$ 且 $\gamma'$ 連續即可使 $\|\gamma'\|$ 可積，弧長有限。**必要條件**方面：$\gamma$ Lipschitz 是弧長有限的必要條件之一，但不是充分條件（Cantor函數弧長無限的變體需另行討論，本章不深入）。
<<<NEW>>>
**充分條件**：$C^1$ 曲線的速度範數在緊區間上連續，故弧長有限。Lipschitz 也是有限弧長的充分條件，但不是必要條件：$\gamma(t)=\sqrt t$ 在 $[0,1]$ 上不是 Lipschitz，弧長仍為 $1$。
<<<END>>>
<<<PATCH 23>>>
<<<OLD>>>
def check(inner, outer, cells=200, segments=720):
    if not (0 <= inner < outer and cells > 0 and segments >= 3):
        raise ValueError("需要 0 <= inner < outer、cells > 0、segments >= 3")
<<<NEW>>>
def check(inner, outer, cells=200, segments=720):
    if (isinstance(inner, bool) or isinstance(outer, bool)
            or not isinstance(inner, (int, float))
            or not isinstance(outer, (int, float))
            or not math.isfinite(inner) or not math.isfinite(outer)
            or not 0 <= inner < outer):
        raise ValueError("半徑須為有限實數，且 0 <= inner < outer")
    if (isinstance(cells, bool) or isinstance(segments, bool)
            or not isinstance(cells, int) or not isinstance(segments, int)
            or cells <= 0 or segments < 3):
        raise ValueError("cells 須為正整數，segments 須為至少 3 的整數")
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
**定理 24.5 (微分交換)** 設 $\{f_n\}$ 定義在 $[a,b]$ 上，滿足：
<<<NEW>>>
**定理 24.5 (微分交換)** 設 $a<b$，且 $\{f_n\}$ 定義在 $[a,b]$ 上，滿足：
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
**函數級數逐項微分定理** 若 $u_n\in C^1([a,b])$，某點 $c\in[a,b]$ 上的數值級數 $\sum_{n=1}^{\infty}u_n(c)$ 收斂，且導數級數 $\sum_{n=1}^{\infty}u_n'$ 在 $[a,b]$ 上一致收斂，則原級數在 $[a,b]$ 上一致收斂；其和函數 $S$ 在 $(a,b)$ 上可微，且
$$ S'(x)=\sum_{n=1}^{\infty}u_n'(x). $$
證明：令部分和 $S_N=\sum_{n=1}^N u_n$。各 $S_N$ 屬於 $C^1([a,b])$，$S_N(c)$ 收斂，而 $S_N'=\sum_{n=1}^N u_n'$ 一致收斂。對函數列 $S_N$ 套用定理 24.5 即得結論。M-判準可用來驗證導數級數一致收斂，但它只是充分條件。僅有原級數一致收斂仍不足以逐項微分：在 $[0,\pi]$ 上，令 $u_1(x)=\sin x$，對 $n\ge2$ 令 $u_n(x)=\sin(nx)/n-\sin((n-1)x)/(n-1)$。原級數的部分和為 $\sin(Nx)/N$，一致趨於零；導數部分和卻為 $\cos(Nx)$，在 $x=\pi$ 不收斂。
<<<NEW>>>
**函數級數逐項微分定理** 設 $a<b$。若 $u_n\in C^1([a,b])$，某點 $c\in[a,b]$ 上的數值級數 $\sum_{n=1}^{\infty}u_n(c)$ 收斂，且導數級數 $\sum_{n=1}^{\infty}u_n'$ 在 $[a,b]$ 上一致收斂，則原級數在 $[a,b]$ 上一致收斂；其和函數 $S$ 在 $(a,b)$ 上可微，且
$$ S'(x)=\sum_{n=1}^{\infty}u_n'(x). $$
證明：令部分和 $S_N=\sum_{n=1}^N u_n$。各 $S_N$ 屬於 $C^1([a,b])$，$S_N(c)$ 收斂，而 $S_N'=\sum_{n=1}^N u_n'$ 一致收斂。對函數列 $S_N$ 套用定理 24.5 即得結論。M-判準可用來驗證導數級數一致收斂，但它只是充分條件。僅有原級數一致收斂仍不足以逐項微分：在 $[0,\pi]$ 上，令 $u_1(x)=\sin x$，對 $n\ge2$ 令 $u_n(x)=\sin(nx)/n-\sin((n-1)x)/(n-1)$。原級數的部分和為 $\sin(Nx)/N$，一致趨於零；導數級數的部分和為 $\cos(Nx)$，在 $x=\pi$ 等於 $(-1)^N$，故不收斂。
<<<END>>>
<<<PATCH 24>>>
<<<OLD>>>
$$ \le \sup_{\substack{x \in [a,b] \\ |s-t| \le |h|}} \left| \frac{\partial f}{\partial t}(x,s) - \frac{\partial f}{\partial t}(x,t) \right| $$
由於 $\frac{\partial f}{\partial t}$ 在緊集 $R$ 上連續，故一致連續。當 $h \to 0$ 時，$\sup$ 範圍縮小至 $s \approx t$，上述 supremum 趨近於 0。
<<<NEW>>>
$$ \le \sup_{\substack{x\in[a,b]\\s\text{ 介於 }t\text{ 與 }t+h\text{ 之間}}}
\left|\frac{\partial f}{\partial t}(x,s)-\frac{\partial f}{\partial t}(x,t)\right| $$
此處只取 $h\ne0$ 且 $t+h\in[c,d]$ 的差商，因此上確界中的 $s$ 均在定義域內。由於 $\frac{\partial f}{\partial t}$ 在緊集 $R$ 上連續，故一致連續；當 $h\to0$ 時，上述上確界趨近於 $0$。在 $t=c,d$ 處，分別只取域內的右側、左側極限。
<<<END>>>