<<<PATCH 01>>>
<<<OLD>>>
**充分條件**：$C^1$ 且 $\gamma'$ 連續即可使 $\|\gamma'\|$ 可積，弧長有限。**必要條件**方面：$\gamma$ Lipschitz 是弧長有限的必要條件之一，但不是充分條件（Cantor函數弧長無限的變體需另行討論，本章不深入）。
<<<NEW>>>
**充分條件**：$C^1$ 曲線的速度範數連續，在緊區間上可積，故弧長有限。Lipschitz 也是有限弧長的充分條件，但不是必要條件：$\gamma(t)=\sqrt t$ 在 $[0,1]$ 上不是 Lipschitz，弧長仍為 $1$。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
import math

def field(x, y):
<<<NEW>>>
import math
from numbers import Integral, Real

def field(x, y):
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
def check(inner, outer, cells=200, segments=720):
    if not (0 <= inner < outer and cells > 0 and segments >= 3):
        raise ValueError("需要 0 <= inner < outer、cells > 0、segments >= 3")
<<<NEW>>>
def check(inner, outer, cells=200, segments=720):
    if (isinstance(inner, bool) or isinstance(outer, bool)
            or not isinstance(inner, Real) or not isinstance(outer, Real)
            or not math.isfinite(inner) or not math.isfinite(outer)
            or not 0 <= inner < outer):
        raise ValueError("半徑須為有限實數，且 0 <= inner < outer")
    if (isinstance(cells, bool) or isinstance(segments, bool)
            or not isinstance(cells, Integral)
            or not isinstance(segments, Integral)
            or cells <= 0 or segments < 3):
        raise ValueError("cells 須為正整數，segments 須為至少 3 的整數")
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
**定理 24.5 (微分交換)** 設 $\{f_n\}$ 定義在 $[a,b]$ 上，滿足：
<<<NEW>>>
**定理 24.5 (微分交換)** 設 $a<b$，$\{f_n\}$ 定義在 $[a,b]$ 上，滿足：
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
$$ \le \sup_{\substack{x \in [a,b] \\ |s-t| \le |h|}} \left| \frac{\partial f}{\partial t}(x,s) - \frac{\partial f}{\partial t}(x,t) \right| $$
由於 $\frac{\partial f}{\partial t}$ 在緊集 $R$ 上連續，故一致連續。當 $h \to 0$ 時，$\sup$ 範圍縮小至 $s \approx t$，上述 supremum 趨近於 0。
<<<NEW>>>
$$ \le \sup_{\substack{x\in[a,b]\\s\text{ 介於 }t\text{ 與 }t+h\text{ 之間}}}
\left|\frac{\partial f}{\partial t}(x,s)-\frac{\partial f}{\partial t}(x,t)\right|. $$
此處只取滿足 $t+h\in[c,d]$ 且 $h\ne0$ 的差商，故上確界中的 $s$ 均在定義域內。由於 $\frac{\partial f}{\partial t}$ 在緊集 $R$ 上連續，故一致連續；當 $h\to0$ 時，上述上確界趨近於 $0$。在 $t=c,d$ 處，分別只取域內的右側、左側極限。
<<<END>>>