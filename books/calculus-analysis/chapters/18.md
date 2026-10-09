# 第18章 不等式約束與KKT橋接

## 學習目標與先備知識

完成本章後，讀者應能以一致的符號寫出不等式約束的 Karush–Kuhn–Tucker（KKT）條件，辨認哪些條件是局部最小值的**必要條件**、哪些足以保證**全域最小值**，並用主動集枚舉核對二維盒約束問題。先備知識包括梯度、凸函數、等式約束的 Lagrange 乘數，以及矩陣秩。這裡梯度一律是直向的列向量；標量函數 $f:\mathbb R^n\to\mathbb R$ 的導數 $Df(x)$ 則是 $1\times n$ 線性泛函，$Df(x)[d]=\nabla f(x)^Td$。

本章的判斷順序是：先查可行性，再查局部最優所需的約束資格，最後判斷凸性是否允許把 KKT 證書提升為全域證書。計算只能核對特定候選點；有限個候選點或浮點輸出，不能自行證明一般定理。

## 問題與直覺

設要最小化 $f(x)$，同時要求 $g_i(x)\leq0$。在可行域內，若某個約束嚴格小於零，它在該點尚有餘裕，通常不阻擋足夠小的移動；若等於零，該約束便稱為**主動**。在光滑邊界的局部最小值，目標函數的下降方向可能被主動約束擋住。乘數描述這種阻擋如何在梯度方程中平衡，而不直接表示現實世界的力。

符號尤其重要。採用 $g_i\leq0$、最小化 $f$ 與 $L=f+\sum_i\lambda_i g_i$ 時，乘數必須非負。若把約束改寫成相反方向卻不更換乘數規則，就可能把最大值誤判為最小值。

以盒 $a\leq x\leq b$ 為例，下界應寫成 $a-x\leq0$，上界寫成 $x-b\leq0$。盒角落可能同時有兩個主動面；但「主動」不等於「乘數嚴格為正」：主動約束的乘數也可以是零。

## 定義、定理與推導

**定義：可行域與 KKT 條件。** 令 $U\subset\mathbb R^n$ 為開集，$f,g_1,\ldots,g_p:U\to\mathbb R$ 為連續可微函數，並令
$$
F=\{x\in U:g_i(x)\leq0,\ i=1,\ldots,p\},\qquad
L(x,\lambda)=f(x)+\sum_{i=1}^p\lambda_i g_i(x).
$$
在候選點 $x^*$，KKT 條件為
$$
\begin{aligned}
&g_i(x^*)\leq0 &&\text{原始可行性},\\
&\lambda_i\geq0 &&\text{對偶可行性},\\
&\lambda_i g_i(x^*)=0 &&\text{互補鬆弛},\\
&\nabla f(x^*)+J_g(x^*)^T\lambda=0 &&\text{站立條件}.
\end{aligned}
$$
此處 $J_g$ 是 $p\times n$ 矩陣，$\lambda$ 是 $p\times1$ 列向量；各維度因而相容。互補鬆弛表示非主動約束的乘數必為零，反向推論則不成立。

**定義：約束資格。** 在 $x^*$，令主動指標集合為 $A(x^*)=\{i:g_i(x^*)=0\}$。若主動梯度 $\{\nabla g_i(x^*):i\in A(x^*)\}$ 線性獨立，稱滿足 LICQ。沒有主動約束時，LICQ 視為成立。LICQ 是保證局部最小值具有 KKT 乘數的一種**充分約束資格**，並非 KKT 成立的必要條件。

**定理：非凸問題的局部必要條件。** 若 $x^*\in F$ 是局部最小值，$f,g_i$ 在其開鄰域連續可微，且 $x^*$ 滿足 LICQ，則存在乘數 $\lambda$ 使 KKT 條件成立。這是必要條件，並未聲稱該點為全域最小值；此定理在此引用，不以有限方向取樣替代證明。

LICQ 並非唯一的約束資格。例如，對只有不等式的問題，另一種條件是存在方向 $d\in\mathbb R^n$，使每條主動約束都滿足 $\nabla g_i(x^*)^Td<0$；這稱為 MFCQ 的純不等式形式。它要求有一個一階方向能同時走入所有主動邊界的內側，卻不要求主動梯度彼此獨立。在連續可微與局部最小的假設下，MFCQ 也足以保障 KKT 乘數存在；此結果同樣在此引用。例如把同一條盒上界重複列出兩次，兩個相同的主動梯度使 LICQ 失效，但仍可能存在共同向內的方向。相反，若主動梯度全為零，便不能憑「該點可行」推定乘數存在。不同約束資格提供不同的**充分途徑**；未通過某一種資格測試，並不能直接判定點不是最小值。

**定義：Slater 條件。** 對凸最佳化問題，若可行域另含仿射等式 $Ax=b$，本章採用一個簡便的充分版本：存在滿足 $Ax=b$、每一個不等式均嚴格滿足 $g_i(x)<0$，且位於相關定義域相對內部的點。對仿射不等式，常見的較一般版本允許適當放寬嚴格性；本章不需要該擴充。純盒問題若每段區間都有正長度，其內點即提供嚴格可行點。Slater 不能拿來使非凸問題的 KKT 駐點自動變成全域最優。

**小命題：凸問題的 KKT 全域充分性。** 設 $C\subset\mathbb R^n$ 為凸集，$f,g_1,\ldots,g_p$ 在包含 $C$ 的開鄰域可微且於 $C$ 凸。若 $x^*\in C$ 與 $\lambda\geq0$ 滿足上述 KKT 條件，則 $x^*$ 在 $F\cap C$ 上為全域最小值。本命題只主張充分性；其證明不需要 Slater。

**證明。** 任取 $y\in F\cap C$。凸函數的一階不等式給出
$$
f(y)\geq f(x^*)+\nabla f(x^*)^T(y-x^*),
$$
以及對每個 $i$，
$$
g_i(y)\geq g_i(x^*)+\nabla g_i(x^*)^T(y-x^*).
$$
以 $\lambda_i\geq0$ 乘第二組不等式並求和，再利用站立條件，可得
$$
f(y)-f(x^*)\geq
-\sum_i\lambda_i\nabla g_i(x^*)^T(y-x^*)
\geq
\sum_i\lambda_i\bigl(g_i(x^*)-g_i(y)\bigr).
$$
互補鬆弛使右端第一部分 $\sum_i\lambda_i g_i(x^*)=0$；又因 $g_i(y)\leq0$、$\lambda_i\geq0$，所以剩餘的 $-\sum_i\lambda_i g_i(y)\geq0$。故 $f(y)\geq f(x^*)$。由於 $y$ 任意，命題得證。$\square$

在標準的可微凸問題中，適當的 Slater 條件可用來保證最優解具有乘數，使 KKT 也成為**必要**條件；其乘數存在定理在此引用。須分清兩條邏輯：凸性加 KKT 給全域充分性；凸性加約束資格協助由最優性取得 KKT。單有 Slater 而無凸性，沒有前一項保證。

盒約束還提供直接的分量判斷。對 $a_j\leq x_j\leq b_j$，設下界乘數 $\alpha_j\geq0$、上界乘數 $\beta_j\geq0$，則
$$
\partial_j f(x)-\alpha_j+\beta_j=0,\qquad
\alpha_j(a_j-x_j)=\beta_j(x_j-b_j)=0.
$$
因此在下界且上界未啟動時，$\partial_j f(x)=\alpha_j\geq0$；在上界時，$\partial_j f(x)=-\beta_j\leq0$；在內部時，$\partial_j f(x)=0$。這些是**候選點的 KKT 符號測試**，不是對任意函數的全域最小判別。

## 逐步手算例題

**例一：有一個主動邊界的凸二次函數。** 在 $0\leq x,y\leq1$ 上最小化
$$
f(x,y)=(x-2)^2+(y-\tfrac14)^2.
$$
無約束駐點 $(2,\tfrac14)$ 不可行。沿 $x$，當 $0\leq x\leq1$ 時有 $2(x-2)<0$，故增大 $x$ 會降低目標，最優點須取 $x=1$；沿 $y$，平方項在 $y=\tfrac14$ 最小。因此 $x^*=(1,\tfrac14)$，$f(x^*)=1$。

按下界 $-x,-y$ 與上界 $x-1,y-1$ 的順序排列約束。此點只有 $x-1=0$，且 $\nabla f(x^*)=(-2,0)^T$、$\nabla(x-1)=(1,0)^T$。站立條件給上界乘數 $2$，其他乘數為零。四種 KKT 條件均成立。目標及仿射約束皆凸，故上面的小命題另行證明它是全域最小值；這一步不是僅靠手算枚舉的猜測。

**例二：角落與主動零乘數。** 在同一盒上最小化
$$
q(x,y)=x^2+y.
$$
因 $x^2\geq0$、$y\geq0$，手算得唯一最小點 $(0,0)$，值為零。其梯度為 $(0,1)^T$，兩條下界 $-x\leq0$、$-y\leq0$ 均主動。站立條件
$$
(0,1)^T-\alpha_x(1,0)^T-\alpha_y(0,1)^T=0
$$
給 $\alpha_x=0$、$\alpha_y=1$。所以「主動就必有正乘數」是錯的；仍須保留零乘數與主動約束的區別。此例的全域結論也可由凸性和 KKT 取得。

## 實作與程式

以下程式對二維盒的三種座標狀態——下界、內部、上界——逐一組合。若某座標被固定在邊界，便解另一座標的無約束站立方程；若兩者均在內部，便解二維站立方程。它只適用於程式所寫的**可分離二次目標**，不是一般非線性 KKT 求解器。程式使用 Python 標準庫，CPU 即可執行；本章未執行它。

每個座標有三種標記，所以總共檢視 $3^2=9$ 種狀態組合。這個數目不是九個互異的可行點：若偏好中心位於盒外，某些「內部」解會先被捨棄；若偏好中心恰好等於端點，不同標記還可能指向同一幾何位置。程式保留通過位置篩選的紀錄，分別檢查原始可行、乘數符號、互補和站立，最後才選出通過 KKT 的紀錄。這種分欄記錄比僅輸出一個最低函數值更容易追查符號錯誤。

```python
from itertools import product
from math import isfinite

def enumerate_box(center, weight, lo=(0.0, 0.0), hi=(1.0, 1.0)):
    """Minimize sum_j weight[j] * (x[j] - center[j])**2 on a 2D box.
    Reports feasible stationary active-set candidates and KKT checks.
    """
    if not all(len(v) == 2 for v in (center, weight, lo, hi)):
        raise ValueError("all inputs must have length two")
    try:
        finite = all(isfinite(v)
                     for seq in (center, weight, lo, hi) for v in seq)
    except TypeError as exc:
        raise ValueError("all entries must be finite numbers") from exc
    if not finite:
        raise ValueError("all entries must be finite numbers")
    if any(lo[j] >= hi[j] for j in range(2)):
        raise ValueError("each box interval must have positive length")
    if any(weight[j] <= 0 for j in range(2)):
        raise ValueError("weights must be positive")

    tol = 1e-12
    records = []
    for status in product(("lower", "free", "upper"), repeat=2):
        x = tuple(
            lo[j] if status[j] == "lower" else
            hi[j] if status[j] == "upper" else center[j]
            for j in range(2)
        )
        if any(x[j] < lo[j] - tol or x[j] > hi[j] + tol
               for j in range(2)):
            continue

        grad = tuple(2.0 * weight[j] * (x[j] - center[j])
                     for j in range(2))
        lower = tuple(grad[j] if status[j] == "lower" else 0.0
                      for j in range(2))
        upper = tuple(-grad[j] if status[j] == "upper" else 0.0
                      for j in range(2))
        primal = all(lo[j] - x[j] <= tol and x[j] - hi[j] <= tol
                     for j in range(2))
        dual = all(v >= -tol for v in lower + upper)
        comp = all(abs(lower[j] * (lo[j] - x[j])) <= tol and
                   abs(upper[j] * (x[j] - hi[j])) <= tol
                   for j in range(2))
        residual = max(abs(grad[j] - lower[j] + upper[j])
                       for j in range(2))
        stationarity = residual <= tol
        kkt = primal and dual and comp and stationarity
        value = sum(weight[j] * (x[j] - center[j]) ** 2
                    for j in range(2))
        records.append({
            "status": status, "x": x, "value": value,
            "lower": lower, "upper": upper,
            "primal": primal, "dual": dual,
            "complementarity": comp,
            "stationarity_residual": residual, "kkt": kkt
        })
    return records

def best_kkt(records):
    valid = [r for r in records if r["kkt"]]
    if not valid:
        raise ValueError("no KKT candidate found")
    return min(valid, key=lambda r: r["value"])

if __name__ == "__main__":
    cases = (
        ("normal", (2.0, 0.25), (1.0, 1.0)),
        ("boundary", (0.0, -1.0), (1.0, 0.5)),
    )
    for name, center, weight in cases:
        records = enumerate_box(center, weight)
        print(name, "checked", len(records), "best", best_kkt(records))
    try:
        enumerate_box((2.0, 0.25), (1.0, 0.0))
    except ValueError as exc:
        print("failure:", exc)
```

用 $w_j>0$ 是為了讓每一個「free」座標的站立解確實為 `center[j]`，並確保嚴格凸性。容差 `1e-12` 是本例量級約為一時的示範性判斷門檻，並非通用的誤差保證；若把資料放大許多倍，固定的絕對容差可能不合適。程式以容差作浮點**核對**，不能由此推得精確等式；接近容差時應回到解析式，或依輸入尺度、計算誤差另訂檢查準則。枚舉對指定目標奏效的數學原因是可分離與正權重，而不是因為電腦「試遍了」連續盒中的所有點。

## 測試與預期結果

**正常測試，未執行的預期：** `normal` 對應例一，最好的 KKT 紀錄為 $x=(1,0.25)$、目標值 $1$、$x$ 上界乘數 $2$。被保留但對偶乘數為負的邊界紀錄可能不是 KKT 點；不能只看其站立殘差。

**邊界測試，未執行的預期：** `boundary` 的目標為 $x^2+\tfrac12(y+1)^2$，最好的 KKT 紀錄在 $(0,0)$，值為 $0.5$；兩個下界主動，其中 $x$ 下界乘數為零、$y$ 下界乘數為 $1$。這也檢查了「主動但零乘數」情形。

**故障測試，未執行的預期：** 零權重觸發 `ValueError`，因為「free 座標必等於 center」的枚舉推導不再由嚴格凸二次項保證。顛倒的盒端點、非有限輸入與錯誤長度亦會被拒絕。拒絕輸入是程式適用範圍的聲明，不表示一般零權重問題沒有最小值。

## 反例與常見陷阱

首先，KKT 不保證非凸問題的最小性。在 $-1\leq x\leq1$ 最小化 $f(x)=-x^2$，內點 $x=0$ 的梯度為零、兩個盒約束均不主動，故它符合 KKT，卻是局部最大點；全域最小值 $-1$ 出現在兩端。於是「駐點一概是全域最小」即使加上盒約束也不成立。

其次，局部最小值可能因約束資格失效而沒有 KKT 乘數。考慮最小化 $f(x)=x$，要求 $g(x)=x^2\leq0$。可行域只有 $\{0\}$，故零點是最小值；但 $g'(0)=0$、$f'(0)=1$，站立方程 $1+\lambda\cdot0=0$ 無解。此處 LICQ 失效，也沒有嚴格可行點。不能把「沒有乘數」直接解讀為「不是最小值」。

最後，從等式約束搬來的切空間二階判據不可原封套用。對不等式，邊界上可行的微小移動通常是單側的；主動梯度所定的可行方向與臨界方向需分別處理。光滑非凸問題若需判別二階局部最小性，還要考慮乘數、臨界錐及額外正則性；本章不以簡單的 $Z^TH_LZ$ 取代該理論。

## AI、幾何與養殖案例

設有兩個**合成**且已無因次化的校準參數 $x,y$，規格各限制在 $[0,1]$。模型給出偏好的無因次位置 $(2,0.25)$，並以平方偏差 $f=(x-2)^2+(y-0.25)^2$ 評分。例一的 KKT 證書表明：在這個明確寫出的凸模型及盒內，最佳解為 $(1,0.25)$。從幾何看，無約束最低點落在盒外；朝它前進的方向在 $x=1$ 被上界阻擋。乘數 $2$ 是按目前無因次目標與約束尺度計算的量，重縮放目標或約束會改變其數值，不宜直接稱為設備的物理壓力。

若原始參數是有單位的感測量，應先指定各自的 SI 單位、基準值及尺度，再用如 $x=(p-p_0)/s_p$、$y=(q-q_0)/s_q$ 轉為無因次參數；返回物理值時使用 $p=p_0+s_px$、$q=q_0+s_qy$。例如可明定 $p$ 以攝氏溫差對應的開爾文差值計、$q$ 以公尺計，並分別選用帶相同單位的正尺度 $s_p,s_q$；實際尺度必須由案例規格提供，不能由本章的抽象盒推測。沒有這些尺度，混合不同單位的平方偏差缺乏可解釋的物理權重。盒內最優僅針對合成評分函數和指定有效範圍，既非現場驗證，也非操作安全證明。AI agent 在此只能唯讀整理模型、界限、乘數與失效條件，不據此控制設備、投餌或加藥。

## 習題

1. **手算。** 在 $0\leq x,y\leq1$ 最小化 $f=(x+\tfrac12)^2+(y-\tfrac34)^2$。求最小點、值及依下界 $-x,-y$、上界 $x-1,y-1$ 排列的乘數，說明全域性的依據。
2. **程式。** 不修改核心函式，給習題一設定 `center` 與 `weight`，寫出呼叫式及未執行時預期的 `best_kkt` 結果。若把第一個權重改為零，預期會怎樣？
3. **反例。** 在盒 $[-1,1]$ 上，給出一個滿足 KKT 卻不是最小值的光滑函數；再給出局部最小卻無 KKT 乘數的單一不等式問題，指出失效條件。
4. **整合。** 對可微凸 $f$ 與可微凸 $g_i\leq0$，已知某可行點及非負乘數滿足站立與互補。某人說：「還須先數值找到 Slater 點，才能斷言此 KKT 點全域最優。」判斷此話；並說明若要從最優性反推乘數，Slater 扮演何種角色。

## 習題解答

1. 無約束最低點為 $(-\tfrac12,\tfrac34)$，投向盒內得 $(0,\tfrac34)$，函數值 $\tfrac14$。該點梯度為 $(1,0)^T$；$x$ 下界乘數為 $1$，其餘三個為零。原始與對偶可行、站立、互補均成立。函數與盒的仿射約束凸，依已證小命題，此點為全域最小值。
2. 呼叫式為 `best_kkt(enumerate_box((-0.5, 0.75), (1.0, 1.0)))`。未執行的預期為點 `(0.0, 0.75)`、值 `0.25`、`lower` 為 `(1.0, 0.0)`、`upper` 為 `(0.0, 0.0)`，且 `kkt` 為 `True`。把第一個權重改成 `0.0` 會觸發 `ValueError`；這是函式的前置條件，不是該新問題不可解。
3. 可取 $f(x)=-x^2$，兩個盒約束寫成 $-1-x\leq0$ 與 $x-1\leq0$。$x=0$ 可行、梯度零、乘數皆零，所以符合 KKT，但 $f(0)=0>-1=f(1)$。另一例為最小化 $x$ 且 $x^2\leq0$：唯一可行點零為最小值，站立方程卻是 $1=0$；主動約束梯度為零，LICQ 失效。
4. 該說法錯誤。已取得滿足條件的 KKT 點時，凸性加一階不等式便足以證全域最優，不需要先找 Slater 點，更不需要以數值搜尋代替證明。反方向若已知最優、想保證乘數存在，Slater 是凸問題常用的充分約束資格；宣稱它成立時仍須有符合嚴格不等式及定義域條件的數學依據。

## 本章小結

KKT 同時檢查原始可行、非負乘數、互補與站立。LICQ 等資格支撐非凸局部最小值的**必要性**；凸性使一個已成立的 KKT 證書具有**全域充分性**；凸問題中的 Slater 則常用於保障從最優解取得乘數。三種用途不能混寫。主動集枚舉能稽核指定二次盒問題，但浮點核對和有限候選輸出不是一般最佳化定理。

## 參考來源

- [A1] Jiří Lebl，*Basic Analysis*，作者教材入口：https://www.jirka.org/ra/
- [A2] MIT OCW，*18.100A Real Analysis*：https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/
- [A3] MIT OCW，*18.02SC Multivariable Calculus*：https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/
- [A6] SciPy，`minimize` 方法與參數文件，供延伸比較最佳化器介面；本章程式未使用 SciPy：https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

以上來源作為延伸入口，並不表示已逐條核對其完整教材或已執行任何外部程式。