# 第13章 Poisson問題與零空間

## 學習目標與先備知識

讀完本章，讀者應能：

1. 說明為何 Poisson 方程常寫成 $-\Delta u=f$，並從邊界條件判斷解是否唯一。
2. 推導純 Neumann 問題的相容條件，辨認常數零模態；遇到不相容資料時拒絕求解，而不是任意固定一點。
3. 在 cell-centered 網格組裝 $A=-L$，使用體積加權均值約束求解相容問題，並檢查真殘差。
4. 分開判斷線性求解、離散近似及物理模型是否可信。

先備知識是控制體通量、中心差分與矩陣乘法。本章的場 $u$ 可代表濃度或一個待求的位勢；**相同的 Poisson 算子不表示相同的物理定律**。以下一維有因次擴散例取 $x$ 單位為 m、濃度 $u$ 為 kg/m³、擴散係數 $D$ 為 m²/s，穩態來源 $s$ 為 kg/(m³·s)。若寫 $-D u''=s$，兩端單位相同。為集中討論零空間，手算與程式另取 $D=1$ 的**無因次**方程；其參考長度、濃度與來源尺度分別為 $L_0,U_0,DU_0/L_0^2$，並不把數值 1 當作池水物性。

本章只研究橢圓邊界值問題；不把迭代器收斂誤認為時間演化穩定，也不聲稱已驗證任何現場模型。

## 問題與直覺

考慮連通區間 $\Omega=(0,1)$ 上的

$$
-u''=f.
$$

指定兩端的 $u$ 值是 **Dirichlet 條件**：整條曲線被端點錨定。指定兩端的外法向導數 $\partial_nu$ 則是 **Neumann 條件**：只規定斜率，將曲線整體加上一個常數不改變斜率，也不改變二階導數。因此純 Neumann 問題即使可解，也不能僅靠原方程決定絕對高度。

這個自由度不是程式「精度不足」，而是算子的零空間。若 $L$ 近似 $\Delta$，週期或齊次 Neumann 邊界下 $L\mathbf1=0$，且 $L$ 為負半定；$A=-L$ 為正半定，仍有 $A\mathbf1=0$。任意把某個格點設為零，可能替換掉原來的一條方程；當資料本來不相容時，這只會藏起守恆矛盾。

![算子與邊界示意](../figures/operators.svg)

## 數學與物理推導

### 邊界、弱恆等式與唯一性

令 $\Omega$ 是有界、連通且邊界足夠規則的區域，$\mathbf n$ 為外法向。對 $-\Delta u=f$ 積分，散度定理給出

$$
-\int_{\partial\Omega}\partial_nu\,dS
=\int_\Omega f\,dV.
$$

所以若純 Neumann 資料寫作 $\partial_nu=g$，必要相容條件是

$$
\boxed{\int_\Omega f\,dV+\int_{\partial\Omega}g\,dS=0.}
$$

一維時，左端外法向為 $-1$，右端為 $+1$；因而 $g(0)=-u'(0)$、$g(1)=u'(1)$，不可把兩端的 $u'$ 都直接當成向外通量。週期邊界的成對邊界通量互相抵消，故要求 $\int_\Omega f=0$。

將兩個具有相同資料的解相減，得 $-\Delta w=0$。乘上 $w$、積分並分部積分：

$$
\int_\Omega|\nabla w|^2\,dV
=\int_{\partial\Omega}w\,\partial_nw\,dS.
$$

若兩解在整個邊界具有相同 Dirichlet 值，$w=0$，右端為零，故 $\nabla w=0$；連通性再配合邊界值給出 $w=0$，解唯一。若兩解具有相同的純 Neumann 資料，$\partial_nw=0$，只得到 $w$ 是常數。指定

$$
\frac{1}{|\Omega|}\int_\Omega u\,dV=m
$$

即可選出唯一代表；通常取 $m=0$。若區域有多個互不連通的分量，各分量各有一個常數自由度，亦各需相容條件與均值約束。

相容是**存在解的必要條件**，均值約束是**可解後選擇唯一代表**的條件，兩者不能互換。對規則區域與適當函數空間，相容條件加上均值約束可得到弱解；本章不把這項分析結論擴張到任意破碎邊界或任意來源資料。

### cell-centered 離散與矩陣零空間

令一維有 $N$ 個 cell，$h=1/N$，中心 $x_i=(i+\tfrac12)h$。兩個相鄰 cell 共用一面，以差分近似面上的梯度。內點有

$$
(Au)_i=\frac{2u_i-u_{i-1}-u_{i+1}}{h^2}.
$$

齊次 Neumann 邊界的外面梯度為零，因此首尾行分別是

$$
(Au)_0=\frac{u_0-u_1}{h^2},\qquad
(Au)_{N-1}=\frac{u_{N-1}-u_{N-2}}{h^2}.
$$

所有行的係數和皆為零。更有

$$
h\,\mathbf1^{\mathsf T}Au=0,\qquad
h\,u^{\mathsf T}Au
=\frac{1}{h}\sum_{i=0}^{N-2}(u_{i+1}-u_i)^2\geq0.
$$

其中 $h\,\mathbf1^{\mathsf T}u$ 是 cell 平均乘長度的積分近似；在二維，每格應乘面積 $dx\,dy$（每單位厚度），不可把 ghost 值算作實際質量。離散相容條件為 $h\sum_i f_i=0$；若各格體積不同，須改用相應體積權重，不能只做未加權平均。

齊次 Dirichlet 邊界也能在 cell-centered 網格處理，但邊界位於 cell 中心之外半格。以西側為例，面梯度為 $(u_0-u_{\rm W})/(h/2)$，因此固定邊界值對第一行係數及右端都有貢獻。不能直接把上面的 Neumann 首行沿用，再聲稱是 Dirichlet 問題。Dirichlet 系統在此網格上正定；純 Neumann 系統只在零均值子空間正定。

對齊次 Neumann，可先檢查 $f$ 相容，再在零均值子空間解 $Au=f$。為示範小矩陣，也可解增廣系統

$$
\begin{pmatrix}A&\mathbf1\\
\mathbf1^{\mathsf T}&0\end{pmatrix}
\begin{pmatrix}u\\\lambda\end{pmatrix}
=
\begin{pmatrix}f\\0\end{pmatrix}.
$$

在均勻格網上，末行要求 $\sum_i u_i=0$。若 $f$ 已相容，理論上 $\lambda=0$；若不相容，增廣系統仍可能產生一個非零 $\lambda$，實際上解了 $Au=f-\lambda\mathbf1$，**不是原問題**。所以必須先拒絕不相容右端，而不能以增廣系統能回傳數值當作成功。

## 逐步手算例題

### 例一：Dirichlet 解的唯一性與來源檢查

在無因次區間取 $-u''=2$，$u(0)=u(1)=0$。

1. 積分兩次：$u''=-2$，故 $u(x)=-x^2+C_1x+C_2$。
2. 左端邊界給 $C_2=0$；右端給 $C_1=1$。
3. 得 $u(x)=x(1-x)$，在 $x=\tfrac12$ 為 $\tfrac14$。
4. 驗算：$-u''=2$；兩端的函數值皆為零。

若改成 $u(0)=u(1)=1$，答案是 $1+x(1-x)$。來源相同而邊界不同，場值便不同；單看來源圖像不能決定解。兩組 Dirichlet 資料各自指定一個唯一解。

### 例二：三格純 Neumann 問題

取 $N=3$、$h=1/3$，左右面皆為零法向導數。則

$$
A=9\begin{pmatrix}
1&-1&0\\
-1&2&-1\\
0&-1&1
\end{pmatrix},\qquad
f=\begin{pmatrix}-9\\0\\9\end{pmatrix}.
$$

1. 相容檢查：$h\sum_i f_i=(-9+0+9)/3=0$。
2. 首行給 $u_0-u_1=-1$；末行給 $u_2-u_1=1$。
3. 零均值條件給 $u_0+u_1+u_2=0$。
4. 因此 $u=(-1,0,1)^{\mathsf T}$；代回中間一行為 $9(1+0-1)=0$。
5. 對任意常數 $c$，$u+c\mathbf1$ 仍滿足原方程與 Neumann 條件，只有上述零均值代表符合附加約束。

現在只把中間來源改為 $1$，令 $\tilde f=(-9,1,9)^{\mathsf T}$。其積分為 $1/3$，但 $h\mathbf1^{\mathsf T}Au$ 永遠是零，故不存在滿足 $Au=\tilde f$ 的解。固定 $u_0=0$ 不會消除這個矛盾；它至多使被替換的那一行不再受檢查。

## 實作與程式

以下程式僅依賴 Python 3.10+ 與 NumPy，在 CPU 上組裝小型一維矩陣。輸入是**無因次** cell 來源；初始猜測為零，穩態問題不需要時間初值。外面邊界為齊次 Neumann，輸出固定為零均值。對一般非零邊界通量，應先將面通量貢獻納入各邊界 cell 的右端，再依新的收支檢查相容；不能直接沿用本函式。

```python
import numpy as np


def neumann_matrix(n, length=1.0):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    if n < 2 or not np.isfinite(length) or length <= 0:
        raise ValueError("need n >= 2 and finite positive length")
    h = length / n
    A = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        A[i, i] += 1.0 / h**2
        A[i + 1, i + 1] += 1.0 / h**2
        A[i, i + 1] -= 1.0 / h**2
        A[i + 1, i] -= 1.0 / h**2
    return A, h


def solve_zero_mean(f, length=1.0, atol=1e-11, rtol=1e-11):
    b = np.asarray(f, dtype=float)
    if b.ndim != 1 or b.size < 2 or not np.all(np.isfinite(b)):
        raise ValueError("f must be a finite one-dimensional array")
    if not np.isfinite(atol) or not np.isfinite(rtol):
        raise ValueError("tolerances must be finite")
    if atol < 0 or rtol < 0:
        raise ValueError("tolerances must be nonnegative")

    A, h = neumann_matrix(b.size, length)
    # 相容容差作用於「積分收支」，與線性真殘差分開。
    imbalance = abs(h * np.sum(b))
    balance_scale = h * np.sum(np.abs(b))
    if imbalance > atol + rtol * balance_scale:
        raise ValueError("incompatible Neumann source")

    n = b.size
    K = np.zeros((n + 1, n + 1), dtype=float)
    K[:n, :n] = A
    K[:n, n] = 1.0
    K[n, :n] = 1.0
    rhs = np.r_[b, 0.0]
    answer = np.linalg.solve(K, rhs)
    u, multiplier = answer[:n], answer[n]
    true_residual = b - A @ u
    residual_norm = np.linalg.norm(true_residual)
    bound = atol + rtol * np.linalg.norm(b)
    if residual_norm > bound or abs(h * np.sum(u)) > bound:
        raise ArithmeticError("original equation or mean check failed")
    return u, multiplier, residual_norm


if __name__ == "__main__":
    A, h = neumann_matrix(3)
    u, lam, residual = solve_zero_mean([-9.0, 0.0, 9.0])
    print("u =", u, "lambda =", lam, "residual =", residual)
    try:
        solve_zero_mean([-9.0, 1.0, 9.0])
    except ValueError as exc:
        print("rejected:", exc)
```

`np.linalg.solve` 在此只是供小矩陣教學用的稠密直接解法，並非大型網格建議。`atol` 與 `rtol` 明確作用於來源積分及真殘差 $\lVert f-Au\rVert_2$；兩種檢查雖使用同名容差，量綱尺度與意義不同。有因次應用宜分別選定收支與代數殘差容差。即使殘差很小，也不能直接斷定連續解誤差很小：誤差還受條件數、網格及模型影響。

## 測試與預期結果

以下均為**依公式推導的預期結果，並非執行紀錄**。

| 類型 | 輸入與檢查 | 預期 |
|---|---|---|
| 正常 | 三格 `[-9, 0, 9]` | $u$ 接近 $[-1,0,1]$，乘子接近零，真殘差符合設定容差 |
| 邊界／零模態 | `A @ np.ones(3)`、`A @ (u + 7)` | 前者接近零；後者與 `A @ u` 相同，但 $u+7$ 不符合零均值 |
| 故障：不相容 | `[-9, 1, 9]` | 在求解前拋出 `ValueError` |
| 故障：非法資料 | 含 `nan`、`inf`，或錯誤形狀 | 拋出 `ValueError`，不輸出看似有效的場 |
| 製造解 | 設 $u_i=x_i-\tfrac12$，令 $f=A u$ | 回求得到同一零均值離散場；這是組裝自洽檢查，**不是**連續誤差階驗證 |

若要檢驗連續精度，必須另選解析 $u(x)$，從連續方程製作 $f=-u''$ 及一致的邊界資料，於多個 $h$ 上比較帶權誤差。用 $f=Au$ 回求只能證明代數流程一致；只看圖平滑也不能證明收斂。此處靜態 Poisson 問題沒有時間步長，故「時間積分穩定」並非測項。零均值場可有負值，除非所建物理量及其邊界、來源滿足額外最大值原理條件，不能因其為負就裁零；裁零又會破壞方程及收支。

## 除錯與常見陷阱

**符號顛倒。** 本卷約定 $L\approx\Delta$，所以 $A=-L$。若無意中用 $L$ 當正定矩陣，能量檢查 $u^{\mathsf T}Au\geq0$ 便會失敗。可先檢查常數場與單一相鄰差值，再交給求解器。

**把 pin 點當成相容條件。** 固定 $u_0=0$ 只選擇參考值；如果來源總和不為零，原方程仍相互矛盾。正確順序是先核對來源與邊界通量收支，再決定用均值或合法的參考值消除零模態。對相容資料，pin 點與零均值一般給出相差常數的解；兩者不可直接逐點比較。

**忽略邊界面方向。** 在擴散模型中，物理通量為 $\mathbf J=-D\nabla u$，而數學 Neumann 資料可定義為 $\partial_nu$；兩者差一個 $-D$。西／南面的外法向又指向負座標方向。先寫清資料究竟是梯度、向外物理通量，還是向內供給，才能檢查積分式的正負號。

**只看求解器訊息。** 奇異矩陣上的某種數值方法即使回報收斂，仍須檢查原始方程的 $r=f-Au$、均值以及來源相容性。小真殘差是代數品質，不保證模型來源有物理根據；解唯一也不保證解非負。

**誤用節點與 cell 索引。** 本章的 $u_i$ 位於 $(i+\tfrac12)h$；端點 $0,1$ 是邊界面，不是首尾未知數。若改用 node-centered 網格，節點與邊界行、積分權重都需重新定義。在二維，本卷採 $q[j,i]$、形狀 $(N_y,N_x)$，$i$ 沿 $+X$、$j$ 沿 $+Y$，展平 $k=jN_x+i$；影像列的向下方向不能直接當作物理 $+Y$。

## 養殖與相場案例

設有**合成**池域的二維每單位厚度穩態濃度模型 $-D\Delta c=s$。若整個邊界都假定零擴散通量，卻在內部處處給正的持續來源，則積分收支要求 $\int_\Omega s\,dA=0$，與設定矛盾。這不是需要更強求解器的問題，而是模型少了出口、消耗項，或根本不應採穩態假設。濃度的 SI 單位為 kg/m³；若資料原為 mg/L，因 $1\ {\rm mg}=10^{-6}\ {\rm kg}$、$1\ {\rm L}=10^{-3}\ {\rm m^3}$，故 $1\ {\rm mg/L}=10^{-3}\ {\rm kg/m^3}$。轉換和來源每秒的單位必須在組裝前完成。不得憑一幅逼真分布圖宣稱已完成現場驗證。

相場中的化學勢也包含 Laplacian，例如本卷無因次約定 $\mu=\phi^3-\phi-\kappa\Delta\phi$。這**不是**說每個相場步驟都能當成獨立 Poisson 方程：非線性項、邊界與演化方程仍須一起處理。不過，週期算子的常數零模態提示我們，往後處理守恆型 Cahn–Hilliard 模型時須檢查均值及來源收支；處理非守恆 Allen–Cahn 模型時則不能照搬「$\phi$ 的總量一定守恆」的結論。養殖管理的溶氧閾值不是物理相變。

## 習題

1. **手算。** 求 $-u''=6x$、$u(0)=u(1)=0$ 的解，並驗算兩端值與來源。
2. **程式。** 對四個均勻 cell 組裝齊次 Neumann 的 $A$。令 $v=(-3,-1,1,3)^{\mathsf T}$、$f=Av$，寫出 $f$，並說明 `solve_zero_mean(f)` 應回傳甚麼。若用 `v+10` 造來源呢？
3. **反例。** 有人聲稱「純 Neumann 不相容時，將第一格未知數固定為零即可得到原問題的解」。以三格 $f=(-9,1,9)^{\mathsf T}$ 反駁，指出被掩蓋的等式。
4. **整合。** 一個長度 $2\ {\rm m}$、截面積 $1\ {\rm m^2}$ 的封閉合成水槽，採穩態一維模型 $-D c''=s$，其中 $D=0.01\ {\rm m^2/s}$，兩端零擴散通量，均勻來源 $s=2\times10^{-6}\ {\rm kg/(m^3\,s)}$。能否得到穩態解？若加入均勻消耗 $q$，取何值才滿足必要相容條件？這是否足以證明所得濃度非負且符合真實池況？

## 習題解答

1. 由 $u''=-6x$ 積分得 $u'=-3x^2+C_1$，再積分得 $u=-x^3+C_1x+C_2$。$u(0)=0$ 給 $C_2=0$；$u(1)=0$ 給 $C_1=1$，故
   $$
   u(x)=x-x^3.
   $$
   代回有 $-u''=6x$，且兩端皆為零。Dirichlet 資料消除了任意加常數的可能。

2. $h=2/4$ 並不適用：本題沿用程式預設長度 $1$，所以 $h=1/4$、$h^{-2}=16$。由相鄰差值依序皆為 $2$，
   $$
   f=16(-2,0,0,2)^{\mathsf T}
    =(-32,0,0,32)^{\mathsf T}.
   $$
   $v$ 的均值是零，因此回傳場應接近 $v$，乘子與真殘差應接近零。$A\mathbf1=0$，故 $A(v+10\mathbf1)=Av=f$；求解器仍應回傳零均值代表 $v$，不是 $v+10\mathbf1$。若自行把區間長度改成 $2$，須相應重算 $h$ 與 $f$，不能混用。

3. 三格齊次 Neumann 矩陣每行相加所形成的恆等式是 $\mathbf1^{\mathsf T}Au=0$。但 $\mathbf1^{\mathsf T}f=-9+1+9=1$，等價地，來源的 cell 積分為 $1/3\ne0$。所以三行不可能同時成立。把第一行改成 $u_0=0$ 後，即使餘下兩行可解，第一行原本要求的通量平衡已被刪掉；所得向量不是 $Au=f$ 的解。

4. 水槽體積為 $2\ {\rm m^3}$。兩端物理擴散通量為零時，穩態積分要求來源淨量為零；原設定的總產生率是 $sV=4\times10^{-6}\ {\rm kg/s}\ne0$，故無穩態解。加入均勻消耗後，淨來源是 $s-q$，必要條件為 $q=s=2\times10^{-6}\ {\rm kg/(m^3\,s)}$。此時 $c''=0$ 且兩端斜率為零，所有常數 $c$ 都滿足方程；仍需一個平均濃度或總量資料才能選出常數。相容性本身既未指定該常數的正負，也未驗證來源、消耗與零通量假設符合真實池況。

## 本章小結

Dirichlet 邊界可固定 Poisson 解；純 Neumann 邊界留下常數零模態，且來源必須先與邊界通量相容。離散矩陣應重現這兩件事：$A\mathbf1=0$、來源的體積加權總和符合收支。均值約束用來選解，不用來修補不相容資料。最後，應分別報告收支、真殘差、離散誤差與物理假設；其中任何一項良好，都不能代替其餘三項。

## 參考來源

- [F1 FiPy：有限體積離散與邊界](https://pages.nist.gov/fipy/en/latest/numerical/discret.html)：控制體面通量及邊界處理的延伸參考。
- [F2 FEniCSx：Poisson 與弱形式](https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html)：分部積分及邊界條件的延伸參考。
- [F3 PETSc：線性系統求解器](https://petsc.org/release/manual/ksp/)：大型系統的求解與診斷參考。
- [F4 SciPy：稀疏線性代數 API](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)：後續稀疏實作的介面參考；本章核心程式不依賴 SciPy。