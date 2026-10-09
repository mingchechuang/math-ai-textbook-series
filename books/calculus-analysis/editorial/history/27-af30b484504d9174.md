# 第27章 線性化、局部穩定與Lyapunov橋接

## 學習目標與先備知識

本章研究有限維自治常微分方程

$$
\dot x=f(x),\qquad x(t)\in\mathbb R^n,
$$

在平衡點附近的行為。完成本章後，讀者應能：

1. 區分Lyapunov穩定、局部漸近穩定、局部指數穩定與不穩定。
2. 使用Jacobian線性化非線性系統，並列明定理所需的定義域、正則性與譜條件。
3. 理解Hurwitz矩陣、Lyapunov矩陣方程與二次Lyapunov函數的關係。
4. 說明為何零實部特徵值使線性化判別不定。
5. 辨認非正規矩陣造成的暫態放大。
6. 使用NumPy檢查特徵值、Lyapunov方程殘差與數值軌跡，同時區分證明與數值證據。
7. 在混合物理單位的模型中先無因次化，再解釋矩陣範數與Lyapunov橢球。

先備知識包括Jacobian、特徵值、正定矩陣、矩陣指數、局部存在唯一性與一階Taylor展開。設$U\subset\mathbb R^n$為開集，$f:U\to\mathbb R^n$在平衡點$x_\ast$附近Fréchet可微。令$z=x-x_\ast$，則

$$
\dot z=Az+r(z),\qquad A=J_f(x_\ast),
$$

其中使用Euclidean範數時，

$$
\frac{\|r(z)\|_2}{\|z\|_2}\to0
\qquad
(z\to0,\ z\ne0).
$$

這個餘項條件才是非線性系統與線性化之間的分析橋梁。只算出偏導數或沿有限方向取樣，不能代替Fréchet可微性及其全方向極限。

---

## 問題與直覺

平衡點$x_\ast$滿足$f(x_\ast)=0$。受到小擾動後，可以依序提出三個問題：

- 軌跡是否一直留在平衡點附近？
- 軌跡是否最後回到平衡點？
- 回去的速度是否能被指數函數控制？

線性系統$\dot z=Az$的解為

$$
z(t)=e^{tA}z(0).
$$

若$A$的所有特徵值實部都小於零，則$e^{tA}$長時間衰減。對非線性系統，若$r(z)=o(\|z\|_2)$，非線性餘項在足夠小的鄰域內比線性主項低階。這導向Hurwitz線性化定理。

然而，若$A$有實部為零的特徵值，線性項可能完全看不見第一個有效非線性項。例如

$$
\dot x=-x^3,\qquad \dot x=x^3
$$

在$x=0$的Jacobian都等於零，穩定性卻相反。因此「線性化不定」只表示該判準沒有結論，不表示系統必然穩定、必然不穩定或必然週期運動。

另一個常見誤解是把暫態放大當成不穩定。即使$A$為Hurwitz，Euclidean長度$\|e^{tA}z_0\|_2$仍可能先增加再減少。非正規矩陣的特徵方向可能高度非正交，使不同模態短時間疊加。適當的Lyapunov函數則提供另一種加權幾何，使對應橢球半徑單調下降。

---

## 定義、定理與推導

### 1. 平衡點與解的存在範圍

設$f$在$x_\ast$附近局部Lipschitz，因此每個鄰近初值$x_0$有唯一最大解

$$
x(\,\cdot\,;x_0):[0,T_{\max})\to U.
$$

**定義（Lyapunov穩定）**  
平衡點$x_\ast$稱為穩定，若對每個$\varepsilon>0$，存在$\delta>0$，使每個滿足$\|x_0-x_\ast\|_2<\delta$的解均可向前延續至所有$t\ge0$，且

$$
\|x(t;x_0)-x_\ast\|_2<\varepsilon
\qquad\text{對所有 }t\ge0.
$$

有些教材先在最大存在區間內定義穩定，再由解留在定義域內的緊子集及ODE延拓定理推出$T_{\max}=\infty$。本章直接把向前全時間存在寫入定義，避免有限時間逃逸造成歧義。

**定義（局部漸近穩定）**  
若$x_\ast$穩定，且存在$\rho>0$，使所有滿足$\|x_0-x_\ast\|_2<\rho$的解皆有

$$
x(t;x_0)\to x_\ast\qquad(t\to\infty),
$$

則稱$x_\ast$局部漸近穩定。

**定義（局部指數穩定）**  
若存在$M\ge1$、$\alpha>0$及$\rho>0$，使所有$\|x_0-x_\ast\|_2<\rho$的解均對$t\ge0$存在，並滿足

$$
\|x(t;x_0)-x_\ast\|_2
\le Me^{-\alpha t}\|x_0-x_\ast\|_2,
$$

則稱$x_\ast$局部指數穩定。

局部指數穩定充分推出局部漸近穩定，後者充分推出Lyapunov穩定；反向一般不成立。

### 2. Hurwitz矩陣與線性化間接法

**定義（Hurwitz矩陣）**  
實方陣$A\in\mathbb R^{n\times n}$稱為Hurwitz矩陣，若其每個複特徵值$\lambda$都滿足

$$
\operatorname{Re}\lambda<0.
$$

**定理（線性化間接法）**  
設$U\subset\mathbb R^n$為開集，$f\in C^1(U,\mathbb R^n)$，且$f(x_\ast)=0$。

1. 若$J_f(x_\ast)$為Hurwitz，則$x_\ast$局部指數穩定。
2. 若$J_f(x_\ast)$至少有一個特徵值的實部大於零，則$x_\ast$不穩定。
3. 若沒有正實部特徵值，但至少有一個特徵值實部等於零，則只靠線性化一般不能判定穩定性。

前兩項是充分判準，但不構成所有情況的必要充分分類。第三種情況稱為非雙曲情況，可能需要高階項、中心流形或直接Lyapunov法。這些延伸工具另有條件，不能只寫出名稱便當成證明。

### 3. Lyapunov直接法與鄰域限制

令$D\subset U$為包含$x_\ast$的開鄰域，$V\in C^1(D,\mathbb R)$。沿軌跡的導數為

$$
\dot V(x)=DV(x)[f(x)]
=\nabla V(x)^Tf(x).
$$

此處$DV(x)$是$1\times n$線性泛函，而Euclidean梯度$\nabla V(x)$是$n\times1$列向量。

**定理（局部Lyapunov判準）**  
假設$f$局部Lipschitz，且存在開鄰域$D$與$V\in C^1(D)$，使

$$
V(x_\ast)=0,\qquad V(x)>0\quad(x\ne x_\ast),
$$

以及

$$
\dot V(x)\le0.
$$

則$x_\ast$穩定。若另外有

$$
\dot V(x)<0\qquad(x\ne x_\ast),
$$

則$x_\ast$局部漸近穩定。

上述局部定理包含一個不可省略的緊子水平集論證。先取$r>0$使

$$
\overline B_r(x_\ast)\subset D,
$$

再選$0<s<r$。有限維Heine–Borel定理保證球面

$$
S_s=\{x:\|x-x_\ast\|_2=s\}
$$

為緊集。因$V$連續且在$S_s$上嚴格為正，

$$
m_s=\min_{\|x-x_\ast\|_2=s}V(x)>0.
$$

選擇$0<c<m_s$，並定義閉子水平集

$$
K_c=
\{x\in\overline B_s(x_\ast):V(x)\le c\}.
$$

$K_c$是緊集，且$K_c\subset B_s(x_\ast)\subset D$，因為球面上的$V$至少為$m_s>c$。若初值滿足$V(x_0)<c$，則在解仍存在時，

$$
V(x(t))\le V(x_0)<c,
$$

所以軌跡不能碰到$K_c$的外側邊界，也不能穿越球面$S_s$。因此軌跡留在$D$內的緊集$K_c$，由ODE延拓定理可排除有限時間逃逸。這同時建立正向不變性與向前全時間存在。

若$\dot V<0$，再對任意不含$x_\ast$的緊環帶利用$\dot V$的嚴格負上界，可排除軌跡長時間停留在該環帶，從而得到$x(t)\to x_\ast$。這是負定導數推出局部漸近穩定的標準緊緻性論證。

若在某鄰域內存在$c_1,c_2,c_3>0$使

$$
c_1\|x-x_\ast\|_2^2
\le V(x)
\le c_2\|x-x_\ast\|_2^2
$$

及

$$
\dot V(x)\le-c_3\|x-x_\ast\|_2^2,
$$

則在閉包完全包含於該鄰域的不變子水平集內，

$$
\dot V\le-\frac{c_3}{c_2}V.
$$

由微分不等式，

$$
V(t)\le e^{-(c_3/c_2)t}V(0),
$$

因此

$$
\|x(t)-x_\ast\|_2
\le
\sqrt{\frac{c_2}{c_1}}\,
e^{-c_3t/(2c_2)}
\|x_0-x_\ast\|_2.
$$

這才給出對所有$t\ge0$成立的局部指數估計。若未先建立不變子水平集，則此估計只能宣稱在解仍留於有效鄰域的時間內成立。

### 4. Lyapunov矩陣方程

對線性系統$\dot z=Az$，考慮

$$
V(z)=z^TPz,\qquad P=P^T\succ0.
$$

沿解有

$$
\dot V(z)=z^T(A^TP+PA)z.
$$

若存在$Q=Q^T\succ0$使

$$
A^TP+PA=-Q,
$$

則

$$
\dot V(z)=-z^TQz<0\qquad(z\ne0).
$$

### 小命題：Hurwitz矩陣產生正定Lyapunov解

**命題**  
設$A\in\mathbb R^{n\times n}$為Hurwitz，且$Q=Q^T\succ0$。則

$$
P=\int_0^\infty e^{A^Tt}Qe^{At}\,dt
$$

收斂，$P=P^T\succ0$，並滿足

$$
A^TP+PA=-Q.
$$

此外，此對稱解唯一。

**證明**  
有限維Hurwitz條件保證存在$M\ge1$及$\alpha>0$使

$$
\|e^{At}\|_2\le Me^{-\alpha t}.
$$

因此

$$
\|e^{A^Tt}Qe^{At}\|_2
\le M^2\|Q\|_2e^{-2\alpha t}.
$$

右側在$[0,\infty)$可積，所以定義$P$的矩陣瑕積分絕對收斂。每個被積矩陣均對稱，故$P$對稱。對任意非零列向量$z$，

$$
z^TPz
=
\int_0^\infty
(e^{At}z)^TQ(e^{At}z)\,dt.
$$

被積函數非負，且$t=0$時等於$z^TQz>0$。由連續性，積分嚴格為正，所以$P\succ0$。

接著保持矩陣乘法次序。因$A^T$與其矩陣指數$e^{A^Tt}$交換，且$A$與$e^{At}$交換，

$$
A^TP+PA
=
\int_0^\infty
\left(
A^Te^{A^Tt}Qe^{At}
+
e^{A^Tt}Qe^{At}A
\right)\,dt.
$$

另一方面，

$$
\frac{d}{dt}
\left(e^{A^Tt}Qe^{At}\right)
=
A^Te^{A^Tt}Qe^{At}
+
e^{A^Tt}Qe^{At}A.
$$

等價地，第一項也可寫成$e^{A^Tt}A^TQe^{At}$，第二項也可寫成$e^{A^Tt}QAe^{At}$；這只使用矩陣與其自身指數交換，並未假設$A$與$Q$交換。因此

$$
A^TP+PA
=
\left[e^{A^Tt}Qe^{At}\right]_{0}^{\infty}.
$$

Hurwitz條件使上限趨於零，而$t=0$時矩陣為$Q$，故

$$
A^TP+PA=-Q.
$$

最後，若$P_1,P_2$都是方程的解，令$R=P_1-P_2$，則

$$
A^TR+RA=0.
$$

考慮$e^{A^Tt}Re^{At}$，其導數為零，故它恆等於$R$。但$t\to\infty$時，Hurwitz條件使該矩陣趨於零，因此$R=0$，解唯一。證畢。

量詞完整的等價敘述是：

- 若$A$為Hurwitz，則對每個$Q=Q^T\succ0$，方程皆有唯一的對稱正定解$P$。
- 若存在某一組$Q=Q^T\succ0$及$P=P^T\succ0$滿足$A^TP+PA=-Q$，則$A$為Hurwitz。

第二點可由複特徵向量驗證。若$Av=\lambda v$且$v\ne0$，則

$$
v^\ast(A^TP+PA)v
=
2\operatorname{Re}(\lambda)\,v^\ast Pv
=
-v^\ast Qv<0.
$$

因$v^\ast Pv>0$，必有$\operatorname{Re}\lambda<0$。這是必要充分判準，而不是有限採樣所得的經驗規則。

---

## 逐步手算例題

### 例1：相同零Jacobian，不同穩定性

考慮

$$
\dot x=-x^3.
$$

對$x_0\ne0$分離變數可得

$$
x(t)=\frac{x_0}{\sqrt{1+2x_0^2t}}.
$$

因此$|x(t)|\le|x_0|$且$x(t)\to0$。取

$$
V(x)=\frac{x^2}{2},
$$

則

$$
\dot V=x\dot x=-x^4<0\qquad(x\ne0).
$$

原點漸近穩定，但不是局部指數穩定。固定非零$x_0$時，精確解只以$t^{-1/2}$階衰減，不可能對所有$t\ge0$被固定常數倍的$e^{-\alpha t}$控制。

再考慮

$$
\dot x=x^3.
$$

其非零解為

$$
x(t)=\frac{x_0}{\sqrt{1-2x_0^2t}},
$$

並在

$$
T_{\max}=\frac{1}{2x_0^2}
$$

發生有限時間爆破。任意小的非零初值最終都離開固定小鄰域，因此原點不穩定。兩個系統都有$f'(0)=0$，說明零Jacobian沒有判別力。

### 例2：非正規暫態與二次Lyapunov函數

令

$$
A=
\begin{pmatrix}
-1&8\\
0&-2
\end{pmatrix}.
$$

$A$為上三角矩陣，特徵值為$-1,-2$，故為Hurwitz。直接計算得

$$
e^{At}=
\begin{pmatrix}
e^{-t}&8(e^{-t}-e^{-2t})\\
0&e^{-2t}
\end{pmatrix}.
$$

取$z_0=(0,1)^T$。在$t=\log2$時，

$$
z(t)=
\begin{pmatrix}
2\\
1/4
\end{pmatrix},
\qquad
\|z(t)\|_2=\frac{\sqrt{65}}4>1=\|z_0\|_2.
$$

Euclidean長度曾經增加，但這不否定漸近穩定。

取$Q=I$並設

$$
P=
\begin{pmatrix}
p_{11}&p_{12}\\
p_{12}&p_{22}
\end{pmatrix}.
$$

由$A^TP+PA=-I$逐項比較得

$$
p_{11}=\frac12,\qquad
p_{12}=\frac43,\qquad
p_{22}=\frac{67}{12}.
$$

其首要主子式為$1/2>0$，且

$$
\det P
=
\frac12\frac{67}{12}
-\left(\frac43\right)^2
=
\frac{73}{72}>0.
$$

故$P\succ0$。雖然$\|z(t)\|_2$可能暫時增加，

$$
V(z)=z^TPz,\qquad
\dot V=-\|z\|_2^2
$$

仍嚴格下降。Lyapunov函數改變的是衡量距離的橢球幾何，不是實際軌跡。

---

## 實作與程式

以下自足程式只使用NumPy與CPU。它求解Lyapunov方程、執行RK4積分，並輸出離散Lyapunov增量。程式未在本章執行；所有輸出描述均為預期結果。

```python
import numpy as np

def is_hurwitz(A, tol=1e-12):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必須是方陣")
    eig = np.linalg.eigvals(A)
    return bool(np.all(np.real(eig) < -tol)), eig

def solve_lyapunov(A, Q):
    """解 A.T @ P + P @ A = -Q，使用 column-major vec。"""
    A = np.asarray(A, dtype=float)
    Q = np.asarray(Q, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必須是方陣")
    n = A.shape[0]
    if Q.shape != (n, n):
        raise ValueError("Q 的形狀必須與 A 相同")
    if not np.allclose(Q, Q.T):
        raise ValueError("Q 必須對稱")

    I = np.eye(n)
    K = np.kron(I, A.T) + np.kron(A.T, I)
    rhs = -Q.reshape(n * n, order="F")
    p = np.linalg.solve(K, rhs)
    P = p.reshape((n, n), order="F")
    return 0.5 * (P + P.T)

def rk4(f, x0, dt, steps):
    if dt <= 0:
        raise ValueError("dt 必須為正")
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps 必須為非負整數")

    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1:
        raise ValueError("程式儲存的 x0 必須是一維陣列")

    out = np.empty((steps + 1, x.size), dtype=float)
    out[0] = x

    for k in range(steps):
        k1 = np.asarray(f(x), dtype=float)
        k2 = np.asarray(f(x + 0.5 * dt * k1), dtype=float)
        k3 = np.asarray(f(x + 0.5 * dt * k2), dtype=float)
        k4 = np.asarray(f(x + dt * k3), dtype=float)

        if not all(v.shape == x.shape for v in (k1, k2, k3, k4)):
            raise ValueError("f(x) 的形狀必須與 x 相同")

        x = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("軌跡出現非有限值")
        out[k + 1] = x

    return out

def numerical_report(A, P, Q, x0, dt, final_time):
    if final_time < 0:
        raise ValueError("final_time 必須非負")
    steps = int(round(final_time / dt))
    traj = rk4(lambda x: A @ x, x0, dt, steps)

    V = np.einsum("bi,ij,bj->b", traj, P, traj)
    norms = np.linalg.norm(traj, axis=1)
    residual = A.T @ P + P @ A + Q

    return {
        "dt": dt,
        "eig_A": np.linalg.eigvals(A),
        "eig_P": np.linalg.eigvalsh(P),
        "residual_inf": np.linalg.norm(residual, ord=np.inf),
        "peak_norm": np.max(norms),
        "max_discrete_V_increment": (
            np.max(np.diff(V)) if steps > 0 else 0.0
        ),
        "final_norm": norms[-1]
    }

if __name__ == "__main__":
    A = np.array([[-1.0, 8.0],
                  [ 0.0, -2.0]])
    Q = np.eye(2)
    x0 = np.array([0.0, 1.0])

    ok, eig = is_hurwitz(A)
    P = solve_lyapunov(A, Q)

    assert ok
    assert np.min(np.linalg.eigvalsh(P)) > 0.0
    assert np.linalg.norm(
        A.T @ P + P @ A + Q, ord=np.inf
    ) < 1e-10

    for dt in (0.1, 0.05, 0.01, 0.001):
        print(numerical_report(
            A, P, Q, x0, dt=dt, final_time=5.0
        ))

    one = rk4(lambda x: A @ x, x0, dt=0.1, steps=0)
    assert one.shape == (1, 2)
    assert np.allclose(one[0], x0)

    print("eigenvalues =", eig)
    print("P =\n", P)
```

NumPy一維陣列的形狀是`(n,)`，只是儲存格式；數學上的$x$仍被視為$n\times1$列向量。不可利用一維陣列的`.T`聲稱已改成橫向量。

一般有限步長RK4不保證保存連續系統的Lyapunov單調性。因此程式不斷言離散$V_k$必定下降，而只輸出`max_discrete_V_increment`。即使該值非正，也只支持指定步長、初值與時間窗內的觀察。

---

## 測試與預期結果

### 正常測試

對例2的$A$與$Q=I$，預期：

- 特徵值在浮點誤差內接近$-1,-2$；
- $P$接近手算矩陣；
- $P$的數值特徵值皆為正；
- Lyapunov方程殘差接近浮點捨入尺度；
- `peak_norm`大於$1$，顯示暫態放大；
- 長時間後`final_norm`小於暫態峰值。

步長掃描可能顯示較小步長下的離散結果更接近連續軌跡，但本章未執行程式，故不宣稱具體數值或觀察到的收斂階。

### 邊界測試

對$\dot x=-x^3$，Jacobian為$[0]$。`is_hurwitz`預期回傳`False`，但這不表示原點不穩定；直接Lyapunov分析已證明原點漸近穩定。

`steps=0`時，`rk4`應只回傳初值。輸出陣列的一個橫列只是資料表中的單筆紀錄，不代表數學列向量變成橫向量。

### 故障測試

1. 非方陣$A$應引發`ValueError`。
2. 非對稱$Q$應引發`ValueError`。
3. `dt<=0`、負步數或負`final_time`應引發`ValueError`。
4. 對$A=\operatorname{diag}(0,-1)$及$Q=I$求解時，Kronecker線性系統奇異，預期`np.linalg.solve`引發例外。
5. 對$\dot x=x^3$積分接近爆破時間時，數值可能急速增大或出現非有限值。這只能作故障警訊，不能精確認定爆破時間。
6. 接近虛軸的特徵值容易受浮點誤差影響；固定容差不能替代解析譜判定。

---

## 反例與常見陷阱

1. **把非Hurwitz等同不穩定。**  
   $\dot x=-x^3$的Jacobian不是Hurwitz，原點仍漸近穩定。

2. **把零實部等同中性穩定。**  
   $\dot x=x^3$在線性層次為$\dot x=0$，原非線性系統卻不穩定。

3. **只檢查$V>0$。**  
   正定函數未必沿軌跡下降，還必須計算$\dot V=\nabla V^Tf$。

4. **由$\dot V\le0$直接宣稱漸近穩定。**  
   旋轉系統
   $$
   \dot x=-y,\qquad \dot y=x
   $$
   取$V=x^2+y^2$時有$\dot V=0$。原點穩定但不是漸近穩定。

5. **忽略閉子水平集與延拓。**  
   局部不等式只在指定鄰域有效。必須把軌跡限制在閉包留於該鄰域內的緊子水平集，才能排除有限時間逃逸。

6. **把Euclidean暫態增長當成不穩定。**  
   指數穩定估計容許$M>1$；非正規系統可以先放大再衰減。

7. **把局部結論升格為全域結論。**  
   全域漸近穩定通常還要求條件在全空間成立、解全域存在，並常配合徑向無界的$V$。

8. **混合不同物理單位。**  
   溫度與濃度不能直接以無權重Euclidean長度作物理解釋，應先無因次化或說明$P$的單位。

9. **把數值軌跡當成定理。**  
   有限初值、有限時間及有限精度不能證明對整個鄰域與所有$t\ge0$成立的命題。

---

## AI、幾何與養殖案例

### AI局部動態

學習參數的連續時間近似有時寫成

$$
\dot\theta=F(\theta).
$$

若平衡參數$\theta_\ast$附近的$J_F(\theta_\ast)$為Hurwitz，可推出該連續模型的局部指數穩定；但這不代表損失函數全域凸，也不保證遠處初始化收斂。神經網路常因參數置換或尺度對稱而出現零特徵值，此時線性化可能不定。

離散更新$\theta_{k+1}=G(\theta_k)$的局部線性穩定條件則是Jacobian特徵值位於單位圓內，不能直接套用連續時間Hurwitz條件。

### Lyapunov橢球的幾何

若$P\succ0$，則

$$
V(z)=z^TPz
$$

的子水平集$V(z)\le c$是橢球。其主軸由$P$的正交特徵向量給出，半軸長與對應特徵值平方根的倒數成正比。非正規系統中，Euclidean圓可能被流動暫時拉長，但某個Lyapunov橢球仍可向內收縮。

### 合成養殖感測模型

令水溫偏差$\Delta T$以$^\circ\mathrm C$計，溶氧偏差$\Delta O$以$\mathrm{mg/L}$計。直接令

$$
x=(\Delta T,\Delta O)^T
$$

並使用$\|x\|_2$，會把不同單位的平方相加。選尺度

$$
T_s=2^\circ\mathrm C,\qquad
O_s=1\,\mathrm{mg/L},
$$

並定義無因次狀態

$$
z_1=\frac{\Delta T}{T_s},\qquad
z_2=\frac{\Delta O}{O_s}.
$$

考慮合成局部模型

$$
\dot z=
\begin{pmatrix}
-0.4&1.2\\
0&-0.7
\end{pmatrix}z+r(z),
$$

其中時間單位為小時，且

$$
\frac{\|r(z)\|_2}{\|z\|_2}\to0.
$$

線性矩陣的特徵值為$-0.4$與$-0.7$每小時，所以在$C^1$模型成立的充分小鄰域內，可推出局部指數穩定。非對角元素表示無因次狀態間的局部耦合，不應脫離尺度定義解讀。

返回物理量時，

$$
\Delta T=T_sz_1,\qquad
\Delta O=O_sz_2.
$$

此案例完全是合成模型。數學穩定不等於現場操作安全；感測偏差、延遲、致動飽和與未建模外擾都可能使實際狀態離開證明有效的鄰域。唯讀分析不控制曝氣、投餌或加藥。

---

## 習題

1. **手算題**  
   對$\dot x=-2x+x^3$判斷原點的局部穩定性，並以$V=x^2/2$找出$\dot V<0$的區域。

2. **程式題**  
   對
   $$
   A_k=
   \begin{pmatrix}
   -1&k\\
   0&-2
   \end{pmatrix},
   \qquad k=0,2,\ldots,20,
   $$
   計算初值$(0,1)^T$的有限時間峰值範數，以及$Q=I$所得$P$的條件數。說明輸出能與不能證明什麼。

3. **反例題**  
   找出一個正定$V$滿足$\dot V\le0$，但原點不是漸近穩定的系統。驗證條件並指出缺少哪項充分條件。

4. **整合題**  
   設
   $$
   \dot z=Az+g(z),\qquad
   A=
   \begin{pmatrix}
   -1&0\\
   0&-3
   \end{pmatrix},
   $$
   且在$\|z\|_2\le\rho$內有$\|g(z)\|_2\le c\|z\|_2^2$。使用$V=\|z\|_2^2/2$找出保證$\dot V<0$的半徑，並建立不離開有效鄰域的初值集合。

---

## 習題解答

### 1. 手算題

因$f'(0)=-2<0$，線性化定理給出原點局部指數穩定。直接計算

$$
\dot V=x(-2x+x^3)
=x^2(x^2-2).
$$

因此$0<|x|<\sqrt2$時$\dot V<0$。若縮小到$|x|\le1$，則

$$
\dot V\le-x^2=-2V.
$$

區間$[-1,1]$正向不變，故對$|x_0|<1$有

$$
V(t)\le e^{-2t}V(0),
\qquad
|x(t)|\le e^{-t}|x_0|.
$$

### 2. 程式題

可使用：

```python
for k in range(0, 21, 2):
    A = np.array([[-1.0, float(k)],
                  [ 0.0, -2.0]])
    P = solve_lyapunov(A, np.eye(2))
    report = numerical_report(
        A, P, np.eye(2),
        np.array([0.0, 1.0]),
        dt=0.001, final_time=5.0
    )
    print(k, report["peak_norm"], np.linalg.cond(P))
```

預期較大的$k$通常帶來較強的暫態放大及較狹長的Lyapunov橢球，但程式未執行，不能宣稱具體數值。有限參數表只能支持所列$k$及時間窗內的觀察，不能證明對所有實數$k$的單調關係。每個$A_k$為Hurwitz則可由其精確特徵值$-1,-2$證明。

### 3. 反例題

取

$$
\dot x=-y,\qquad \dot y=x,
$$

以及

$$
V(x,y)=x^2+y^2.
$$

$V$正定，且

$$
\dot V=2x(-y)+2y(x)=0.
$$

每條非零軌跡都位於圓$x^2+y^2=\text{常數}$上，因此不收斂至原點。原點穩定但不是漸近穩定。缺少的是$x\ne0$時$\dot V<0$的負定條件。

### 4. 整合題

沿軌跡有

$$
\dot V=z^TAz+z^Tg(z).
$$

因$A=\operatorname{diag}(-1,-3)$，

$$
z^TAz\le-\|z\|_2^2.
$$

由Cauchy–Schwarz不等式，

$$
z^Tg(z)
\le\|z\|_2\|g(z)\|_2
\le c\|z\|_2^3.
$$

所以

$$
\dot V
\le-\|z\|_2^2(1-c\|z\|_2).
$$

若$c>0$，則在

$$
0<\|z\|_2<\min\{\rho,1/c\}
$$

內有$\dot V<0$。選

$$
r<\min\{\rho,1/c\},
$$

並限制$\|z(0)\|_2<r$。因$V$下降，

$$
\|z(t)\|_2\le\|z(0)\|_2<r,
$$

所以軌跡留在有效球內，局部不等式可持續使用；其閉包是定義域內的緊集，故可由延拓定理得到向前全時間存在。若$c=0$，則不需要$1/c$限制。

---

## 本章小結

非線性系統在平衡點附近可寫成

$$
\dot z=Az+r(z),\qquad r(z)=o(\|z\|_2).
$$

若$A$為Hurwitz，平衡點局部指數穩定；若$A$有正實部特徵值，平衡點不穩定；若有零實部特徵值，線性化一般不能判定。$\dot x=-x^3$與$\dot x=x^3$展示了相同零Jacobian可能對應完全不同的行為。

Lyapunov直接法以正定函數及沿軌跡導數研究穩定性。局部條件必須配合留在有效鄰域內的緊閉子水平集與解的延拓，才能得到對所有$t\ge0$的結論。Hurwitz矩陣與Lyapunov矩陣方程則連接了譜判準、能量下降與橢球幾何。

非正規Hurwitz系統可以有Euclidean暫態放大。有限模擬、浮點特徵值與離散Lyapunov下降都只是數值證據，不能取代具有完整量詞與鄰域條件的分析證明。

---

## 參考來源

1. Hassan K. Khalil，*Nonlinear Systems*，第3版，Prentice Hall。可參考平衡點穩定性、Lyapunov直接法與線性化間接法。
2. Lawrence Perko，*Differential Equations and Dynamical Systems*，Springer。可參考平衡點、線性化與局部動力系統。
3. Jiří Lebl，*Basic Analysis*，作者教材入口：<https://www.jirka.org/ra/>
4. MIT OpenCourseWare，18.100A Real Analysis：<https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/>
5. MIT OpenCourseWare，18.02SC Multivariable Calculus：<https://ocw.mit.edu/courses/18-02sc/multivariable-calculus-fall-2010/>
6. SciPy，`scipy.linalg.expm`文件：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html>

前兩項直接對應本章的ODE穩定性與Lyapunov理論；其餘來源提供分析、微積分及矩陣指數背景。來源列舉不表示已逐頁核對所有版本。本章未執行程式，SciPy亦非核心程式依賴。