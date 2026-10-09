# 第16章 梯度下降、線搜尋與Newton法

## 學習目標與先備知識

本章把第12至14章的二階資訊真正轉成可執行的迭代演算法。讀完本章，你應該能：

1. 用Fréchet可微的定義寫出下降引理，並分清「方向導數為負」與「函數值真的下降」的差別。
2. 說明步長條件（含Armijo充分下降）如何把無窮小的下降推廣到有限步。
3. 寫出Newton法的更新式$x_{k+1}=x_k-H_f(x_k)^{-1}\nabla f(x_k)$，並指出它在哪些假設下才具有局部二次收斂。
4. 辨識不定Hessian的兩種典型後果：Newton方向可能不是下降方向、或是不穩定的跳躍步。
5. 實作一個只用NumPy的小型CPU最佳化器，含梯度下降、Armijo線搜尋與Newton法。
6. 以正常、邊界、故障三類測試驗證程式，並在報告中區分「數值觀察」與「定理結論」。

先備知識為：第6章的Fréchet導數與Jacobian、第12章的Hessian與二階Taylor展開、第13章關於正定與半正定二次型的判別、第14章凸性與$L$-Lipschitz梯度的定義。本章不預設讀者已學過信賴域或內點法。

---

## 問題與直覺

無約束最佳化問題寫成

$$
\min_{x \in \mathbb{R}^n} f(x)
$$

直覺上，函數在$x$附近的最佳線性近似是$f(x+h)\approx f(x)+\nabla f(x)^{\top}h$。若$\nabla f(x)\neq 0$，令$h=-t\nabla f(x)$、$t>0$小，則方向導數為

$$
\nabla f(x)^{\top}h=-t\,\|\nabla f(x)\|^2<0
$$

也就是「沿負梯度走一小段，函數值會下降」。把這個無窮小敘述變成演算法只需要兩步：選方向（負梯度或Newton方向），選步長（線搜尋）。第三章分析與第八章微積分橋接的餘項控制，在此處第一次變成演算法收斂論的具體輸入。

但直覺會出錯。$\nabla f(x)^{\top}h<0$只在固定方向上保證$t$足夠小時函數下降，並不保證$t=1$下降。Newton方向$-H_f(x)^{-1}\nabla f(x)$在$H_f(x)$不定時甚至可能使$\nabla f^{\top}d>0$。此外，梯度下降的線性收斂率由$L/\mu$（條件數）控制，病態問題上可能極慢。

---

## 定義、定理與推導

### 定義 16.1（下降方向）

設$f:U\subset\mathbb{R}^n\to\mathbb{R}$在開集$U$上Fréchet可微，$x\in U$。向量$d\in\mathbb{R}^n\setminus\{0\}$稱為$f$在$x$處的**下降方向**，若存在$\bar t>0$使得對所有$t\in(0,\bar t)$皆有$f(x+td)<f(x)$。

### 命題 16.1（下降引理）

設$f:U\subset\mathbb{R}^n\to\mathbb{R}$在開集$U$上Fréchet可微，$x\in U$且$\nabla f(x)\neq 0$。則$d=-\nabla f(x)$是$f$在$x$處的下降方向。

**證明.** 由Fréchet可微的定義，存在餘項$r(h)$滿足

$$
f(x+h)=f(x)+\nabla f(x)^{\top}h+r(h),\qquad \frac{\|r(h)\|}{\|h\|}\xrightarrow[h\to 0]{}0
$$

取$h=td$，$t>0$：

$$
f(x+td)=f(x)+t\,\nabla f(x)^{\top}d+r(td)=f(x)-t\|\nabla f(x)\|^2+r(td)
$$

因為$d=-\nabla f(x)$且$\|d\|=\|\nabla f(x)\|$，對$t>0$有$\|td\|=t\|\nabla f(x)\|$，於是

$$
\left|\frac{r(td)}{t}\right|=\|d\|\cdot\frac{\|r(td)\|}{\|td\|}\xrightarrow[t\to 0^+]{}0
$$

因此

$$
\lim_{t\to 0^+}\frac{f(x+td)-f(x)}{t}=-\|\nabla f(x)\|^2<0
$$

取左邊極限為負值，必存在$\bar t>0$使得當$0<t<\bar t$時該商為負，即$f(x+td)<f(x)$。$\square$

**評註.** 證明只用到$x$點的Fréchet可微，不需要$f$在鄰域內$C^1$。但要把這個局部性質轉成「從$x_0$出發的整個迭代收斂到駐點」，就需要全域界：$f$有下界、$\nabla f$一致連續，或$f$為$L$-光滑等。這些條件在第14章已交代，本章不重複。

### 定義 16.2（Armijo充分下降條件）

設$d$為$x$處的下降方向，$c_1\in(0,1)$。步長$\alpha>0$滿足**Armijo條件**，若

$$
f(x+\alpha d)\le f(x)+c_1\alpha\, \nabla f(x)^{\top}d
$$

由於右端嚴格小於$f(x)$，Armijo條件是「函數值下降至少與預測下降成比例」的充分條件。背後的定理是：

### 命題 16.2（Armijo步長存在性）

設$f:U\to\mathbb{R}$在$x$處Fréchet可微，$d$為下降方向且$\nabla f(x)^{\top}d<0$，$c_1\in(0,1)$。則存在$\bar t>0$使得所有$\alpha\in(0,\bar t)$均滿足Armijo條件。

**證明思路.** 由可微性，$\dfrac{f(x+\alpha d)-f(x)}{\alpha}\to\nabla f(x)^{\top}d$當$\alpha\to 0^+$。記$g:=\nabla f(x)^{\top}d$，$g<0$。取$c_1\in(0,1)$，則$c_1g>g$（因$g<0$而$c_1<1$）。由極限定義，存在$\bar t>0$使當$0<\alpha<\bar t$時$\frac{f(x+\alpha d)-f(x)}{\alpha}<c_1g$，整理即得Armijo條件。$\square$

實作上常採用**回溯線搜尋**：從$\alpha_0=1$開始，令$\alpha\leftarrow\rho\alpha$（$0<\rho<1$，常取$\rho=1/2$）直到Armijo成立，或達最大步數後宣告線搜尋失敗。

**弱Wolfe條件**再加一項$\nabla f(x+\alpha d)^{\top}d\ge c_2\nabla f(x)^{\top}d$（$c_1<c_2<1$）才能保證BFGS等擬Newton方法的收斂，Newton法用Armijo已足夠。

### 定義 16.3（Newton法）

設$f:U\to\mathbb{R}$為$C^2$，$x_k\in U$且$H_f(x_k)$可逆。**Newton迭代**為

$$
x_{k+1}=x_k-H_f(x_k)^{-1}\nabla f(x_k)
$$

在Newton法中，步長為$\alpha=1$、方向$d_k=-H_f(x_k)^{-1}\nabla f(x_k)$。實作上不顯式求逆，而解線性方程$H_f(x_k)d_k=-\nabla f(x_k)$。

### 命題 16.3（Newton局部二次收斂，標準假設）

設$f:U\subset\mathbb{R}^n\to\mathbb{R}$在$x^*$的開鄰域$B(x^*,r)\subset U$上$C^2$，$\nabla f(x^*)=0$，$H_f(x^*)$可逆，且$H_f$在$B(x^*,r)$上Lipschitz連續（常數$\gamma$）。則存在$\delta\in(0,r)$與$C>0$，使得當$x_0\in B(x^*,\delta)$時，Newton迭代有定義且

$$
\|x_{k+1}-x^*\|\le C\|x_k-x^*\|^2
$$

**說明.** 證明略去，屬引用結果；核心條件為：$C^2$鄰域、$H_f(x^*) $可逆、$H_f$局部Lipschitz。這三者合起來使Newton迭代在$x^*$附近落在收斂吸引球內。$\|H_f(x^*)^{-1}\|$與$\gamma$共同控制常數$C$。若$H_f(x^*)$奇異，該結論失效（見習題）。

### 定義 16.4（二次模型下降量）

給定$x$、梯度$g=\nabla f(x)$、Hessian $H=H_f(x)$與步長$\alpha$，二次模型

$$
m_\alpha(d)=f(x)+\alpha g^{\top}d+\tfrac{\alpha^2}{2}d^{\top}H d
$$

Newton步$d_N$恰使$\nabla m_1(d)=0$：$H d_N=-g$。若$H$正定，$d_N$是$m_1$的嚴格全域極小；若$H$不定，$d_N$可能是鞍點，對應的二次模型值是$f(x)-\tfrac12 g^{\top}H^{-1}g$，但$H^{-1}$有負特徵值時該值可能比$f(x)$高。

---

## 逐步手算例題

### 例16.1（梯度下降在二次目標上的精確線搜尋）

取$f(x_1,x_2)=x_1^2+2x_2^2$，起始點$x_0=(2,1)$。

**第一步**：$\nabla f(x_0)=(2x_1,4x_2)=(4,4)$，$\|\nabla f(x_0)\|^2=32$，下降方向$d_0=(-4,-4)$。

沿$d_0$的單變量函數

$$
\phi(t)=f(x_0+td_0)=(2-4t)^2+2(1-4t)^2=6-32t+48t^2
$$

$\phi'(t)=-32+96t=0\Rightarrow t^*=1/3$。$\phi(1/3)=6-32/3+48/9=2/3$。新點

$$
x_1=(2,1)+\tfrac13(-4,-4)=\left(\tfrac23,-\tfrac13\right),\quad f(x_1)=\tfrac23
$$

**第二步**：$\nabla f(x_1)=(4/3,-4/3)$，$\|\nabla f(x_1)\|^2=32/9$，$d_1=(-4/3,4/3)$。單變量函數

$$
\psi(t)=f\left(\tfrac23-\tfrac{4t}{3},-\tfrac13+\tfrac{4t}{3}\right)=\tfrac{6}{9}-\tfrac{32t}{9}+\tfrac{48t^2}{9}
$$

$\psi'(t)=0\Rightarrow t^*=1/3$，$\psi(1/3)=2/27$，$x_2=(2/9,1/9)$。

對這個目標，$f(x_k)=6\cdot 9^{-k}$，精確線搜尋梯度下降的收斂比為$1/9$。這是二次函數沿梯度的精確線搜尋所給的比值$(1-\kappa^{-1})^2$的一個特例，其中$\kappa=L/\mu$為條件數（$L=4$、$\mu=2$，$\kappa=2$；此處$x_0$恰沿對角，實際比$1/9$而非$(1-1/2)^2=1/4$，說明收斂常數還依賴初始方向與特徵軸的相對位置）。

### 例16.2（Newton法一步收斂與二次目標的關係）

取$f(x_1,x_2)=x_1^2+3x_1x_2+5x_2^2$，起始點$x_0=(1,1)$。

$H=\begin{pmatrix}2&3\\3&10\end{pmatrix}$，$\det H=11$，$H^{-1}=\frac{1}{11}\begin{pmatrix}10&-3\\-3&2\end{pmatrix}$。

$\nabla f(1,1)=(2\cdot 1+3\cdot 1,\ 3\cdot 1+10\cdot 1)=(5,13)$。

Newton步

$$
d_0=-H^{-1}\nabla f(x_0)=-\frac{1}{11}\begin{pmatrix}10\cdot 5-3\cdot 13\\ -3\cdot 5+2\cdot 13\end{pmatrix}=-\frac{1}{11}\begin{pmatrix}11\\11\end{pmatrix}=(-1,-1)
$$

$x_1=(0,0)$，$f(x_1)=0$，確為全域最小值。對任何正定二次目標，Newton法從任意$x_0$出發一步到達唯一駐點；這是二次模型精確的結果，性質無法直接推廣到非二次目標。

### 例16.3（不定Hessian反例手算）

取$f(x,y)=(x^2-1)^2+y^2$，在$x_0=(0.5,0.5)$：

$\nabla f(0.5,0.5)=(4x(x^2-1),2y)\big|_{(0.5,0.5)}=(-1.5,1)$。

$H_f=\begin{pmatrix}12x^2-4&0\\0&2\end{pmatrix}$，$H_f(0.5,0.5)=\begin{pmatrix}-1&0\\0&2\end{pmatrix}$為不定矩陣。

Newton步$d_N=-H^{-1}\nabla f=-\begin{pmatrix}-1&0\\0&0.5\end{pmatrix}(-1.5,1)^{\top}=-(-1.5,0.5)^{\top}=(1.5,-0.5)$。

方向導數$\nabla f^{\top}d_N=-1.5\cdot 1.5+1\cdot(-0.5)=-2.75<0$，本例方向仍是下降方向，但這是因為梯度在負曲率方向上已有分量；若將起始點移到$(0,0.5)$，$\nabla f(0,0.5)=(0,1)$、$H_f(0,0.5)=(-4,0;0,2)$，$d_N=(0,-0.5)$仍是下降。真正的危險情形是當梯度在負曲率方向分量很小時Newton步可能近似沿正曲率方向上升；習題16.3給出一個明確構造。

---

## 實作與程式

下列程式只用NumPy（標準庫以外的唯一依賴），不安裝任何套件、不呼叫SciPy、不執行GPU。目標函數介面為`f(x)->float`、`grad(x)->np.ndarray shape (n,1)`、`hess(x)->np.ndarray shape (n,n)`。

```python
import numpy as np

def quad_factory(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1, 1)
    def f(x):
        x = x.reshape(-1, 1)
        return 0.5 * float(x.T @ A @ x) - float(b.T @ x)
    def grad(x):
        x = x.reshape(-1, 1)
        return A @ x - b
    def hess(x):
        return A.copy()
    return f, grad, hess

def armijo(f, grad, x, d, fx, alpha0=1.0, rho=0.5, c1=1e-4, max_ls=60):
    g = grad(x)
    gTd = float(g.T @ d)
    if gTd >= 0.0:
        return None, None, fx, "not_descent"
    alpha = alpha0
    for _ in range(max_ls):
        xnew = x + alpha * d
        fnew = f(xnew)
        if fnew <= fx + c1 * alpha * gTd:
            return alpha, xnew, fnew, "ok"
        alpha *= rho
    return None, None, fx, "ls_exhausted"

def gradient_descent(x0, f, grad, tol=1e-10, max_iter=2000, **ls_kw):
    x = np.asarray(x0, dtype=float).reshape(-1, 1)
    fx = f(x)
    log = []
    for k in range(max_iter):
        g = grad(x)
        gn = float(np.linalg.norm(g))
        if gn < tol:
            log.append((k, fx, gn, None, "grad_tol"))
            return x, log
        d = -g
        alpha, xnew, fnew, msg = armijo(f, grad, x, d, fx, **ls_kw)
        if alpha is None:
            log.append((k, fx, gn, None, msg))
            return x, log
        x, fx = xnew, fnew
        log.append((k, fx, gn, alpha, "ok"))
    log.append((max_iter, fx, float(np.linalg.norm(grad(x))), None, "max_iter"))
    return x, log

def newton(x0, f, grad, hess, tol=1e-10, max_iter=200,
           reg=0.0, alpha0=1.0, rho=0.5, c1=1e-4):
    x = np.asarray(x0, dtype=float).reshape(-1, 1)
    fx = f(x)
    log = []
    for k in range(max_iter):
        g = grad(x)
        gn = float(np.linalg.norm(g))
        if gn < tol:
            log.append((k, fx, gn, None, "grad_tol"))
            return x, log
        H = hess(x)
        if reg > 0.0:
            H = H + reg * np.eye(H.shape[0])
        try:
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            log.append((k, fx, gn, None, "singular_hessian"))
            return x, log
        if float(g.T @ d) >= 0.0:
            log.append((k, fx, gn, None, "newton_not_descent"))
            return x, log
        alpha, xnew, fnew, msg = armijo(f, grad, x, d, fx,
                                        alpha0=alpha0, rho=rho, c1=c1)
        if alpha is None:
            log.append((k, fx, gn, None, msg))
            return x, log
        x, fx = xnew, fnew
        log.append((k, fx, gn, alpha, "ok"))
    log.append((max_iter, fx, float(np.linalg.norm(grad(x))), None, "max_iter"))
    return x, log
```

要點：`newton`在`np.linalg.solve`失敗或方向非下降時立即返回並在log中標記原因；不做未定義的「修正」，留給呼叫者決定重新初始化策略。`reg>0`時是對Hessian加$\text{reg}\cdot I$的Tikhonov型正則化，屬啟發式，不提供收斂證明。

---

## 測試與預期結果

以下是本章應達到的三類測試（未在本環境實際執行，僅預期）。任何一次執行結果必須自報版本、NumPy版本與硬體環境，才能作為證據。

### 正常測試

$A=\begin{pmatrix}4&1\\1&3\end{pmatrix}$，$b=(1,1)^{\top}$，唯一最小值$x^*=A^{-1}b=\frac{1}{11}(2,3)^{\top}\approx(0.1818,0.2727)$。

- `gradient_descent`從$x_0=(10,-5)$、`tol=1e-10`應在數百步內收斂，log最後一行`status=="grad_tol"`，$f$單調不增。
- `newton`從同一$x_0$應在2步內達到$10^{-12}$級梯度範數（一步跳到精確解附近，另一步歸零殘差）；`alpha=1.0`。

### 邊界測試

- 起始點$x_0=x^*+\varepsilon(1,1)$，$\varepsilon=10^{-8}$：`gradient_descent`第0步即`grad_tol`；`newton`同樣第0步停止。驗證停止條件在近最優點的退化行為。
- 步長$t$的邊界：設`alpha0=1e-15`，預期`gradient_descent`在第一步由於步長太小、Armijo右端$c_1\alpha g^{\top}d$過小，可能先被接受但幾乎不移動，之後各步需累積才能接近$x^*$，日誌顯示$f$幾乎不下降。此為數值現象，不構成收斂證明。

### 故障測試

- 取$f(x,y)=(x^2-1)^2+y^2$，$x_0=(0,0.5)$，$H_f(0,0.5)=$`diag(-4,2)`。`newton`應在某一步報告`newton_not_descent`或`singular_hessian`（視$x$演化而定）；用`reg=0.5`時$H+\text{reg}\cdot I=\text{diag}(-3.5,2.5)$仍不定，故未必自動痊癒，需靠Armijo。若啟用`reg=10`，步驟方向將轉為近梯度方向。
- 取$f(x,y)=x^2-y^2$（馬鞍），$x_0=(1,0.1)$。$\nabla f=(2x,-2y)$、$H_f=\text{diag}(2,-2)$。Newton方向$-H^{-1}\nabla f=-(x,-y)=( -x,y)$，方向導數$=2x^2-2y^2<0$在$|x|>|y|$時仍下降，但$H_f$不定意味著Newton步可能使$y$分量變號甚至震盪；應觀察log顯示$y_k$在$0$附近跳動而$f$不單調。

**任何測試若未被執行，報告必須明說「僅為預期」，不得虛構時間、步數或收斂階。**

---

## 反例與常見陷阱

**(一) 下降方向不是下降實作.** 命題16.1只保證存在$\bar t>0$。若直接令$\alpha=1$、$d=-\nabla f(x)$，在梯度尺度遠大於局部曲率尺度時反而上升。例如$f(x)=10^{12}x^2$在$x=1$處，梯度和二階導數的尺度完全失衡，任何固定步長都會爆炸。

**(二) Newton方向不是下降方向.** 即使$H_f$可逆且不定，$d_N$可能使$g^{\top}d_N>0$。構造如下：設$H=\text{diag}(1,-1)$、$g=(\epsilon,1)$且$0<\epsilon\ll 1$。$-H^{-1}g=(-\epsilon,1)$，$g^{\top}d_N=-\epsilon^2+1>0$。這種情形對應於梯度幾乎落在正曲率方向，而負曲率方向比例大。修正策略包括修正Hessian（加$\tau I$使$H+\tau I$正定，$\tau>\max(0,-\lambda_{\min}(H))$）或改用信賴域方法，兩者都超出本章範圍。

**(三) 把線性收斂誤當二次收斂.** 只有在$x_0$足夠接近$x^*$、$H_f(x^*)$可逆且$H_f$局部Lipschitz時，Newton法才有命題16.3的二次收斂。全域Newton法（例如加Armijo）的觀察收斂階是「最終二次」，前期可能線性甚至次線性。

**(四) 「函數值下降」不等於「收斂到駐點」.** 梯度下降的$f(x_k)$單調遞減且下界存在，只保證$f(x_k)\to f^*$。若無其他條件（如凸性、Łojasiewicz條件或梯度一致性），$\nabla f(x_k)\to 0$不成立，$x_k$可能無極限點。

**(五) 條件數與步長.** $L/\mu$大時梯度下降的步長上界$1/L$變小，收斂率$1-\mu/L$接近1。把$x$任意縮放（如把單位由公尺換成毫米）會改變$L/\mu$，因此「演算法快慢」不是座標無關的量；報告必須明示尺度與單位。感測校準的SI情境中，不同成分的Jacobian元素單位不同，直接取歐氏範數會混合單位，見第8章。

---

## AI、幾何與養殖案例

以合成雙參數飼料校準為例。模型預測值為

$$
\hat y(t) = \theta_1\, \phi_1(t) + \theta_2\,\phi_2(t),\qquad t\in[0,T]
$$

$\phi_1$、$\phi_2$為兩個不同頻率的無因次激發波形，$\theta_1$、$\theta_2$分別帶有「每公斤飼料对应應答」的單位。損失

$$
f(\theta)=\tfrac12\sum_{i=1}^N\bigl(\hat y(t_i)-y_i^{\text{obs}}\bigr)^2
$$

$\nabla f(\theta)=J_\phi(\theta)^{\top}(\hat y-y^{\text{obs}})$，此處$J_\phi\in\mathbb{R}^{N\times 2}$。Hessian為$J_\phi^{\top}J_\phi$加（若考慮Gauss-Newton則無此項）殘餘加權項。$J_\phi^{\top}J_\phi$的正定性等價於$\phi_1$、$\phi_2$在採樣時間點上線性無關，也就是**可辨識性**；若兩個波形在$t_i$上近似成比例，$H$接近奇異，Newton方向被雜訊放大。

對此合成案例，可行的檢核鏈為：

1. 檢查$J_\phi^{\top}J_\phi$的特徵值；小特徵值對應弱可辨識方向。
2. 對同一組$y^{\text{obs}}$分別跑梯度下降與Newton，比較迭代步數與收斂點；兩者應趨近同一個$\theta^*$（在凸二次情形）。
3. 對殘餘$r(\theta)=\hat y(\theta)-y^{\text{obs}}$做尺度檢查：若單位混用，$f$的數值大小不具物理意義，且預測誤差歸因會失真。

必須明示：數學上收斂與否是相對於合成模型；不等於實地飼養試驗的統計顯著性或操作安全。Agent在此角色只整理合成證據，不修改設備、投餌或加藥。

---

## 習題

**習題16.1（手算）** 對$f(x_1,x_2)=3x_1^2+x_1x_2+2x_2^2$，$x_0=(2,-1)$。
(a) 寫出$H_f$並判斷正定性。
(b) 用精確線搜尋沿$d_0=-\nabla f(x_0)$算$x_1$。
(c) 用Newton法從$x_0$出發算$x_1$，並與(a)比較。

**習題16.2（程式）** 用本章`gradient_descent`與`newton`，在$A=\text{diag}(1,100)$、$b=(1,100)^{\top}$上從$x_0=(10,10)$出發。記錄每個方法達到$\|\nabla f\|<10^{-8}$所需的迭代數與總函數求值次數。解釋兩者的差距為何與$L/\mu$相關。

**習題16.3（反例）** 構造$f:\mathbb{R}^2\to\mathbb{R}$與點$x_0$使$\nabla f(x_0)\neq 0$、$H_f(x_0)$可逆且不定，且Newton方向$d_N=-H_f(x_0)^{-1}\nabla f(x_0)$滿足$\nabla f(x_0)^{\top}d_N>0$。寫出顯式公式並驗證條件。

**習題16.4（整合）** 令$f(x)=\frac12\|Ax-b\|^2+\lambda\|x\|_1$（$\lambda>0$、$A\in\mathbb{R}^{m\times n}$滿行秩）。寫出$f$在$\|x\|_1$可微處的梯度與Hessian，並說明為何本章Newton法在$x_i=0$處不直接適用。

---

## 習題解答

**16.1解.**
(a) $\nabla f=(6x_1+x_2,\ x_1+4x_2)$，$H_f=\begin{pmatrix}6&1\\1&4\end{pmatrix}$。$\text{tr}=10>0$、$\det=23>0$，故正定。

(b) $\nabla f(2,-1)=(12-1,\ 2-4)=(11,-2)$，$d_0=(-11,2)$。沿$d_0$：

$$
\phi(t)=f(2-11t,-1+2t)
$$

展開：
$x_1^2=4-44t+121t^2$，$x_2^2=1-4t+4t^2$，$x_1x_2=(2-11t)(-1+2t)=-2+4t+11t-22t^2=-2+15t-22t^2$。

$$
\phi(t)=3(4-44t+121t^2)+(-2+15t-22t^2)+2(1-4t+4t^2)
$$
$$
=12-132t+363t^2-2+15t-22t^2+2-8t+8t^2=12-125t+349t^2
$$

$\phi'(t)=-125+698t=0\Rightarrow t^*=125/698\approx0.1791$。$x_1=(2-11t^*,-1+2t^*)\approx(0.0301,-0.6418)$。

(c) $H_f$正定，Newton步$d_N=-H_f^{-1}\nabla f$。$H_f^{-1}=\frac{1}{23}\begin{pmatrix}4&-1\\-1&6\end{pmatrix}$，$H_f^{-1}(11,-2)=\frac{1}{23}(44+2,\ -11-12)=\frac{1}{23}(46,-23)=(2,-1)$，$\nabla f^{\top}d_N=11\cdot 2+(-2)(-1)=24>0$。

這裡方向$d_N=-(2,-1)=(-2,1)$恰為負梯度方向的比例縮放？檢查：$-H_f^{-1}\nabla f=-(2,-1)=(-2,1)$，而負梯度$-g=(-11,2)$。二者不同；Newton方向$-H^{-1}g=(-2,1)$。核對下降性質：$g^{\top}d_N=11(-2)+(-2)(1)=-24<0$。Newton步為$x_1=(2,-1)+(-2,1)=(0,0)$，$f(0,0)=0$。這是二次目標一步到$0$的最小值。

**16.2解（預期）.** $A=\text{diag}(1,100)$，$b=(1,100)$，$x^*=(1,1)$，$f^*=0$。條件數$\kappa=100$。

梯度下降：步長$1/L=1/100$，收斂率$1-\mu/L=0.99$，從$x_0=(10,10)$起$\|x_k-x^*\|_\infty$約以$0.99^k$衰減；要達到$10^{-8}$大約需$\log(10^{-8}/9)/\log(0.99)\approx 3000$步量級。

Newton：Hessian為常數$A$，一步到$x^*$，第二步殘差歸零。`newton`的`max_iter=200`足夠。

求值次數：梯度下降每次迭代1次`grad`（+若干次線搜尋函數值）；Newton每次迭代1次`grad`、1次`hess`、1次`solve`。$\kappa$大時梯度下降迭代數約正比於$\kappa\log(1/\epsilon)$，Newton在二次目標上與$\kappa$無關。注意：此結論依賴目標為精確二次，不能推廣到一般$f$。

**16.3解.** 令$H=\text{diag}(1,-1)$、$g=(\epsilon,1)^{\top}$，$0<\epsilon<1$。$-H^{-1}g=-(\epsilon,-1)=(-\epsilon,1)$。

$$
g^{\top}d_N=\epsilon\cdot(-\epsilon)+1\cdot 1=1-\epsilon^2>0
$$

故$d_N$非下降方向。取$f(x_1,x_2)=\frac12 x_1^2-\frac12 x_2^2+\epsilon x_1+x_2$，則$\nabla f(x)=Hx+g$，在$x_0=(0,0)$處$\nabla f(x_0)=(\epsilon,1)$、$H_f(x_0)=H$，滿足假設。檢查：$\epsilon=1/2$時$H^{-1}=\text{diag}(1,-1)$，$-H^{-1}g=(-1/2,1)$，$g^{\top}d_N=1-1/4=3/4>0$。

**16.4解.** 對$x_i\neq 0$，$\|x\|_1$沿$x_i$的偏導為$\text{sign}(x_i)$；對$x_i=0$處，$\|x\|_1$在該座標方向右導數$=1$、左導數$=-1$，不可微。故

$$
\nabla f(x)=A^{\top}(Ax-b)+\lambda\,\text{sign}(x)\quad(x_i\neq 0)
$$

$H_f(x)=A^{\top}A$在$\|x\|_1$二次可微處；在$x_i=0$處$H_f$無定義。

Newton法在$\|x\|_1$不可微點不適用，因為其更新式依賴$f$的局部二次模型；線搜尋的方向導數需要$\nabla f$存在。可改用近端梯度法或座標下降，其條件與收斂論超出本章範圍，屬第17章之後。

---

## 本章小結

1. 下降引理是Fréchet可微性的直接推論，只用一階餘項控制；「$\nabla f(x)\neq 0\Rightarrow-\nabla f$為下降方向」為局部結論（命題16.1）。
2. Armijo條件是步長存在的充分條件，其檢核式為$f(x+\alpha d)\le f(x)+c_1\alpha\nabla f(x)^{\top}d$；回溯線搜尋依$0<\rho<1$縮減$\alpha$直到成立或耗盡允許步數。
3. Newton法以$-H_f(x)^{-1}\nabla f(x)$為方向，在$C^2$鄰域、$H_f(x^*)$可逆、$H_f$局部Lipschitz下具有二次收斂；全域化需要外加線搜尋或信賴域，本節不展開。
4. 不定Hessian有兩類後果：Newton方向可能非下降、步長可能震盪；用Hessian正則化或信賴域處理，屬延伸內容。
5. 手算例16.1、16.2、16.3展示精確線搜尋、Newton一步與不定Hessian的最小反例。
6. 可稽核的最佳化報告必須包含：演算法、停止條件、單位與尺度、日誌中每步$f$、$\|\nabla f\|$、$\alpha$與狀態碼；數值實驗只作為證據標籤`evidence:numeric`，不是定理證明。

---

## 參考來源

- [A1] Jiří Lebl, *Basic Analysis*（作者目錄與教材入口），<https://www.jirka.org/ra/>。
- [A2] MIT OCW 18.100A Real Analysis，<https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/>。
- [A3] MIT OCW 18.02SC Multivariable Calculus，<https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/>。
- [A4] JAX Autodiff Cookbook: JVP／VJP，<https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html>。
- [A5] SciPy `scipy.linalg.expm`，<https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html>。
- [A6] SciPy `scipy.optimize.minimize`，<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>。

來源狀態：A1為作者首頁；A2、A3為課程概要頁；A4–A6為API文件。上述頁面於2026-10-05已取得，教材PDF未逐頁查證；A5、A6為延伸參考，本章程式不依賴SciPy。