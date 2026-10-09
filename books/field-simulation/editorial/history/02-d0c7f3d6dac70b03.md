# 第02章 多變量微積分與Taylor橋接

## 學習目標與先備知識

完成本章後，讀者應能：

1. 區分純量場、向量場與其座標表示，並檢查輸入值、座標及單位是否一致。
2. 正確計算偏導數、方向導數、全微分、梯度、Jacobian與Hessian。
3. 理解「各偏導數存在」不等於「函數可微」，並能用定義檢查反例。
4. 由多變量Taylor公式推導有限差分近似及其截斷誤差。
5. 分辨模型誤差、離散截斷誤差與浮點捨入誤差。
6. 以解析梯度核對有限差分，辨識步長過大與過小的不同故障。
7. 建立具有輸入驗證、單位紀錄與可重現設定雜湊的CPU小程式。

先備知識包括一元微積分、矩陣乘法、Euclidean範數與基本Python/NumPy。本文採右手座標系，$X$向右、$Y$向上、$Z$朝向觀者。所有向量均視為列向量（column vector）。

---

## 問題與直覺

場是「每一空間位置與時間都配置一個量」的數學物件。例如池域溫度

$$
T(x,y,t)
$$

是純量場，單位為K；給定流速

$$
\mathbf{u}(x,y,t)=
\begin{bmatrix}
u_x\\u_y
\end{bmatrix}
$$

是向量場，單位為$\mathrm{m/s}$。若模型同時輸出溫度與溶氧濃度，可寫成

$$
\mathbf{q}(x,y,t)=
\begin{bmatrix}
T\\C
\end{bmatrix},
$$

但兩個分量單位不同，不能直接把$\|\mathbf q\|_2$當成有明確物理意義的大小；必須先無因次化。

多變量微積分回答三類問題：

- 沿座標方向移動時，場值如何變化？
- 沿任意方向移動時，場值如何變化？
- 有限但很小的位移，能否由局部導數預測？

Taylor展開正是連續微積分與數值差分之間的橋梁。然而「步長越小越準」只在忽略浮點誤差時成立。步長太大會產生截斷誤差，太小則會因兩個近似相等的浮點數相減而喪失有效位數。

模型也可能本身不準確。即使數值解已收斂到離散方程，若反應律、邊界資料或幾何假設錯誤，物理預測仍然錯誤。因此必須分開記錄：

1. **模型誤差**：方程與現實之差。
2. **離散誤差**：網格與時間步造成的近似誤差。
3. **代數誤差**：迭代求解尚未收斂造成的誤差。
4. **浮點誤差**：有限精度運算造成的誤差。

---

## 數學與物理推導

### 偏導數、梯度與方向導數

對$f:\mathbb R^n\rightarrow\mathbb R$，第$i$個偏導數定義為

$$
\frac{\partial f}{\partial x_i}(\mathbf x)
=
\lim_{h\to0}
\frac{f(\mathbf x+h\mathbf e_i)-f(\mathbf x)}{h}.
$$

若各分量存在，梯度寫成列向量

$$
\nabla f=
\begin{bmatrix}
\partial f/\partial x_1\\
\vdots\\
\partial f/\partial x_n
\end{bmatrix}.
$$

若$f$在$\mathbf x$可微，沿單位向量$\mathbf v$的方向導數為

$$
D_{\mathbf v}f(\mathbf x)=\nabla f(\mathbf x)^T\mathbf v.
$$

梯度指向局部增加最快的方向；最大增加率為$\|\nabla f\|_2$。這項結論要求可微，不能僅因偏導存在便直接使用。

### 全微分與可微性

$f$在$\mathbf x$可微，是指存在線性映射，使

$$
f(\mathbf x+\mathbf h)
=
f(\mathbf x)+\nabla f(\mathbf x)^T\mathbf h+r(\mathbf h),
$$

且

$$
\lim_{\|\mathbf h\|\to0}
\frac{|r(\mathbf h)|}{\|\mathbf h\|}=0.
$$

因此全微分為

$$
df=\nabla f^T d\mathbf x.
$$

偏導數只沿座標軸探測函數；可微性則要求所有接近方向共同服從同一個線性近似。若偏導數在鄰域內存在且於該點連續，則可推出可微；但這是充分條件，不是必要條件。

### Jacobian與鏈式法則

對映射$\mathbf F:\mathbb R^n\rightarrow\mathbb R^m$，Jacobian為$m\times n$矩陣

$$
J_{\mathbf F}(\mathbf x)_{ij}
=
\frac{\partial F_i}{\partial x_j}.
$$

小擾動滿足

$$
\mathbf F(\mathbf x+\mathbf h)
=
\mathbf F(\mathbf x)+J_{\mathbf F}(\mathbf x)\mathbf h
+o(\|\mathbf h\|).
$$

若$\mathbf y=\mathbf F(\mathbf x)$且$g=g(\mathbf y)$，則

$$
\nabla_{\mathbf x}(g\circ\mathbf F)
=
J_{\mathbf F}^T\nabla_{\mathbf y}g.
$$

Jacobian的第$j$縱行（column）描述只改變$x_j$時所有輸出的反應。其行空間（column space）位於$\mathbb R^m$；列空間（row space）位於$\mathbb R^n$。

### Hessian與二階Taylor公式

純量函數的Hessian為

$$
H_f(\mathbf x)_{ij}
=
\frac{\partial^2f}{\partial x_i\partial x_j}.
$$

若二階偏導在鄰域連續，混合偏導相等，Hessian對稱。二階Taylor式為

$$
f(\mathbf x+\mathbf h)
=
f(\mathbf x)+\nabla f(\mathbf x)^T\mathbf h
+\frac12\mathbf h^TH_f(\mathbf x)\mathbf h
+R_2.
$$

若線段上三階導數有界，則$R_2=O(\|\mathbf h\|^3)$。一階式的餘項則是$O(\|\mathbf h\|^2)$。大$O$敘述必須附帶足夠光滑性及局部有界條件。

### Taylor公式導出有限差分

在一維，

$$
f(x+h)=f(x)+hf'(x)+\frac{h^2}{2}f''(x)+O(h^3),
$$

故前向差分為

$$
\frac{f(x+h)-f(x)}{h}=f'(x)+O(h).
$$

同理展開$f(x-h)$並相減：

$$
\frac{f(x+h)-f(x-h)}{2h}
=f'(x)+O(h^2).
$$

中心差分雖為二階一致，但總誤差約可粗略寫成

$$
E(h)\approx C_1h^2+C_2\frac{\epsilon_{\mathrm{mach}}}{h}.
$$

第一項隨$h$縮小，第二項反而增加。因此誤差對$h$通常呈先降後升的U形，而不是單調下降。

---

## 逐步手算例題

### 例一：梯度、Hessian與局部預測

令

$$
f(x,y)=x^2+xy+2y^2,
$$

在$\mathbf x_0=(1,2)^T$附近估計$f(1.01,1.98)$。

**第一步：求梯度。**

$$
\nabla f=
\begin{bmatrix}
2x+y\\
x+4y
\end{bmatrix},
\qquad
\nabla f(1,2)=
\begin{bmatrix}
4\\9
\end{bmatrix}.
$$

**第二步：求Hessian。**

$$
H_f=
\begin{bmatrix}
2&1\\
1&4
\end{bmatrix}.
$$

**第三步：設位移。**

$$
\mathbf h=
\begin{bmatrix}
0.01\\-0.02
\end{bmatrix}.
$$

一階變化為

$$
\nabla f^T\mathbf h=4(0.01)+9(-0.02)=-0.14.
$$

二階修正為

$$
\frac12\mathbf h^TH_f\mathbf h
=\frac12(0.0014)=0.0007.
$$

因$f$本身是二次多項式，三階導數為零，故二階Taylor式精確：

$$
f(1.01,1.98)=f(1,2)-0.14+0.0007=10.8607.
$$

這也說明「局部線性化」與「二階曲率修正」是不同層次的近似。

### 例二：偏導存在但不可微

定義

$$
g(x,y)=
\begin{cases}
\dfrac{x^2y}{x^4+y^2},&(x,y)\ne(0,0),\\
0,&(x,y)=(0,0).
\end{cases}
$$

沿$x$軸，$g(h,0)=0$，故$g_x(0,0)=0$；沿$y$軸，$g(0,h)=0$，故$g_y(0,0)=0$。

若只看偏導，可能誤判其梯度為零並可微。然而沿曲線$y=x^2$，

$$
g(x,x^2)=\frac{x^4}{2x^4}=\frac12
$$

對所有$x\ne0$成立。當$x\to0$時函數不趨近$0$，因此連續性已失敗，更不可能可微。此例顯示座標軸測試遠遠不足。

---

## 實作與程式

以下自足程式只需Python 3.10+與NumPy。它比較解析梯度、前向差分及中心差分，並拒絕非有限、形狀錯誤或非法步長。設定雜湊用來辨識實驗輸入，不代表物理正確性。

```python
import json
import hashlib
import numpy as np

def validate_point(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (2,):
        raise ValueError("x必須是形狀(2,)的座標")
    if not np.all(np.isfinite(x)):
        raise ValueError("座標必須全部有限")
    return x

def field(x):
    x = validate_point(x)
    a, b = x
    return a*a + a*b + 2.0*b*b

def analytic_gradient(x):
    x = validate_point(x)
    a, b = x
    return np.array([2.0*a + b, a + 4.0*b])

def numerical_gradient(fun, x, h, method="central"):
    x = validate_point(x)
    h = float(h)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h必須是有限正數")
    grad = np.empty(2)
    f0 = fun(x)
    if not np.isfinite(f0):
        raise ValueError("函數值非有限")

    for k in range(2):
        step = np.zeros(2)
        step[k] = h
        if method == "forward":
            grad[k] = (fun(x + step) - f0) / h
        elif method == "central":
            grad[k] = (fun(x + step) - fun(x - step)) / (2.0*h)
        else:
            raise ValueError("method必須是forward或central")

    if not np.all(np.isfinite(grad)):
        raise FloatingPointError("數值梯度含非有限值")
    return grad

def config_hash(config):
    text = json.dumps(
        config, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def main():
    x = np.array([1.0, 2.0])
    exact = analytic_gradient(x)
    steps = [1e-1, 1e-3, 1e-6, 1e-10, 1e-14]

    config = {
        "point_m": x.tolist(),
        "steps_m": steps,
        "field": "x^2+x*y+2*y^2",
        "dtype": "float64"
    }
    print("config_sha256 =", config_hash(config))
    print("analytic =", exact)

    for h in steps:
        for method in ("forward", "central"):
            approx = numerical_gradient(field, x, h, method)
            error = np.linalg.norm(approx - exact, ord=np.inf)
            print(h, method, approx, error)

if __name__ == "__main__":
    main()
```

這裡把座標記為公尺只是輸入契約；由於$f$未指定物理量綱，其係數其實隱含相應單位。真實模型不應把有量綱座標直接代入未標示單位的多項式。較妥善的做法是令$\hat x=x/L$、$\hat y=y/L$，再以參考場值$f_\mathrm{ref}$建立無因次函數。

---

## 測試與預期結果

以下均為依推導所得的**預期結果**，並未在此執行程式。

### 正常測試

在$(1,2)$，解析梯度應為$[4,9]^T$。對二次函數，中心差分在精確算術下會完全消去二次項，因此除浮點誤差外應接近解析值。前向差分的誤差應與$h$成正比。

### 步長過大

取$h=10^{-1}$時，前向差分會保留明顯的$O(h)$誤差。對一般非二次函數，中心差分亦會有可見的$O(h^2)$誤差。不能以「結果看似平滑」取代誤差比較。

### 步長過小

當$h$接近$10^{-14}$時，$f(x+h)-f(x-h)$是近似相等數相減；float64捨入誤差可能主導，數值梯度反而偏離解析值。不同硬體或函式尺度可使最低誤差點改變。

### 故障與拒絕測試

下列輸入應明確失敗：

```python
# 預期：ValueError
numerical_gradient(field, [1.0, np.nan], 1e-4)

# 預期：ValueError
numerical_gradient(field, [1.0, 2.0], 0.0)

# 預期：ValueError
numerical_gradient(field, [1.0, 2.0, 3.0], 1e-4)

# 預期：ValueError
numerical_gradient(field, [1.0, 2.0], 1e-4, "unknown")
```

不可微函數的有限差分可能回傳有限數字，但這不證明導數存在。應額外比較多方向、多路徑與多個步長。

---

## 除錯與常見陷阱

1. **把偏導存在當成可微。**  
   應檢查連續性及線性餘項是否為$o(\|\mathbf h\|)$。

2. **把梯度寫成橫列後混用矩陣乘法。**  
   本書梯度是列向量，方向導數寫作$\nabla f^T\mathbf v$。

3. **盲目縮小步長。**  
   應掃描一系列對數間隔步長，尋找截斷與捨入誤差的平衡區。

4. **只測一個點或一個方向。**  
   錯誤公式可能在對稱點偶然通過。至少測一般點、零點、尺度不同的點及邊界附近。

5. **混淆單位。**  
   若$x$以m、$y$以s表示，$\partial f/\partial x$與$\partial f/\partial y$單位不同，不能不經縮放直接比較大小。

6. **把有限差分核對當成物理驗證。**  
   梯度核對只驗證導數實作，既不證明模型適合池域，也不證明參數來自現場。

7. **對非光滑函數宣稱二階收斂。**  
   Taylor階數依賴足夠光滑性；尖點、跳躍與分段模型可能降低甚至破壞觀測階。

---

## 養殖與相場案例

### 合成池域的溫度敏感度

設合成溫度場

$$
T(x,y)=298+0.4\frac{x}{L_x}-0.2\left(\frac{y}{L_y}\right)^2\ \mathrm{K}.
$$

則

$$
\nabla T=
\begin{bmatrix}
0.4/L_x\\
-0.4y/L_y^2
\end{bmatrix}\mathrm{K/m}.
$$

若$L_x=10\,\mathrm m$、$L_y=5\,\mathrm m$，在$y=2\,\mathrm m$處，

$$
\nabla T=
\begin{bmatrix}
0.04\\-0.032
\end{bmatrix}\mathrm{K/m}.
$$

這只描述局部空間變化，不代表熱通量；要得到熱通量還需導熱係數及Fourier定律。合成場也不是現場驗證資料。

### 相場自由能密度

本卷採無因次序參量$\phi$及

$$
W(\phi)=\frac{(\phi^2-1)^2}{4}.
$$

其導數與二階導數為

$$
W'(\phi)=\phi^3-\phi,\qquad
W''(\phi)=3\phi^2-1.
$$

在$\phi=\pm1$，$W'=0$且$W''=2>0$，為局部極小；在$\phi=0$，$W'=0$且$W''=-1<0$，為局部極大。這個Hessian判別只針對局部自由能密度。完整能量還含$\kappa|\nabla\phi|^2/2$，其變分會產生空間耦合。養殖溶氧跨越管理閾值並不是這種物理相變，不可借用雙井術語混淆。

---

## 習題

1. **手算題**  
   對$f(x,y)=e^x\cos y$，求$(0,0)$的梯度、Hessian，並用二階Taylor式估計$f(0.1,0.2)$。

2. **程式題**  
   修改程式以核對$f(x,y)=\sin x\,e^y$的梯度。使用$h=10^{-1},\ldots,10^{-12}$，分別記錄前向與中心差分誤差，不宣稱未實際算得的收斂率。

3. **反例題**  
   判斷$f(x,y)=\sqrt{x^2+y^2}$在原點的兩個偏導是否存在，以及函數是否可微。

4. **整合題**  
   設$\mathbf F(x,y)=[x^2y,\;x+\sin y]^T$。求Jacobian，並計算在$(1,0)$受到$\mathbf h=(0.01,-0.02)^T$擾動時的一階輸出變化。說明此預測不包含哪些誤差。

---

## 習題解答

### 第一題

$$
\nabla f=
\begin{bmatrix}
e^x\cos y\\
-e^x\sin y
\end{bmatrix},
\qquad
\nabla f(0,0)=
\begin{bmatrix}
1\\0
\end{bmatrix}.
$$

$$
H_f=
\begin{bmatrix}
e^x\cos y&-e^x\sin y\\
-e^x\sin y&-e^x\cos y
\end{bmatrix},
\qquad
H_f(0,0)=
\begin{bmatrix}
1&0\\0&-1
\end{bmatrix}.
$$

令$\mathbf h=(0.1,0.2)^T$：

$$
f(\mathbf h)\approx
1+0.1+\frac12(0.1^2-0.2^2)=1.085.
$$

這是二階近似，餘項受三階導數控制。

### 第二題

解析梯度為

$$
\nabla f=
\begin{bmatrix}
\cos x\,e^y\\
\sin x\,e^y
\end{bmatrix}.
$$

將程式中的`field`與`analytic_gradient`替換即可。預期前向差分在截斷誤差主導區呈一階趨勢，中心差分呈二階趨勢；極小$h$處兩者都可能因捨入誤差惡化。實際觀測階必須由運行資料計算，不能預先捏造。

### 第三題

沿$x$方向，

$$
\frac{f(h,0)-f(0,0)}{h}=\frac{|h|}{h},
$$

左右極限分別為$1$與$-1$，故$f_x(0,0)$不存在。同理$f_y(0,0)$也不存在，因此不可微。雖然函數在原點連續，但連續不保證可微。

### 第四題

$$
J_{\mathbf F}(x,y)=
\begin{bmatrix}
2xy&x^2\\
1&\cos y
\end{bmatrix}.
$$

在$(1,0)$，

$$
J_{\mathbf F}=
\begin{bmatrix}
0&1\\
1&1
\end{bmatrix}.
$$

故

$$
\Delta\mathbf F\approx J_{\mathbf F}\mathbf h
=
\begin{bmatrix}
-0.02\\
-0.01
\end{bmatrix}.
$$

此結果忽略二階及更高階Taylor餘項；若輸入、參數或模型本身不準確，還另有資料誤差與模型誤差。若用有限差分計算Jacobian，則再加入截斷及浮點誤差。

---

## 本章小結

偏導數只描述座標方向變化；可微性則要求所有小擾動共享同一個線性近似。純量函數以梯度表達一階變化，以Hessian表達二階曲率；向量映射則由Jacobian連接輸入與輸出擾動。Taylor公式不只是形式展開，也是有限差分、一致性分析與局部敏感度的根據。

數值核對時必須同時防範步長過大造成的截斷誤差，以及步長過小造成的浮點消去。解析梯度與有限差分一致，只能證明局部實作通過一項verification，不能證明物理模型有效。不可微反例又提醒我們：得到一個有限數值，不等於相應數學導數存在。

---

## 參考來源

1. FiPy文件，〈Finite Volume Discretization〉：https://pages.nist.gov/fipy/en/latest/numerical/discret.html  
   本章僅以其作為後續有限體積離散背景；本章Taylor推導自足。
2. NumPy文件，數值陣列與浮點運算相關API：https://numpy.org/doc/stable/  
3. NumPy Fourier變換慣例：https://numpy.org/doc/stable/reference/routines.fft.html  
   留供後續週期算子章使用，本章不依賴FFT。
4. FiPy Cahn–Hilliard示例：https://pages.nist.gov/fipy/en/latest/generated/examples.cahnHilliard.mesh2D.html  
   其序參量與參數慣例未必等同本卷的$[-1,1]$雙井定義，不可直接移植。