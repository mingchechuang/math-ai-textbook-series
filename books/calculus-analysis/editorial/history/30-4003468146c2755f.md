# 第30章 整合專題：可稽核的感測校準與分析

## 學習目標與先備知識

本章把局部微分、可辨識性、約束最佳化、積分單位與動態敏感度，組成一條可檢查、可重現的校準證據鏈。案例是完全合成的雙參數感測校準，不代表任何現場設備的性能，也不構成操作建議。

完成本章後，讀者應能：

1. 以明確的輸入、輸出和單位，寫出校準模型與殘差。
2. 以 Jacobian 的秩和奇異值判斷局部可辨識性的條件與限制。
3. 區分局部線性近似、有限差分核對與可微性證明。
4. 在參數界限下描述最小平方問題，並辨認約束活躍時的必要條件。
5. 追蹤時間積分的單位，並分清儀器讀值、模型預測與累積量。
6. 設計只讀分析流程，保存假設、尺度、誤差和失敗訊號，而不控制設備。

先備知識包括向量與矩陣、偏導與 Jacobian、梯度、Taylor 線性近似、最小平方和定積分。所有資料與參數均為合成值；SI 單位依量綱書寫。

## 問題與直覺

設有兩個無因次校準參數 $\theta_1,\theta_2$，以及兩支輸出單位為 $\mathrm{mg/L}$ 的感測通道。讀值不只由參數決定，也會隨合成環境量 $T$ 改變。模型取為

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

為了避免把量測直接當成真值，另外給定獨立合成參考值 $y(T)\in\mathbb R^2$。多個校準時刻的資料為 $(T_k,y_k)$，模型殘差定義為

$$
r_k(\theta)=s(T_k;\theta)-y_k.
$$

參數以無因次尺度表示，兩個通道則同以 $\mathrm{mg/L}$ 表示，因此本例中的 Euclidean 殘差範數至少具有一致的輸出單位。若通道單位不同或量測不確定度不同，應先明示標準化或權重；不能直接把混合物理單位的數字當成有意義的距離。

直覺上，若兩參數對輸出的影響方向幾乎相同，改變一個參數便能由另一個抵消，校準會對雜訊很敏感。增加資料點並非必然解決問題：重複同一環境下的同一種訊息，可能只降低隨機誤差，卻沒有提供新的參數方向。可稽核分析因此不只問「最小平方值多少」，還要問「資料在什麼條件下能區分參數」、「近似在哪個範圍有效」與「失敗時能否被辨認」。

## 定義、定理與推導

### 校準映射、Jacobian 與單位

對一般映射 $f:U\subset\mathbb R^n\to\mathbb R^m$，若 $U$ 開，$f$ 在 $\theta$ 可微是指存在線性映射 $J_f(\theta)\in\mathbb R^{m\times n}$，使得

$$
f(\theta+h)=f(\theta)+J_f(\theta)h+\rho(h),
\qquad
\frac{\|\rho(h)\|_2}{\|h\|_2}\longrightarrow 0
\quad(h\to0,\ h\ne0).
$$

這裡的 $h$ 是 $n\times1$ 直向量。Jacobian 的第 $i,j$ 元單位為「第 $i$ 個輸出單位／第 $j$ 個輸入單位」。本例參數無因次，所以 $J_s$ 每一元素的單位為 $\mathrm{mg/L}$；環境量的偏導數則另帶有每攝氏度的單位。

令

$$
A(T)=
\begin{pmatrix}
1+0.1T&0.2\\
0.3&0.9-0.05T
\end{pmatrix},
\qquad
b=
\begin{pmatrix}
1\\0.8
\end{pmatrix}.
$$

模型 $s(T;\theta)=b+A(T)\theta$ 對 $\theta$ 是仿射函數，故其參數 Jacobian 為

$$
J_\theta s(T;\theta)=A(T).
$$

對不同 $T_k$ 堆疊資料，定義

$$
F(\theta)=
\begin{pmatrix}
s(T_1;\theta)\\
\vdots\\
s(T_N;\theta)
\end{pmatrix},
\qquad
Y=
\begin{pmatrix}
y_1\\
\vdots\\
y_N
\end{pmatrix}.
$$

則 $F(\theta)=B\theta+c$，其中 $B$ 是把各個 $A(T_k)$ 依序縱向堆疊而成的 $2N\times2$ 矩陣，$c$ 則堆疊相同的基準向量 $b$。無權重平方損失為

$$
L(\theta)=\frac12\|F(\theta)-Y\|_2^2,
\qquad
\nabla L(\theta)=B^T(F(\theta)-Y),
\qquad
\nabla^2L(\theta)=B^TB.
$$

若有可信的誤差標準差 $\sigma_i>0$，可令 $W=\operatorname{diag}(\sigma_i^{-2})$，使用 $L_W=\tfrac12(F-Y)^TW(F-Y)$。此時權重提供無因次的加權殘差平方；其意義取決於標準差模型是否合理，不能只為改善數值結果而任意調整。

### 定義：局部可辨識性的線性化檢查

若 $J_F(\theta_*)$ 對參數具有滿列秩 $n$，則在 $\theta_*$ 附近，參數的小變化會產生彼此可區分的線性化輸出變化。這是局部可辨識的常用充分檢查，不等於整個非線性模型在全域一對一。

本章案例對參數是仿射的，因此若 $B$ 滿列秩，映射 $\theta\mapsto B\theta+c$ 在全域一對一。反過來，若 $B$ 不滿列秩，存在非零 $v$ 使 $Bv=0$，於是所有 $\theta$ 與 $\theta+tv$ 都有同一預測輸出，參數便無法由這批資料唯一決定。對一般非線性模型，即使某點 Jacobian 可逆，也只支持該點附近的局部結論，不排除遠處存在另一組參數得到相同輸出。

秩是定性的；奇異值則描述穩定性。最小奇異值很小時，資料雖可能在精確算術下滿秩，反演仍可能放大量測誤差。須一併報告參數尺度、輸出尺度、單位與所用範數，才可解讀奇異值與條件數。

### 命題：滿列秩保證本例無約束平方損失有唯一極小點

**命題。** 若合成設計矩陣 $B\in\mathbb R^{m\times n}$ 滿列秩，則對任意 $Y,c\in\mathbb R^m$，

$$
L(\theta)=\frac12\|B\theta+c-Y\|_2^2
$$

有唯一全域極小點 $\widehat\theta$，且滿足 $B^T(B\widehat\theta+c-Y)=0$。

**證明。** 對任意非零直向量 $v\in\mathbb R^n$，因 $B$ 滿列秩，$Bv\ne0$。故

$$
v^TB^TBv=\|Bv\|_2^2>0.
$$

所以 $B^TB$ 正定，損失的 Hessian 正定。由 $L$ 是有限維二次函數可知其梯度為 $B^T(B\theta+c-Y)$。正定 Hessian 表示 $L$ 嚴格凸，且二次項沿每個非零方向均正；因此損失在遠離原點時趨向無窮，必有極小點，嚴格凸性又保證該點唯一。極小點的梯度必為零，得到所述方程。這是本模型下的解析結論；若 $B$ 不滿列秩，該命題的關鍵前提失效，不能沿用唯一性結論。$\square$

### 參數界限與約束最佳化

若參數具有合成容許範圍，例如

$$
0\le\theta_1\le2,\qquad 0\le\theta_2\le2,
$$

則問題是在線段組成的閉矩形上最小化 $L$。由於該集合在有限維歐氏空間中閉且有界，故緊緻；連續函數 $L$ 在其上取得最小值。但全域最小點可能位於邊界，不能只解無約束梯度方程。

可以列舉內部、四條邊與四個角落候選點，或使用具界限的最佳化方法。所得結果須核對參數是否可行、梯度是否符合邊界方向的必要條件、損失是否未增加，以及是否遭遇病態或數值失敗。求解器回報成功不會自動證明資料合理、模型正確或解是全域最佳。

### 積分量與動態敏感度的證據界線

若某通道的濃度讀值 $C(t)$ 以 $\mathrm{mg/L}$ 表示，水量 $Q(t)$ 以 $\mathrm{L/s}$ 表示，則流量加權輸送率

$$
p(t)=C(t)Q(t)
$$

的單位為 $\mathrm{mg/s}$，其時間積分 $\int_{t_0}^{t_1}p(t)\,dt$ 的單位為 $\mathrm{mg}$。若要把讀值換成區間總量，必須說明時間單位、取樣代表方式、缺值處理，以及 $C,Q$ 是否在區間內經過適當校準。梯形法只是給定端點資料下的一種近似；未有光滑性或誤差界條件時，有限網格的答案不能冒充真實積分的證明。

對動態狀態 $z(t)$，若狀態方程和輸入也依參數 $\theta$ 改變，動態敏感度 $S(t)=\partial z(t)/\partial\theta$ 通常須由變分方程、差分近似或其他有明確假設的方法取得。有限時間模擬或參數有限差分只能提供數值證據。它不能單獨證明對所有時間穩定，也不能把數學模型的穩定性等同操作安全。

## 逐步手算例題

### 例一：單一環境點的局部可辨識性

取 $T=0^\circ\mathrm C$，則

$$
A(0)=
\begin{pmatrix}
1&0.2\\
0.3&0.9
\end{pmatrix},
\qquad
\det A(0)=0.9-0.06=0.84\ne0.
$$

因此兩個通道的線性變化方向獨立，這一環境下的兩參數線性模型滿秩。在本例的仿射模型中，若無雜訊且兩個輸出參考值一致，參數能唯一反解。

例如取 $\theta=(0.5,1)^T$，預測為

$$
s(0;\theta)=
\begin{pmatrix}
1\\0.8
\end{pmatrix}
+
\begin{pmatrix}
1&0.2\\0.3&0.9
\end{pmatrix}
\begin{pmatrix}0.5\\1\end{pmatrix}
=
\begin{pmatrix}1.7\\1.85\end{pmatrix}\ \mathrm{mg/L}.
$$

小參數擾動 $h=(0.01,-0.02)^T$ 造成輸出變化

$$
A(0)h=
\begin{pmatrix}0.006\\-0.015\end{pmatrix}\ \mathrm{mg/L}.
$$

因為模型對參數恰為仿射，這個線性變化不是近似，而是精確變化。不能將此精確性推廣到未寫進模型的真實感測過程。

### 例二：兩個時刻的合成平方損失

取 $T_1=0^\circ\mathrm C$、$T_2=10^\circ\mathrm C$。堆疊設計矩陣為

$$
B=
\begin{pmatrix}
1&0.2\\
0.3&0.9\\
2&0.2\\
0.3&0.4
\end{pmatrix}.
$$

取合成參數 $\theta_*=(0.5,1)^T$，並假設參考值正好等於模型值，則 $Y=B\theta_*+c$。對估計值 $\theta=(0.6,0.9)^T$，誤差為 $h=(0.1,-0.1)^T$，堆疊殘差為

$$
Bh=
\begin{pmatrix}
0.08\\-0.06\\0.18\\-0.01
\end{pmatrix}\ \mathrm{mg/L}.
$$

因此

$$
L(\theta)=\frac12(0.08^2+0.06^2+0.18^2+0.01^2)
=0.0225\ (\mathrm{mg/L})^2.
$$

手算顯示損失是加總所有輸出殘差平方，不是先把各輸出平均後再平方。若不同通道的誤差尺度不同，需先說明權重。

此設計的兩個欄向量為 $u=(1,0.3,2,0.3)^T$、$v=(0.2,0.9,0.2,0.4)^T$。其 Gram 矩陣為

$$
B^TB=
\begin{pmatrix}
5.18&0.99\\
0.99&1.05
\end{pmatrix}.
$$

行列式為

$$
5.18(1.05)-0.99^2=4.4589>0.
$$

首項為正且行列式為正，故此對稱矩陣正定；這再次核對滿列秩與命題中的唯一性條件。

## 實作與程式

以下程式只使用 Python 標準函式庫與 NumPy，在 CPU 上計算本章的合成模型、解析 Jacobian、中心有限差分核對、殘差比、梯形積分與簡單的盒限制投影梯度下降。此程式不是通用最佳化器，不檢查真實儀器，也不連接任何設備。此處未執行程式；結果是理論預期，須在實際執行環境獨立核對。

```python
import numpy as np

# theta: 無因次；s: mg/L；T: 攝氏度。
def A_of_T(T):
    T = float(T)
    return np.array([
        [1.0 + 0.1 * T, 0.2],
        [0.3, 0.9 - 0.05 * T],
    ], dtype=float)

def predict(theta, T):
    theta = np.asarray(theta, dtype=float).reshape(2, 1)
    base = np.array([[1.0], [0.8]])
    return (base + A_of_T(T) @ theta).reshape(2)

def stacked_prediction(theta, temperatures):
    return np.concatenate([predict(theta, T) for T in temperatures])

def design_matrix(temperatures):
    return np.vstack([A_of_T(T) for T in temperatures])

def residual_vector(theta, temperatures, references):
    references = np.asarray(references, dtype=float).reshape(-1)
    return stacked_prediction(theta, temperatures) - references

def loss_and_gradient(theta, temperatures, references):
    B = design_matrix(temperatures)
    r = residual_vector(theta, temperatures, references)
    loss = 0.5 * float(r @ r)
    gradient = B.T @ r
    return loss, gradient

def central_jacobian(fun, x, step=1e-6):
    x = np.asarray(x, dtype=float).reshape(-1)
    y0 = np.asarray(fun(x), dtype=float).reshape(-1)
    J = np.empty((y0.size, x.size), dtype=float)
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = step
        J[:, j] = (
            np.asarray(fun(x + e)).reshape(-1)
            - np.asarray(fun(x - e)).reshape(-1)
        ) / (2.0 * step)
    return J

def projected_gradient_descent(
    temperatures, references, initial, lower, upper,
    step_size=0.05, max_iter=2000, grad_tol=1e-10
):
    theta = np.clip(
        np.asarray(initial, dtype=float),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    history = []

    for _ in range(max_iter):
        loss, grad = loss_and_gradient(theta, temperatures, references)
        trial = np.clip(theta - step_size * grad, lower, upper)
        trial_loss, _ = loss_and_gradient(trial, temperatures, references)
        history.append((loss, float(np.linalg.norm(grad)), theta.copy()))

        # 固定步長版本遇到不下降便停止並回報，不暗中稱為收斂。
        if trial_loss > loss + 1e-14:
            return theta, history, "step_rejected"
        if np.array_equal(trial, theta):
            return theta, history, "projected_step_zero"
        theta = trial
        if np.linalg.norm(grad) <= grad_tol:
            return theta, history, "gradient_tolerance"

    return theta, history, "iteration_limit"

def trapezoid_integral(time_s, rate_mg_per_s):
    time_s = np.asarray(time_s, dtype=float)
    rate_mg_per_s = np.asarray(rate_mg_per_s, dtype=float)
    if time_s.ndim != 1 or rate_mg_per_s.ndim != 1:
        raise ValueError("輸入必須是一維時間序列")
    if time_s.size != rate_mg_per_s.size or time_s.size < 2:
        raise ValueError("時間與率必須等長，且至少含兩點")
    if not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(rate_mg_per_s)):
        raise ValueError("不接受 NaN 或無限值")
    dt = np.diff(time_s)
    if np.any(dt <= 0):
        raise ValueError("時間必須嚴格遞增")
    return float(np.sum(0.5 * (rate_mg_per_s[:-1] + rate_mg_per_s[1:]) * dt))

if __name__ == "__main__":
    temperatures = np.array([0.0, 10.0])
    true_theta = np.array([0.5, 1.0])
    references = stacked_prediction(true_theta, temperatures)

    theta0 = np.array([0.6, 0.9])
    B = design_matrix(temperatures)
    analytic_J = B
    numeric_J = central_jacobian(
        lambda th: stacked_prediction(th, temperatures), theta0
    )
    jacobian_error = np.linalg.norm(analytic_J - numeric_J, ord=2)

    residual = residual_vector(theta0, temperatures, references)
    loss, gradient = loss_and_gradient(theta0, temperatures, references)
    linear_prediction = stacked_prediction(theta0, temperatures) + B @ (
        true_theta - theta0
    )
    remainder = (
        stacked_prediction(true_theta, temperatures) - linear_prediction
    )
    denominator = np.linalg.norm(true_theta - theta0)
    remainder_ratio = np.linalg.norm(remainder) / denominator

    fit, history, status = projected_gradient_descent(
        temperatures, references,
        initial=np.array([0.0, 0.0]),
        lower=np.array([0.0, 0.0]),
        upper=np.array([2.0, 2.0]),
    )

    area_mg = trapezoid_integral(
        np.array([0.0, 1.0, 2.0]),
        np.array([2.0, 2.0, 2.0]),
    )

    print("解析 Jacobian：\n", analytic_J)
    print("有限差分 Jacobian 誤差：", jacobian_error)
    print("參數擾動：", true_theta - theta0)
    print("殘差：", residual)
    print("損失：", loss)
    print("梯度：", gradient)
    print("局部近似餘項比：", remainder_ratio)
    print("估計參數、狀態：", fit, status)
    print("常率積分預期為 4 mg；程式值：", area_mg)
```

在這個線性參數模型中，只要浮點誤差可忽略，參數方向的餘項應接近零；在任意步長與有限精度下，不應要求浮點結果恰為零。有限差分 Jacobian 誤差也受步長與捨入影響，不宜只挑一個步長就宣稱導數已被證明。程式中的固定步長投影法只適合此小型教學案例；若回報 `step_rejected` 或 `iteration_limit`，應記為失敗訊號，而不是略過。

## 測試與預期結果

| 類別 | 測試 | 預期與判讀 |
|---|---|---|
| 正常 | 以兩個時刻建立 $B$，比較解析 Jacobian 與中心差分 | Jacobian 形狀應為 $4\times2$；差異應受浮點精度與差分步長影響 |
| 正常 | 使用例二的 $\theta_*$ 作參考，再由界限內初值估計 | 損失應下降並接近零；不能以「接近」取代參數可辨識條件的檢查 |
| 正常 | 對常率 $2\,\mathrm{mg/s}$，在 $[0,2]$ 秒作梯形積分 | 預期為 $4\,\mathrm{mg}$ |
| 邊界 | 令初值超出參數界限 | 投影後的初值應落在界限上；若解位於邊界，應報告活躍界限 |
| 邊界 | 只用單一通道且設計矩陣只有一列 | 兩個參數不可能由一個純量方程唯一決定；檢查程序不得假稱滿列秩 |
| 故障 | 時間點重複、倒序、資料含 NaN | 積分函數預期明確拒絕輸入，不應回傳看似有效的總量 |
| 故障 | 固定步長令試探步造成損失增加 | 最佳化器應回報 `step_rejected`；應調整演算法或檢查尺度，不能偽報成功 |
| 故障 | 用多組高度相似的敏感度欄向量校準 | 矩陣即使形式上滿秩，最小奇異值仍可能很小；應標記病態與對雜訊敏感 |

表列的是預期測試，不是已執行的紀錄。實際報告應保留程式版本、資料、參數界限、停止狀態、Jacobian 檢查方法、損失定義與未通過測試，不得事後刪除不利結果。

## 反例與常見陷阱

1. **有限個方向不能證明可微。** 即使沿多條線測得的導數吻合，也不能推出所有趨近方向的餘項比都趨零。必須證明所需的極限，或清楚標示結果僅是數值支持。
2. **偏導存在不等於可辨識。** 可微模型在某點的 Jacobian 若不滿列秩，線性化有不可區分的參數方向；即便殘差為零，也可能有多組參數。
3. **滿秩不等於穩定。** 例如兩欄幾乎平行的設計矩陣仍可能滿秩，但微小量測誤差足以造成很大的參數改變。應檢查尺度化後的奇異值與條件數。
4. **局部可逆不等於全域唯一。** 對一般非線性校準模型，可逆 Jacobian 保證的是適當鄰域內的局部結論，不足以排除其他相距甚遠的可行解。
5. **梯度為零不代表受限最小。** 在參數盒的邊界，需檢查可行方向上的一階條件；只將無約束梯度解截斷，不等於解了受約束問題。
6. **積分近似不等於積分真值。** 若以不規則取樣資料套用梯形法，漏掉資料缺口或時間單位換算，結果即使數值精確也可能量綱錯誤。沒有誤差估計時須標為近似。
7. **模型穩定不等於現場安全。** 離散資料上的殘差、敏感度或狀態軌跡，只支持指定模型和指定範圍內的數學檢查，不能替代儀器驗證、風險評估或合格人員判斷。
8. **用未加權 Euclidean 範數混合單位。** 若一個分量以攝氏度、另一個以毫克每升表示，直接平方相加沒有一致的物理意義。應依量測不確定度標準化，並同時保留原始單位報告。
9. **把差分吻合說成解析證明。** 數值導數取決於步長、捨入與實際程式路徑；差分只能核對特定點附近的實作，不能證明全域性質。

## AI、幾何與養殖案例

### 可稽核證據鏈

合成養殖案例可假設通道輸出是某種濃度指標，環境量是水溫，參數是兩個無因次校準係數。應將下列資訊一併保存：

- 原始讀值、參考值、單位、時間戳記與資料缺失標記。
- 前處理、模型公式、參數尺度與參數界限。
- Jacobian 的解析式或計算方式、數值核對方法與測試步長。
- 殘差、損失、秩、奇異值與病態警訊。
- 積分的被積函數、時間單位、取樣間隔與近似方法。
- 求解器停止狀態、失敗測試、人工覆核狀態與限制說明。

AI agent 在此案例中是**唯讀分析器**：可以整理資料、重新計算診斷量、產生差異報告，並指出證據不足的環節；不可以直接寫入儀器參數、啟動控制程序、改變投餌或加藥。當校準結果異常、模型不一致、資料缺失，或程式測試未通過時，應停止自動結論並交由適當人員覆核。

### 幾何與數值尺度

在此問題中，$B$ 的每個欄向量代表一個參數對多個輸出的影響方向。兩欄的夾角越小，參數越難區分；但夾角與條件數會受參數及輸出的尺度選擇影響。因此，應先說明參數如何無因次化、輸出如何按誤差尺度標準化，再解讀列空間或奇異值。若採用變數替換 $\theta=Dq$，則新 Jacobian 為 $BD$；這會改變數值尺度，不會憑空增加實際量測資訊。

### 動態敏感度

若校準映射取用一段時間的讀值而非單一時刻，觀測映射可寫為 $F(\theta)=\mathcal O(z(\cdot;\theta))$，其中 $z$ 是動態狀態，$\mathcal O$ 是取樣或積分觀測方式。此時 Jacobian 的每欄可解讀為一個參數擾動對觀測序列的局部影響。要使這個解讀成立，須先明確說明狀態方程、初始條件、觀測時間和參數依賴關係。有限差分軌跡只能核對所選參數與時間範圍，不能替代動態方程的存在唯一條件或穩定性證明。

## 習題

1. **手算。** 對 $T=0$，求模型在 $\theta=(0.5,1)^T$ 的預測。再計算參數擾動 $h=(0.02,0)^T$ 對兩個輸出的變化，並解釋其單位。
2. **手算與可辨識性。** 單獨使用 $T=10$ 時刻的兩個通道，求 $A(10)$ 的行列式，判斷本例仿射模型能否唯一辨認兩個參數。
3. **反例。** 一個校準實驗只有一個通道讀值，依賴兩個參數而且其 Jacobian 是 $J=(1,2)$。證明單筆精確讀值仍不能唯一決定兩參數；指出哪項秩條件失敗。
4. **程式分析。** 說明中心差分 Jacobian 為何要逐個輸入座標擾動。若把差分步長從 $10^{-6}$ 改成極小數，結果一定更準嗎？列出兩個相反方向的誤差來源。
5. **整合與單位。** 假設 $C(t)$ 以 $\mathrm{mg/L}$ 表示、$Q(t)$ 以 $\mathrm{L/s}$ 表示，求 $C(t)Q(t)$ 和 $\int_0^{60} C(t)Q(t)\,dt$ 的單位。若把時間以分鐘輸入而仍把 $Q$ 當作每秒流量，會發生什麼錯誤？
6. **整合與失效報告。** 一次受界限校準給出邊界解，最小奇異值很小，且固定步長程式回報 `step_rejected`。寫出一段不少於三項的可稽核報告內容，並說明為何不能直接宣稱校準成功。

## 習題解答

1. **解。** 預測為
   $$
   \begin{pmatrix}1\\0.8\end{pmatrix}
   +
   \begin{pmatrix}1&0.2\\0.3&0.9\end{pmatrix}
   \begin{pmatrix}0.5\\1\end{pmatrix}
   =
   \begin{pmatrix}1.7\\1.85\end{pmatrix}\ \mathrm{mg/L}.
   $$
   因模型對參數仿射，擾動造成的變化精確等於
   $$
   A(0)\begin{pmatrix}0.02\\0\end{pmatrix}
   =
   \begin{pmatrix}0.02\\0.006\end{pmatrix}\ \mathrm{mg/L}.
   $$
   參數無因次，因此 Jacobian 的輸出單位為 $\mathrm{mg/L}$，擾動後輸出仍以該單位表示。
2. **解。** $A(10)=\begin{pmatrix}2&0.2\\0.3&0.4\end{pmatrix}$，所以
   $$
   \det A(10)=2(0.4)-0.2(0.3)=0.74\ne0.
   $$
   該單一環境時刻的兩通道矩陣可逆。因此對本例的無雜訊仿射模型，若參考讀值相容，兩個參數唯一。這個結論不代表實際量測不受雜訊影響，也不保證其他非線性模型全域一對一。
3. **解。** 方程只提供 $\theta_1+2\theta_2=d$ 一個條件。若 $(a,b)$ 是一組解，則對任意實數 $t$，$(a-2t,b+t)$ 也滿足同一方程。$1\times2$ 的 Jacobian 不可能滿列秩，故參數不可唯一辨識。
4. **解。** 每欄對應一個輸入座標方向的偏導，故需逐欄分別施加擾動。步長過大會使有限差分的截斷誤差增大；步長過小則可能使兩個幾乎相同的函數值相減而放大浮點捨入誤差。故不能只說步長越小越準；須結合尺度、精度、步長掃描與解析結果核對。
5. **解。** $CQ$ 的單位是 $\mathrm{mg/s}$；以秒為積分變數積分後是 $\mathrm{mg}$。若 $Q$ 仍按每秒解讀，時間卻以分鐘數值積分，數值會少一個 $60$ 倍的換算因子。應把時間換成秒，或把率改寫成每分鐘，並在報告中標明。
6. **解。** 報告至少應記錄：邊界參數值及哪些界限活躍；參數界限與目標函數定義；尺度化方法及最小奇異值診斷；固定步長拒絕步驟的狀態；殘差與資料品質；未通過測試及下一步覆核要求。小奇異值代表參數可能對雜訊高度敏感，`step_rejected` 又表示所用迭代程序沒有接受試探步；兩者都不能略去。邊界解也必須依受限問題的條件判讀。因此應報告「校準程序未能以此證據確認成功」，而不是只憑一個輸出參數宣稱有效。

## 本章小結

感測校準的可信度不由單一最小損失決定。可稽核的證據鏈至少包括明確的模型與單位、具有適當秩的 Jacobian、對條件數和尺度的檢查、正確處理參數約束、可說明誤差界線的局部近似，以及符合量綱的時間積分。

對本章的仿射模型，滿列秩可證明無約束平方損失具有唯一全域極小點；這是一項有清楚假設的數學結論。中心差分、求解器狀態與有限取樣則只是對程式和特定資料的檢查，不能冒充微分性、全域唯一性或現場可靠性的證明。所有資料皆為合成資料，AI agent 僅作唯讀分析，不控制設備或改變養殖操作。

## 參考來源

- A1. Jiří Lebl, *Basic Analysis*，作者目錄與教材入口：<https://www.jirka.org/ra/>
- A2. MIT OpenCourseWare, *18.100A Real Analysis*：<https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/>
- A3. MIT OpenCourseWare, *18.02SC Multivariable Calculus*：<https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/>
- A4. JAX, *Autodiff Cookbook: JVP/VJP*，延伸參考：<https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html>
- A6. SciPy, `minimize` API，延伸參考：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>

本章的核心程式僅使用 Python 與 NumPy，不要求 SciPy、JAX 或 GPU。參考來源提供延伸閱讀，不表示已逐條核對教材中的所有定理。