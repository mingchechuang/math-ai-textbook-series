# 第15章 預條件、GMRES與求解診斷

## 學習目標與先備知識

本章討論如何解非對稱稀疏線性系統，特別是由平流、反應與擴散離散後形成的系統。完成本章後，應能：

1. 說明為什麼非對稱矩陣通常不能直接使用共軛梯度法（CG），以及 GMRES 如何在 Krylov 子空間中尋找近似解。
2. 推導左、右預條件的殘差定義，分辨預條件殘差與原系統的真殘差。
3. 實作一個小型、重啟式 GMRES，並以 $r=b-Ax$、絕對及相對容差檢查解的可信度。
4. 用診斷辨識尺度不平衡、停滯、重啟設定不當與病態矩陣；不把殘差小誤當作解誤差必然小。
5. 區分代數求解誤差、離散誤差與模型誤差。

先備知識為矩陣向量乘法、線性方程組、向量範數，以及稀疏矩陣的基本概念。本文程式只需 Python 3.10+ 與 NumPy，限於 CPU 小問題；不要求 SciPy。所有程式輸出皆為依方程推導的預期結果，並非已執行或實測結果。

## 問題與直覺

設離散後的系統為

$$
Ax=b,
$$

其中 $A\in\mathbb{R}^{n\times n}$、$x,b\in\mathbb{R}^n$。若 $A$ 對稱正定，CG 可利用對稱性與正定性，高效地建立近似解。但平流項具有方向性，常使離散矩陣不對稱；此時 CG 的理論條件不成立，迭代甚至可能失去穩定的下降行為。

GMRES（generalized minimal residual method，廣義最小殘差法）在由起始殘差產生的 Krylov 子空間中，選擇令殘差二範數最小的近似解。其直覺不是「每一步都讓解更接近真解」，而是在可搜尋的方向組合裡，挑出令線性方程不平衡程度最小的候選解。理想算術下，未重啟 GMRES 最多在 $n$ 個步驟內得到精確解；但它需要儲存逐步擴大的基底，而且每次迭代的正交化成本會增加。

重啟 GMRES 限制每段迭代最多建立 $m$ 個 Krylov 方向，到達 $m$ 後以目前解重新開始。這降低記憶體及計算成本，但會丟棄已建立的搜尋方向，可能造成停滯。

預條件則試圖把問題轉換成較容易迭代的形式。若 $P$ 近似 $A$，理想情形下 $P^{-1}A$ 比 $A$ 更容易處理；但 $P^{-1}$ 不必真的形成矩陣，通常只需能有效解 $Pz=v$。預條件也不是越強越好：建構及套用成本、平行效率、數值穩定性，都可能抵銷迭代次數的減少。

## 數學與物理推導

給定初值 $x_0$，真殘差與相對殘差定義為

$$
r_0=b-Ax_0,\qquad
r_k=b-Ax_k,\qquad
\eta_k=\frac{\|r_k\|_2}{\|b\|_2}.
$$

若 $\|b\|_2=0$，相對殘差不能按此式定義，應改用明確的絕對殘差判準，或另行規定分母尺度。常見停止條件是

$$
\|b-Ax_k\|_2\le \mathrm{atol}+\mathrm{rtol}\,\|b\|_2,
$$

其中 $\mathrm{atol}$ 與 $\mathrm{rtol}$ 必須明示。小殘差代表 $x_k$ 符合方程至一定程度，卻不必然代表 $x_k$ 接近精確解。若 $A$ 可逆且 $\kappa_2(A)=\|A\|_2\|A^{-1}\|_2$，則

$$
\|x_k-x_*\|_2\le \|A^{-1}\|_2\|r_k\|_2.
$$

相對誤差界還受到條件數影響；對相對擾動，可得與 $\kappa_2(A)$ 相關的界，但其形式需以 $A$ 可逆及誤差相對量的適用條件為前提。病態系統即使殘差很小，解仍可能對右端微小誤差十分敏感。

### Krylov 子空間與最小殘差

令 $r_0=b-Ax_0$。第 $m$ 步 Krylov 子空間為

$$
\mathcal{K}_m(A,r_0)
=\operatorname{span}\{r_0,Ar_0,A^2r_0,\ldots,A^{m-1}r_0\}.
$$

GMRES 在 $x_0+\mathcal{K}_m(A,r_0)$ 中選擇 $x_m$，使 $\|b-Ax_m\|_2$ 最小。以 Arnoldi 正交化建立正交基底 $V_{m+1}$，使

$$
AV_m=V_{m+1}\overline{H}_m,
$$

其中 $\overline{H}_m$ 是上 Hessenberg 矩陣。若 $\beta=\|r_0\|_2$ 且 $v_1=r_0/\beta$，寫 $x_m=x_0+V_my$，則

$$
\|b-Ax_m\|_2
=\|\beta e_1-\overline{H}_m y\|_2.
$$

因此每個 GMRES 內迭代只需解一個小型最小平方問題，而不是重新解完整的 $n\times n$ 系統。

### 左、右預條件及真殘差

左預條件求解

$$
P^{-1}Ax=P^{-1}b,
$$

並在變換後的系統中最小化 $\|P^{-1}(b-Ax)\|_2$。右預條件則令 $x=P^{-1}y$，求解

$$
AP^{-1}y=b,
$$

其 Krylov 迭代最小化原系統殘差 $\|b-AP^{-1}y\|_2$。兩者的停止行為與最小化的量不同：左預條件 GMRES 的內部殘差可以很小，但原系統真殘差未必符合指定容差。因此工程實作應定期重新計算 $b-Ax$，並以真殘差作最終判定。

預條件在實務上通常不是組成 $P^{-1}$，而是提供近似求解器 $z\approx P^{-1}v$。本章小程式用稠密矩陣列出運算，是為了教學及小型測試；大問題應以稀疏矩陣乘法和稀疏預條件器處理。

### 平流擴散的非對稱來源

考慮一維有因次對流擴散方程

$$
\frac{\partial c}{\partial t}
+u\frac{\partial c}{\partial x}
=D\frac{\partial^2c}{\partial x^2}+s,
$$

其中 $x$ 為公尺、$t$ 為秒、速度 $u$ 為 $\mathrm{m/s}$、擴散係數 $D$ 為 $\mathrm{m^2/s}$。若 $c$ 為 $\mathrm{kg/m^3}$，則來源 $s$ 為 $\mathrm{kg/(m^3\,s)}$。隱式 Euler 及空間離散可形成

$$
\left(I+\Delta t\,uD_x-\Delta t\,D L\right)c^{n+1}
=c^n+\Delta t\,s^{n+1},
$$

其中 $D_x$ 是所選的一階差分算子，$L$ 是 Laplacian 離散。上式中平流項的正負號需依原方程移項及差分定義一致決定；迎風閉合、邊界閉合與網格都會影響矩陣係數。方向性平流通常造成非對稱性，因此不能只因矩陣稀疏便推定 CG 適用。

**逐步例題一：判斷對稱性與預條件必要性。** 令

$$
A=\begin{bmatrix}2&-1&0\\-3&4&-1\\0&-2&3\end{bmatrix}.
$$

第一步比較 $A_{12}=-1$ 與 $A_{21}=-3$，可知 $A\ne A^T$。第二步確認其來源可代表含方向性的鄰格耦合；這種結構不符合 CG 的對稱條件，故考慮 GMRES。第三步可用對角預條件 $P=\operatorname{diag}(2,4,3)$ 作為基線。它便宜且易解，但只調整各分量尺度，不能保證處理非對稱耦合。第四步求出候選解後，必須計算 $r=b-Ax$，不可只檢查預條件系統的內部指標。

**逐步例題二：手算殘差、停止門檻與病態警訊。** 取

$$
A=\begin{bmatrix}2&1\\0&1\end{bmatrix},
\quad
b=\begin{bmatrix}1\\1\end{bmatrix},
\quad
x_0=\begin{bmatrix}0\\0\end{bmatrix}.
$$

第一步得到 $r_0=b=[1,1]^T$，所以 $\|r_0\|_2=\sqrt{2}$。第二步令候選值 $x=[0.45,1]^T$，則

$$
Ax=\begin{bmatrix}1.9\\1\end{bmatrix},
\qquad
r=b-Ax=\begin{bmatrix}-0.9\\0\end{bmatrix},
\qquad
\|r\|_2=0.9.
$$

第三步若 $\mathrm{atol}=10^{-8}$、$\mathrm{rtol}=10^{-2}$，門檻為 $10^{-8}+10^{-2}\sqrt{2}\approx0.0141$，所以此候選解明確不合格。第四步解精確方程可得 $x_*=[0,1]^T$；即使某個近似殘差達標，也要注意矩陣條件數及資料尺度，不能將單一殘差指標當成模型正確性的證明。

## 逐步手算例題

下例展示一次無預條件 GMRES 如何轉成小型最小平方問題。取

$$
A=\begin{bmatrix}2&0\\1&1\end{bmatrix},
\quad b=\begin{bmatrix}1\\0\end{bmatrix},
\quad x_0=0.
$$

第一步為 $r_0=[1,0]^T$、$\beta=1$、$v_1=[1,0]^T$。第二步算得 $Av_1=[2,1]^T$。將其投影到 $v_1$ 得 $h_{11}=2$，扣除投影後餘量為 $[0,1]^T$，故 $h_{21}=1$、$v_2=[0,1]^T$。此時

$$
\overline H_1=\begin{bmatrix}2\\1\end{bmatrix},
\qquad
y_1=\arg\min_y\left\|
\begin{bmatrix}1\\0\end{bmatrix}
-\begin{bmatrix}2\\1\end{bmatrix}y\right\|_2.
$$

最小平方解為 $y_1=(2\cdot1+1\cdot0)/(2^2+1^2)=2/5$。因此 $x_1=x_0+v_1y_1=[0.4,0]^T$。真殘差為 $b-Ax_1=[0.2,-0.4]^T$，其範數為 $\sqrt{0.2}$。第二個 Krylov 方向可讓此二維例子解到精確值；但停止仍應依明示容差及真殘差，而不是只依步數。

## 實作與程式

以下程式使用 modified Gram–Schmidt 正交化與 Givens 旋轉，實作小型右預條件、重啟式 GMRES。`apply_Pinv` 表示預條件器套用函式，預設為恆等映射。實際的大型計算需注意有限精度下正交性可能退化，必要時採重新正交化或成熟函式庫。

```python
import numpy as np


def gmres(A, b, x0=None, restart=10, maxiter=100,
          atol=1e-12, rtol=1e-8, apply_Pinv=None):
    """教學用右預條件 GMRES；A 為小型稠密矩陣。"""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = b.size
    if A.shape != (n, n) or not np.all(np.isfinite(A)):
        raise ValueError("A 必須是有限值方陣")
    if not np.all(np.isfinite(b)):
        raise ValueError("b 必須是有限值向量")
    if restart < 1 or maxiter < 0 or atol < 0 or rtol < 0:
        raise ValueError("restart、迭代上限或容差不合法")

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    if x.shape != (n,) or not np.all(np.isfinite(x)):
        raise ValueError("x0 必須是有限值且形狀相符")
    if apply_Pinv is None:
        apply_Pinv = lambda v: v.copy()

    bnorm = np.linalg.norm(b)
    tol = atol + rtol * bnorm
    history = []
    iterations = 0

    while iterations < maxiter:
        r = b - A @ x
        true_norm = np.linalg.norm(r)
        history.append(true_norm)
        if true_norm <= tol:
            return x, history, True
        beta = true_norm
        m = min(restart, maxiter - iterations, n)
        V = np.zeros((n, m + 1))
        Z = np.zeros((n, m))
        H = np.zeros((m + 1, m))
        cs = np.zeros(m)
        sn = np.zeros(m)
        g = np.zeros(m + 1)
        V[:, 0] = r / beta
        g[0] = beta
        used = 0

        for j in range(m):
            z = np.asarray(apply_Pinv(V[:, j]), dtype=float)
            if z.shape != (n,) or not np.all(np.isfinite(z)):
                raise ValueError("預條件器輸出不合法")
            Z[:, j] = z
            w = A @ z

            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w -= H[i, j] * V[:, i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] > 0:
                V[:, j + 1] = w / H[j + 1, j]

            for i in range(j):
                a = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                q = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j], H[i + 1, j] = a, q

            den = np.hypot(H[j, j], H[j + 1, j])
            if den == 0:
                cs[j], sn[j] = 1.0, 0.0
            else:
                cs[j] = H[j, j] / den
                sn[j] = H[j + 1, j] / den
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            iterations += 1
            used = j + 1

            y = np.linalg.solve(H[:used, :used], g[:used])
            trial = x + Z[:, :used] @ y
            true_norm = np.linalg.norm(b - A @ trial)
            history.append(true_norm)
            if true_norm <= tol:
                return trial, history, True
            if iterations >= maxiter or den == 0:
                x = trial
                break

        if used == 0:
            break
        if iterations < maxiter:
            x = trial

    return x, history, np.linalg.norm(b - A @ x) <= tol


def main():
    A = np.array([[2.0, -1.0, 0.0],
                  [-3.0, 4.0, -1.0],
                  [0.0, -2.0, 3.0]])
    b = np.array([1.0, 2.0, 0.0])
    Pdiag = np.diag(A)
    apply_Pinv = lambda v: v / Pdiag

    x, history, ok = gmres(
        A, b, restart=3, maxiter=30,
        atol=1e-12, rtol=1e-10,
        apply_Pinv=apply_Pinv
    )
    true_r = b - A @ x
    print("converged:", ok)
    print("x:", x)
    print("true residual norm:", np.linalg.norm(true_r))
    print("direct-solve difference:",
          np.linalg.norm(x - np.linalg.solve(A, b)))
    print("residual history:", history)


if __name__ == "__main__":
    main()
```

這是教學實作，不是通用稀疏求解器。迴圈內為了方便展示，每一步都直接重算真殘差，成本較高；大型系統通常依可靠的遞推量更新，並在週期性檢查或停止時重算真殘差。程式沒有宣稱任何執行結果。若預期成功，殘差應低於給定門檻，且解應與同一矩陣的直接解相符至浮點誤差範圍；若實際不符，應先檢查程式版本、矩陣條件數、停滯及容差，不可將預期當成測試證據。

## 測試與預期結果

下列測試設計供讀者加入 `main` 或獨立測試程式。它們是可推導的預期，不表示已執行。

| 類別 | 測試 | 預期診斷 |
|---|---|---|
| 正常 | 以章內非對稱 $A$、有限 $b$ 求解，計算 `b - A @ x` | 若達停止條件，真殘差應小於 $\mathrm{atol}+\mathrm{rtol}\|b\|_2$；與直接解差異應受條件數及浮點精度影響。 |
| 邊界 | $b=0$、$x_0=0$ | 初始真殘差為零，應立即回報成功；不應除以 $\|b\|_2$。 |
| 邊界 | `restart=1`，限制 `maxiter` | 每段只保留一個方向，可能收斂慢或停滯；應回傳實際真殘差及未達標狀態，而非假報成功。 |
| 故障 | 對角線含零而仍用 `v / Pdiag` | 預條件器會產生非有限值；本教學程式應拒絕非有限輸出。呼叫者應改用有效預條件，而非忽略警告。 |
| 故障 | 傳入 NaN 的右端或不相符形狀 | 輸入檢查應拒絕。 |
| 故障 | 用 CG 求解非對稱平流矩陣 | CG 假設不成立；即使某次偶然得到小殘差，也不是有效的一般收斂保證。 |
| 停滯 | 故意使用很短重啟、粗糙預條件或高條件數系統 | 可能反覆達不到真殘差門檻；應記錄每次重啟的真殘差，並檢查預條件效果與尺度。 |

若做尺度測試，可把方程某一列乘常數，並相應調整 $b$，使精確解不變。此變換可能改變迭代數值行為及殘差範數；故應交代殘差採用何種尺度，並避免只看一個未縮放殘差。若為了診斷使用行尺度矩陣 $S$，則須分清測試的是 $SAx=Sb$，還是原系統 $Ax=b$；最後仍應回報原系統的 $b-Ax$。

## 除錯與常見陷阱

**把預條件殘差當真殘差。** 左預條件的停止量可能是 $\|P^{-1}(b-Ax)\|_2$。其量小不保證原系統真殘差小，特別是 $P$ 對不同分量縮放很不均勻時。輸出至少列出兩者，並以原系統真殘差決定最終合格與否。

**停止容差寫得含糊。** 只說「殘差小於 $10^{-8}$」而未說是絕對、相對或預條件殘差，無法比較不同尺度的問題。若 $b$ 很小，相對殘差也可能不合適；應同時明示 $\mathrm{atol}$、$\mathrm{rtol}$ 及使用的範數。

**誤把收斂當成物理解正確。** GMRES 收斂只說離散線性系統解得足夠準。網格截斷誤差、邊界條件、離散模型與物理假設仍可能錯。對平流擴散，應另做網格細化、邊界與守恆檢查；對反應濃度亦不能靠裁零掩蓋負值。

**重啟不當造成停滯。** 重啟過短會捨棄有效方向；只盲目增加最大迭代次數未必有用。可比較不同重啟長度及預條件器，觀察每段末端的真殘差；同時記錄記憶體成本與單步費用。重啟越長通常儲存越多基底，也會增加正交化成本。

**預條件器未必可逆或適用。** 對角預條件若有零對角元素便失效；不完整分解也可能因破壞性樞軸而失敗。預條件器的建立成本、作用成本、穩定性與對問題結構的適配，都應列入診斷，而非只報迭代步數。

**混淆縮放與改變問題。** 左行縮放、右變數縮放可以改善數值尺度，但輸出仍需映回原變數，且應在原方程中核對殘差。若縮放後誤將求解器的新殘差當作原問題殘差，便會錯判求解精度。

## 養殖與相場案例

在合成池域的溫度或溶氧傳輸模型中，給定速度場後以隱式時間步處理擴散，並以迎風或其他明確的平流離散處理搬運，可能得到稀疏非對稱系統。即使線性求解器將代數殘差降至門檻以下，仍要另外檢查邊界通量、來源與反應收支、單位、網格敏感度及濃度非負性。溶氧濃度低於管理閾值是管理或模型判讀，不是物理相變。對實際養殖資料或操作建議，本章的合成模型不提供現場門檻；唯讀 agent 只能呈現設定、求解診斷與證據，不能操作設備或改寫模型。

對無因次 Allen–Cahn 或 Cahn–Hilliard 相場，非線性時間離散與化學勢計算也可能在每步產生線性子問題。GMRES 可作為該子問題的線性求解器，但這本身不保證相場時間步能量下降。Cahn–Hilliard 在週期或適當無通量邊界下要求質量守恆；線性子問題殘差小不代表整體離散質量與能量檢查已通過。應分別記錄非線性迭代殘差、線性真殘差、總質量及離散自由能，並註明時間格式與步長條件。不能靠裁零或平滑把振盪隱去。

## 習題

1. **手算：** 對
   $$
   A=\begin{bmatrix}3&1\\0&2\end{bmatrix},
   \qquad b=\begin{bmatrix}4\\2\end{bmatrix},
   $$
   從 $x_0=0$ 出發。計算初始殘差，並說明為何不能僅憑矩陣對角元素為正就認定 CG 適用。

2. **程式：** 將章內程式的預條件改為恆等映射，並比較有、無對角預條件時的真殘差歷程。規定絕對、相對容差及重啟長度，報告 `b - A @ x`。不可把某次預期的步數寫成已執行結果。

3. **反例：** 說明「預條件殘差低於 $10^{-8}$，所以原系統解一定準確」為何錯。請分別討論左預條件與病態矩陣情況。

4. **整合：** 一維合成對流擴散模型形成非對稱線性系統，GMRES 回報未收斂。列出至少五項依合理順序進行的診斷，並區分代數、離散及模型層級問題。

## 習題解答

1. 初始殘差為
   $$
   r_0=b-Ax_0=\begin{bmatrix}4\\2\end{bmatrix},
   \qquad \|r_0\|_2=\sqrt{20}.
   $$
   $A_{12}=1$ 而 $A_{21}=0$，所以 $A$ 不對稱。對角元素為正並不足以使矩陣對稱正定；CG 的適用條件不能以「正對角線」取代。

2. 把 `apply_Pinv=lambda v: v.copy()` 傳入即為無預條件版本。另以 `lambda v: v / np.diag(A)` 作為對角預條件時，需先確認對角元素非零。兩次都使用相同矩陣、初值、容差及迭代上限，並重新計算 `np.linalg.norm(b - A @ x)`。預期在達標時兩者真殘差均滿足同一停止門檻；哪一者迭代較少須依矩陣及實際執行結果判定，不能預先宣稱。若停滯，應記錄未達標，而非只比較預條件後的量。

3. 左預條件 GMRES 最小化的是 $\|P^{-1}(b-Ax)\|_2$，若 $P^{-1}$ 對某些方向大幅縮小，該量小仍可能掩蓋原殘差。即使原殘差很小，病態 $A$ 也可能使 $\|A^{-1}r\|$ 不小，故解誤差不一定小。要判定求解器是否達成原方程容差，需計算真殘差；要評估解對擾動的敏感度，需考慮條件數、資料誤差或可靠的誤差估計。

4. 合理診斷順序如下：  
   (一) 檢查 $A,b,x_0$ 是否有限、形狀相符，並確認矩陣與來源項量綱一致。這是輸入與代數設定問題。  
   (二) 以原系統重算 $b-Ax$，核對 $\mathrm{atol}$、$\mathrm{rtol}$ 及分母尺度，不把內部殘差冒充真殘差。  
   (三) 檢查是否有零對角、無效預條件器或錯誤套用左右預條件；比較預條件前後的真殘差。  
   (四) 檢查矩陣尺度、條件性、重啟長度與停滯記錄，必要時以小型直接解作參照。這屬代數求解診斷。  
   (五) 檢查平流差分方向、流入邊界、擴散符號與矩陣組裝，並對網格細化及邊界通量作一致性檢查；這屬離散驗證。  
   (六) 最後檢查速度場、擴散係數、反應式與模型假設是否適合該合成情境；這屬模型檢驗，不能用增加 GMRES 迭代次數取代。

## 本章小結

非對稱平流系統適合考慮 GMRES，而非未滿足條件的 CG。GMRES 在 Krylov 子空間中最小化殘差，重啟則以較低記憶體成本交換可能的停滯。預條件能改善迭代問題，但左、右預條件最小化的量不同。最終判定應計算原系統真殘差 $b-Ax$，並明示絕對及相對容差；小殘差不等於小解誤差，更不等於模型與物理正確。求解診斷應分開處理代數、離散與模型層級，並保留可重現的矩陣、參數、容差與殘差紀錄。

## 參考來源

- [F3] PETSc，KSP 線性系統求解器手冊：<https://petsc.org/release/manual/ksp/>
- [F4] SciPy，稀疏線性代數 API：<https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>
- [F1] FiPy，有限體積離散與邊界：<https://pages.nist.gov/fipy/en/latest/numerical/discret.html>

本章提供的是小型教學用 NumPy 實作，沒有執行程式，也沒有宣稱任何實測效能、收斂階或物理驗證結果。API 文件可能比讀者本機套件新；實際使用時應核對所用版本。