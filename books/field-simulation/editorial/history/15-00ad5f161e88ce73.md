# 第15章 預條件、GMRES與求解診斷

## 學習目標與先備知識

本章說明如何求解非對稱稀疏線性系統，尤其是平流、反應與擴散離散後常見的系統。讀完後，應能說明 GMRES 如何在 Krylov 子空間中尋找近似解，分辨左、右預條件及其殘差，並用原系統真殘差判定求解是否達標。也應能診斷尺度不平衡、停滯、重啟設定不當與病態矩陣，區分代數求解誤差、離散誤差及模型誤差。

先備知識為矩陣向量乘法、線性方程組、向量範數及稀疏矩陣的基本概念。以下程式使用 Python 3.10+ 與 NumPy，限於 CPU 小問題，不需 SciPy。程式未執行；文中結果均為推導的預期，不是實測。

## 問題與直覺

設離散後的系統為

$$
Ax=b,
$$

其中 $A\in\mathbb{R}^{n\times n}$，而 $x,b\in\mathbb{R}^n$。若 $A$ 對稱正定，共軛梯度法（CG）可利用其結構求解。平流項具有方向性，常使離散矩陣不對稱；此時 CG 的適用條件不成立，即使偶然得到小殘差，也不能因此認定一般收斂保證適用。

GMRES（generalized minimal residual method，廣義最小殘差法）在由起始殘差生成的 Krylov 子空間中，選擇使殘差二範數最小的近似解。理想算術下，未重啟 GMRES 至多經 $n$ 個有效方向可得到精確解；代價是基底儲存量增加，正交化成本也會隨迭代上升。重啟式 GMRES 每累積 $m$ 個方向便以目前解重新開始，減少儲存與正交化負擔，但會捨棄已建立的搜尋方向，可能停滯。

預條件試圖把問題轉成較易求解的形式。若 $P$ 近似 $A$，理想情況下 $P^{-1}A$ 比 $A$ 更適合迭代；實務上通常只提供「近似求解 $Pz=v$」的程序，不形成 $P^{-1}$。預條件的建構及套用成本、穩定性和問題結構適配度，都會影響是否值得使用。

## 數學與物理推導

給定初值 $x_0$，原系統的真殘差與相對殘差為

$$
r_k=b-Ax_k,\qquad
\eta_k=\frac{\|r_k\|_2}{\|b\|_2}.
$$

若 $\|b\|_2=0$，相對殘差無法按此式定義，應使用絕對殘差或明訂其他尺度。常見停止條件為

$$
\|b-Ax_k\|_2\le \mathrm{atol}+\mathrm{rtol}\,\|b\|_2.
$$

容差必須明確標示。小殘差代表線性方程的未平衡量小，不必然代表近似解接近精確解。若 $A$ 可逆，精確解為 $x_*$，則

$$
\|x_k-x_*\|_2\le \|A^{-1}\|_2\|r_k\|_2.
$$

因此解誤差還受矩陣條件數影響；病態系統可能對右端的小擾動十分敏感。殘差、解誤差、離散誤差與模型誤差是不同診斷量。

### Krylov 子空間與最小殘差

令 $r_0=b-Ax_0$，第 $m$ 階 Krylov 子空間為

$$
\mathcal{K}_m(A,r_0)
=\operatorname{span}\{r_0,Ar_0,\ldots,A^{m-1}r_0\}.
$$

GMRES 在 $x_0+\mathcal{K}_m(A,r_0)$ 中選擇使 $\|b-Ax_m\|_2$ 最小的 $x_m$。Arnoldi 正交化建立正交基底 $V_{m+1}$，滿足

$$
AV_m=V_{m+1}\overline H_m,
$$

其中 $\overline H_m$ 為上 Hessenberg 矩陣。令 $\beta=\|r_0\|_2$，且 $v_1=r_0/\beta$。寫成 $x_m=x_0+V_my$，則

$$
\|b-Ax_m\|_2
=\|\beta e_1-\overline H_m y\|_2.
$$

因此每步只需解一個小型最小平方問題。

### 左、右預條件與真殘差

左預條件求解

$$
P^{-1}Ax=P^{-1}b,
$$

GMRES 在變換後的系統中最小化 $\|P^{-1}(b-Ax)\|_2$。右預條件令 $x=P^{-1}y$，求解

$$
AP^{-1}y=b.
$$

右預條件 GMRES 最小化的是原系統殘差 $\|b-AP^{-1}y\|_2$。左預條件的內部殘差即使很小，原系統真殘差仍可能不符合使用者指定的容差。因此求解後應重算 $b-Ax$，並依原系統真殘差作最終判定。

### 平流擴散離散的非對稱性

考慮一維有因次對流擴散方程

$$
\frac{\partial c}{\partial t}
+u\frac{\partial c}{\partial x}
=D\frac{\partial^2c}{\partial x^2}+s.
$$

其中 $x$ 以公尺計，$t$ 以秒計，$u$ 的單位為 $\mathrm{m/s}$，$D$ 為 $\mathrm{m^2/s}$。若 $c$ 為 $\mathrm{kg/m^3}$，來源項 $s$ 的單位是 $\mathrm{kg/(m^3\,s)}$。以隱式 Euler 離散可寫成

$$
\left(I+\Delta t\,uD_x-\Delta t\,D L\right)c^{n+1}
=c^n+\Delta t\,s^{n+1},
$$

其中 $D_x$ 為所選一階差分算子，$L$ 為 Laplacian 離散。此式的符號取決於平流項移項方式與 $D_x$ 的定義，須在推導和程式中一致。方向性平流通常造成非對稱矩陣；迎風格式及邊界閉合亦會影響其係數。

**逐步例題一：判斷矩陣結構。** 令

$$
A=\begin{bmatrix}
2&-1&0\\
-3&4&-1\\
0&-2&3
\end{bmatrix}.
$$

比較 $A_{12}=-1$ 與 $A_{21}=-3$，可知 $A\ne A^T$，故不符合 CG 的對稱條件。取對角預條件 $P=\operatorname{diag}(2,4,3)$ 作為便宜的基線，能調整分量尺度，卻不保證有效處理非對稱耦合。求解後仍須直接計算 $r=b-Ax$。

**逐步例題二：手算殘差與門檻。** 取

$$
A=\begin{bmatrix}2&1\\0&1\end{bmatrix},
\qquad
b=\begin{bmatrix}1\\1\end{bmatrix},
\qquad x_0=\begin{bmatrix}0\\0\end{bmatrix}.
$$

初始殘差是 $r_0=[1,1]^T$，範數為 $\sqrt{2}$。令候選解 $x=[0.45,1]^T$，則

$$
Ax=\begin{bmatrix}1.9\\1\end{bmatrix},
\qquad
b-Ax=\begin{bmatrix}-0.9\\0\end{bmatrix},
\qquad
\|b-Ax\|_2=0.9.
$$

若 $\mathrm{atol}=10^{-8}$、$\mathrm{rtol}=10^{-2}$，停止門檻是 $10^{-8}+10^{-2}\sqrt{2}\approx0.0141$，故此候選解不合格。精確解為 $x_*=[0,1]^T$。即使某近似解通過殘差門檻，仍不能據此證明模型或物理解正確。

## 逐步手算例題

取

$$
A=\begin{bmatrix}2&0\\1&1\end{bmatrix},
\qquad
b=\begin{bmatrix}1\\0\end{bmatrix},
\qquad x_0=0.
$$

初始殘差 $r_0=[1,0]^T$，故 $\beta=1$、$v_1=[1,0]^T$。計算 $Av_1=[2,1]^T$，投影至 $v_1$ 得 $h_{11}=2$；扣除投影後的向量為 $[0,1]^T$，因此 $h_{21}=1$、$v_2=[0,1]^T$。一步 GMRES 的小型最小平方問題為

$$
y_1=\arg\min_y
\left\|
\begin{bmatrix}1\\0\end{bmatrix}
-\begin{bmatrix}2\\1\end{bmatrix}y
\right\|_2.
$$

解得 $y_1=2/5$，故 $x_1=[0.4,0]^T$。此時

$$
b-Ax_1=\begin{bmatrix}0.2\\-0.4\end{bmatrix},
\qquad \|b-Ax_1\|_2=\sqrt{0.2}.
$$

第二個有效 Krylov 方向可令此二維問題精確求解；但一般問題仍須依明確容差檢查真殘差，而非依迭代步數判定。

## 實作與程式

以下是教學用途的右預條件、重啟式 GMRES。矩陣以稠密格式儲存，僅適合小型測試；大問題應採稀疏矩陣和合適的稀疏預條件器。程式檢查輸入形狀與有限值，並在迴圈前先處理初始殘差，使 `maxiter=0` 時也能正確判定初值是否已達標。遇到零 Arnoldi 樞軸時，不會對奇異的小型 Hessenberg 矩陣呼叫 `solve`，而是保留目前解並回報真殘差及未收斂狀態。

```python
import numpy as np


def gmres(A, b, x0=None, restart=10, maxiter=100,
          atol=1e-12, rtol=1e-8, apply_Pinv=None):
    """教學用右預條件 GMRES；A 為小型稠密方陣。"""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必須是方陣")
    n = A.shape[0]
    if b.shape != (n,):
        raise ValueError("b 必須是形狀 (n,) 的向量")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("A 與 b 必須全部為有限值")

    # bool 是 int 的子類別，但不是合理的迭代數設定，故明確拒絕。
    if (not isinstance(restart, (int, np.integer))
            or isinstance(restart, (bool, np.bool_))
            or restart < 1):
        raise ValueError("restart 必須是正整數")
    if (not isinstance(maxiter, (int, np.integer))
            or isinstance(maxiter, (bool, np.bool_))
            or maxiter < 0):
        raise ValueError("maxiter 必須是非負整數")
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("atol 與 rtol 必須是有限非負值")

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    if x.shape != (n,) or not np.all(np.isfinite(x)):
        raise ValueError("x0 必須是有限值且形狀為 (n,)")

    if apply_Pinv is None:
        apply_Pinv = lambda v: v.copy()

    tol = atol + rtol * np.linalg.norm(b)
    history = []
    r = b - A @ x
    true_norm = np.linalg.norm(r)
    history.append(true_norm)

    # 先判初值；因此 b=0、x0=0 或 maxiter=0 都有明確結果。
    if true_norm <= tol:
        return x, history, True
    if maxiter == 0:
        return x, history, False

    iterations = 0
    while iterations < maxiter:
        r = b - A @ x
        beta = np.linalg.norm(r)
        if beta <= tol:
            history.append(beta)
            return x, history, True

        m = min(int(restart), int(maxiter - iterations), n)
        V = np.zeros((n, m + 1))
        Z = np.zeros((n, m))
        H = np.zeros((m + 1, m))
        cs = np.zeros(m)
        sn = np.zeros(m)
        g = np.zeros(m + 1)
        V[:, 0] = r / beta
        g[0] = beta
        used = 0
        trial = x.copy()
        breakdown = False

        for j in range(m):
            z = np.asarray(apply_Pinv(V[:, j]), dtype=float)
            if z.shape != (n,) or not np.all(np.isfinite(z)):
                raise ValueError("預條件器輸出必須是有限值且形狀為 (n,)")
            Z[:, j] = z
            w = A @ z

            # Modified Gram–Schmidt。
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w -= H[i, j] * V[:, i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] > 0:
                V[:, j + 1] = w / H[j + 1, j]

            # 套用先前建立的 Givens 旋轉。
            for i in range(j):
                a = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                q = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j], H[i + 1, j] = a, q

            den = np.hypot(H[j, j], H[j + 1, j])
            if den == 0:
                # 沒有可解的小型最小平方樞軸；不可呼叫 solve。
                breakdown = True
                break

            cs[j] = H[j, j] / den
            sn[j] = H[j + 1, j] / den
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]

            used = j + 1
            iterations += 1
            y = np.linalg.solve(H[:used, :used], g[:used])
            trial = x + Z[:, :used] @ y

            # 用原系統真殘差檢查，不只依賴遞推殘差。
            true_norm = np.linalg.norm(b - A @ trial)
            history.append(true_norm)
            if true_norm <= tol:
                return trial, history, True

            # Arnoldi 子空間可能封閉；候選解仍已計算，接著判定真殘差。
            if H[j + 1, j] == 0:
                breakdown = True
                break

        if used > 0:
            x = trial
        if breakdown:
            return x, history, np.linalg.norm(b - A @ x) <= tol
        if used == 0:
            return x, history, False

    return x, history, np.linalg.norm(b - A @ x) <= tol


def main():
    A = np.array([[2.0, -1.0, 0.0],
                  [-3.0, 4.0, -1.0],
                  [0.0, -2.0, 3.0]])
    b = np.array([1.0, 2.0, 0.0])

    diagonal = np.diag(A)
    if np.any(diagonal == 0):
        raise ValueError("對角預條件不可用：A 有零對角元素")
    apply_Pinv = lambda v: v / diagonal

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

此程式每個內迭代都重算原系統真殘差，便於教學但成本較高。大型問題常以遞推量監控，再定期重算真殘差。程式預期在可解且數值行為良好的小系統上，若回報成功，真殘差應不超過指定門檻；解與直接解的差異則受浮點精度及條件數影響。這些是預期，不是已執行的測試結果。

## 測試與預期結果

| 類別 | 測試 | 預期診斷 |
|---|---|---|
| 正常 | 使用章內非對稱矩陣，有限值右端，計算 `b - A @ x` | 若回報成功，真殘差應符合 $\mathrm{atol}+\mathrm{rtol}\|b\|_2$。直接解比較是另一項診斷。 |
| 邊界 | $b=0$、$x_0=0$、`maxiter=0` | 初始真殘差為零，應立即回報成功，且歷程中有初始殘差。 |
| 邊界 | $b\ne0$、`maxiter=0` | 應回報未收斂及初始真殘差，不應進入迭代。 |
| 邊界 | `restart=1` 且迭代上限有限 | 可能收斂慢或停滯；應以回傳真殘差判定，不可假報成功。 |
| 故障 | 令 $A=0$、$b\ne0$，使用恆等預條件 | Arnoldi 方向無法產生有效樞軸；應回報未收斂，不應在小型奇異系統呼叫 `solve`。 |
| 故障 | 傳入形狀 `(n,1)` 的 $b$ | 應在矩陣運算前因形狀不符而拒絕，不應發生 NumPy 廣播。 |
| 故障 | 傳入 NaN 的 $A$、$b$ 或預條件器輸出 | 應拒絕非有限輸入或輸出。 |
| 故障 | 對角線含零仍直接使用 `v / diagonal` | 呼叫者應先拒絕無效對角預條件；不能忽略非有限結果。 |
| 停滯 | 短重啟、粗糙預條件或高條件數系統 | 可能不達門檻；記錄真殘差歷程，並檢查尺度、重啟及預條件效果。 |

尺度測試可把方程某列及對應右端同乘常數，精確解不變，但殘差範數及迭代行為可能改變。若使用行縮放矩陣 $S$，須分清求解的是 $SAx=Sb$，並在原系統計算 $b-Ax$ 作最終檢查。

## 除錯與常見陷阱

**預條件殘差不等於真殘差。** 左預條件可能最小化 $\|P^{-1}(b-Ax)\|_2$；若 $P^{-1}$ 大幅縮小特定方向，該量小仍可能掩蓋原殘差。應同時記錄預條件殘差（若可取得）及原系統真殘差。

**容差須定義清楚。** 「殘差小於 $10^{-8}$」未說明是絕對、相對還是預條件殘差，無法比較不同尺度的問題。若 $b$ 很小，相對殘差亦可能不適用；應明示 $\mathrm{atol}$、$\mathrm{rtol}$ 與範數。

**求解器收斂不等於物理正確。** GMRES 收斂只表示離散線性系統解至指定精度。差分符號、邊界條件、網格截斷誤差與物理假設仍可能錯。濃度的非負性、收支守恆與網格收斂須另外檢查，不能靠裁零掩蓋失敗。

**重啟可能造成停滯。** 重啟過短會捨棄有效方向；盲目提高迭代上限不一定有效。可比較不同重啟長度及預條件器，觀察每段末端的真殘差，同時考慮基底記憶體與正交化成本。

**預條件器可能無效。** 對角預條件在有零對角時失效；不完整分解也可能因樞軸問題失敗。應檢查其適用性、建立成本、套用成本與穩定性，不能只比較迭代數。

**縮放不是改變物理問題的許可。** 左行縮放或右變數縮放可改善數值尺度，但需正確映回原變數，最後仍要在原方程核對殘差。

## 養殖與相場案例

在合成池域的溫度或溶氧傳輸中，給定速度場、以隱式時間步處理擴散，並採明確的平流離散，可能得到稀疏非對稱系統。即使 GMRES 將代數殘差降到門檻以下，也須另查邊界通量、來源與反應收支、單位、網格敏感度及濃度非負性。溶氧跨越管理閾值不是物理相變。本章不提供現場門檻或設備操作建議；唯讀 agent 只可呈現合成設定與求解證據，不自行改模型、投餌、加藥或操作設備。

對無因次 Allen–Cahn 或 Cahn–Hilliard 相場，非線性時間離散可能在每步形成線性子問題，GMRES 可用於該子問題。但線性子問題殘差小，不保證相場時間離散能量下降。Cahn–Hilliard 在週期或適當無通量邊界下須檢查質量守恆；亦應分別記錄非線性殘差、線性真殘差、總質量與離散自由能，並列出時間格式與步長條件。不能以裁零或平滑掩蓋振盪。

## 習題

1. **手算：** 對
   $$
   A=\begin{bmatrix}3&1\\0&2\end{bmatrix},
   \qquad b=\begin{bmatrix}4\\2\end{bmatrix},
   $$
   從 $x_0=0$ 出發，計算初始殘差，並說明為何正對角元素不足以判定 CG 適用。

2. **程式：** 將章內程式的預條件改成恆等映射，與對角預條件比較。使用相同的容差、重啟與迭代上限，列出真殘差。不得把推測的迭代數寫成已執行結果。

3. **反例：** 說明「預條件殘差低於 $10^{-8}$，所以原系統解一定準確」為何錯，分別討論左預條件與病態矩陣。

4. **整合：** 一維合成對流擴散系統的 GMRES 未收斂。列出至少五項依序診斷，並區分代數、離散及模型層級問題。

## 習題解答

1. 因 $x_0=0$，$r_0=b=[4,2]^T$，故 $\|r_0\|_2=\sqrt{20}$。又 $A_{12}=1$ 而 $A_{21}=0$，矩陣不對稱。正對角線不足以確保對稱正定，所以不能據此認定 CG 適用。

2. 無預條件可使用 `apply_Pinv=lambda v: v.copy()`；對角預條件則可用 `lambda v: v / np.diag(A)`，但須先檢查對角線無零。兩次採用相同初值、容差、重啟及迭代上限，並計算 `np.linalg.norm(b - A @ x)`。預期若回報成功，兩次真殘差都符合同一停止門檻；哪一者迭代較少須由實際結果決定，不能預先宣稱。

3. 左預條件最小化的是 $\|P^{-1}(b-Ax)\|_2$，$P^{-1}$ 可能縮小某些方向，使預條件殘差很小但原殘差仍大。此外，病態 $A$ 可令小殘差對應顯著解誤差。應重算真殘差，並考慮條件數及資料擾動。

4. 先檢查 $A,b,x_0$ 的形狀與有限值；再以原系統重算 $b-Ax$，核對容差定義；接著檢查左右預條件的使用方式及是否含零對角；再檢查尺度、條件性、重啟長度及停滯記錄，必要時以直接解作小型參照。以上屬輸入與代數診斷。其後查平流方向、流入邊界、擴散符號和矩陣組裝，再作網格細化與邊界通量檢查，屬離散驗證。最後才檢查速度、擴散係數、反應式和模型假設是否適合合成情境，屬模型檢驗；增加迭代次數不能代替這些檢查。

## 本章小結

GMRES 適用於一般非對稱系統，並在 Krylov 子空間中最小化殘差；重啟降低記憶體需求，卻可能造成停滯。預條件能改善迭代，但左右形式最小化的量不同。最終判定須計算原系統真殘差 $b-Ax$，明示絕對與相對容差，並把解誤差、離散誤差和模型誤差分開診斷。

## 參考來源

- [F3] PETSc，KSP 線性系統求解器手冊：<https://petsc.org/release/manual/ksp/>
- [F4] SciPy，稀疏線性代數 API：<https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>
- [F1] FiPy，有限體積離散與邊界：<https://pages.nist.gov/fipy/en/latest/numerical/discret.html>

本章程式未執行，未提供實測效能或收斂結果。實際使用時應核對所用套件版本及求解器輸出。