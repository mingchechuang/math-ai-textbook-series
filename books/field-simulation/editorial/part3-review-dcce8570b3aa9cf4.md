## 重算結論

前輪四項阻擋均已修正：第14章例外範圍、第15章Arnoldi容差、第16章集中質量敘述與NumPy積分API皆已同步改善。跨章的 $A=-L$、座標軸、矩陣shape、Neumann符號及零空間處理大致一致。

仍有以下實質問題。

## 必須修正事項

### 1. 第16章表面熱通量的量綱換算錯誤

> 「若給定的是表面熱通量 $q$（$\mathrm{W/m^2}$），需除以 $\rho c_p$ 與單位厚度換算為 $\mathrm{K/s}$ 才能進入右端。」

**重算：**

$$
\frac{q}{\rho c_p}
=
\frac{\mathrm{J/(s\,m^2)}}{\mathrm{J/(m^3\,K)}}
=
\mathrm{K\,m/s}.
$$

這與 $aT'$ 的單位相同，是**邊界上的Neumann資料**，不是體積來源 $f$ 的 $\mathrm{K/s}$。單位厚度也不會把邊界熱通量直接變成一維體積來源；若要等效為體積來源，還需明列受熱體積或面積／體積比。

**最小修法：** 改成：「$q/(\rho c_p)$ 的單位為 $\mathrm{K\,m/s}$，應作為 $aT'$ 的Neumann邊界資料；只有經另行推導的面積／體積換算後，才可轉成 $\mathrm{K/s}$ 的體積來源。」

### 2. 第16章 `p1_load` 作為公開函式時未驗證網格契約

> ```python
> x = np.asarray(x, dtype=float)
> f = np.asarray(f, dtype=float)
> if f.shape != x.shape:
> ```

**原因：** 函式未檢查 `x.ndim == 1`、至少兩點、有限值及嚴格遞增。直接呼叫時，二維同shape輸入會產生非預期廣播；含 `nan` 的座標會生成非有限載荷；遞減或重複座標會產生負權重或零長單元。`solve_p1_dirichlet` 先呼叫 `p1_matrices` 雖能間接保護主流程，但不能補足 `p1_load` 自身的接口契約。

**最小修法：** 在 `p1_load` 加入與 `p1_matrices` 相同的座標檢查：

```python
if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
    raise ValueError(...)
if np.any(np.diff(x) <= 0.0):
    raise ValueError(...)
if f.ndim != 1 or f.shape != x.shape:
    raise ValueError(...)
```

並加入重複節點、遞減節點及非有限座標故障測試。

### 3. 第15章新增的 `epsilon_arnoldi` 缺少純量shape契約

> ```python
> not np.isfinite(epsilon_arnoldi) or epsilon_arnoldi < 0
> ```

**原因：** 若傳入長度大於一的NumPy陣列，條件判斷會因布林值不明確而拋出非契約性的例外；布林值也會被當成數值接受。這與同章對 `restart`、`maxiter` 的明確純量型別檢查不一致。

**最小修法：** 先轉為零維陣列並拒絕布林值，確認 `ndim == 0`、有限且非負，再轉成 `float`。`atol`、`rtol` 若要完整一致，也宜採相同純量檢查。

## 非阻擋文字修正

第16章：

> 「把 Dirichlet 當 Neumann 則矩陣缺一列」

標準組裝通常不是「缺一列」，而是未施加本質約束，使純擴散矩陣保留常數零模態而奇異。宜改為「矩陣缺少消除零模態的本質約束」。

除上述項目外，未發現新的跨章符號、Fourier軸向、Jacobian、通量或手算阻擋。

VERDICT: REVISE