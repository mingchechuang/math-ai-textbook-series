# 審稿結果

本稿的連續方程量綱大致正確，但壓力投影的符號、MAC 陣列 shape、週期重複面、Taylor–Green 初始化及能量預期彼此不一致。依目前程式，投影可能把散度放大而非消除，因此核心數值結論尚不能成立。

## 阻擋性問題

### 1. 壓力投影的 Poisson 右端符號相反

**可定位原句：**

> `# We solve: -Laplacian(p) = div(u*) / dt * rho`

以及

> `rhs = div_flat / self.dt * self.rho`

> `self.u = self.u - (self.dt / self.rho) * du_dx`

稿中定義 $A=-L$，而速度校正為

$$
\mathbf u^{n+1}=\mathbf u^*-\frac{\Delta t}{\rho}G p.
$$

因此

$$
D\mathbf u^{n+1}
=D\mathbf u^*-\frac{\Delta t}{\rho}DGp.
$$

若 $DG=L$，要使校正後散度為零，必須有

$$
Lp=\frac{\rho}{\Delta t}D\mathbf u^*,
$$

等價於

$$
Ap=-\frac{\rho}{\Delta t}D\mathbf u^*.
$$

目前程式卻解 $Ap=+\rho D\mathbf u^*/\Delta t$。理想相容算子下會得到 $Lp=-\rho D\mathbf u^*/\Delta t$，故校正後散度成為原散度的兩倍。

**最小修法：**

將程式改成：

```python
rhs = -self.rho * div_flat / self.dt
```

並在文字中統一寫成

$$
Lp=\frac{\rho}{\Delta t}D\mathbf u^*
$$

或

$$
(-L)p=-\frac{\rho}{\Delta t}D\mathbf u^*.
$$

另須加入投影前後 $\|D\mathbf u\|$ 與真殘差 $\|b-Ap\|$ 的預期檢查。

---

### 2. `v` 的 shape 與全卷 MAC 慣例相反，索引因而混亂

**可定位原句：**

> `# v: (nx, ny+1)`

> `self.v = np.zeros((self.nx, self.ny + 1))`

> `div[j, i] += (self.v[i, j+1] - self.v[i, j]) / self.dy`

共同慣例要求：

- `u_x[j, i_face]` shape 為 `(Ny, Nx+1)`
- `u_y[j_face, i]` shape 為 `(Ny+1, Nx)`

本稿把 `v` 儲存成 `(Nx, Ny+1)`，不但違反全卷資料契約，也使後續 `v[i,j]` 與物理網格 `q[j,i]` 不一致。當 `nx != ny` 時，更容易暴露 shape 錯誤。

**最小修法：**

改為：

```python
self.v = np.zeros((self.ny + 1, self.nx))
```

相應索引統一為：

```python
div[j, i] += (self.v[j + 1, i] - self.v[j, i]) / self.dy
```

壓力 $y$ 梯度、Laplacian、初始化與能量計算也都改用 `v[j_face, i]`。測試至少使用一次 `nx != ny`，避免方形網格掩蓋轉置錯誤。

---

### 3. 週期邊界面存兩份，卻未同步；Laplacian 還把重複端點當成不同未知量

**可定位原句：**

> `u: (ny, nx+1)`

> `ip = (i + 1) % (self.nx + 1)`

以及對 `v` 使用：

> `jp = (j + 1) % (self.ny + 1)`

週期 MAC 網格中的 `u[:,0]` 與 `u[:,nx]` 表示同一物理面；同理，正確 shape 下 `v[0,:]` 與 `v[ny,:]` 是同一物理面。程式卻將這兩份資料當成長度 `nx+1` 或 `ny+1` 的獨立週期環，沒有同步，造成錯誤鄰接與非物理自由度。

這也使故障測試所稱的「故意設置非週期邊界值」實際上是破壞資料表示契約，而不是合法邊界條件測試。

**最小修法：**

只在獨立面上更新，例如 `u[:, :nx]`、`v[:ny, :]`，每次更新及校正後明確同步：

```python
u[:, nx] = u[:, 0]
v[ny, :] = v[0, :]
```

Laplacian 的週期索引應對獨立面的 `nx`、`ny` 取模，而不是對 `nx+1`、`ny+1` 取模。總量與能量計算不得重複計入重複面。

---

### 4. Taylor–Green 流場的程式符號與正文更正後的無散度解不一致

**可定位原句：**

正文最後得到：

> `u = sin x cos y, v = -cos x sin y`

但程式初始化為：

> `solver.u[j, i] = -np.sin(x) * np.cos(y)`

> `solver.v[i, j] = -np.cos(x) * np.sin(y)`

兩個分量同為負號時，

$$
\partial_xu+\partial_yv
=-2\cos x\cos y\neq0.
$$

這正是正文已發現並更正的錯誤形式，但程式仍保留錯誤符號。因此它不是所宣稱的無散度 Taylor–Green 初值。

**最小修法：**

採一致形式，例如

```python
u =  np.sin(x) * np.cos(y)
v = -np.cos(x) * np.sin(y)
```

或同時反轉兩者並重新核對散度；不可只反轉其中與正文不一致的描述而不手算。

---

### 5. MAC 面座標錯置半格

**可定位原句：**

> `u[j,i] is at x = (i - 0.5)*dx`

> `v[i,j] is at y = (j - 0.5)*dy`

依本卷慣例，cell 中心為 $((i+1/2)\Delta x,(j+1/2)\Delta y)$，故：

- `u[j,i_face]` 位於 $(i\Delta x,(j+1/2)\Delta y)$；
- `v[j_face,i]` 位於 $((i+1/2)\Delta x,j\Delta y)$。

目前初始化把兩種面都錯移半格，且與 `compute_pressure_grad()` 所採「`u[j,i]` 是 cell $i$ 左面」的定義矛盾。

**最小修法：**

使用：

```python
x_u = i * dx
y_u = (j + 0.5) * dy

x_v = (i + 0.5) * dx
y_v = j * dy
```

並刪除對 `i == 0` 的特例。

---

### 6. 對奇異 Poisson 系統加 $\epsilon I$ 再扣均值，並不等於正確處理零空間

**可定位原句：**

> `self.A_reg = self.A + self.eps * np.eye(n_cells)`

> `self.A_inv = np.linalg.inv(self.A_reg)`

> `self.p -= np.mean(self.p)`

正則化改變了原 Poisson 方程。事後扣除壓力均值不會消除正則化造成的方程殘差，也不能修復不相容右端。稿件還宣稱：

> `修正後...離散散度應接近零... < 10^{-10}`

但在符號錯誤、$\epsilon I$ 正則化及未檢查相容性的情況下，這個容差沒有依據。

**最小修法：**

先執行相容性檢查：

```python
if abs(rhs.mean()) > atol + rtol * np.max(np.abs(rhs)):
    raise ValueError("incompatible periodic Poisson RHS")
rhs -= rhs.mean()
```

再以均值約束的增廣系統求解，或在明確的零均值子空間求解。最後報告原系統真殘差 `rhs - A @ p`，不得只看正則化系統殘差。

---

### 7. 「故障測試」的預期與實際離散散度性質不符

**可定位原句：**

> `壓力 Poisson 方程的右端向量將不再與零空間正交`

對面通量差分

$$
(D\mathbf u)_{j,i}
=\frac{u_{j,i+1}-u_{j,i}}{\Delta x}
+\frac{v_{j+1,i}-v_{j,i}}{\Delta y},
$$

全域求和後會留下週期重複面的差：

$$
\sum D\mathbf u
=\frac{1}{\Delta x}\sum_j(u_{j,N_x}-u_{j,0})
+\frac{1}{\Delta y}\sum_i(v_{N_y,i}-v_{0,i}).
$$

因此不同步確實「可能」造成不相容，但不必然如此；也不會必然產生「奇異值」。此外，現有程式使用已正則化的 `A_reg` 直接求逆，通常不會拋出 `Singular matrix`，與測試預期矛盾。

**最小修法：**

把測試改成確定不相容的構造，例如僅令一個週期重複面相差已知值，手算右端總和非零；程式應在求解前因相容性檢查而明確拒絕，而不是期待條件數或求逆偶然報錯。

---

### 8. 動能衰減率混淆速度振幅與能量

**可定位原句：**

> `動能 E ... 應按指數衰減：E(t) ∼ e^{-2ν k^2 t}`

此式對單一 Laplacian 特徵值為 $-k^2$ 的速度模態成立：振幅按 $e^{-\nu k^2t}$，能量按 $e^{-2\nu k^2t}$。

但 Taylor–Green 場滿足

$$
\Delta u=-2u,\qquad \Delta v=-2v,
$$

所以速度振幅為 $e^{-2\nu t}$，動能應為

$$
E(t)=E(0)e^{-4\nu t}.
$$

習題 2 卻要求動能曲線與 $e^{-2\nu t}$ 重疊，錯把振幅衰減當成能量衰減。

**最小修法：**

習題 2 攧為比較

$$
E(t)/E(0)
$$

與 $e^{-4\nu t}$；若比較速度範數比，才使用 $e^{-2\nu t}$。

---

### 9. 習題的 face 能量計算重複計入週期面，且不是所定義的 cell 積分

**可定位原句：**

> `e_u = np.sum(self.u**2) * (self.dx * self.dy)`

> `e_v = np.sum(self.v**2) * (self.dx * self.dy)`

`u` 與 `v` 含週期重複邊界面，因此直接求和會重複計入；而且此式沒有說明 staggered face 的控制體權重。這與本卷「週期邊界面存兩份時必須同步且不可重複計算總量」相違。

**最小修法：**

若採 cell-centered 診斷，先插值且使用正確 shape：

```python
uc = 0.5 * (u[:, :nx] + u[:, 1:nx+1])
vc = 0.5 * (v[:ny, :] + v[1:ny+1, :])
E = 0.5 * rho * np.sum(uc**2 + vc**2) * dx * dy
```

並說明這是每單位厚度的二維動能近似。或者為 face-based 能量明列不重複面的離散內積。

---

### 10. 程式並未實作所宣稱的 Navier–Stokes／製造來源測試

**可定位原句：**

> `Simple explicit advection`

但函式內容為：

> `pass`

以及：

> `source_func`  
> `self.u, self.v = source_func(...)`

此介面直接覆寫速度，而不是加入量綱為 $m/s^2$ 的動量來源。因而既未離散 $(\mathbf u\cdot\nabla)\mathbf u$，也未完成「製造流場檢查動量來源」的章節實驗。當前實作實際上只是顯式速度擴散加投影，不能支持小結所稱的 Navier–Stokes 數值實現。

**最小修法：**

二選一：

1. 明確將程式限定為週期 Stokes 擴散投影示範，刪除已實作平流及製造來源的宣稱；或
2. 實作完整、量綱清楚的 face 動量 RHS，讓 `source_func` 回傳加速度陣列，並以  
   `u_star = u + dt * (... + f_u)`、`v_star = ...` 更新。

章節 lab 要求仍需至少給出一個自足的製造解：列出 $\mathbf u,p,\mathbf f$，手算 $\nabla\cdot\mathbf u=0$，並檢查離散動量殘差隨網格細化的預期，而不能只檢查平滑圖或峰值。

---

### 11. 顯式二維黏性穩定條件寫成一維條件

**可定位原句：**

> `Δt < dx^2/(2ν)`

程式使用二維五點 Laplacian；其基本 FTCS 條件應為

$$
\nu\Delta t
\left(\frac{1}{\Delta x^2}+\frac{1}{\Delta y^2}\right)
\le \frac12.
$$

若 $\Delta x=\Delta y=h$，即 $\nu\Delta t/h^2\le1/4$，不是一維的 $1/2$。

**最小修法：**

將註解與時間步檢查改為上述二維條件，並另列平流 CFL；兩者均須滿足，但 CFL 必要性不應被宣稱為任意離散的充分穩定保證。

---

### 12. 壓力壁面條件與無滑移敘述不正確

**可定位原句：**

> `在固壁邊界，通常施加法向速度為零（無滑移），切向速度由動量方程決定。`

法向速度為零只是不可穿透；無滑移還要求切向速度等於壁面切向速度。壓力 Neumann 式

> `∂p/∂n = ρ (u·∇)(u·n) - ρ f·n`

也不是一般固定壁面投影法可直接使用的完整條件；其形式取決於速度邊界、壁面加速度、黏性項與離散投影步驟。

**最小修法：**

改為：固定無滑移壁面施加 $\mathbf u=\mathbf0$；壓力／壓力增量的法向條件由動量方程及所採投影格式一致導出。本章若只實作週期域，避免給出未推導的一般壁面公式，並明示固壁留待下一章。

## 其他必要修正

- 「壓力 $\mathbf p$」應為純量 $p$，不是向量。
- 「週期純 Neumann（或週期 Dirichlet）」用語錯誤；週期邊界不是週期 Dirichlet。改為「週期問題與純 Neumann 問題皆可能具有常數零空間」。
- 「在半正定意義上是 SPD」自相矛盾。應寫「原矩陣為對稱半正定；限制於零均值子空間後為正定，方可在該子空間使用 CG」。
- 手算例題 1 宣稱求投影，最後卻放棄壓力解；不算完整投影手算。至少解出 $p$ 與校正後速度，並使用與程式相同的 face divergence／gradient，不能混用 node-centered 中心差分。
- 習題 1 題目未給 $\Delta t$，解答先寫「設 $\Delta t=1$」又立刻改成 $0.25$；應在題目直接指定 $\Delta t=0.25$。
- 養殖池通常不能僅因「小規模」便近似為低 Reynolds 數；$Re$ 取決於 $UL/\nu$。應稱其為合成低 $Re$ 案例，不推論實際池塘。
- 「相場案例」實際內容是金融類比，與本卷相場共同術語無關，容易混淆 phase field。最小修法是刪除此小節，或改成流體與相場耦合的模型限制說明。
- 來源清單沒有直接支撐本章 Taylor–Green、投影法及壁面壓力條件；參考來源不能替代稿內推導與離散驗證。

VERDICT: REVISE