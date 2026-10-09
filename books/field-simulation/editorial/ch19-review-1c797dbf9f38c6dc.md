# 審稿結果

本次稿件仍未修正核心阻擋事項；以下為重算後的必要修改。

## 阻擋性問題

### 1. 投影方程符號前後矛盾

**原句：**

> $\nabla^2p=-\frac{\rho}{\Delta t}\nabla\cdot\mathbf u^*$

由速度校正

$$
\mathbf u^{n+1}=\mathbf u^*-\frac{\Delta t}{\rho}\nabla p
$$

應得

$$
\nabla^2p=\frac{\rho}{\Delta t}\nabla\cdot\mathbf u^*,
\qquad
Ap=-\frac{\rho}{\Delta t}\nabla\cdot\mathbf u^*
$$

其中 $A=-\nabla^2$。稿件後段雖自行更正，但錯誤式與未定義的 $\mathbf u^*_{div}$ 仍留在正文。

**最小修法：**刪除自我推翻段落，只保留上述一致推導。另將錯誤 Einstein 式改為 $\partial_i(u_j\partial_j u_i)$。

---

### 2. 例題 1 的代數步驟錯誤

**原句：**

> 由 (4): $2p_0-p_2+2p_3=0\Rightarrow p_2=2(p_0+p_3)$

正確為

$$
-p_2+2p_3-p_0=0,
\qquad p_2=2p_3-p_0.
$$

最終壓力向量可代回原系統，但不能由所列推導得到。

**最小修法：**重寫消去步驟；並將「投影平滑」改成「投影移除非零散度分量」，因本例未含黏性更新。

---

### 3. Taylor–Green 壓力與 Stokes 模型不一致

**原句：**

> $p=-\frac14(\cos2x+\cos2y)e^{-4\nu t}$

對稿中速度，完整 Navier–Stokes 解的壓力應為

$$
p=\frac{\rho}{4}(\cos2x+\cos2y)e^{-4\nu t}+C(t).
$$

若忽略平流而採 Stokes 方程，則 $\nabla p=0$，壓力應為空間常數。

**最小修法：**現有程式只做黏性擴散，故改稱 Stokes 黏性模態並令 $p=C(t)$；否則須實作平流並修正壓力符號。

---

### 4. 程式會因 shape 錯誤中止

**原句：**

```python
lap_u_x = (
    self.u[:, 1:self.nx]
    - 2*self.u[:, :self.nx-1]
    + self.u[:, :self.nx-2]
) / self.dx**2
```

三項 shape 分別為 `(ny,nx-1)`、`(ny,nx-1)`、`(ny,nx-2)`，不能 broadcasting。

**最小修法：**刪除此未使用切片及前面的空迴圈，只對 `u[:,:nx]` 計算週期 Laplacian，再同步 `lap_u[:,nx]=lap_u[:,0]`。

---

### 5. 二維 $v$ 分量未實作

**原句：**

```python
# self.v = self.v + self.dt * self.nu * lap_v
# self.v = self.v - (self.dt / self.rho) * dp_dy
```

Poisson 右端含 $D_yv$，但 $v$ 未預測也未校正，故一般不能使完整散度歸零。Taylor–Green 的 $v$ 亦非零。

**最小修法：**完整實作 `lap_v`、$v$ 預測、壓力校正及週期同步。核心二維算法不能留作習題。

---

### 6. 動能診斷不完整

**原句：**

```python
E = 0.5 * self.rho * np.sum(uc**2) * self.dx * self.dy
```

此式漏掉 $v$。應為

```python
vc = 0.5 * (self.v[:self.ny, :] + self.v[1:self.ny+1, :])
E = 0.5*self.rho*np.sum(uc**2 + vc**2)*self.dx*self.dy
```

並明示這是每單位厚度的二維動能。

---

### 7. 未完成製造來源與可稽核測試

**原句：**

> 平流項設為 0 或手動製造

稿中沒有 $\mathbf f$ 的解析式、單位、程式介面或動量殘差；也沒有檢查真殘差 $r=b-Ap$。固定宣稱散度小於 $10^{-10}$，未給 `atol/rtol` 或尺度基準。

**最小修法：**

- 明列一組 $\mathbf u,p,\mathbf f$ 並檢查離散動量殘差；
- 報告 $r=b-Ap$；
- 明示 `atol/rtol`；
- 未執行時只寫「預期」，不得在小結稱「驗證了」。

---

### 8. 習題 1 的週期矩陣與答案錯誤

**原句：**

> $A=[[2,-1],[-1,2]]$，$p=[0,0]$

對 $N=2,\Delta x=1$，左右週期鄰居重合，故

$$
A=
\begin{bmatrix}
2&-2\\
-2&2
\end{bmatrix}.
$$

非零右端不可能由 $p=0$ 滿足。題目也缺 $\rho,\Delta t$。

**最小修法：**例如指定 $\rho=\Delta t=1$，則

$$
b=[2,-2]^T,\qquad p=[1/2,-1/2]^T
$$

為零均值解，再完成速度校正。

---

### 9. 習題解答不完整

**原句：**

> `(Code omitted, similar to u).`

> `動能增加或振盪。`

這不符合四類習題均須有完整解答的規範。習題 4 也未真正列出矩陣，且混淆左零空間與零空間。

**最小修法：**補齊程式、具體超限參數與放大因子、矩陣 shape 及零空間證明。

---

### 10. 穩定性手算及無因次化缺項

**原句：**

> `0.001 * 256 ~ 0.25`

實際為

$$
0.1(0.01)\left[2/(2\pi/16)^2\right]\approx0.01297.
$$

此外應定義

$$
\mathbf f^*=\frac{L}{U^2}\mathbf f.
$$

有外力時，解不只取決於 $Re$ 與幾何。

---

### 11. 未達最低交付量

輸入標示正文計數為 2509，低於最低 3000 中文字；且仍缺完整二維程式、製造來源實驗與完整習題解答。

**最小修法：**用上述必要技術內容補足篇幅，不以保留錯誤推導充字。

## 其他必要修正

- 常密度不可壓模型中，不宜稱壓力「抵抗局部高密度區」。
- 高 $Re$ 不必然等同湍流或混沌。
- 「相場」小節實為金融類比，與本卷 phase field 術語不符。
- 本審稿未執行程式；來源清單亦不能替代離散推導與測試。

VERDICT: REVISE