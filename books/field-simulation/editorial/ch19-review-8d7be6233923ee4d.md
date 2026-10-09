# 審稿結果

本版仍與前稿實質相同，核心錯誤尚未修正。重算後有以下阻擋事項。

## 阻擋事項

1. **投影符號錯誤仍留在正文**

   **原句：**

   > $\nabla^2p=-\frac{\rho}{\Delta t}\nabla\cdot\mathbf u^*$

   正確為

   $$
   \nabla^2p=\frac{\rho}{\Delta t}\nabla\cdot\mathbf u^*,
   \qquad
   Ap=-\frac{\rho}{\Delta t}\nabla\cdot\mathbf u^*
   $$

   其中 $A=-\nabla^2$。後文雖已自行更正，但前面的錯式與未定義的 $\mathbf u^*_{div}$ 必須刪除。平流項指標也應改為 $\partial_i(u_j\partial_j u_i)$。

2. **例題 1 中間代數錯誤**

   **原句：**

   > $2p_0-p_2+2p_3=0\Rightarrow p_2=2(p_0+p_3)$

   由第 (4) 式實際應得

   $$
   -p_2+2p_3-p_0=0,\qquad p_2=2p_3-p_0.
   $$

   最終向量雖可代回原系統，中間推導仍須重算。

3. **Taylor–Green 與 Stokes 模型混用**

   **原句：**

   > $p=-\frac14(\cos2x+\cos2y)e^{-4\nu t}$

   完整 Navier–Stokes 對應壓力應為正號：

   $$
   p=\frac{\rho}{4}(\cos2x+\cos2y)e^{-4\nu t}+C(t).
   $$

   若忽略平流而測 Stokes 黏性模態，則應令 $p=C(t)$。兩種模型必須擇一。

4. **程式在 Laplacian 計算時會因 shape 不符而中止**

   **原句：**

   ```python
   self.u[:, 1:self.nx]
   - 2*self.u[:, :self.nx-1]
   + self.u[:, :self.nx-2]
   ```

   shape 分別為 `(ny,nx-1)`、`(ny,nx-1)`、`(ny,nx-2)`，不能相加。

   **最小修法：**刪除這段未使用切片及空 `pass` 迴圈，只對 `u[:,:nx]` 計算，再同步最後一面。

5. **$v$ 分量未更新或投影**

   **原句：**

   ```python
   # self.v = self.v + self.dt * self.nu * lap_v
   # self.v = self.v - (self.dt / self.rho) * dp_dy
   ```

   Poisson 右端包含 $D_yv$，但速度校正不含 $v$，所以一般不能得到零散度。必須完整實作 `lap_v`、預測、校正與週期同步。

6. **動能漏掉 $v$**

   **原句：**

   ```python
   E = 0.5 * self.rho * np.sum(uc**2) * self.dx * self.dy
   ```

   應加入

   ```python
   vc = 0.5 * (self.v[:self.ny, :] + self.v[1:self.ny+1, :])
   E = 0.5*self.rho*np.sum(uc**2 + vc**2)*self.dx*self.dy
   ```

   並明示為每單位厚度的二維動能。

7. **沒有製造來源與動量殘差**

   **原句：**

   > 平流項設為 0 或手動製造

   稿中未給 $\mathbf f$、其量綱、來源程式或離散動量殘差。須新增完整的 Stokes 或 Navier–Stokes 製造解，不能只以解析初值代替。

8. **習題 1 的週期矩陣及答案錯誤**

   **原句：**

   > $A=[[2,-1],[-1,2]]$，$p=[0,0]$

   對 $N=2,\Delta x=1$，左右週期鄰居重合，應為

   $$
   A=
   \begin{bmatrix}
   2&-2\\
   -2&2
   \end{bmatrix}.
   $$

   若指定 $\rho=\Delta t=1$，則 $b=[2,-2]^T$ 的零均值解為 $p=[1/2,-1/2]^T$，不是零向量。

9. **習題解答未完成**

   **原句：**

   > `(Code omitted, similar to u).`

   > `動能增加或振盪。`

   這不符合四類習題均須提供完整解答的要求。須補上 `lap_v` 程式、具體超限參數與放大因子、以及明確的散度矩陣和零空間證明。

10. **測試不可稽核**

    `pinv` 未指定截斷準則，亦未檢查真殘差 $r=b-Ap$。固定宣稱散度小於 $10^{-10}$ 不能普遍視為機器精度。

    **最小修法：**明示 `atol/rtol`，同時檢查 Poisson 真殘差及投影後散度的尺度化範數。

11. **穩定性手算錯誤**

    **原句：**

    > `0.001 * 256 ~ 0.25`

    實際為

    $$
    0.1(0.01)\left[2/(2\pi/16)^2\right]\approx0.01297.
    $$

12. **無因次外力未定義**

    應補上

    $$
    \mathbf f^*=\frac{L}{U^2}\mathbf f.
    $$

    有外力時，解也取決於無因次外力、初始及邊界條件。

13. **篇幅及交付條件未達標**

    稿件標示正文 2509 字，低於最低 3000 字，且缺完整二維程式、製造來源實驗及完整習題解答。應以補足實質內容達標，不保留已自行推翻的錯誤推導充字。

此外，小結的「驗證了」應改成「提出預期測試」；本稿未提供執行結果，本審稿亦未執行程式。

VERDICT: REVISE