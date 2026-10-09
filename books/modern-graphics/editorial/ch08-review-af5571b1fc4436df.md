# 第08章〈法線、切線與逆轉置〉審查報告

## 一、總體判斷

逐項核對維度、座標系、法線、切線、TBN、鏡射、程式、手算例題、習題與引用界線後，**未發現數學真錯誤**。本章的核心推導、數值案例與程式斷言在紙上均可重算通過，教學內容完整。以下先列真錯誤（無），再列可選文風／篇幅建議，最後附重算驗證。

---

## 二、真錯誤（無）

以下各點經逐項重算，確認正確，僅列出以資佐證。

### 1. 逆轉置推導（「數學與幾何推導」）

原句：「令 $\mathbf n'=A^{-T}\mathbf n$，則 $(A^{-T}\mathbf n)^TA\mathbf v=\mathbf n^TA^{-1}A\mathbf v=\mathbf n^T\mathbf v=0$。」

核對：$(A^{-T}\mathbf n)^T=\mathbf n^T(A^{-T})^T=\mathbf n^TA^{-1}$，與 $A\mathbf v$ 相乘得 $\mathbf n^T\mathbf v$。維度 $(1\times3)(3\times3)(3\times1)$ 一致。**正確。**

### 2. $A=RS$ 的分解（「數學與幾何推導」）

原句：「$A^{-T}=R\,\operatorname{diag}(1/s_x,1/s_y,1/s_z)$。」

核對：$(RS)^{-T}=R^{-T}S^{-T}=R\,S^{-1}$（$R$ 正交），$S^{-1}=\operatorname{diag}(1/s_x,1/s_y,1/s_z)$。**正確。**

### 3. 切線／副切線公式（「切線、UV 與 TBN」）

原句：
$$\mathbf T=\frac{\mathbf e_1\Delta v_2-\mathbf e_2\Delta v_1}{D},\qquad \mathbf B=\frac{\mathbf e_2\Delta u_1-\mathbf e_1\Delta u_2}{D}.$$

核對：由
$$\begin{pmatrix}\mathbf e_1&\mathbf e_2\end{pmatrix}=\begin{pmatrix}\mathbf T&\mathbf B\end{pmatrix}\begin{pmatrix}\Delta u_1&\Delta u_2\\ \Delta v_1&\Delta v_2\end{pmatrix}$$
取逆後 $M^{-1}=\frac{1}{D}\begin{pmatrix}\Delta v_2&-\Delta u_2\\-\Delta v_1&\Delta u_1\end{pmatrix}$，$D=\Delta u_1\Delta v_2-\Delta v_1\Delta u_2$，展開確與原式一致。**正確。**

### 4. 外積恆等式（「鏡射須分清兩件事」）

原句：「$(A\mathbf e_1)\times(A\mathbf e_2)=\det(A)\,A^{-T}(\mathbf e_1\times\mathbf e_2)$。」

核對：以 $A=\operatorname{diag}(-1,1,1)$、$\mathbf e_1=(1,0,0)$、$\mathbf e_2=(0,1,0)$ 代入，左邊 $=(-1,0,0)\times(0,1,0)=(0,0,-1)$，右邊 $\det A\cdot A^{-T}(0,0,1)=-1\cdot(0,0,1)=(0,0,-1)$。**正確。**

### 5. 手算例一（非均勻縮放）

- $\mathbf n=(1,-1,0)^T/\sqrt2$，$\mathbf v_1=(1,1,0)$，$\mathbf v_2=(0,0,1)$：$\mathbf n\cdot\mathbf v_1=\mathbf n\cdot\mathbf v_2=0$ ✓
- $A\mathbf n=(2,-1,0)^T$，$A\mathbf v_1=(2,1,0)$，內積 $=3\ne0$ ✓
- $A^{-T}\mathbf n=(1/2,-1,0)$，內積 $=(1/2)(2)+(-1)(1)=0$ ✓

**正確。**

### 6. 手算例二（鏡射與 TBN 手性）

- $A=\operatorname{diag}(-1,1,1)$，$A^{-T}\mathbf n_g=(0,0,1)^T$ ✓
- $(-1,0,0)\times(0,1,0)=(0,0,-1)^T$ ✓
- UV $(0,0),(0,1),(1,0)$：$D=-1$，$\mathbf T=(0,1,0)$，$\mathbf B=(1,0,0)$，$\mathbf N\times\mathbf T=(-1,0,0)$，手性 $h=-1$ ✓

**正確。**

### 7. 程式（「實作與程式」）

逐行核對：

- `unit`：對零長或非有限值丟 `ValueError`，維度無誤。
- `geometry_normal`：以 $\|\mathbf e_1\times\mathbf e_2\|$ 作面積量級檢查，量綱為 $m^2$，文中已明示。
- `normal_matrix`：`inv(A).T` 即 $(A^{-1})^T=A^{-T}$；`det_eps` 保護奇異輸入。
- `tangent_frame`：先解未正規化 $\mathbf T_{\text{raw}}$、$\mathbf B_{\text{raw}}$，再對 $\mathbf N$ 做 Gram–Schmidt，最後以 $\operatorname{sign}\big((\mathbf N\times\hat{\mathbf T})\cdot\mathbf B_{\text{raw}}\big)$ 決定手性。
- 主程式：$p_0=(0,0,0)$、$p_1=(1,1,0)$、$p_2=(0,0,1)$、UV $(0,0),(1,0),(0,1)$。 $D=1$、$\mathbf T=(1,1,0)/\sqrt2$、$\mathbf B=(0,0,1)$、$h=1$，與斷言相符。

四條斷言（兩條垂直、三條正交、奇異拒絕）在紙上皆成立。**正確。**

### 8. 習題與解答

- 習題 1：$(A\mathbf n)\cdot(A\mathbf v)=16$；$(A^{-T}\mathbf n)\cdot(A\mathbf v)=0$；原內積 $=0$。**正確。**
- 習題 2：$\det A=24>0$，故 $\mathbf N_{\text{from\_edges}}=\mathbf N_{\text{from\_matrix}}$，斷言方向合理。**正確。**
- 習題 3：真正原因是 $\det A<0$ 反轉外積方向，不是逆轉置公式錯；UV 鏡射須另存手性，交換索引不足以修復 TBN。**正確。**
- 習題 4：$\mathbf n_g=(0,0,1)$、$D=-1$、$\mathbf T=(0,1,0)$、$\mathbf B=(1,0,0)$、$h=-1$；施加 $A$ 後先 $\mathbf T_w=A\mathbf T$、$\mathbf N_w=\operatorname{unit}(A^{-T}\mathbf n_g)$，再正交化並用 $h$ 建立副切線。**正確。**

### 9. 座標系與慣例一貫性

- 右手世界系、$+Z$ 朝向觀者、三角形逆時針為正面，全文一致。
- column vector、$p_h$、方向 $w=0$、$M=TRS$ 由右作用、像素中心 $(u+0.5,v+0.5)$，均與慣例一致。
- UV $v$ 向上、法線 $A^{-T}$、normal/depth 貼圖不套 sRGB（本章未涉 sRGB，但未與之衝突）。
- 數值單位為公尺、角度用弧度，文中已聲明。

### 10. 引用界線

「實作與程式」明示「本書未執行它」；「測試與預期結果」明示「上述是根據公式給出的**預期**，不是已執行的輸出」；「養殖數位分身案例」明示「不證明魚體生物形態、材質或水下照明與真實養殖場一致」。符合「不得虛構驗證、GPU 效能、物理準確度或來源查核」的要求。`source_notes` 亦聲明 G7 沿用第一卷、G8 未逐節查核。

---

## 三、可選文風／篇幅建議（不影響批准）

以下非真錯誤，僅供作者參考。

1. **篇幅低於目標。** 實測 `measured_characters = 3069`，滿足下限 3000，但低於每章約 4500 的目標。可考慮：
   - 在「數學與幾何推導」補一段「為何法線不是單純乘 $A$」的幾何圖像說明；
   - 在「實作與程式」加一個小節「世界空間 TBN 的組裝」，示範 $4\times4$ 矩陣切片與 $\mathbf T_w$ 的完整呼叫；
   - 習題可再加一題「連續兩次非均勻縮放後的法線矩陣」。

2. **來源與正文標註對齊。** 章末「參考來源」列 G1、G2、G5、G7，但 G2（PBRT：Reflection Models）與本章主題關聯較弱；G3、G4、G6、G8 未列入章末，卻出現在 `source_notes`。建議：要嘛在正文關鍵處加 `[G1]`／`[G5]` 等行內標記，要嘛把章末清單縮到真正被用到的 G1、G5、G7，避免「清單多於實際引用」。

3. **詞語微調。** 「測試與預期結果」第 1 點說錯誤作法「通常不再接近零」；在例一的具體設定下是**必然**不為零（值為 $3/\sqrt2$）。可改為「在此例中不再接近零」，語氣更精確。

4. **手性判斷的退化情形。** `tangent_frame` 中 `np.dot(np.cross(N, T), B_raw) >= 0` 在 $B_{\text{raw}}$ 恰與 $\mathbf N\times\mathbf T$ 垂直（罕見退化）時會勉強歸為 $+1$。可在註解中加一句「真正退化時應先拒絕，本函式僅在 $\mathbf B_{\text{raw}}\ne0$ 且不與 $\mathbf N\times\mathbf T$ 垂直時使用」。

5. **`det_eps` 的絕對容差。** 文中已聲明是「簡化保護」，但可補一句：對尺度懸殊的矩陣，應改用 $\det(A)/\big(\|A\|^3\big)$ 之類的相對量，避免大尺度矩陣被誤判為奇異。

---

## 四、重算總表

| 項目 | 輸入 | 期望 | 重算 |
|---|---|---|---|
| 例一錯誤變換 | $A=\operatorname{diag}(2,1,1)$，$\mathbf n=(1,-1,0)/\sqrt2$ | $(A\mathbf n)\cdot(A\mathbf v_1)=3$ | 3 ✓ |
| 例一正確變換 | $A^{-T}\mathbf n=(1/2,-1,0)$ | 內積 $=0$ | 0 ✓ |
| 例二鏡射外積 | $A=\operatorname{diag}(-1,1,1)$ | $(A\mathbf e_1)\times(A\mathbf e_2)=(0,0,-1)$ | (0,0,-1) ✓ |
| 例二 UV 手性 | UV $(0,0),(0,1),(1,0)$ | $D=-1$，$h=-1$ | $-1,-1$ ✓ |
| 程式主例 | $p_0,p_1,p_2$ 如上 | $D=1$，$h=1$ | $1,1$ ✓ |
| 習題 1 | $A=\operatorname{diag}(3,1,2)$ | 錯 $16$、對 $0$ | $16,0$ ✓ |
| 習題 4 | $p_0,p_1,p_2$ 平面三角形 | $n_g=(0,0,1)$，$D=-1$，$h=-1$ | 一致 ✓ |

---

## 五、待人工與程式驗證事項（非拒稿理由）

- 依規範，本章「未執行」的程式與數值屬**預期**；實際在 Python 3.10+ 與 NumPy 2.2.6 環境執行時，仍應確認 `np.cross`、`np.linalg.inv` 的回傳形狀與 `assert` 容差是否與此處手算一致。
- 參考來源 G1、G5、G7 的頁面內容未由本審查獨立查核；`source_notes` 已聲明此界線。
- 手性與面繞序在真實 DCC 工具（如 Blender）輸出時，可能因座標系或索引慣例而有差異，需以導入工具實測。

以上均屬讀者／作者應自行完成的驗證，不影響本章數學正確性與教學完整性。

VERDICT: APPROVE