## 審查結論

前次指出的行列式手性判定與 UV 絕對容差問題，已以 `slogdet`、矩陣尺度正規化及奇異值比例判定處理。主文推導、手算例題及習題答案大致正確，且明確說明程式未執行。

不過，**目前程式有一項會讓新增 UV 測試失敗的實作錯誤，須修正後批准。**

## 必須修正：UV 正規化後解出的切線多除了一次尺度

**定位：**`triangle_tangent_frame`：

```python
U_scaled = U / sigma_max
...
tb = np.linalg.solve(U_scaled, edges) / sigma_max
```

**原因：**原方程為 $U X=E$。令 $U_{\text{scaled}}=U/\sigma_{\max}$，則 $U=\sigma_{\max}U_{\text{scaled}}$，所以正確解是：

$$
X=U^{-1}E
=\frac{1}{\sigma_{\max}}U_{\text{scaled}}^{-1}E.
$$

程式卻在 `solve` 後再除以 `sigma_max`，造成結果多縮小一次。對章內新增的 UV 邊長 $10^{-6}$ 案例，切線會比正確結果再小 $10^6$ 倍；隨後切線正規化可能觸發 `VECTOR_EPS`，或導致錯誤結果。這也違背測試所預期的小 UV 島應得出與標準 UV 相同的單位切線。

**修法：**改成先縮放右側再解：

```python
tb = np.linalg.solve(U_scaled, edges / sigma_max)
```

並補充核對 $U X=E$ 的殘差，且以小尺度 UV 案例確認輸出的切線、副切線方向與標準 UV 案例一致。

## 其餘核對

- **維度與座標：**矩陣式 UV 推導尺寸相符；TBN 縱行與 column-vector 慣例一致。
- **法線與色彩：**逆轉置法線、鏡射手性和 normal map 不套 sRGB 的說明正確。
- **手算與習題：**鏡射 UV、非均勻縮放、法線解碼及奇異值比例的手算結果合理。
- **邊界與引用：**奇異／病態情形有說明；來源清單未被表述成逐條獨立查證。
- **驗證界線：**文中明示未執行程式，因此不構成虛稱；但上述錯誤表示「預期結果」須在修正程式後重新核對。

VERDICT: REVISE