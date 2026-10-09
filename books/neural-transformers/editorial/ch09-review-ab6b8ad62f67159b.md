## 獨立審稿結論

本次修訂已正確修復最嚴重的 `(B,)` 標籤廣播問題，並加入 logits 版 BCE、參數展平及方向導數介面；手算中的 bias 方向也已修正。這些是實質進步。然而目前仍不能核准，因為：

1. 穩定 BCE 的 loss 與裁切 sigmoid 的 backward 仍非同一函數；
2. 命題 9.1 的證明把向量參數與純量擾動混用，不是有效證明；
3. 宣稱有 train／validation／test，程式卻只有 train／test；
4. 尚未真正完成「小資料過擬合檢查」；
5. 邊界與故障測試仍只是敘述，沒有納入自足程式的可執行斷言；
6. ReLU 習題解答含未定義變數；
7. 正文依提供的計數仍只有 2125 中文字，低於最低 3000。

以下為靜態重算結果。我沒有執行程式，不能確認輸出的 `PASS`、收斂情形或任何實際誤差值。

---

## 一、已修正且可保留的部分

### 1. 標籤 shape 已修正

**原句／程式：**

> `Y_train = (X_train[:, [0]] > 0).astype(np.float64)`

> `Y_test = (X_test[:, [0]] > 0).astype(np.float64)`

使用 `[:, [0]]` 會保留第二維，因此標籤為 `(B,1)`，與 `Y_pred:(B,1)` 同形。這避免了舊稿中 `(B,1)-(B,)` 被廣播成 `(B,B)` 的致命錯誤。

`_check_shapes` 也明確拒絕 `(B,)` 標籤，方向正確。

### 2. logits BCE 公式正確

正文給出：

$$
L_i=\operatorname{logaddexp}(0,z_i)-y_i z_i.
$$

對 $z_i$ 微分：

$$
\frac{\partial L_i}{\partial z_i}
=\sigma(z_i)-y_i.
$$

批次沿全部 $B$ 筆樣本平均後：

$$
dZ_2=\frac{\sigma(Z_2)-Y_{\mathrm{true}}}{B}.
$$

因輸出維度固定為 1，`np.mean(loss_per_sample)` 的分母確實為 $B$。因此除一次 $B$ 的 reduction 約定成立。

### 3. 兩個手算的主要數值一致

以 $z=0.19486$ 重算：

$$
\sigma(z)\approx0.54856,
$$

$$
\log(1+e^z)-z\approx0.60046.
$$

稿中因中間數值只保留四位，寫成約 $0.60034$，誤差屬手算捨入範圍。後續

$$
dZ_2\approx-0.45144,
$$

$$
dW_2\approx
\begin{bmatrix}
-0.2424\\
-0.0450
\end{bmatrix}
$$

亦一致。

### 4. 舊故障測試的自我修正正確

稿中已承認：

> 「在這種特定情況下（輸出維度為 1），逐元素乘法與矩陣乘法結果相同。」

這項判斷正確。因為

$$
dZ_2:(B,1),\qquad W_2^T:(1,D_{hid}),
$$

兩者逐元素廣播所得外積，恰好等於矩陣乘法。改用漏掉 tanh 導數作為故障 mutation 是合理方向。

---

## 二、仍須修正的數學問題

### 1. 命題 9.1 的敘述仍不成立

**原句：**

> 「若前向傳播計算正確，且使用中心差分……則解析梯度與數值梯度應在浮點精度與步長誤差範圍內一致。」

**原因：**

前向正確並不推出解析梯度正確。若 backward 漏掉 tanh 導數，前向仍完全正確，但解析梯度與中心差分不一致。此命題至少要加入「解析梯度是該損失的真導數」；然而加入後又近乎把結論放進前提，不能作為反向程式正確性的證明。

有限差分的角色是提供局部數值核對，不是由「前向正確」推出「反向正確」。

**最小修法：**

把命題改為純數學命題：

> 若純量函數 $f$ 在 $x$ 鄰域三階可微，則中心差分  
> $[f(x+h)-f(x-h)]/(2h)$ 與 $f'(x)$ 的差為 $O(h^2)$。

然後另起一段說明：將此命題套用到每個參數座標或方向函數，只能形成反向實作的局部數值證據。

---

### 2. 證明混淆純量與向量 shape

**原句：**

> 「設損失函數 $L(\theta)$ 在某點 $\theta$ 附近三階可微。」

> $$L(\theta+h)=L(\theta)+h\nabla L(\theta)+\frac{h^2}{2}\nabla^2L(\theta)h+O(h^3)$$

**原因：**

本章的 $\theta$ 是展平後的參數向量，而 $h$ 在後文是純量步長。此時：

- $L(\theta+h)$ 沒有指定將純量 $h$ 加到哪個座標；
- $L(\theta)$ 是純量；
- $h\nabla L(\theta)$ 是向量；
- 純量與向量不能相加；
- $\nabla^2L(\theta)h$ 在 $h$ 是純量時也不是所需的二次型。

因此目前的 Taylor 展開 shape 不成立。

**最小修法一：座標版本。**

固定第 $j$ 個標準基底 $e_j$，定義

$$
f(t)=L(\theta+t e_j).
$$

則對純量 $t$ 展開：

$$
f(h)=f(0)+hf'(0)+\frac{h^2}{2}f''(0)
+\frac{h^3}{6}f'''(\xi_+),
$$

$$
f(-h)=f(0)-hf'(0)+\frac{h^2}{2}f''(0)
-\frac{h^3}{6}f'''(\xi_-).
$$

相減並除以 $2h$，才得到 $f'(0)+O(h^2)$。

**最小修法二：方向版本。**

固定與 $\theta$ 同 shape 的方向 $v$，定義

$$
\phi(t)=L(\theta+tv).
$$

則

$$
\phi'(0)=\nabla L(\theta)^Tv,
$$

中心差分核對的正是此方向導數。這也能直接銜接程式。

---

### 3. $h\approx10^{-5}$ 到 $10^{-6}$ 不能稱為普遍「最佳步長」

**原句：**

> 「故最佳步長 $h\approx10^{-5}$ 至 $10^{-6}$。」

**原因：**

從

$$
E(h)\approx C_1h^2+\frac{C_2u}{h}
$$

只能得到

$$
h_\star=\left(\frac{C_2u}{2C_1}\right)^{1/3}.
$$

它仍取決於 $C_1,C_2$、參數尺度、損失尺度和局部三階導數。$u^{1/3}$ 只是量級啟發，不是所有參數共用的最佳值。

**最小修法：**

改成「若常數與尺度約為 1，float64 的候選步長常落在 $10^{-5}$ 至 $10^{-6}$；實務上應比較數個 $h$，確認誤差先下降、後受捨入誤差影響。」

---

### 4. 穩定 BCE 與 sigmoid backward 仍有函數不一致

**原句／程式：**

> `Z2_safe = np.clip(self.Z2, -500, 500)`

> `self.Y_pred = 1.0 / (1.0 + np.exp(-Z2_safe))`

> `loss_per_sample = np.logaddexp(0.0, self.Z2) - self.Y_true * self.Z2`

> `dZ2 = (self.Y_pred - self.Y_true) / B`

**原因：**

loss 使用未裁切的 `self.Z2`，但 backward 使用 $\sigma(\operatorname{clip}(Z_2))$。解析上，真正的 logits BCE 梯度應為 $\sigma(Z_2)-Y_{\mathrm{true}}$，不是裁切後 sigmoid。

在 $|z|\le500$ 時兩者相同；超出範圍後則不是同一函數的導數。差異可能因飽和而極小，但數學契約仍然錯誤，尤其本章主題就是逐項梯度驗證，不能以「數值上可能很小」略過。

**最小修法：**

使用不裁切輸入的分支式穩定 sigmoid：

```python
def stable_sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out
```

然後：

```python
self.Y_pred = stable_sigmoid(self.Z2)
```

如此 loss 與 backward 才對應同一個 logits BCE。不得以任意 clip 假裝精確。

---

### 5. loss 註解含明顯錯誤敘述

**原句：**

> `For large negative Z2, log(1 + exp(Z2)) ~ Z2`

**原因：**

當 $z\to-\infty$ 時，

$$
\log(1+e^z)\sim e^z\to0,
$$

不是近似 $z$。近似 $z$ 是 $z\to+\infty$ 時的結果。後面的註解自行產生「loss 可能為負」的疑問，雖再口頭否定，仍留下錯誤推導。

**最小修法：**

刪除整段自問自答式註解，改成：

- $z\gg0$ 時，`logaddexp(0,z) ≈ z`；
- $z\ll0$ 時，`logaddexp(0,z) ≈ exp(z) ≈ 0`。

---

## 三、程式自足性與介面問題

### 1. `directional_derivative_check` 並非真正自足

**原句／程式：**

> `grad_flat = np.concatenate([self.dW1.flatten(), ...])`

函數接收 `X, Y_true`，卻沒有在內部執行 `forward(X,Y_true)` 和 `backward()`。它依賴呼叫者事先建立與相同資料對應的梯度 cache。主程式目前確實先做了 forward/backward，但方法介面本身容易讀取過期梯度。

**最小修法：**

在 `directional_derivative_check` 開頭加入：

```python
self.forward(X, Y_true)
self.backward()
```

完成後恢復原參數並重新 forward/backward，讓 cache 與原參數一致。這也避免後續程式誤用 `theta_minus` 的 cache。

---

### 2. `unpack_params` 沒有驗證向量長度

**原句／程式：**

> `self.b2 = params[-1:].reshape(self.b2.shape)`

**原因：**

若 `params` 過長，前面切片可能忽略額外元素，而 `[-1:]` 仍能取最後一項；若架構將來改成非單一輸出，這個寫法也不通用。pack/unpack 應是明確互逆，不能默許錯誤長度。

**最小修法：**

先計算總長度並要求：

```python
expected = self.W1.size + self.b1.size + self.W2.size + self.b2.size
if params.ndim != 1 or params.size != expected:
    raise ValueError(...)
```

再依累積 offset 切出 `b2`，不要用 `[-1:]`。

---

### 3. 空 batch 拒絕位置太晚

**原句／程式：**

> `if B == 0: raise ValueError(...)`

這只在 `backward` 發生。空 batch 已可先通過 `_check_shapes`，進入 forward，接著 `loss()` 的 `np.mean` 會產生非有限結果或警告。

**最小修法：**

在 `_check_shapes` 中加入：

```python
if X.shape[0] == 0:
    raise ValueError("Batch size cannot be 0")
```

使 forward、loss 與 backward 共用同一契約。

---

### 4. 未檢查模型參數是否有限

`_check_shapes` 只檢查 `X` 與 `Y_true`。若參數在更新後成為 NaN 或無窮，只有 `loss()` 最後檢查 `Z2`；`W1` 的 NaN 會間接被抓到，但錯誤位置與原因不清楚。

最小修法是在 forward 入口對四個參數做有限值檢查，或提供統一的 `_check_params`。學習率也應要求有限且正值。

---

### 5. 數值擾動缺少 `try/finally`

**原句／程式：**

> `param[idx] = original_value + h`

> `...`

> `param[idx] = original_value`

若中間的 `forward` 或 `loss` 拋出例外，參數不會還原，模型狀態受污染。

**最小修法：**

以 `try/finally` 保證原值一定恢復，並檢查 `h` 是有限正數。

---

## 四、訓練、切分與評估範圍

### 1. 宣稱三分割，但程式沒有 validation

**原句：**

> 「嚴格執行訓練集、驗證集與保留測試集的切分」

**實際程式：**

只有 `X_train/Y_train` 與 `X_test/Y_test`，沒有 validation set。

**原因：**

文字與程式契約不一致。學習率 `0.1`、100 步及隱藏維度 8 若經任何結果觀察後調整，理應只使用 validation，而不能使用 test。

**最小修法：**

二選一：

1. 真正生成 train／validation／test 三份獨立資料，用 validation 選訓練步數或學習率，test 最後只評估一次；
2. 若本章不做超參數選擇，刪除「驗證集」宣稱，清楚說所有設定預先固定，test 僅最後使用。

由於正文後面的養殖案例也明列 Train、Validation、Test，較一致的修法是補上 validation。

---

### 2. 尚未完成「小資料過擬合檢查」

**原句／程式：**

> `B_train, B_test = 100, 20`

> `if final_train_loss > initial_train_loss * 0.1: print("Warning...")`

**原因：**

100 筆線性邊界資料配 100 次全批更新，是一般訓練示例，不是明確的小資料 overfit 除錯。過擬合檢查通常選極少量樣本，例如 4、8 或 10 筆，確認模型能把訓練 loss 壓低並達高訓練準確率。

目前也只有 warning，沒有失敗斷言；而「降到初始 loss 的 10%」是否能在固定 100 步達成未經執行，不能當作可靠門檻。

**最小修法：**

另外建立 `X_tiny, Y_tiny`，使用獨立模型訓練固定上限步數，檢查：

- 最終 loss 小於明定門檻；
- 或 tiny-set accuracy 達到明定門檻；
- 未達門檻時拋出 `AssertionError`。

所有數字在沒有執行紀錄時只能稱為預期門檻，不能聲稱已達成。

---

### 3. test set 被用來發出模型品質警告

**原句／程式：**

> `if test_acc < 0.5: print("Warning: Model performance on test set is poor.")`

最終一次報告 test 指標並非資料洩漏；但若作者依此警告回頭改模型、epoch 或學習率，test 就參與調參。章稿應明示這個判斷只是最終報告，不據此修改模型。需要迭代決策時應用 validation。

### 4. 「以測試集檢測過擬合」表述不精確

**原句：**

> 「保留測試集以檢測過擬合」

訓練／驗證落差可用於診斷和模型選擇；test 應保留作最終無偏估計。最小修法是改成「validation 用於診斷與選擇；test 在設定鎖定後只作最終評估」。

---

## 五、測試仍未達章規要求

「測試與預期結果」目前主要是文字，沒有實際放進自足程式的測試函數或斷言。章規要求正常、邊界、故障測試，不能只說應發生什麼。

至少應補入可執行但不宣稱已執行的測試：

1. 正常：全部參數逐項梯度與方向導數低於容差。
2. shape：`Y.shape == (B,)` 時確實拋出 `ValueError`。
3. 空 batch：forward 立即拒絕。
4. 非法標籤：`-1`、`0.5`、`2` 被拒絕。
5. 非有限資料：NaN、正負無窮被拒絕。
6. 極端 logits：$z=\pm100$ 且標籤正確／錯誤時，loss 與梯度有限且方向正確。
7. pack/unpack：往返後每個參數值與 shape 不變。
8. 錯誤向量長度：unpack 拋出例外。
9. 故障 mutation：漏掉 `1-H**2` 後梯度檢查失敗。
10. `B=1`：batch 軸與輸出軸均保留，bias 梯度 shape 與參數相同。

故障測試目前寫：

> 「`max_rel_err` 將遠大於 $10^{-5}$。」

由於沒有執行紀錄，而且初始化或資料可能使 tanh 導數接近 1，不能保證「遠大於」。最小修法是寫「預期至少一項誤差超過設定容差；測試資料應選擇使 $H$ 不接近零，以避免 mutation 偶然不明顯」。

---

## 六、習題解答含程式錯誤

**原句：**

> `將 self.H = np.tanh(Z1) 改為 self.H = np.maximum(0, Z1)`

> `導數改為 dZ1 = dH * (Z1 > 0)`

**原因：**

在類別的 `forward` 中定義的是 `self.Z1`，沒有區域變數 `Z1`。照解答修改會得到 `NameError`。backward 同樣應使用 `self.Z1`。

**最小修法：**

```python
self.H = np.maximum(0.0, self.Z1)
```

及

```python
dZ1 = dH * (self.Z1 > 0.0)
```

另應指出有限差分若跨越 ReLU 的零點，中心差分不再核對單一可微導數；初始化與測試資料應避免 $Z_1$ 太靠近零，或明定零點次梯度約定。

---

## 七、敘述與來源問題

### 1. 框架敘述仍過度概括

**原句：**

> 「若我們在反向傳播中錯誤地處理了轉置……框架不會報錯」

一般自動微分框架會從前向圖建立導數，使用者不必手寫線性層 backward；shape 不合法時也常會直接報錯。真正不一定報錯的是「shape 合法但語義錯誤的前向圖」或「自訂 backward」。

**最小修法：**

改成：「高階框架能自動計算已建立計算圖的梯度，但不能保證使用者建立的模型語義、reduction、標籤 shape 或自訂梯度正確。」

### 2. 來源仍不可視為已核對

參考來源中的 NumPy 與 PyTorch 已標「未逐條核對」，這點誠實；但：

> 「PyTorch Documentation on Numerical Stability」

沒有明確文件名、版本或網址，且本章並未使用 PyTorch。最小修法是刪除無法定位的條目，或提供精確可核對來源並保留「未逐條核對」狀態。Goodfellow Chapter 6 的具體內容也不可在未核對下用作本章數值差分證明的唯一依據。

---

## 八、篇幅與最低章規

提供的資料明示：

> `measured_characters: 2125`

以及：

> 「正文不足，需补足推導及實作說明」

這仍低於每章最低 3000 中文字，而且公式、程式、英文與參考來源不計入正文。核准前必須補足有效中文正文。最適合補寫而非灌水的內容包括：

1. 用純量方向函數重做中心差分證明；
2. 說明逐參數差分與方向導數各自能發現什麼、不能證明什麼；
3. 補完整的 pack/unpack slice 表及互逆條件；
4. 詳述 stable sigmoid 的正負分支；
5. 補 train／validation／test 使用契約；
6. 補小資料 overfit 的目的與失敗解讀；
7. 把正常、邊界、故障測試寫成程式斷言並逐項解釋預期。

---

## 九、核准前最小修訂清單

1. 將裁切 sigmoid 改成正負分支的未裁切穩定 sigmoid。
2. 以 $f(t)=L(\theta+te_j)$ 或 $\phi(t)=L(\theta+tv)$ 重寫命題與證明。
3. 刪除 large negative softplus 近似為 $z$ 的錯誤註解。
4. 讓方向導數函數自行 forward/backward，並在結束後恢復 cache。
5. 驗證 unpack 向量的 ndim、總長度與有限值。
6. 在 forward 前拒絕空 batch，並檢查參數及學習率。
7. 用 `try/finally` 保證有限差分擾動一定還原。
8. 補 validation，或刪除三分割宣稱並禁止依 test 調參。
9. 另做真正的 tiny-set overfit check，不以 100 筆一般訓練取代。
10. 把正常、邊界及故障案例加入自足程式的可執行斷言。
11. 修正 ReLU 解答中的 `Z1` 為 `self.Z1`。
12. 將正文補至至少 3000 中文字，並精確化來源。

VERDICT: REVISE