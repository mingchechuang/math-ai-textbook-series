## 獨立審稿結論

本輪有實質進步：命題 9.1 已改成正確的方向函數形式，原先純量／向量混用的 Taylor 展開已修復；程式也改成未裁切的分支式 sigmoid，使 logits BCE 與 backward 一致，並補上空 batch、輸入 feature 維度及 `unpack_params` 長度檢查。正文 3172 字亦已通過最低篇幅門檻。

但仍不能核准。主要阻擋項是：方向導數方法仍依賴外部舊梯度且恢復參數後不恢復 cache；有限差分沒有例外安全；所謂相對誤差仍定義錯誤；章綱要求的 tiny-set 過擬合與可執行邊界／故障測試尚未實作；正文宣稱 train／validation／test，程式仍只有 train/test；習題解答仍含未定義變數及省略號。

以下是靜態重算；我沒有執行程式，因此不確認 `PASS`、收斂或準確率。

---

## 一、已正確修復的部分

### 1. 命題 9.1 現在成立

**原句：**

> 「固定參數向量 $\theta$ 與方向 $v$，令 $\phi(t)=L(\theta+tv)$。若 $\phi$ 在零附近三階連續可微，則……」

這次改成對純量函數 $\phi$ 作 Taylor 展開，shape 正確。相減後常數項與二次項抵消，三階餘項除以 $2h$ 後為 $O(h^2)$；再由：

$$
\phi'(0)=\nabla L(\theta)^Tv
$$

得到方向導數公式。取 $v=e_j$ 亦可導出逐參數中心差分。此證明可以保留。

後續將 $10^{-5}$ 至 $10^{-6}$ 降格為「可試量級」，而非普遍最佳步長，也已修正前稿的過度宣稱。

### 2. 穩定 sigmoid 與 BCE 現已一致

目前程式對 $z\ge0$ 使用：

$$
\sigma(z)=\frac{1}{1+e^{-z}},
$$

對 $z<0$ 使用：

$$
\sigma(z)=\frac{e^z}{1+e^z}.
$$

兩個分支都計算原始 $\sigma(z)$，沒有裁切 logits，因此：

$$
\frac{\partial}{\partial z}
\left[\operatorname{logaddexp}(0,z)-yz\right]
=\sigma(z)-y
$$

與 backward 一致。舊稿中 $|z|>500$ 時 loss 與梯度分屬不同函數的問題已消失。

softplus 的負大輸入註解也已修正為趨近零，這一點正確。

### 3. `unpack_params` 的基本契約已修正

現有程式已檢查：

- `params.ndim == 1`
- 元素總數完全相等
- 所有元素有限

並以固定順序 $W_1,b_1,W_2,b_2$ 切割，不再使用綁死 `b2.size == 1` 的 `params[-1:]`。每段 `.copy()` 也避免參數繼續共享外部展平向量的底層儲存。這些修正可保留。

### 4. 空 batch 與輸入 feature 維度已處理

**原句／程式：**

> `if X.shape[0] != Y_true.shape[0] or X.shape[0] == 0:`

> `if X.shape[1] != self.W1.shape[0]:`

這已避免空 batch 進入 `np.mean`，並在矩陣乘法前提供明確 feature shape 契約。

---

## 二、方向導數方法仍不是自足函數

### 1. 仍使用外部舊梯度

**逐字程式：**

> `theta = self.pack_params()`

> `grad_flat = np.concatenate([self.dW1.flatten(), self.db1.flatten(), self.dW2.flatten(), self.db2.flatten()])`

方法接收 `X,Y_true`，但沒有先呼叫：

```python
self.forward(X, Y_true)
self.backward()
```

主程式目前恰好在呼叫前做了 forward/backward，但類別方法本身的契約仍不安全。例如：

1. 先對資料批次 A 執行 backward；
2. 再呼叫 `directional_derivative_check` 並傳入資料批次 B；
3. 解析方向導數來自 A，數值方向導數來自 B。

程式不一定報錯，只會得到錯誤比較。

**最小修法：**

在方法開頭自行對傳入的 `X,Y_true` 執行 forward/backward，再展平梯度。如此主程式也不必在外面重複建立 cache。

### 2. 恢復參數後沒有恢復 cache

方向導數方法依次執行：

1. `unpack_params(theta_plus)` 並 forward；
2. `unpack_params(theta_minus)` 並 forward；
3. `unpack_params(theta)`，但未 forward。

因此方法返回時，參數已是原始 $\theta$，但 `X,Z1,H,Z2,Y_pred` 仍是 `theta_minus` 對應的 cache。若呼叫者接著直接呼叫 `loss()` 或 `backward()`，會把原參數與舊 cache 混用。

主程式之後會重新 forward 訓練集，所以眼前流程未必立即出錯；但方法介面仍不自足。

**最小修法：**

恢復 $\theta$ 後執行：

```python
self.forward(X, Y_true)
self.backward()
```

### 3. 缺少 $v$、$h$、`tol` 驗證

目前沒有拒絕：

- `v` 長度錯誤；
- `v` 不是一維；
- `v` 含 NaN／無窮；
- $v=0$；
- $h\le0$ 或非有限；
- `tol <= 0`。

**最小修法：**

在方法入口明確驗證。零方向雖會使解析與數值方向導數都為零，但不提供任何驗證資訊，應拒絕。

---

## 三、有限差分仍沒有例外安全

新增正文已明確寫道：

> 「任何一步拋出例外時也要還原參數」

但程式仍是：

```python
param[idx] = original_value + h
...
param[idx] = original_value - h
...
param[idx] = original_value
```

若任一 `forward` 或 `loss` 拋出例外，最後的還原不會執行，模型可能永久停留在加 $h$ 或減 $h$ 的狀態。文字契約與實作仍不一致。

**最小修法：**

以 `try/finally` 包住擾動，保證：

```python
finally:
    param[idx] = original_value
    self.forward(self.X, self.Y_true)
    self.backward()
```

方向導數的整體參數替換也應採相同原則。另應檢查 $h$ 是有限正數，並確認 `loss_plus_val`、`loss_minus_val` 及數值梯度有限。

---

## 四、梯度誤差仍被錯稱為對稱相對誤差

**逐字程式：**

> `# Use symmetric relative error to handle small gradients`

> `denom = max(1.0, abs(num_grad), abs(ana_grad))`

> `rel_err = abs_err / denom`

正文已正確指出分母至少為 1 時，小梯度區域所得接近絕對誤差；但程式仍未同步修改命名。

若 $|g_n|,|g_a|<1$，則：

$$
\text{rel\_err}=|g_n-g_a|,
$$

並非通常意義的對稱相對誤差。

**最小修法：**

二選一：

1. 保留公式，但改名為 `scaled_err`；
2. 使用：
   $$
   \frac{|g_n-g_a|}
   {\max(\tau,|g_n|+|g_a|)}
   $$
   並明定 $\tau$。

目前程式雖計算 `max_abs_err`，通過條件卻只使用 `max_rel_err`。應明確規定兩種誤差何時各自生效，不要只改輸出名稱。

另建議記錄最大誤差的 `param_name` 與索引，否則「逐項核對」失敗後仍難定位。

---

## 五、正常測試存在，但邊界與故障測試仍未實作

主程式確實呼叫了全部參數的 `check_gradients` 和方向導數，這可算正常測試。然而「邊界測試」仍只是文字預期，沒有程式斷言。

至少應實作：

1. `(B,)` 標籤被 `_check_shapes` 拒絕；
2. 空 batch 被拒絕；
3. feature 維度錯誤被拒絕；
4. 標籤含 $0.5$、$-1$ 或 $2$ 被拒絕；
5. 輸入含 NaN／無窮被拒絕；
6. $B=1$ 時 `Y_pred.shape == (1,1)`；
7. pack/unpack 往返保持值與 shape；
8. 過短、過長及非有限展平向量被拒絕；
9. $z=100,y=0$ 與 $z=-100,y=1$ 的 loss／梯度預期；
10. 非法 `h`、`tol`、`v` 被拒絕。

### 故障測試仍是手動修改說明

**逐字原句：**

> 「正確的故障測試：例如漏掉 `1 - H**2`，即 `dZ1 = dH`。」

這不是自足故障測試，因為讀者必須手動破壞正式類別。應提供測試專用的錯誤梯度函數或 subclass，然後斷言檢查結果超過容差。

**逐字原句：**

> 「`max_rel_err` 將遠大於 $10^{-5}$。」

這仍屬未經執行的量化宣稱。若測試資料使 $H$ 接近零，漏掉 $1-H^2$ 的差異可能不大。

**最小修法：**

改成「預期至少一個參數的誤差超過容差」，並選固定參數使 $|H|$ 明顯不接近零。

---

## 六、小資料過擬合仍只存在於說明

新增正文已正確寫道：

> 「可另取少量固定訓練樣本，使用獨立模型更新參數，並在程式中以預先指定的 loss 或準確率門檻作斷言。」

但主程式仍只有：

> `B_train, B_test = 100, 20`

以及：

> `if final_train_loss > initial_train_loss * 0.1: print("Warning...")`

這不是 tiny-set overfit check，也沒有斷言。章綱明定「小資料過擬合檢查」，不能只在正文說明應如何做。

**最小修法：**

另建例如 8 筆固定資料與獨立 `tiny_model`：

- 記錄初始 tiny loss；
- 執行完整訓練 loop；
- 計算最終 tiny loss 和 accuracy；
- 未達預先設定門檻時拋出 `AssertionError`。

由於沒有執行紀錄，章稿只能把門檻與結果稱為預期。

---

## 七、train／validation／test 仍不一致

正文開頭宣稱：

> 「嚴格執行訓練集、驗證集與保留測試集的切分」

新增說明也正確區分三者；但實際程式只有 train 和 test，沒有 validation。

程式還依 test 指標發出：

> `Warning: Model performance on test set is poor.`

若作者看到這個 warning 後調整學習率、步數或隱藏維度，test 就參與了調參。

**最小修法：**

二選一：

1. 新增 validation set，所有超參數比較只看 validation，test 最後一次評估；
2. 若本章不做超參數選擇，刪除「validation」宣稱，並明示設定在看 test 前已鎖定。

小結中的：

> 「保留測試集以檢測過擬合」

也應改成「validation 用於診斷過擬合，test 用於最終評估」。

另有註解：

> `Use different seeds for train and test to ensure they are distinct`

不同 seed 建立不同隨機流，但不邏輯保證每個樣本值都不重複。最小修法是改成「independent synthetic draws」。

---

## 八、建構與更新參數仍缺少非法值測試

`__init__` 沒有拒絕：

- `dim_in <= 0`
- `dim_hid <= 0`
- 非整數維度

其中 `dim_in=0` 會直接使 `1.0/dim_in` 非法。

`train_one_step` 亦未拒絕：

- `lr <= 0`
- `lr` 為 NaN／無窮

雖然較晚可能由非有限 logits 間接發現問題，但應在更新前拒絕非法超參數。更新後也可檢查所有參數仍有限，避免壞狀態延續。

---

## 九、習題與完整解答仍不合格

### 1. 第一題題目資訊不足

**題目原句：**

> 「給定 $B=2,D_{in}=2,D_{hid}=2$，計算 $dW_2$ 和 $db_2$。」

只有維度，無法得到唯一數值。解答自行引入 $X,W_1,b_1,W_2,b_2,Y_{\mathrm{true}}$，這些資料應出現在題目中。

### 2. ReLU 解答含未定義變數

**逐字原句：**

> `self.H = np.maximum(0, Z1)`

> `dZ1 = dH * (Z1 > 0)`

類別內是 `self.Z1`，沒有區域變數 `Z1`。照此修改會發生 `NameError`。

**最小修法：**

```python
self.H = np.maximum(0.0, self.Z1)
dZ1 = dH * (self.Z1 > 0.0)
```

並明定零點導數約定。

### 3. 第三題條件不充分

**題目原句：**

> 「當 $Y=1$ 且 $W_2$ 極大時，梯度可能趨近 0」

$W_2$ 的絕對值大不保證 logit 是大正值。若真標籤 $y=1$ 但 $z\to-\infty$，則：

$$
dZ_2=\sigma(z)-1\to-1,
$$

不會趨零。

**最小修法：**

把條件改成「真標籤 $y=1$ 且 logit $z\to+\infty$」。

### 4. 第四題解答含省略號

**逐字原句：**

> `dZ1 = ...`

本卷要求完整解答，且程式不得留 omitted／省略內容。應給出完整可執行的 $B=1$ 測試，包括 forward、backward、手動重算及 shape/value 斷言。

---

## 十、來源與能力陳述

**逐字原句：**

> 「框架不會報錯，但模型將永遠無法收斂」

這仍過度概括。自動微分框架會依前向圖產生梯度，非法 shape 通常會報錯；可能靜默存在的是合法但語義錯誤的廣播、reduction、資料 shape 或自訂 backward。「永遠無法收斂」也沒有證據。

參考來源中的 NumPy與 PyTorch 條目仍明示「未逐條核對」，不能用作已查證依據。「PyTorch Documentation on Numerical Stability」沒有精確標題、版本或網址，而且本章不使用 PyTorch，應刪除或精確定位。

---

## 核准前最小修訂清單

1. 讓方向導數方法自行 forward/backward，並在還原參數後重建 cache。
2. 驗證方向 $v$、`h`、`tol`。
3. 對逐項與方向擾動使用 `try/finally`。
4. 修正「symmetric relative error」名稱與判定規則。
5. 將邊界與故障案例寫成可執行斷言。
6. 實作獨立 tiny-set overfit check。
7. 補 validation，或刪除三分割與相關能力宣稱。
8. 驗證建構維度及學習率。
9. 修正四項習題／解答問題。
10. 精確化框架敘述及參考來源。

VERDICT: REVISE