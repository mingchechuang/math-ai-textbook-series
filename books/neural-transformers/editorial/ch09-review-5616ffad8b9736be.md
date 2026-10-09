## 獨立審稿結論

本輪所附稿件與上一輪實質相同；我仍重新核算了公式、shape、reduction、有限差分、訓練流程及測試範圍。標籤 `(B,1)`、主要反向公式、logits BCE、參數展平與方向導數骨架已大致正確，但仍有數學證明不成立、loss 與 backward 不完全一致、測試未實作、小資料過擬合缺失、validation 宣稱與程式矛盾，以及正文不足 3000 字等阻擋問題。

以下僅為靜態審查。我沒有執行程式，不能確認 `PASS`、訓練收斂、準確率或任何誤差門檻。

---

## 一、重新核算後可保留的內容

現有資料建立方式：

> `Y_train = (X_train[:, [0]] > 0).astype(np.float64)`

會保留輸出軸，故 `Y_train.shape == (B,1)`。這與：

- $Z_2:(B,1)$
- $Y_{\mathrm{pred}}:(B,1)$
- $Y_{\mathrm{true}}:(B,1)$

相容，不再產生 `(B,1)-(B,) -> (B,B)` 的錯誤廣播。

對每筆樣本的 logits BCE：

$$
\ell_i=\operatorname{logaddexp}(0,z_i)-y_i z_i
$$

以及 batch mean：

$$
L=\frac{1}{B}\sum_{i=1}^B\ell_i,
$$

正確梯度為：

$$
dZ_2=\frac{\sigma(Z_2)-Y_{\mathrm{true}}}{B}.
$$

其餘梯度：

$$
dW_2=H^TdZ_2,\qquad db_2=\sum_{i=1}^BdZ_{2,i},
$$

$$
dH=dZ_2W_2^T,
$$

$$
dZ_1=dH\odot(1-H^2),
$$

$$
dW_1=X^TdZ_1,\qquad db_1=\sum_{i=1}^BdZ_{1,i}
$$

均與程式中的矩陣 shape 相符。兩個手算例題的結果也只存在正常的四位小數捨入差異。

---

## 二、命題 9.1 仍不是有效證明

### 1. 命題前提不能推出結論

**逐字原句：**

> 「若前向傳播計算正確，且使用中心差分進行數值梯度檢查，則解析梯度與數值梯度應在浮點精度與步長誤差範圍內一致。」

前向計算正確不能保證 backward 正確。例如把：

```python
dZ1 = dH * (1 - self.H ** 2)
```

錯寫成：

```python
dZ1 = dH
```

完全不影響前向，但會使解析梯度錯誤。中心差分是檢查工具，不是保證一致的條件。

**最小修法：**

把命題縮小為中心差分截斷誤差命題，再另行說明：解析梯度與有限差分一致，只是反向實作的局部數值證據，不能證明前向模型、資料語義或 loss 選擇皆正確。

### 2. Taylor 展開發生純量／向量 shape 錯誤

**逐字原句：**

> $$L(\theta+h)=L(\theta)+h\nabla L(\theta)+\frac{h^2}{2}\nabla^2L(\theta)h+O(h^3)$$

本章的 $\theta$ 是完整參數向量，$h$ 是純量。式中 $L(\theta)$ 是純量，但 $h\nabla L(\theta)$ 是向量，兩者不能相加；`theta + h` 也未指定沿哪個方向擾動。Hessian 項亦未形成正確的二次型。

**最小修法：**

固定方向 $v$，定義純量函數：

$$
\phi(t)=L(\theta+tv).
$$

若 $\phi$ 在零附近三階可微，則：

$$
\phi(h)=\phi(0)+h\phi'(0)+\frac{h^2}{2}\phi''(0)
+\frac{h^3}{6}\phi'''(\xi_+),
$$

$$
\phi(-h)=\phi(0)-h\phi'(0)+\frac{h^2}{2}\phi''(0)
-\frac{h^3}{6}\phi'''(\xi_-).
$$

相減並除以 $2h$：

$$
\frac{\phi(h)-\phi(-h)}{2h}
=\phi'(0)+O(h^2),
$$

而鏈式法則給出：

$$
\phi'(0)=\nabla L(\theta)^Tv.
$$

這才與章內的方向導數程式一致。

### 3. 「最佳步長」仍屬過度宣稱

**逐字原句：**

> 「故最佳步長 $h\approx10^{-5}$ 至 $10^{-6}$。」

若總誤差模型為：

$$
E(h)\approx C_1h^2+\frac{C_2u}{h},
$$

則最小點是：

$$
h_\star=\left(\frac{C_2u}{2C_1}\right)^{1/3}.
$$

其值依賴局部高階導數、函數尺度、參數尺度及捨入誤差常數，不能只由 float64 機械精度得到普遍最佳值。

**最小修法：**

改成「在尺度與常數約為一時，$10^{-5}$ 至 $10^{-6}$ 可作候選步長」，並建議比較多個 $h$。

---

## 三、loss 與 backward 仍不是同一函數的精確導數

**逐字程式：**

> `Z2_safe = np.clip(self.Z2, -500, 500)`

> `self.Y_pred = 1.0 / (1.0 + np.exp(-Z2_safe))`

> `loss_per_sample = np.logaddexp(0.0, self.Z2) - self.Y_true * self.Z2`

loss 使用原始 $Z_2$，但 backward 使用裁切後的 sigmoid：

$$
\sigma(\operatorname{clip}(Z_2,-500,500)).
$$

對 logits BCE 而言，正確導數必須使用 $\sigma(Z_2)$。當 $|Z_2|>500$ 時，解析梯度與有限差分所微分的函數不再完全一致。差異即使在飽和區可能很小，仍違反本章梯度驗證的數學契約。

**最小修法：**

改用不裁切輸入的分支式 sigmoid：

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

### loss 註解還包含錯誤漸近式

**逐字原句：**

> `For large negative Z2, log(1 + exp(Z2)) ~ Z2`

當 $z\to-\infty$ 時，正確結果是：

$$
\log(1+e^z)\sim e^z\to0,
$$

不是近似 $z$。只有 $z\to+\infty$ 時才近似 $z$。

**最小修法：**

刪除後續自問自答式註解，改為正確說明正、負兩側的 softplus 漸近行為。

---

## 四、有限差分與方向導數介面仍有狀態風險

### 1. `directional_derivative_check` 使用外部舊梯度

**逐字程式：**

> `grad_flat = np.concatenate([self.dW1.flatten(), self.db1.flatten(), ...])`

方法雖接收 `X, Y_true`，卻未先自行執行 `forward(X,Y_true)` 與 `backward()`。若呼叫者先前使用另一批資料，這裡會靜默讀取舊梯度。主程式目前剛好預先呼叫，但方法本身不自足。

**最小修法：**

方法開頭自行 forward/backward；還原 $\theta$ 後也重新 forward/backward，使 cache 與原始參數一致。

### 2. 方向 $v$ 未驗證

應檢查：

- `v.ndim == 1`
- `v.size == theta.size`
- 所有元素有限
- 範數非零
- `h` 與 `tol` 是有限正數

否則 `np.dot` 可能拋出不清楚的 shape 錯誤，或零方向使測試失去意義。

### 3. `unpack_params` 沒有長度契約

**逐字程式：**

> `self.b2 = params[-1:].reshape(self.b2.shape)`

此寫法假設 `b2.size == 1`，且沒有拒絕過長或非有限向量。pack/unpack 應為明確互逆。

**最小修法：**

先計算期望總長度，要求 `params` 是同長度一維有限向量，再以累積 offset 取出四段；不要用 `[-1:]`。

### 4. 參數擾動沒有例外安全

`numerical_gradient` 若在計算 `loss_plus_val` 或 `loss_minus_val` 時拋錯，原參數不會恢復。方向導數中的多次 unpack 也有相同問題。

**最小修法：**

以 `try/finally` 保證恢復原參數及 cache。

---

## 五、梯度誤差指標命名不正確

**逐字程式：**

> `denom = max(1.0, abs(num_grad), abs(ana_grad))`

> `rel_err = abs_err / denom`

註解稱這是：

> `Use symmetric relative error`

但當兩個梯度都小於 1 時，分母固定為 1，所得其實是絕對誤差。它可作尺度化誤差，但不是通常意義的 symmetric relative error。

**最小修法：**

二選一：

1. 改名為 `scaled_err`，並說明分母定義；
2. 使用
   $$
   \frac{|g_a-g_n|}{\max(\tau,|g_a|+|g_n|)}
   $$
   作對稱相對誤差。

正常驗收最好同時檢查最大絕對誤差與尺度化誤差，避免只用一個數值掩蓋近零梯度問題。

---

## 六、輸入與模型契約仍不完整

### 1. 空 batch 在 loss 之後才可能被拒絕

**逐字程式：**

> `if B == 0: raise ValueError("Batch size cannot be 0")`

此檢查只在 backward。空 batch 可先通過 `_check_shapes`，而 `loss()` 會對空陣列做 `np.mean`。

**最小修法：**

在 `_check_shapes` 中立即拒絕 `X.shape[0] == 0`。

### 2. 未驗證 feature 維度

應明確要求：

```python
X.shape[1] == self.W1.shape[0]
```

而不是等待 `@` 發出低階錯誤。

### 3. 建構參數未驗證

`dim_in=0` 會導致 `1.0 / dim_in` 非法。`dim_in`、`dim_hid` 應為正整數。`train_one_step` 也應拒絕非有限或非正學習率。

### 4. 未驗證參數有限性

`_check_shapes` 只檢查輸入和標籤。如果某次更新使參數成為 NaN／無窮，應在 forward 開始時明確拒絕，而不是等待 `Z2` 間接出錯。

---

## 七、正常、邊界與故障測試仍沒有完整程式

「測試與預期結果」目前主要是敘述。主程式沒有真正執行以下斷言：

- `(B,)` 標籤被拒絕；
- 空 batch 被拒絕；
- 非二元標籤被拒絕；
- NaN／無窮被拒絕；
- $B=1$ 時輸出軸不消失；
- 極端有限 logits 的 loss 與梯度方向正確；
- pack/unpack 往返保持所有值和 shape；
- 錯誤展平向量長度被拒絕；
- 漏掉 tanh 導數確實使梯度檢查失敗。

**逐字原句：**

> 「正確的故障測試：例如漏掉 `1 - H**2`」

這只是要求讀者手動改壞正式程式，不是自足故障測試。

**逐字原句：**

> 「`max_rel_err` 將遠大於 $10^{-5}$。」

沒有執行紀錄或普遍數學保證。若 $H$ 很接近零，$1-H^2$ 很接近 1，漏掉該因子造成的差異可能不大。

**最小修法：**

提供獨立的故障 backward 或測試 subclass，並選固定資料使 $|H|$ 明顯不接近零。未執行時只能寫「預期至少一項誤差超過容差」。

---

## 八、小資料過擬合檢查仍缺失

**逐字程式：**

> `B_train, B_test = 100, 20`

> `for i in range(100):`

> `if final_train_loss > initial_train_loss * 0.1: print(...)`

這是一般全批訓練，不是章綱要求的 tiny-set overfit check。小資料過擬合的目的，是使用極少數固定樣本確認模型、loss、backward 和更新步驟能共同運作。

此外，現稿只印出 warning，不會使驗收失敗；固定 100 步是否足以使 loss 降至初始值的 10%，也沒有執行證據。

**最小修法：**

另建獨立模型與 4 至 10 筆固定資料，執行完整訓練 loop，最後以斷言檢查預先指定的 loss 或 accuracy 門檻。結果只能標為預期，不能宣稱已通過。

---

## 九、train／validation／test 宣稱與程式矛盾

**逐字原句：**

> 「嚴格執行訓練集、驗證集與保留測試集的切分」

程式實際只有 train 與 test，沒有 validation。隱藏維度、學習率、步數及警告門檻若根據 test 結果調整，test 就參與了調參。

**逐字原句：**

> 「保留測試集以檢測過擬合」

較正確的角色是：

- train：更新參數；
- validation：選設定、診斷過擬合；
- test：設定鎖定後只作一次最終評估。

**最小修法：**

補 validation set；若本章刻意不做模型選擇，則刪除「驗證集」宣稱，並明說所有設定事先固定且不依 test 結果修改。

另有註解：

> `Use different seeds for train and test to ensure they are distinct`

不同 seed 表示不同隨機流，不是資料絕不重複的邏輯保證。可改為「產生獨立的合成抽樣」，不要寫「確保每筆值皆不同」。

---

## 十、習題解答含未定義變數

**逐字原句：**

> `self.H = np.maximum(0, Z1)`

> `dZ1 = dH * (Z1 > 0)`

類別中定義的是 `self.Z1`，沒有區域變數 `Z1`。照此解答修改會出現 `NameError`。

**最小修法：**

```python
self.H = np.maximum(0.0, self.Z1)
dZ1 = dH * (self.Z1 > 0.0)
```

並明確約定 ReLU 在零點的導數。有限差分測試還應避免擾動跨越零點。

---

## 十一、敘述與來源仍需收斂

**逐字原句：**

> 「框架不會報錯，但模型將永遠無法收斂，或者收斂到局部極小值。」

這是過度概括。自動微分框架通常會從前向圖生成梯度，非法 shape 常會報錯；真正可能靜默錯誤的是合法但語義錯誤的廣播、reduction、標籤 shape 或自訂 backward。「永遠無法收斂」也沒有證據。

**最小修法：**

改為「框架能微分既有計算圖，但不保證模型語義、資料 shape、reduction 或自訂 backward 正確」。

參考來源中的 NumPy 與 PyTorch 已標「未逐條核對」，不能作為已查證依據。「PyTorch Documentation on Numerical Stability」沒有明確標題、版本或網址，而且本章沒有使用 PyTorch，應刪除或改成可定位來源。

---

## 十二、正文仍未達硬性下限

提供的檢查資料為：

> `measured_characters: 2125`

並明示：

> 「正文不足，需补足推導及實作說明」

本卷要求每章至少 3000 中文字，且程式、公式、英文及來源不計。現稿仍未達最低門檻。

可用下列實質內容補足，而非灌水：

1. 正確的方向函數 Taylor 證明；
2. 逐項有限差分與方向導數的差異；
3. stable sigmoid 的正負分支原理；
4. pack/unpack 的 offset、shape 與互逆條件；
5. tiny-set overfit 的診斷意義；
6. train／validation／test 的使用規則；
7. 正常、邊界、故障測試的完整斷言與預期失敗說明。

---

## 核准前最小修訂清單

1. 以分支式 stable sigmoid 取代 logit clip。
2. 用 $\phi(t)=L(\theta+tv)$ 重寫命題及證明。
3. 將「最佳步長」改成有條件的候選量級。
4. 修正負大輸入時 softplus 的錯誤註解。
5. 驗證 pack/unpack 的 ndim、長度、有限性及互逆性。
6. 讓方向導數函數自行 forward/backward 並恢復 cache。
7. 對擾動使用 `try/finally`。
8. 拒絕空 batch、錯 feature 維度、非法建構維度與非法學習率。
9. 正確命名梯度誤差指標。
10. 將邊界及故障案例寫成可執行斷言。
11. 實作獨立 tiny-set overfit check。
12. 補 validation，或刪除三分割宣稱。
13. 修正 ReLU 解答中的 `Z1`。
14. 精確化來源及框架能力敘述。
15. 將有效中文正文補至至少 3000 字。

VERDICT: REVISE