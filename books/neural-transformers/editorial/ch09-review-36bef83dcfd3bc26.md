## 獨立審稿結論

本輪稿件與前一輪所附稿件實質相同；前輪指出的證明、穩定 sigmoid、測試、資料切分及篇幅問題均仍存在。現稿已具備兩層 tanh 網路、完整前反傳、逐參數有限差分、參數展平、方向導數及訓練 loop 的主要骨架，也已修正早期 `(B,)` 標籤造成的錯誤廣播；但尚不能稱為符合本章驗收契約的自足實作。

本審查只作靜態重算，沒有執行程式；因此不能確認稿中預期的 `PASS`、損失下降或準確率。

---

## 一、重算結果

現有標籤由：

> `Y_train = (X_train[:, [0]] > 0).astype(np.float64)`

產生，shape 為 $(B,1)$，與預測輸出一致。正確的前反傳 shape 是：

- $X:(B,D_{in})$
- $W_1:(D_{in},D_{hid})$
- $b_1:(D_{hid},)$
- $Z_1,H:(B,D_{hid})$
- $W_2:(D_{hid},1)$
- $b_2:(1,)$
- $Z_2,Y_{\mathrm{pred}},Y_{\mathrm{true}}:(B,1)$

對 logits BCE：

$$
L=\frac{1}{B}\sum_{i=1}^B
\left[\operatorname{logaddexp}(0,z_i)-y_i z_i\right],
$$

梯度是：

$$
dZ_2=\frac{\sigma(Z_2)-Y_{\mathrm{true}}}{B},
$$

$$
dW_2=H^TdZ_2,\qquad db_2=\sum_{i=1}^BdZ_{2,i},
$$

$$
dZ_1=(dZ_2W_2^T)\odot(1-H^2),
$$

$$
dW_1=X^TdZ_1,\qquad db_1=\sum_{i=1}^BdZ_{1,i}.
$$

正文與 backward 的矩陣 shape 基本符合這些公式。兩個手算例題的主要數值也在捨入誤差範圍內。

---

## 二、阻擋核准的數學錯誤

### 1. 命題的前提不能推出結論

**逐字原句：**

> 「若前向傳播計算正確，且使用中心差分進行數值梯度檢查，則解析梯度與數值梯度應在浮點精度與步長誤差範圍內一致。」

前向正確並不保證解析 backward 正確。漏掉 tanh 導數時，前向仍正確，解析梯度卻會錯。中心差分是檢測此錯誤的方法，不是保證兩者一致的前提。

**最小修法：**

將命題改為純粹的中心差分截斷誤差命題；另外說明梯度核對只是局部數值證據，不是正確性保證。

### 2. Taylor 展開的 shape 不合法

**逐字原句：**

> $$L(\theta+h)=L(\theta)+h\nabla L(\theta)+\frac{h^2}{2}\nabla^2L(\theta)h+O(h^3)$$

文中的 $\theta$ 是參數向量，$h$ 是純量。因此 $L(\theta)$ 是純量，$h\nabla L(\theta)$ 卻是向量，兩者不能相加；$L(\theta+h)$ 也未指定擾動哪一個座標。Hessian 項同樣缺少方向向量形成的二次型。

**最小修法：**

固定方向 $v$，定義：

$$
\phi(t)=L(\theta+tv).
$$

對純量 $t$ 展開：

$$
\phi(h)=\phi(0)+h\phi'(0)+\frac{h^2}{2}\phi''(0)
+\frac{h^3}{6}\phi'''(\xi_+),
$$

$$
\phi(-h)=\phi(0)-h\phi'(0)+\frac{h^2}{2}\phi''(0)
-\frac{h^3}{6}\phi'''(\xi_-).
$$

相減後得到：

$$
\frac{\phi(h)-\phi(-h)}{2h}
=\phi'(0)+O(h^2),
$$

且

$$
\phi'(0)=\nabla L(\theta)^Tv.
$$

這才是與方向導數程式一致的完整小命題。

### 3. 「最佳步長」宣稱過強

**逐字原句：**

> 「故最佳步長 $h\approx10^{-5}$ 至 $10^{-6}$。」

由

$$
E(h)\approx C_1h^2+\frac{C_2\epsilon_{\mathrm{mach}}}{h}
$$

只能推出：

$$
h_\star=
\left(\frac{C_2\epsilon_{\mathrm{mach}}}{2C_1}\right)^{1/3}.
$$

$C_1,C_2$ 取決於局部導數、參數尺度與損失尺度。$10^{-5}$ 至 $10^{-6}$ 只能說是尺度約為一時的候選量級。

**最小修法：**

把「最佳」改成「候選」，並建議以多個步長觀察誤差是否先下降後上升。

---

## 三、loss 與 backward 仍不完全一致

**逐字程式：**

> `Z2_safe = np.clip(self.Z2, -500, 500)`

> `self.Y_pred = 1.0 / (1.0 + np.exp(-Z2_safe))`

> `loss_per_sample = np.logaddexp(0.0, self.Z2) - self.Y_true * self.Z2`

loss 使用原始 $Z_2$，但 backward 使用 $\sigma(\operatorname{clip}(Z_2))$。當 $|Z_2|>500$ 時，程式中的解析梯度並非該 loss 的精確導數。本章專門驗證梯度，不能保留此定義差異。

**最小修法：**

改成不裁切輸入的分支式 sigmoid：

```python
def stable_sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out
```

然後令 `self.Y_pred = stable_sigmoid(self.Z2)`。

### loss 註解另有明確錯誤

**逐字原句：**

> `For large negative Z2, log(1 + exp(Z2)) ~ Z2`

實際上當 $z\to-\infty$：

$$
\log(1+e^z)\sim e^z\to0.
$$

只有 $z\to+\infty$ 時才近似 $z$。後續自問「loss 是否為負」正是由此錯誤近似引起。

**最小修法：**

刪除該段自問自答，改寫正負兩側的正確漸近結果。

---

## 四、參數展平與方向導數仍不夠自足

### 1. `unpack_params` 未驗證長度

**逐字程式：**

> `self.b2 = params[-1:].reshape(self.b2.shape)`

這假設 `b2.size == 1`，也可能默許過長向量。沒有檢查 `params.ndim`、總長度及有限性。

**最小修法：**

計算四組參數的總元素數，要求輸入為同長度的一維有限向量，再以累積 offset 切割；不要用 `[-1:]`。

### 2. 方向導數依賴呼叫者預先建立梯度

**逐字程式：**

> `grad_flat = np.concatenate([self.dW1.flatten(), ...])`

`directional_derivative_check` 接收 `X,Y_true`，但沒有自行呼叫 forward/backward。若 cache 來自另一批資料，程式會靜默比較不相干的解析梯度。

**最小修法：**

方法開頭自行執行：

```python
self.forward(X, Y_true)
self.backward()
```

並檢查 `v` 的 shape、長度、有限性及非零範數。還原原參數後重新 forward/backward，使 cache 與原參數一致。

### 3. 擾動失敗時不保證還原

`numerical_gradient` 和方向導數檢查都未使用 `try/finally`。若中間 loss 拋出例外，模型可能停留在擾動後的參數。

**最小修法：**

將恢復原參數放入 `finally`，並拒絕非有限或非正的 `h`。

---

## 五、輸入與狀態契約不完整

### 1. 空 batch 拒絕得太晚

**逐字程式：**

> `if B == 0: raise ValueError("Batch size cannot be 0")`

它位於 backward。空 batch 可先通過 `_check_shapes`，然後 `loss()` 對空陣列取 `mean`。

**最小修法：**

在 `_check_shapes` 內立即拒絕 `X.shape[0] == 0`。

### 2. 未檢查 feature 維度

`_check_shapes` 沒有要求：

```python
X.shape[1] == self.W1.shape[0]
```

雖然錯誤維度通常會由矩陣乘法拋錯，但自足介面應提供明確契約與錯誤訊息。

### 3. 建構參數未驗證

`dim_in=0` 會使初始化中的 `1.0/dim_in` 非法；負維度及非整數也應被拒絕。`train_one_step` 也未拒絕非有限、零或負學習率。

### 4. loss/backward 可在 forward 前被呼叫

此時 cache 是 `None`，錯誤訊息不清楚。最小修法是在 loss/backward 開頭檢查必要 cache 是否存在，或改成 `loss(X,Y)` 的無狀態介面。

---

## 六、梯度誤差名稱與判定需要修正

**逐字程式：**

> `denom = max(1.0, abs(num_grad), abs(ana_grad))`

> `rel_err = abs_err / denom`

正文註解稱它是「symmetric relative error」，但分母固定至少為 1。當兩個梯度絕對值都小於 1 時，`rel_err` 就等於絕對誤差，不是通常所稱的對稱相對誤差。

這種縮放指標本身可以使用，但應正確命名為 scaled error，並同時以最大絕對誤差與縮放誤差判定。若要真正的對稱相對誤差，可使用：

$$
\frac{|g_a-g_n|}
{\max(\tau,|g_a|+|g_n|)}.
$$

其中 $\tau$ 必須明確給定。

---

## 七、測試仍只有文字，未構成自足故障測試

主程式實際只執行正常梯度檢查與方向導數檢查。下列內容沒有程式斷言：

- `(B,)` 標籤被拒絕；
- 空 batch 被拒絕；
- 非二元標籤被拒絕；
- NaN／無窮被拒絕；
- 極端 logits 的 loss 與梯度有限；
- pack/unpack 往返保持值與 shape；
- 錯誤展平向量長度被拒絕；
- $B=1$ 保留 batch 與輸出軸；
- 漏掉 tanh 導數確實被梯度檢查抓出。

**逐字原句：**

> 「例如漏掉 `1 - H**2`」

這只是要求讀者手動破壞正式程式，不是可重現故障測試。另稱：

> 「`max_rel_err` 將遠大於 $10^{-5}$」

也沒有執行或數學保證；若 $H$ 接近零，錯誤可能不明顯。

**最小修法：**

提供獨立故障函數或測試用 subclass，使用固定資料使 $|H|$ 明顯偏離零，再斷言錯誤超過容差。無執行紀錄時只能寫「預期失敗」。

---

## 八、小資料過擬合驗收仍缺失

**逐字程式：**

> `B_train, B_test = 100, 20`

> `for i in range(100):`

> `if final_train_loss > initial_train_loss * 0.1: print("Warning...")`

100 筆資料的一般訓練不等於 tiny-set overfit check。章綱要求的是以極少量資料檢查模型、loss、梯度及更新是否能共同工作。現稿也只印 warning，不會使驗收失敗。

**最小修法：**

另建獨立模型，取例如 4 至 10 筆固定合成樣本，執行訓練後以斷言檢查訓練 loss 或 accuracy 的預定門檻。門檻需標為預期，不能宣稱已通過。

---

## 九、validation 與 test 的角色仍矛盾

**逐字原句：**

> 「嚴格執行訓練集、驗證集與保留測試集的切分」

但程式只有 train/test，沒有 validation。學習率、100 次迭代、隱藏維度及警告門檻若根據 test 結果調整，就會使 test 參與調參。

**逐字原句：**

> 「保留測試集以檢測過擬合」

較正確的契約是：

- train：參數更新；
- validation：診斷過擬合及選超參數；
- test：設定全部鎖定後只評估一次。

**最小修法：**

補 validation set，或刪除三分割宣稱並聲明所有設定事先固定、test 結果不造成任何修改。使用不同 seed 也不能嚴格「確保」兩個隨機樣本集合沒有相同值，只能建立獨立隨機流；在連續分布下精確重複通常極不可能，但不是由 seed 不同作邏輯保證。

---

## 十、習題解答仍不可直接執行

**逐字原句：**

> `self.H = np.maximum(0, Z1)`

> `dZ1 = dH * (Z1 > 0)`

類別中只有 `self.Z1`，沒有區域變數 `Z1`，照解答修改會發生 `NameError`。

**最小修法：**

```python
self.H = np.maximum(0.0, self.Z1)
dZ1 = dH * (self.Z1 > 0.0)
```

並明定 ReLU 在零點的導數約定，且梯度檢查資料須避免擾動跨過零點。

---

## 十一、敘述、來源及能力邊界

**逐字原句：**

> 「框架不會報錯，但模型將永遠無法收斂，或者收斂到局部極小值。」

這是過度概括。自動微分框架通常會依前向圖產生梯度，非法 shape 常會報錯；真正可能靜默存在的是合法但語義錯誤的廣播、reduction、標籤或自訂 backward。「永遠無法收斂」也沒有依據，錯誤梯度可能產生多種行為。

最小修法是縮小陳述，不把 NumPy 教學動機建立在不準確的框架能力描述上。

參考來源中的 NumPy與 PyTorch 條目明標未逐條核對，不能用作已驗證依據。「PyTorch Documentation on Numerical Stability」亦無精確標題、版本與網址，且本章不使用 PyTorch，應刪除或改為可定位來源。

---

## 十二、硬性篇幅要求仍未達成

提供的檢查結果仍是：

> `measured_characters: 2125`

並明示：

> 「正文不足，需补足推導及實作說明」

最低要求為 3000 中文字，且公式、程式、英文與來源不充字數。現稿仍低於硬性下限。應以有效內容補足，例如：

1. 正確的方向函數 Taylor 證明；
2. 逐項差分與方向導數各自的證據範圍；
3. pack/unpack 的 offset 與互逆條件；
4. stable sigmoid 正負分支；
5. tiny-set overfit 的目的與失敗診斷；
6. train／validation／test 的資料使用契約；
7. 可執行的正常、邊界與故障斷言。

---

## 核准前最小修訂清單

1. 以分支式 stable sigmoid 取代 logit clip。
2. 用 $\phi(t)=L(\theta+tv)$ 重寫命題及證明。
3. 修正 softplus 的負大輸入漸近註解。
4. 驗證 pack/unpack 的長度、shape 與有限值。
5. 讓方向導數方法自行建立解析梯度並恢復 cache。
6. 對所有參數擾動使用 `try/finally`。
7. 拒絕空 batch、錯 feature 維度、非法建構維度及非法學習率。
8. 正確命名並定義梯度誤差指標。
9. 把邊界與故障案例寫成可執行斷言。
10. 增加獨立 tiny-set overfit check。
11. 補 validation，或刪除不實的三分割宣稱。
12. 修正 ReLU 解答中的未定義變數。
13. 精確化來源及框架能力敘述。
14. 將有效中文正文補至至少 3000 字。

VERDICT: REVISE