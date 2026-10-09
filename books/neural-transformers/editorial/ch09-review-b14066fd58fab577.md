## 獨立審稿結論

本輪提交的章稿與上一輪所示內容相同，前次指出的主要阻擋項並未落實到正文或程式。現稿雖已修復最初的標籤廣播錯誤，並具備兩層網路、完整訓練 loop、參數展平與方向導數的基本骨架，但數學證明仍有 shape 錯誤，loss 與 backward 仍未嚴格對應同一函數，且小資料過擬合、驗證集及可執行故障測試仍未完成。依本卷契約不能核准。

以下均為靜態重算；我未執行程式，因此不確認 `PASS`、損失下降、準確率或任何門檻是否實際達成。

---

## 一、重算後確認正確的部分

### 1. 標籤與輸出的 shape 現已一致

程式使用：

> `Y_train = (X_train[:, [0]] > 0).astype(np.float64)`

雙中括號保留第二維，因此：

- `Y_train.shape == (B_train, 1)`
- `Y_pred.shape == (B_train, 1)`
- `dZ2.shape == (B_train, 1)`

不再發生舊稿的 $(B,1)-(B,)\to(B,B)$ 錯誤廣播。`_check_shapes` 拒絕 `(B,)` 標籤也是必要修正。

### 2. 主要反向公式正確

在標籤為 $(B,1)$ 且 BCE 沿 batch 平均的前提下：

$$
dZ_2=\frac{\sigma(Z_2)-Y_{\mathrm{true}}}{B},
$$

$$
dW_2=H^TdZ_2,\qquad db_2=\sum_{i=1}^B dZ_{2,i},
$$

$$
dH=dZ_2W_2^T,
$$

$$
dZ_1=dH\odot(1-H^2),
$$

$$
dW_1=X^TdZ_1,\qquad db_1=\sum_{i=1}^B dZ_{1,i}.
$$

程式中的矩陣乘法與 reduction 軸符合這組公式，梯度 shape 亦應分別對應參數 shape。

### 3. BCE logits 公式正確

正文與程式採用：

$$
L_i=\operatorname{logaddexp}(0,z_i)-y_i z_i.
$$

這是穩定的 logits BCE。當 $z=100,y=0$ 時損失約為 100；當 $z=-100,y=1$ 時損失也約為 100。邊界測試中的這兩項預期方向正確。

### 4. 手算主要數值合理

依例題數值：

$$
Z_1=[0.6,0.1],
$$

$$
H\approx[0.53705,0.09967],
$$

$$
Z_2\approx0.19488,
$$

$$
\sigma(Z_2)\approx0.54857.
$$

故 $y=1$ 時：

$$
dZ_2\approx-0.45143,
$$

$$
dW_2\approx[-0.24245,-0.04499]^T.
$$

稿中結果只受四位小數的中間捨入影響，沒有實質錯誤。

---

## 二、數學證明仍不成立

### 1. 命題前提無法推出結論

**逐字原句：**

> 「若前向傳播計算正確，且使用中心差分（Central Difference）進行數值梯度檢查，則解析梯度與數值梯度應在浮點精度與步長誤差範圍內一致。」

**原因：**

前向正確不表示 backward 正確。若反向漏掉 $1-H^2$，前向與 loss 都可完全正確，但解析梯度仍會錯。中心差分只是拿來檢查這個差異，不能由「用了中心差分」推出「兩者應一致」。

「浮點精度與步長誤差範圍」也沒有定義具體範數、絕對／相對尺度或常數，不能作為嚴格命題。

**最小修法：**

將命題縮小為中心差分本身的截斷誤差命題，不宣稱前向正確可保證反向正確。例如固定方向 $v$，定義：

$$
\phi(t)=L(\theta+tv).
$$

若 $\phi$ 在零附近三階可微且三階導數有界，則：

$$
\frac{\phi(h)-\phi(-h)}{2h}
=\phi'(0)+O(h^2),
$$

而

$$
\phi'(0)=\nabla L(\theta)^Tv.
$$

之後另行說明：數值核對通過只是支持 backward 的局部證據，不是完整正確性證明。

---

### 2. Taylor 展開有純量／向量 shape 錯誤

**逐字原句：**

> $$L(\theta+h)=L(\theta)+h\nabla L(\theta)+\frac{h^2}{2}\nabla^2L(\theta)h+O(h^3)$$

**原因：**

正文已將 $\theta$ 定義為參數向量，而 $h$ 是純量。此式中：

- $L(\theta)$ 是純量；
- $h\nabla L(\theta)$ 是向量；
- 純量不能與向量相加；
- $L(\theta+h)$ 也未說明純量 $h$ 加到哪個座標；
- Hessian 項應是方向二次型，例如 $h^2v^T\nabla^2L(\theta)v/2$，而不是目前的寫法。

因此這不是格式小問題，而是證明中的 shape 不合法。

**最小修法：**

使用純量函數 $\phi(t)=L(\theta+tv)$ 展開：

$$
\phi(h)=\phi(0)+h\phi'(0)+\frac{h^2}{2}\phi''(0)
+\frac{h^3}{6}\phi'''(\xi_+),
$$

$$
\phi(-h)=\phi(0)-h\phi'(0)+\frac{h^2}{2}\phi''(0)
-\frac{h^3}{6}\phi'''(\xi_-).
$$

兩式相減後才可得中心差分的 $O(h^2)$ 截斷誤差。這也恰好能與方向導數程式銜接。

---

### 3. 「最佳步長」仍過度宣稱

**逐字原句：**

> 「故最佳步長 $h\approx10^{-5}$ 至 $10^{-6}$。」

**原因：**

由

$$
E(h)\approx C_1h^2+\frac{C_2\epsilon_{\mathrm{mach}}}{h}
$$

求得的是：

$$
h_\star=
\left(\frac{C_2\epsilon_{\mathrm{mach}}}{2C_1}\right)^{1/3}.
$$

步長仍依賴 $C_1,C_2$、參數尺度、loss 尺度及局部三階導數。$10^{-5}$ 至 $10^{-6}$ 只能作為尺度約為一時的候選範圍，不能稱為普遍最佳。

**最小修法：**

將「最佳」改為「常見候選量級」，並建議以多個 $h$ 比較誤差趨勢。

---

## 三、loss 與 backward 仍不是同一函數的精確導數

**逐字原句／程式：**

> `Z2_safe = np.clip(self.Z2, -500, 500)`

> `self.Y_pred = 1.0 / (1.0 + np.exp(-Z2_safe))`

> `loss_per_sample = np.logaddexp(0.0, self.Z2) - self.Y_true * self.Z2`

> `dZ2 = (self.Y_pred - self.Y_true) / B`

**原因：**

loss 對未裁切的 $Z_2$ 計算，因此其導數是：

$$
\frac{\partial L}{\partial Z_2}
=\frac{\sigma(Z_2)-Y_{\mathrm{true}}}{B}.
$$

但程式使用的是：

$$
\frac{\sigma(\operatorname{clip}(Z_2,-500,500))-Y_{\mathrm{true}}}{B}.
$$

當 $|Z_2|>500$ 時兩者不是同一函數。即使差異在飽和區非常小，本章主題是「逐項梯度驗證」，不能容許 loss 與 backward 在定義上不一致。

**最小修法：**

不用 clip，改成正負分支的穩定 sigmoid：

```python
def stable_sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out
```

然後令：

```python
self.Y_pred = stable_sigmoid(self.Z2)
```

如此 logits BCE 與 backward 才完全一致。

---

## 四、loss 註解仍含錯誤數學

**逐字原句：**

> `For large negative Z2, log(1 + exp(Z2)) ~ Z2`

**原因：**

當 $z\to-\infty$：

$$
\log(1+e^z)\sim e^z\to0,
$$

不是近似 $z$。只有當 $z\to+\infty$ 時才有：

$$
\log(1+e^z)\sim z.
$$

其後的：

> `If Y=0, loss ~ Z2 (which is negative? No.)`

正是由前述錯誤近似造成的自我矛盾。即使最後說公式正確，錯誤註解仍會誤導讀者。

**最小修法：**

刪除這段自問自答，換成正確的兩側漸近說明。

---

## 五、參數展平與方向導數介面仍有缺口

### 1. `unpack_params` 沒有檢查輸入長度

**逐字原句：**

> `self.b2 = params[-1:].reshape(self.b2.shape)`

**原因：**

程式沒有要求 `params` 是一維、長度恰等於全部參數總數，也沒有拒絕 NaN／無窮。若向量過長，部分多餘元素可能被忽略；`[-1:]` 也把實作綁死在 `b2.size == 1`。

**最小修法：**

先驗證：

```python
expected = self.W1.size + self.b1.size + self.W2.size + self.b2.size
if params.ndim != 1 or params.size != expected:
    raise ValueError(...)
if not np.all(np.isfinite(params)):
    raise ValueError(...)
```

再用累積 offset 切割四個參數。

### 2. 方向導數函數依賴外部舊 cache

**逐字原句：**

> `grad_flat = np.concatenate([self.dW1.flatten(), ...])`

**原因：**

`directional_derivative_check(X,Y_true,v)` 接收了資料，卻未先對這份資料執行 `forward` 與 `backward`。若呼叫者先前的梯度來自另一批資料，解析方向導數會靜默使用舊 cache。主程式目前恰好先執行一次，但方法本身不自足。

**最小修法：**

在方法開始時自行：

```python
self.forward(X, Y_true)
self.backward()
```

並驗證：

- `v.ndim == 1`
- `v.size == theta.size`
- `v` 全部有限
- `np.linalg.norm(v) > 0`
- `h` 與 `tol` 為有限正數

結束恢復 $\theta$ 後，再對原參數執行一次 forward/backward，使 cache 與當前參數一致。

### 3. 擾動沒有 `try/finally`

**逐字原句：**

> `param[idx] = original_value + h`

> `...`

> `param[idx] = original_value`

若中途因非有限值或 shape 問題拋出例外，參數不會恢復。方向導數的多次 `unpack_params` 也有同樣風險。

**最小修法：**

以 `try/finally` 保證還原，且在擾動前驗證 `h > 0` 且有限。

---

## 六、正常／邊界／故障測試仍未真正納入程式

章稿的「測試與預期結果」主要是文字敘述。主程式只實際呼叫正常梯度檢查與方向導數檢查；以下邊界並沒有可執行測試：

- `(B,)` 標籤應拋錯；
- 空 batch 應拋錯；
- 非二元標籤應拋錯；
- NaN／無窮應拋錯；
- 極端 logits 應保持有限；
- pack/unpack 應互逆；
- 錯誤長度的展平向量應被拒絕；
- 漏掉 tanh 導數的故障版本應失敗；
- $B=1$ 時所有軸與梯度 shape 應保留。

### 空 batch 的檢查位置尤其不正確

**逐字原句／程式：**

> `if B == 0: raise ValueError("Batch size cannot be 0")`

這只放在 `backward`。空 batch 可以先通過 `_check_shapes`，進入 `loss()`，使 `np.mean` 對空陣列產生非有限結果或警告。

**最小修法：**

在 `_check_shapes` 立即拒絕 `X.shape[0] == 0`。

### 故障測試仍沒有執行實體

**逐字原句：**

> 「正確的故障測試：例如漏掉 `1 - H**2`」

這只是告訴讀者手動修改正式程式，並不是自足測試。也不能無執行證據地宣稱：

> 「`max_rel_err` 將遠大於 $10^{-5}$。」

若初始化恰使 $H$ 接近零，$1-H^2$ 接近 1，故障可能不夠明顯。

**最小修法：**

增加獨立的錯誤 backward 函數或 mutation 測試，選定使 $|H|$ 不接近零的固定小張量，再斷言誤差超過容差。未執行時只寫「預期超過容差」，不寫已通過。

---

## 七、所謂小資料過擬合檢查仍未完成

**逐字原句／程式：**

> `B_train, B_test = 100, 20`

> `for i in range(100):`

> `if final_train_loss > initial_train_loss * 0.1: print("Warning...")`

**原因：**

這是對 100 筆線性可分資料做一般全批訓練，不是章綱指定的 tiny-set overfit check。小資料過擬合是除錯驗收：選取例如 4 至 10 筆資料，確認模型是否有能力把這極小集合的 loss 壓低或準確率提升到預設門檻。

目前也只是印 warning，不會使錯誤測試失敗；且固定 100 步能否降至初始 loss 的 10% 沒有執行證據。

**最小修法：**

另建獨立 `tiny_model` 與極小資料集，執行完整訓練 loop，最後以斷言檢查預先指定的 loss 或 accuracy 門檻。此檢查不可拿 test set 參與訓練或門檻調整。

---

## 八、資料切分與評估契約仍自相矛盾

**逐字原句：**

> 「嚴格執行訓練集、驗證集與保留測試集的切分」

**實際程式：**

只有 train 與 test，沒有 validation。

**原因：**

學習率、迭代次數、隱藏維度及門檻若需要選擇，應由 validation 決定。test 應在設定鎖定後只評估一次。程式卻直接根據：

> `if test_acc < 0.5: print("Warning...")`

評判模型品質。若作者據此回頭改設定，就形成 test 調參。

**最小修法：**

補出 validation set，明確規定：

- train：更新參數；
- validation：選學習率、迭代數或其他設定；
- test：設定鎖定後只評估一次。

若不想增加 validation，就必須刪除「三分割」宣稱，並說明所有設定事先固定、test 結果不造成任何修改。

本章小結中的：

> 「保留測試集以檢測過擬合」

也應改為「以 validation 診斷／選擇，以保留 test 作最終評估」。

---

## 九、習題解答仍有不可執行程式

**逐字原句：**

> `self.H = np.maximum(0, Z1)`

> `dZ1 = dH * (Z1 > 0)`

**原因：**

類別方法中只有 `self.Z1`，不存在區域變數 `Z1`。照此解答修改會得到 `NameError`。

**最小修法：**

改為：

```python
self.H = np.maximum(0.0, self.Z1)
```

及：

```python
dZ1 = dH * (self.Z1 > 0.0)
```

並明定 ReLU 在零點採哪個次梯度。有限差分測試還要避免擾動跨過零點，否則中心差分不對應單一可微導數。

---

## 十、敘述、能力與來源

### 1. 對自動微分框架的描述不準確

**逐字原句：**

> 「若我們在反向傳播中錯誤地處理了轉置……框架不會報錯」

標準自動微分框架通常根據前向圖自動產生 backward，使用者並不直接手寫線性層反向。shape 不合法時框架也通常會報錯。真正可能靜默錯誤的是前向語義、錯誤 reduction、合法但錯誤的廣播，或自訂 backward。

**最小修法：**

縮小為上述可成立的說法，不要籠統宣稱框架不會報錯。

### 2. 來源不可視為已核對

NumPy 與 PyTorch 條目已標「未逐條核對」，這是誠實的，但：

> 「PyTorch Documentation on Numerical Stability」

沒有可定位的標題、版本或網址，且本章使用 NumPy。應刪除或改成精確來源。Goodfellow 第六章是否直接支持本章的中心差分誤差證明，也不能在未核對時暗示已驗證。

本章沒有執行紀錄，因此正文中的 `PASS`、loss 降低及準確率都只能是程式預期輸出，不能當作已完成實驗。

---

## 十一、正文篇幅仍不合格

提供的檢查資料仍為：

> `measured_characters: 2125`

並明示：

> 「正文不足，需补足推導及實作說明」

本卷最低要求是 3000 中文字，且公式、程式、英文與參考來源不計數。現稿未達硬性下限。

最適合補足的實質內容是：

1. 用方向函數完整重寫中心差分證明；
2. 解釋逐項梯度與方向導數的證據範圍；
3. 列出 pack/unpack 的 offset、shape 與互逆條件；
4. 說明正負分支 stable sigmoid；
5. 詳述 train／validation／test 的禁止事項；
6. 補 tiny-set overfit 的目的與失敗診斷；
7. 將正常、邊界、故障案例寫成可執行斷言。

---

## 十二、核准前最小修訂清單

1. 用未裁切的分支式 stable sigmoid，使 loss 與 backward 完全一致。
2. 以 $\phi(t)=L(\theta+tv)$ 重寫命題與完整證明。
3. 將「最佳步長」降格為具條件的候選量級。
4. 刪除負大 logits 時 softplus 近似為 $z$ 的錯誤註解。
5. 驗證 `unpack_params` 的 ndim、長度及有限性。
6. 讓方向導數方法自行建立解析梯度並恢復 cache。
7. 對數值擾動使用 `try/finally`。
8. 在 forward 前拒絕空 batch，並檢查參數、方向與學習率。
9. 把邊界與故障案例加入自足程式的斷言。
10. 實作獨立 tiny-set overfit check。
11. 補 validation，或刪除三分割與驗證集宣稱。
12. 修正 ReLU 解答中的未定義 `Z1`。
13. 精確化來源，且不得宣稱未執行測試已通過。
14. 將有效中文正文補至至少 3000 字。

VERDICT: REVISE