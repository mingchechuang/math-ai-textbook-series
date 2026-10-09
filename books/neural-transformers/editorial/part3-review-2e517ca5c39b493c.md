## 審稿結論

本輪依現稿重新核算，未執行任何程式或外部工具。上一輪第16章的主要阻擋問題已大致修復：

- 已明列第14、15章依賴。
- 已補完整多頭反向。
- $dX$ 已正確累加 Q、K、V 三路。
- 已補全部權重、偏置與輸入的中央差分設計。
- mask 的 batch/head 廣播軸已檢查。
- `dv` 已統一為每頭 value 寬度。
- PyTorch 權重轉置與 `.double().cpu()` 已補。
- 無紀錄支持的日期及「已核對」措辭已移除。

重算第16章反向可得：

$$
dO=dYW_O^{\mathsf T},
$$

$$
dA=dO_hV_h^{\mathsf T},\qquad
dV_h=A^{\mathsf T}dO_h,
$$

$$
dS=A\odot\left(dA-\sum_kA\odot dA\right),
$$

$$
dQ_h=\frac{dSK_h}{\sqrt{d_h}},
\qquad
dK_h=\frac{dS^{\mathsf T}Q_h}{\sqrt{d_h}},
$$

以及：

$$
dX=dQW_Q^{\mathsf T}+dKW_K^{\mathsf T}+dVW_V^{\mathsf T}.
$$

現稿程式與上述公式一致，$d_v\ne d_h$ 時各 shape 也能閉合。然而仍有數項可造成介面錯誤或驗收不足的問題，尚不宜批准。

---

# 必須修訂

## 1. 第15章位置索引會靜默把浮點數截斷成整數

### 原句

```python
q_pos = np.asarray(q_pos, dtype=np.int64)
k_pos = np.asarray(k_pos, dtype=np.int64)
```

### 原因

這會讓非法位置靜默改變，例如：

```python
q_pos = [1.9]
k_pos = [0.2, 2.8]
```

會被轉成 `[1]` 與 `[0,2]`。因果判斷遂由原輸入的比較變成截斷後的比較。這不是單純 dtype 便利，而會改變哪些 key 被允許讀取。

第17章的 RoPE 位置介面已嚴格拒絕非整數位置：

```python
if not np.issubdtype(positions.dtype, np.integer):
    raise TypeError(...)
```

因此第15章 causal mask 與第17章 RoPE 對「絕對位置」有跨章契約不一致：同一組浮點位置可被第15章接受並截斷，卻被第17章拒絕。cache 中 mask 與 RoPE 可能因而使用不同位置語義。

非有限浮點值轉成 `int64` 還可能產生警告或平台相關的極端整數，而非章內定義的明確錯誤。

### 最小修法

先保留 dtype，再嚴格檢查：

```python
q_pos = np.asarray(q_pos)
k_pos = np.asarray(k_pos)

for name, pos in (("q_pos", q_pos), ("k_pos", k_pos)):
    if pos.ndim != 1:
        raise ValueError(f"{name} 必須是一維陣列")
    if not np.issubdtype(pos.dtype, np.integer):
        raise TypeError(f"{name} 必須是整數位置")
    if np.issubdtype(pos.dtype, np.bool_):
        raise TypeError(f"{name} 不可使用布林位置")
    if np.any(pos < 0):
        raise ValueError(f"{name} 不得含負位置")
```

然後才視需要：

```python
q_pos = q_pos.astype(np.int64, copy=False)
k_pos = k_pos.astype(np.int64, copy=False)
```

需增加浮點、布林、負數位置故障測試。若有意允許負的相對位置，則應另立介面；目前章稿定義的是非負絕對位置。

---

## 2. 第15章 `causal` 沒有嚴格布林契約

### 原句

```python
def make_allowed(q_pos, k_pos, key_is_valid, causal=True):
...
if causal:
    allowed &= ...
```

### 原因

`causal=1`、`causal="False"`、非空 list 等都可能進入 truth-value 判定；字串 `"False"` 甚至會被視為真。這使呼叫端的設定錯誤可能靜默改變資料是否能讀取未來位置。

對防止未來資訊洩漏的開關，不應依賴 Python 一般 truthiness。

### 最小修法

```python
if not isinstance(causal, (bool, np.bool_)):
    raise TypeError("causal 必須是布林值")
```

增加 `causal=1` 與 `causal="False"` 的故障測試。

---

## 3. 第16章測試沒有直接驗收「禁止位置的權重與分數梯度為零」

### 原句

> 「以部分硬遮罩再測一次」

以及：

```python
for dv, mask in ((None,None),(None,causal),(3,causal)):
    ...
    np.testing.assert_allclose(numerical,analytical,...)
```

### 原因

這已能驗證固定遮罩下輸入及參數的整體梯度，但尚未直接驗收本章文字聲明：

> 「固定硬遮罩位置的$A=0$，故$dS$也為零。」

目前 `backward` 不回傳或暫存 `dS`，測試也沒有直接斷言：

```python
A[~broadcast_mask] == 0
dS[~broadcast_mask] == 0
```

整體參數有限差分未必能隔離禁止 score 的梯度：每個 Q/K 參數通常同時影響允許與禁止位置。如果錯誤實作讓禁止位置產生小梯度，而允許位置梯度同時變化，定位會困難。本卷驗收特別要求多頭 attention 的 shape／mask／gradient，因此 mask-gradient 接口應有直接故障測試。

### 最小修法

不必更改公開 backward 回傳值，可在教學 cache 或私有除錯欄位保存最近的 `dS`：

```python
self._last_dS = dS.copy()
```

遮罩 forward 時亦保存廣播後的 mask。測試加入：

```python
allowed4 = np.broadcast_to(causal[None, None], A.shape)
assert np.all(A[~allowed4] == 0.0)

att.backward(upstream)
assert np.all(att._last_dS[~allowed4] == 0.0)
np.testing.assert_allclose(
    att._last_dS.sum(axis=-1), 0.0, atol=...
)
```

若不希望保存除錯狀態，可把 SDPA backward 抽成內部函式，在測試中直接取得 `dS`。

---

## 4. 第16章小結只列預設 $d_v=d_h$，與章內一般介面不完整一致

### 原句

> 「形狀契約：`Q_h,K_h,V_h ∈ (B,H,T,dh)`，`S,A ∈ (B,H,T,T)`，`O_h ∈ (B,H,T,dh)`，合併後 `(B,T,D)`。」

### 原因

本章程式已正式支援每頭 value 寬度 `dv != dh`，而非只把它留作未實作習題。一般契約應為：

$$
Q_h,K_h\in\mathbb R^{B\times H\times T\times d_h},
$$

$$
V_h,O_h\in\mathbb R^{B\times H\times T\times d_v},
$$

$$
O\in\mathbb R^{B\times T\times Hd_v}.
$$

目前小結重新把 $V_h,O_h$ 寫死成 $d_h$，容易讓讀者誤以為 `dv` 推廣沒有成為本章正式介面。

### 最小修法

將小結改成一般形式，括註「預設 $d_v=d_h$ 時才有 $Hd_v=D$」。

---

## 5. 第16章邊界測試對 `W_O` 的敘述只適用於預設 `dv`

### 原句

> 「`D_out ≠ D`：若在 `__init__` 指定 `D_out`，`Y.shape == (B,T,D_out)`，`W_O.shape == (D, D_out)`。」

### 原因

程式的一般 shape 是：

```python
self.W_O.shape == (H * self.dv, self.D_out)
```

只有預設 `dv=dh=D/H` 時第一軸才等於 $D$。同一章已把 `dv` 列為正式建構參數，因此這段需標出條件，否則與程式契約矛盾。

### 最小修法

改為：

> 預設 `dv=dh` 時 `W_O.shape==(D,D_out)`；一般情形為 `(H*dv,D_out)`。

---

# 建議同步修訂

## 6. 第16章類別 docstring 仍把 V 與 O 寫死為 D

### 原句

```python
Q,K,V:(B, T, D) 投影後 -> 拆分 (B,H,T,dh)
...
O_h:  (B, H, T, dh)
O:    (B, T, D)
```

### 原因

這與實際 `dv` 介面不一致。Q、K 的總寬為 $D$，但 V 的總寬是 $Hd_v$。

### 最小修法

改為：

```python
Q,K: (B,T,D) -> (B,H,T,dh)
V:   (B,T,H*dv) -> (B,H,T,dv)
Oh:  (B,H,T,dv)
O:   (B,T,H*dv)
```

---

## 7. 第14章 backward 未檢查 `dY` 型別及有限性

### 原句

```python
if dY.shape != expected_dy_shape:
    raise ValueError("dY shape mismatch")
```

### 原因

若傳入 list，會產生 `AttributeError`；若傳入含 NaN/inf 的上游梯度，三路梯度會靜默變成非有限值。第16章已對 `dY` 做嚴格檢查，第14章作為其數學依賴卻較弱，形成跨章接口不一致。

### 最小修法

```python
if not isinstance(dY, np.ndarray):
    raise TypeError("dY must be a NumPy array")
if dY.shape != expected_dy_shape:
    raise ValueError("dY shape mismatch")
if not np.all(np.isfinite(dY)):
    raise ValueError("dY must be finite")
```

加入 list 與 NaN 上游故障測試。

---

## 8. 第16章 `split_heads`／`merge_heads` 的 `width` 仍可接受非法型別

### 原句

```python
width = self.dh if width is None else width
...
X.shape[-1] != self.H*width
```

### 原因

目前內部只傳入已驗證的 `self.dh` 或 `self.dv`，所以正常路徑安全；但它們是公開方法，`width=True`、`width=1.5` 或負數可能產生混亂的 shape 比較或 reshape 錯誤。

### 最小修法

增加與 `dv` 相同的非 bool 正整數檢查，或把這兩個方法改成私有方法並聲明只接受已驗證的內部 width。

---

# 已核對為正確的部分

1. 第13章 embedding scatter-add 與 tied weight 梯度兩路相加正確。
2. 第14章 SDPA 的 $dQ,dK,dV$ 及 softmax VJP 正確。
3. 第15章 `True=允許`、softmax 前遮罩、全遮罩拒絕、有效 token 平均均正確；上一輪非布林 mask 靜默轉換問題已修復。
4. 第16章參數數量一般式正確：
   $$
   2D^2+2D+(D+1)Hd_v+(Hd_v+1)D_{\text{out}}.
   $$
5. 第16章完整反向的 shape 與 reduction 軸正確；偏置均沿 $(B,T)$ 求和，$dX$ 三路正確累加。
6. 第16章 PyTorch 轉接中的四個權重轉置正確，布林 `attn_mask` 與本卷 allowed mask 的反相亦已明示。
7. 第17章 RoPE 旋轉符號、相對位置內積、共同位置平移與 cache 起點均正確；來源聲明已不再虛構查閱紀錄。
8. 第18章 LN、RMSNorm、殘差與 FFN 反向公式正確。

本輪已接近批准；主要剩餘阻擋是第15章絕對位置的靜默 dtype 截斷，以及第16章尚未直接驗收硬遮罩位置的 $dS=0$。兩者都可局部修正，無須重寫章稿。

VERDICT: REVISE