## 審稿結論

本輪重新由公式、索引與程式控制流程核算，未執行任何程式、未使用外部工具，也不把章內 `assert`、輸出文字或固定 seed 當成實際通過紀錄。前輪阻擋批准的問題已完成實質修正；目前剩餘事項屬介面防禦與測試覆蓋可再加強，不影響本部核心數學、跨章契約或多頭 attention 驗收，因此本輪可批准。

---

# 一、跨章座標與依賴

第13至18章的推進關係已閉合：

1. 第13章建立 embedding、位置加法廣播與共享權重梯度。
2. 第14章建立單頭 SDPA、softmax VJP 與 $Q,K,V$ 梯度。
3. 第15章建立 padding、causal、loss mask 及絕對位置契約。
4. 第16章明確依賴第14章反向及第15章遮罩契約，並擴展成完整多頭模組。
5. 第17章在第16章的 $(B,H,T,d_h)$ 座標上加入 RoPE，並沿用第15章的絕對位置與 cache 規則。
6. 第18章處理 attention 之外的 norm、殘差與 FFN 構件。

第16章原先錯寫「前一章的縮放點積注意力」，現已改成：

> 「第14章的縮放點積注意力與反向傳播、第15章的布林遮罩及全遮罩拒絕策略」

此依賴正確。

---

# 二、shape、broadcast 與 reduction 重算

## 第13章

查表：

$$
X_{b,t,d}=E_{I_{b,t},d}
$$

與 one-hot 乘法一致。梯度：

$$
\frac{\partial L}{\partial E_{v,d}}
=
\sum_{b,t}\mathbf1[I_{b,t}=v]G_{b,t,d}
$$

正確沿 $B,T$ 累加，沒有錯沿特徵軸求和。位置表 $P\in\mathbb R^{T_{\max}\times D}$ 沿 batch 軸廣播，反向：

$$
dP_{t,d}=\sum_b dZ_{b,t,d}
$$

亦正確。

共享 embedding 的兩路梯度範例重算為：

$$
L=E_0\cdot E_1,
$$

所以：

$$
\nabla_{E_0}L=E_1=(3,4),\qquad
\nabla_{E_1}L=E_0=(1,2),
$$

與正文一致。

## 第14章

形狀為：

$$
Q:(B,T_q,d_k),\quad
K:(B,T_k,d_k),\quad
V:(B,T_k,d_v),
$$

$$
S=QK^{\mathsf T}/\sqrt{d_k}:(B,T_q,T_k),
$$

softmax 沿最後 key 軸，輸出：

$$
Y=AV:(B,T_q,d_v).
$$

反向：

$$
dV=A^{\mathsf T}dY,
\qquad
dA=dYV^{\mathsf T},
$$

$$
dS=A\odot\left(dA-\sum_kA\odot dA\right),
$$

$$
dQ=\frac{dSK}{\sqrt{d_k}},
\qquad
dK=\frac{dS^{\mathsf T}Q}{\sqrt{d_k}}
$$

均正確。新增的 `dY` 型別、shape 與有限性檢查已使其和第16章接口一致。

## 第15章

`make_allowed` 現在嚴格要求：

- `q_pos`、`k_pos` 是一維非布林整數；
- 位置不得為負；
- `key_is_valid` 是 bool；
- `causal` 是 bool。

因此不再把浮點位置靜默截斷。因果規則：

$$
A_{i,j}=[k_j\le q_i]
$$

與第17章 cache 位置契約一致。

矩形例：

$$
q=[4,5],\qquad k=[0,1,2,3,4,5]
$$

確實得到：

$$
\begin{bmatrix}
1&1&1&1&1&0\\
1&1&1&1&1&1
\end{bmatrix}.
$$

loss reduction 亦正確：

$$
L=
\frac{\sum_{b,t}m_{b,t}\ell_{b,t}}
{\sum_{b,t}m_{b,t}},
$$

只除以有效 token 總數一次，零有效 token 明確拒絕。

## 第16章

一般 $d_v$ 契約現已清楚：

$$
Q_h,K_h\in\mathbb R^{B\times H\times T\times d_h},
$$

$$
V_h,O_h\in\mathbb R^{B\times H\times T\times d_v},
$$

$$
O\in\mathbb R^{B\times T\times Hd_v},
\qquad
W_O\in\mathbb R^{Hd_v\times D_{\text{out}}}.
$$

程式 docstring、邊界條件與小結已統一。預設 $d_v=d_h=D/H$ 時才有 $Hd_v=D$。

mask 可接受：

- `(T,T)`；
- `(B,T,T)`；
- `(1|B,1|H,T,T)`。

四維前導軸會先檢查，再廣播到：

$$
(B,H,T,T).
$$

全遮罩檢查作用於廣播後每個 $(b,h,q)$ 列，正確。

## 第17章

RoPE 使用相鄰偶奇配對，旋轉：

$$
(x_{2i}',x_{2i+1}')
=
(x_{2i}\cos\theta-x_{2i+1}\sin\theta,\,
x_{2i}\sin\theta+x_{2i+1}\cos\theta).
$$

其內積：

$$
(R_pq)^{\mathsf T}(R_rk)
=q^{\mathsf T}R_{r-p}k
$$

符號正確。共同把 query 與 key 位置平移同一常數時，相對差不變。cache suffix 從 cache 長度開始，而不是局部零開始，契約正確。

## 第18章

LN 沿最後 feature 軸：

$$
z=\frac{x-\mu}{\sqrt{v+\epsilon}},
$$

反向：

$$
dx=\frac1s\left(u-\operatorname{mean}(u)
-z\operatorname{mean}(uz)\right)
$$

正確，即使 $\epsilon>0$ 亦成立。

RMSNorm：

$$
r=\sqrt{\operatorname{mean}(x^2)+\epsilon},
$$

$$
dx=\frac ur-xr^{-3}\operatorname{mean}(ux)
$$

等價於正文索引公式。殘差：

$$
dx=dy+d_{\text{transform path}}x
$$

亦正確累加。

---

# 三、第16章完整梯度驗收

這是前稿最重要的缺口，現已補齊。

輸出投影：

$$
dW_O=O_{\rm flat}^{\mathsf T}dY_{\rm flat},
\qquad
db_O=\sum_{b,t}dY,
$$

$$
dO=dYW_O^{\mathsf T}.
$$

注意力內部：

$$
dA=dO_hV_h^{\mathsf T},
\qquad
dV_h=A^{\mathsf T}dO_h,
$$

$$
dS=A\odot\left(dA-\sum_kA\odot dA\right).
$$

Q/K：

$$
dQ_h=\frac{dSK_h}{\sqrt{d_h}},
\qquad
dK_h=\frac{dS^{\mathsf T}Q_h}{\sqrt{d_h}}.
$$

投影參數：

$$
dW_Q=X_{\rm flat}^{\mathsf T}dQ_{\rm flat},
$$

K、V 同理；偏置沿 $(B,T)$ 求和。輸入梯度：

$$
dX=dQW_Q^{\mathsf T}
+dKW_K^{\mathsf T}
+dVW_V^{\mathsf T}.
$$

程式的 `backward` 與上述公式一致，並保存 forward 時的權重快照，避免 forward 後參數改動使同一反向混合新舊權重。

測試使用非對稱固定上游，對以下全部對象設計中央差分：

- $X$；
- $W_Q,W_K,W_V,W_O$；
- $b_Q,b_K,b_V,b_O$；
- 預設 $d_v=d_h$；
- $d_v\ne d_h$；
- 無 mask；
- 部分 causal mask。

此外直接檢查：

$$
A_{\text{forbidden}}=0,
\qquad
dS_{\text{forbidden}}=0,
\qquad
\sum_kdS_k=0.
$$

這已滿足本卷要求的「多頭 attention 的形狀／mask／梯度」驗收。章稿沒有宣稱這些測試已實際通過，只寫預期與測試設計，符合執行聲明限制。

---

# 四、概率、loss 與遮罩

第14至16章均採 softmax 前硬遮罩，`True=允許`。全遮罩 query 明確拒絕，不讓：

$$
-\infty-(-\infty)
$$

產生 NaN 後繼續流動。

第13章 CE 由 logits 的 log-sum-exp 計算，未使用 `softmax` 後再 `log`，也未加任意 epsilon。第15、17章已區分：

- key padding mask；
- causal mask；
- padding query 輸出策略；
- target loss mask。

因此沒有把 attention mask 當成 loss mask。

---

# 五、cache、position 與資料洩漏

第15與17章均使用絕對位置，沒有把矩形 cache mask 無條件寫成 `tril(Tq,Tk)`。第17章也明確指出不同樣本間必須重設 cache。

資料評估方面，各章一致要求：

1. 先按文件、群組或時間切分；
2. 再只用訓練集建立詞表及擬合統計量；
3. 再於各集合內切窗口；
4. 重疊窗口不可跨集合；
5. 驗證集選模型；
6. 測試集只作保留評估。

perplexity 定義亦正確：

$$
\operatorname{PPL}
=
\exp\left(
\frac{\text{有效 token 總 NLL}}
{\text{有效 token 總數}}
\right).
$$

沒有平均不同 token 數批次的 perplexity。

---

# 六、來源、能力與執行聲明

本稿沒有再聲稱特定日期取得來源或已逐條核對外部文件。第16、17章均把來源連結明確描述為入口，而非查閱或執行證據。

PyTorch 轉接題已注明：

- 選用依賴；
- CPU；
- float64；
- `.eval()`；
- `dropout=0.0`；
- PyTorch `(out,in)` 與本卷 `(in,out)` 間的轉置；
- MHA bool `attn_mask` 與本卷 allowed mask 的相反語義；
- 未實際執行。

沒有虛構設備、版本驗證、誤差實測、訓練結果或收斂率。

---

# 七、非阻擋的後續強化建議

以下不影響批准，但編校時可順手補強。

## 1. 第15章新增位置故障測試

實作已拒絕浮點、布林、負位置及非布林 `causal`，但 `run_checks()` 尚未直接覆蓋這四個新分支。可增加對應測試，使故障清單與程式分支完全一致。

## 2. 第16章 width 防禦

`split_heads`、`merge_heads` 的 `width` 在內部只收到已驗證的 `dh/dv`，所以目前核心路徑安全。若保留為公開方法，可再拒絕 bool、浮點及非正 width。

## 3. 第13章 target dtype

`tied_forward_backward` 可明確要求 target 是非布林整數 NumPy 陣列，避免浮點 target 最後由 NumPy 索引層拋錯。這是錯誤訊息品質與防禦性問題，不影響正常計算及現有推導。

## 4. 第16章 `_last_dS` 的失敗狀態

現在新 forward 會清成 `None`，已避免讀到上一輪梯度。若追求更嚴格，可在全部反向有限性檢查完成後才賦值 `_last_dS`；目前有限輸入與有限上游下正常路徑不受影響。

---

# 最終判定

第13至18章的定義、shape、軸、廣播、reduction、梯度、概率、mask、cache、位置與資料評估範圍已能跨章一致。第16章原本缺失的完整多頭反向及梯度驗收已補齊，且 $d_v\ne d_h$、硬遮罩梯度、PyTorch 權重座標轉接均已閉合。剩餘項目不構成數學錯誤或核心驗收缺口。

VERDICT: APPROVE