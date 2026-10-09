<<<PATCH 01>>>
<<<OLD>>>
**檢查**：兩個頭對同一 token 的輸出不同，說明這個例子中兩頭學到了不同的「模式」。若把 `X` 改成兩頭得到相同投影的情形（例如 `X = [[1,0,1,0],[0,1,0,1]]`），兩頭的輸出一模一樣——這正是「頭塌縮」的小型演示。
<<<NEW>>>
**檢查**：兩個頭對同一 token 的輸出不同，只說明這組手工指定的輸入與恆等投影產生了不同結果；此例沒有訓練，不能說兩頭「學到了」不同模式。若改用 `X = [[1,0,1,0],[0,1,0,1]]`，兩頭取得相同輸入與投影，因而有相同輸出；這只能演示相同頭的可能性，不能證明訓練後發生了頭塌縮。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
4. **非布林遮罩**：傳入 `float32` mask → 拋 `ValueError`；這避免用 0/1 混過布林約定。
<<<NEW>>>
4. **非布林遮罩**：傳入 `float32` mask → 拋 `TypeError`；這避免用 0/1 混過布林約定。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
**16.4** 讀者應執行：

```python
import torch, numpy as np
torch.manual_seed(0)
B, T, D, H = 2, 3, 8, 2
m = torch.nn.MultiheadAttention(D, H, batch_first=True, bias=True, dropout=0.0)
m.eval()
X = torch.randn(B, T, D, dtype=torch.float64)
W = m.in_proj_weight.detach().numpy()  # (3D, D)，切分為 Q,K,V 三塊
```

`in_proj_weight` 的切分順序是 PyTorch 的實作約定，須在該版本對照；不要假設與本章一致。切分後複製到 `MultiHeadAttention` 的四個矩陣與 `out_proj`，在相同 `X`（float64）下比較。
<<<NEW>>>
**16.4** 讀者可在已安裝 PyTorch 與 NumPy 的 CPU 環境執行以下對照；寫作流程未執行：

```python
import torch
import numpy as np

torch.manual_seed(0)
B, T, D, H = 2, 3, 8, 2
m = torch.nn.MultiheadAttention(
    D, H, batch_first=True, bias=True, dropout=0.0
).double().eval()
X = torch.randn(B, T, D, dtype=torch.float64)
with torch.no_grad():
    y_torch, _ = m(X, X, X, need_weights=False)

att = MultiHeadAttention(D, H)
wq, wk, wv = m.in_proj_weight.detach().numpy().reshape(3, D, D)
bq, bk, bv = m.in_proj_bias.detach().numpy().reshape(3, D)
# PyTorch 線性層用 x @ weight.T；本章用 x @ W。
att.W_Q, att.W_K, att.W_V = wq.T.copy(), wk.T.copy(), wv.T.copy()
att.b_Q, att.b_K, att.b_V = bq.copy(), bk.copy(), bv.copy()
att.W_O = m.out_proj.weight.detach().numpy().T.copy()
att.b_O = m.out_proj.bias.detach().numpy().copy()
y_numpy, _ = att.forward(X.detach().numpy())
error = np.max(np.abs(y_numpy - y_torch.detach().numpy()))
print(error)
```

這裡依本題所用模組的 Q、K、V 拼接順序切分權重；若更換框架版本或配置，須先核對其參數布局。`.double()` 使模型參數與輸入同為 float64，轉置則對接兩者的矩陣乘法約定。
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
3. **錯誤順序**：故意先 `transpose` 後 `reshape`，形狀可能仍正確，但 `merge(split(X)) != X` 會被命題 16.1 的數值檢驗抓到。
<<<NEW>>>
3. **錯誤順序**：用習題 16.3 的錯誤拆分，再以本章正確的 `merge_heads` 合併，結果不等於原 `X`。若錯誤拆分另配一個專門設計的逆操作，單測往返也可能通過；因此還要逐索引核對 `split(X)[b,h,t,i] == X[b,t,h*dh+i]`。
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
- **縮放因子用 `√dh` 而非 `√D`：每個頭的內積來自 `dh` 維，若各維獨立同分布、均值 0、變異數 1，則內積方差維 `dh`，除以 `√dh` 後方差歸一。
<<<NEW>>>
- 縮放因子用 `√dh` 而非 `√D`：每個頭的內積來自 `dh` 維。若每一對 query、key 分量彼此獨立，各分量均值為 0、變異數為 1，且各維乘積互不相關，則內積方差為 `dh`；除以 `√dh` 後方差為 1。這是尺度估計的假設，不保證訓練中的投影符合它。
<<<END>>>