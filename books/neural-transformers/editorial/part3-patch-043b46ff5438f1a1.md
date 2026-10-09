<<<PATCH 01>>>
<<<OLD>>>
    allowed = np.asarray(allowed, dtype=bool)
<<<NEW>>>
    allowed = np.asarray(allowed)
    if allowed.dtype != np.bool_:
        raise TypeError("allowed 必須是布林；True=允許")
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
- **參數數量**：多頭用 `4(D^2 + D)` 個參數（含偏置），與單頭在相同維度下相同。差異是 `W_Q` 等被隱式等分為 `H` 個 `D×dh` 塊，每塊供一個頭使用。
<<<NEW>>>
- **參數數量**：一般情況含偏置共 `3D^2 + 3D + D·D_out + D_out` 個參數；當 `D_out = D` 時，化為 `4(D^2 + D)`。在總投影寬度固定為 `D` 時，改變頭數 `H` 本身不改變參數總數。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
m = torch.nn.MultiheadAttention(D, H, batch_first=True, bias=True, dropout=0.0)
m.eval()
X = torch.randn(B, T, D, dtype=torch.float64)
W = m.in_proj_weight.detach().numpy()  # (3D, D)，切分為 Q,K,V 三塊
<<<NEW>>>
m = torch.nn.MultiheadAttention(D, H, batch_first=True, bias=True, dropout=0.0).double()
m.eval()
X = torch.randn(B, T, D, dtype=torch.float64)
W = m.in_proj_weight.detach().numpy()  # (3D, D)，切分為 Q,K,V 三塊
with torch.no_grad():
    y_torch, _ = m(X, X, X, need_weights=False)
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
**檢查**：兩個頭對同一 token 的輸出不同，說明這個例子中兩頭學到了不同的「模式」。若把 `X` 改成兩頭得到相同投影的情形（例如 `X = [[1,0,1,0],[0,1,0,1]]`），兩頭的輸出一模一樣——這正是「頭塌縮」的小型演示。
<<<NEW>>>
**檢查**：兩個頭對同一 token 的輸出不同，說明固定的兩個特徵子空間在此輸入上產生不同注意力結果；本例沒有訓練，因此不能說兩頭已學到語義模式。若把 `X` 改成兩頭得到相同投影的情形（例如 `X = [[1,0,1,0],[0,1,0,1]]`），兩頭的輸出一模一樣；這是兩頭數值相同的退化示例，不能據此描述訓練後的現象。
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
**來源說明**：2026-10-06 取得 N1、N2 摘要頁，未完整閱讀論文；N3 已取得明確 2.14 API 全文並核對「mask True=參與、evaluation 須 dropout_p=0、矩形因果遮罩左上對齊」。N4–N6 目前僅為待核對延伸入口，**不可宣稱已查證其內容**。本章未執行任何外部程式、未下載模型或語料、未安裝任何套件；所有數值皆為手算或預期。
<<<NEW>>>
**來源說明**：N1–N3 為題目提供的來源入口；本章依所列介面契約說明相關注意事項，但不把來源連結當成本機版本、查閱紀錄或執行證據。N4–N6 為待核對延伸入口。本章未執行任何外部程式、未下載模型或語料、未安裝任何套件；所有數值皆為手算或預期。
<<<END>>>