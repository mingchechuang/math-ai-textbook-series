<<<PATCH 15>>>
<<<OLD>>>
    allowed = np.asarray(allowed, dtype=bool)
<<<NEW>>>
    allowed = np.asarray(allowed)
    if allowed.dtype != np.bool_:
        raise TypeError("allowed 必須是布林；True=允許")
<<<END>>>
<<<PATCH 16>>>
<<<OLD>>>
**先備知識**：前一章的縮放點積注意力、固定張量軸約定 `B/T/D/H/dh`、穩定 softmax 的減最大值技巧、布林遮罩 True=允許的約定、矩陣微分的基本鏈式法則。
<<<NEW>>>
**先備知識**：第14章的縮放點積注意力與反向傳播、第15章的布林遮罩及全遮罩拒絕策略、固定張量軸約定 `B/T/D/H/dh`、穩定 softmax 的減最大值技巧，以及矩陣微分的基本鏈式法則。
<<<END>>>
<<<PATCH 16>>>
<<<OLD>>>
- **參數數量**：多頭用 `4(D^2 + D)` 個參數（含偏置），與單頭在相同維度下相同。差異是 `W_Q` 等被隱式等分為 `H` 個 `D×dh` 塊，每塊供一個頭使用。
<<<NEW>>>
- **參數數量**：一般情況含偏置共 `3D^2 + 3D + D·D_out + D_out` 個參數；當 `D_out = D` 時，化為 `4(D^2 + D)`。在總投影寬度固定為 `D` 時，改變頭數 `H` 本身不改變參數總數。
<<<END>>>
<<<PATCH 16>>>
<<<OLD>>>
            m = mask
            if m.dtype != np.bool_:
                raise ValueError("mask 必須為布林；True=允許")
            if m.ndim == 2:
                m = m[None, None]
            elif m.ndim == 3:
                m = m[:, None]
            elif m.ndim != 4:
                raise ValueError(f"mask 維度應為 2/3/4，得到 {m.ndim}")
            if m.shape[-2:] != (T, T):
                raise ValueError(f"mask 最後兩維應為 ({T},{T})，得到 {m.shape[-2:]}")
            if np.any(np.all(~m, axis=-1)):
                raise ValueError("存在全遮罩 query 列；拒絕輸出 NaN")
            S = np.where(m, S, -np.inf)
<<<NEW>>>
            if not isinstance(mask, np.ndarray):
                raise TypeError("mask 必須是 NumPy 陣列")
            m = mask
            if m.dtype != np.bool_:
                raise TypeError("mask 必須為布林；True=允許")
            if m.ndim == 2:
                m = m[None, None]
            elif m.ndim == 3:
                m = m[:, None]
            elif m.ndim != 4:
                raise ValueError(f"mask 維度應為 2/3/4，得到 {m.ndim}")
            if m.shape[-2:] != (T, T):
                raise ValueError(f"mask 最後兩維應為 ({T},{T})，得到 {m.shape[-2:]}")
            if m.shape[0] not in (1, B) or m.shape[1] not in (1, self.H):
                raise ValueError("mask 的 batch/head 軸不可廣播至 (B,H)")
            m = np.broadcast_to(m, S.shape)
            if np.any(np.all(~m, axis=-1)):
                raise ValueError("存在全遮罩 query 列；拒絕輸出 NaN")
            S = np.where(m, S, -np.inf)
<<<END>>>
<<<PATCH 16>>>
<<<OLD>>>
**來源說明**：2026-10-06 取得 N1、N2 摘要頁，未完整閱讀論文；N3 已取得明確 2.14 API 全文並核對「mask True=參與、evaluation 須 dropout_p=0、矩形因果遮罩左上對齊」。N4–N6 目前僅為待核對延伸入口，**不可宣稱已查證其內容**。本章未執行任何外部程式、未下載模型或語料、未安裝任何套件；所有數值皆為手算或預期。
<<<NEW>>>
**來源說明**：N1–N3 為題目提供的來源入口；使用時仍須依實際版本核對相關內容。來源連結不是本機版本、查閱紀錄或程式執行證據。N4–N6 為待核對延伸入口。本章未執行任何外部程式、未下載模型或語料、未安裝任何套件；所有數值皆為手算或預期。
<<<END>>>