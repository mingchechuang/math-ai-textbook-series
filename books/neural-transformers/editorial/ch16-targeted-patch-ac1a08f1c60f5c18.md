<<<PATCH 01>>>
<<<OLD>>>
3. **遮罩形狀錯**：`mask.shape=(T,)` 或 `(B,T,T+1)` → 拋 `ValueError`。
4. **非布林遮罩**：傳入 `float32` mask → 拋 `ValueError`；這避免用 0/1 混過布林約定。
5. **錯誤順序**：故意先 `transpose` 後 `reshape`，形狀可能仍正確，但 `merge(split(X)) != X` 會被命題 16.1 的數值檢驗抓到。
<<<NEW>>>
3. **遮罩形狀錯**：`mask.shape=(T,)` 或 `(B,T,T+1)` → 拋 `ValueError`。
4. **非布林遮罩**：傳入 `float32` mask → 拋 `TypeError`；這避免用 0/1 混過布林約定。
5. **錯誤順序**：故意先 `transpose` 後 `reshape`，形狀可能仍正確，但 `merge(split(X)) != X` 會被命題 16.1 的數值檢驗抓到。
<<<END>>>