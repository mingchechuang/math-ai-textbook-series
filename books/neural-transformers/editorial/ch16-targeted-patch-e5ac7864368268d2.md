<<<PATCH 01>>>
<<<OLD>>>
4. **非布林遮罩**：傳入 `float32` mask → 拋 `ValueError`；這避免用 0/1 混過布林約定。
<<<NEW>>>
4. **非布林遮罩**：傳入 `float32` mask → 拋 `TypeError`；這避免用 0/1 混過布林約定。
<<<END>>>