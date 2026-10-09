<<<PATCH 01>>>
<<<OLD>>>
**輸出**：Logits $Z \in \mathbb{R}^{B \times T \times V}$，表示每個位置對詞彙表中每個詞元的未歸一化對數概率。
<<<NEW>>>
**輸出**：Logits $Z \in \mathbb{R}^{B \times T \times V}$，是每個位置對各詞元的未正規化實數分數；沿詞彙軸套用 softmax 後才得到條件機率。
<<<END>>>
<<<PATCH 02>>>
<<<OLD>>>
    在 Softmax 前施加遮罩，將不允許注意的位置分數設為 $-\infty$：
    $$ S'_{ij} = \begin{cases} \frac{Q_i \cdot K_j}{\sqrt{d_h}} & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases} $$
    *註：本卷約定布林 True=允許注意。*

*   **Softmax & Weighted Sum**：
    對分數沿最後一個維度（key 軸 $j$）進行 Softmax：
    $$ A_{ij} = \frac{\exp(S'_{ij})}{\sum_{k=0}^{T-1} \exp(S'_{ik})} \in \mathbb{R}^{B \times H \times T \times T} $$
    由於 $S'_{ij} = -\infty$ 當 $j > i$，故 $\exp(S'_{ij}) = 0$，因此 $A_{ij} = 0$ 當 $j > i$。
<<<NEW>>>
    在 Softmax 前施加遮罩，將不允許注意的位置分數設為 $-\infty$：
    $$ S'_{b,h,i,j} = \begin{cases} \frac{Q_{b,h,i,:} \cdot K_{b,h,j,:}}{\sqrt{d_h}} & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases} $$
    *註：本卷約定布林 True=允許注意。*

*   **Softmax & Weighted Sum**：
    沿最後一個維度（key 軸）進行 Softmax。整體 $A \in \mathbb{R}^{B \times H \times T \times T}$；固定四個索引的權重是純量：
    $$ A_{b,h,i,j} = \frac{\exp(S'_{b,h,i,j})}{\sum_{k=0}^{T-1} \exp(S'_{b,h,i,k})} $$
    由於 $S'_{b,h,i,j} = -\infty$ 當 $j > i$，故 $\exp(S'_{b,h,i,j}) = 0$，因此 $A_{b,h,i,j} = 0$ 當 $j > i$。
<<<END>>>
<<<PATCH 03>>>
<<<OLD>>>
    將 $D$ 維拆分為 $H$ 個 $d_h$ 維，需要兩步：首先用 `view/reshape` 將最後一軸 $D$ 拆分為 $(H, d_h)$，得到形狀 $(B, T, H, d_h)$；接著用 `transpose(1, 2)` 交換序列軸與頭軸，得到 $(B, H, T, d_h)$。注意：`reshape` 只改變形狀，不交換軸的順序；`transpose` 才交換軸。兩者缺一不可，reshape 不會自行把 $(B, T, H, d_h)$ 變成 $(B, H, T, d_h)$。`reshape` 也可能回傳副本而非視圖，取決於原始 layout；它不應被概括為「改變記憶體布局的邏輯視圖」。合併頭時先將 $O_h$ 從 $(B, H, T, d_h)$ `transpose(1, 2)` 回 $(B, T, H, d_h)$，再用 `.contiguous().view(B, T, D)` 得到 $(B, T, D)$；`.contiguous()` 不可省略，因為 `transpose` 後張量通常非連續，直接 `view` 會失敗或給出錯誤的記憶體映射：
<<<NEW>>>
    將 $D$ 維拆分為 $H$ 個 $d_h$ 維，需要兩步：首先用 `view/reshape` 將最後一軸 $D$ 拆分為 $(H, d_h)$，得到形狀 $(B, T, H, d_h)$；接著用 `transpose(1, 2)` 交換序列軸與頭軸，得到 $(B, H, T, d_h)$。注意：`reshape` 只改變形狀，不交換軸的順序；`transpose` 才交換軸。兩者缺一不可，reshape 不會自行把 $(B, T, H, d_h)$ 變成 $(B, H, T, d_h)$。`reshape` 也可能回傳副本而非視圖，取決於原始 layout；它不應被概括為「改變記憶體布局的邏輯視圖」。合併頭時先將 $O_h$ 從 $(B, H, T, d_h)$ `transpose(1, 2)` 回 $(B, T, H, d_h)$，再用 `.contiguous().view(B, T, D)` 得到 $(B, T, D)$。本程式使用 `view`，所以先呼叫 `.contiguous()`；`transpose` 後的 stride 通常與此 `view` 不相容，直接呼叫會拋錯。若改用 `reshape`，框架可在必要時建立副本：
<<<END>>>
<<<PATCH 04>>>
<<<OLD>>>
    # 7. Gradient Flow Test
    x_grad = torch.randint(1, V, (2, 5))
    t_grad = torch.randint(1, V, (2, 5))
    model.train()
    _, loss_grad = model(x_grad, t_grad)
    loss_grad.backward()
    
    # 檢查代表性參數的梯度存在、形狀與參數相同且為有限值
    params_to_check = [
        model.tok_emb.weight,
        model.blocks[0].attn.w_q.weight,
        model.blocks[0].attn.w_k.weight,
        model.blocks[0].attn.w_v.weight,
        model.blocks[0].attn.w_o.weight,
        model.blocks[0].ffn.fc1.weight,
        model.blocks[0].ffn.fc2.weight,
        model.out.weight,
    ]
    for p in params_to_check:
        assert p.grad is not None, f"Gradient missing for shape {p.shape}"
        assert p.grad.shape == p.shape, f"Gradient shape mismatch for {p.shape}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {p.shape}"
    pad_grad = model.tok_emb.weight.grad[model.PAD_ID]
    assert torch.all(pad_grad == 0), "PAD embedding gradient should be zero."
    print("7. Gradient Flow Test (extended) Passed.")
<<<NEW>>>
    # 7. Gradient Flow Test：讓 PAD 真正出現在輸入中
    x_grad = torch.tensor([[1, 2, 0]], dtype=torch.long)
    t_grad = torch.tensor([[2, 3, 0]], dtype=torch.long)
    model.train()
    model.zero_grad(set_to_none=True)
    _, loss_grad = model(x_grad, t_grad)
    loss_grad.backward()

    # 巡訪全部參數；有限且同形狀不等於數值梯度已獲驗證
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Gradient missing: {name}"
        assert p.grad.shape == p.shape, f"Gradient shape mismatch: {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient: {name}"
    pad_grad = model.tok_emb.weight.grad[model.PAD_ID]
    assert torch.all(pad_grad == 0), "PAD embedding gradient should be zero."
    print("7. Gradient Flow Test (extended) Passed.")
<<<END>>>
<<<PATCH 05>>>
<<<OLD>>>
    print("8. Fault Tests Passed.")

    print("All tests completed.")
<<<NEW>>>
    print("8. Fault Tests Passed.")

    # 9. 右側 PAD 契約：正常、序列結束與三種違規
    for idx_list, target_list in [
        ([1, 2, 0, 0], [2, 3, 0, 0]),
        ([1, 2, 3], [2, 3, 0]),
    ]:
        _, checked_loss = model(
            torch.tensor([idx_list]), torch.tensor([target_list])
        )
        assert torch.isfinite(checked_loss)
    for idx_list, target_list, msg in [
        ([1, 0, 2, 0], None, "right side"),
        ([1, 2, 3, 4], [2, 0, 3, 0], "right side"),
        ([1, 2, 0], [2, 3, 4], "PAD input positions"),
    ]:
        try:
            model(torch.tensor([idx_list]),
                  None if target_list is None else torch.tensor([target_list]))
        except ValueError as e:
            assert msg in str(e)
        else:
            raise AssertionError("Invalid padding was accepted")
    print("9. Padding Contract Tests Passed.")

    print("All tests completed.")
<<<END>>>