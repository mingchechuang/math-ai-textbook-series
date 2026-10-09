# 參考來源

2026-10-06取得N1與N2摘要頁，未完整閱讀論文；N3 stable入口為HTML轉址，已取得明確2.14 API全文並核對mask True=參與、evaluation須dropout_p=0及矩形因果遮罩的左上對齊。這不是本機PyTorch版本或執行證據。N4–N6目前只是待核對延伸入口，不可宣稱已查證其內容；未執行外部程式或下載模型。

- [N1] [Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [N2] [Hu et al., LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [N3] [PyTorch 2.14 scaled_dot_product_attention API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [N4] [NumPy broadcasting使用指南（待逐條核對）](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [N5] [Dive into Deep Learning（延伸入口，未逐章核對）](https://d2l.ai/)
- [N6] [PyTorch reproducibility（延伸入口，未逐條核對）](https://docs.pytorch.org/docs/stable/notes/randomness.html)