<<<PATCH 19>>>
<<<OLD>>>
        if (idx == self.PAD_ID).any():
            non_pad = (idx != self.PAD_ID).long()
            flipped = torch.flip(non_pad, dims=[1])
            cummax = torch.cummax(flipped, dim=1).values
            violation = (flipped == 0) & (cummax == 1)
            if violation.any():
                raise ValueError("PAD tokens must be at the right side of the sequence.")
<<<NEW>>>
        def check_right_padding(tokens, name):
            non_pad = tokens.ne(self.PAD_ID).long()
            flipped = torch.flip(non_pad, dims=[1])
            cummax = torch.cummax(flipped, dim=1).values
            violation = (flipped == 0) & (cummax == 1)
            if violation.any():
                raise ValueError(f"{name} PAD tokens must be at the right side.")

        check_right_padding(idx, "Input")
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
            if torch.any(targets < 0) or torch.any(targets >= self.vocab_size):
                raise ValueError("Target indices out of bounds")
            
            # 有效 token 掩碼：基於 TARGETS
<<<NEW>>>
            if torch.any(targets < 0) or torch.any(targets >= self.vocab_size):
                raise ValueError("Target indices out of bounds")
            check_right_padding(targets, "Target")
            if torch.any(idx.eq(self.PAD_ID) & targets.ne(self.PAD_ID)):
                raise ValueError("PAD input positions must have PAD targets")
            
            # 有效 token 掩碼：基於 TARGETS
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
**輸出**：Logits $Z \in \mathbb{R}^{B \times T \times V}$，表示每個位置對詞彙表中每個詞元的未歸一化對數概率。
<<<NEW>>>
**輸出**：Logits $Z \in \mathbb{R}^{B \times T \times V}$ 是每個位置對詞彙表中每個詞元的未歸一化實數分數；沿詞彙軸套用 softmax 後得到條件機率，套用 log-softmax 後得到對數機率。
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
    $$ A_{ij} = \frac{\exp(S'_{ij})}{\sum_{k=0}^{T-1} \exp(S'_{ik})} \in \mathbb{R}^{B \times H \times T \times T} $$
    由於 $S'_{ij} = -\infty$ 當 $j > i$，故 $\exp(S'_{ij}) = 0$，因此 $A_{ij} = 0$ 當 $j > i$。
<<<NEW>>>
    $$ A \in \mathbb{R}^{B \times H \times T \times T}, \qquad A_{b,h,i,j} = \frac{\exp(S'_{b,h,i,j})}{\sum_{k=0}^{T-1} \exp(S'_{b,h,i,k})} $$
    由於 $S'_{b,h,i,j} = -\infty$ 當 $j > i$，故 $\exp(S'_{b,h,i,j}) = 0$，因此 $A_{b,h,i,j} = 0$ 當 $j > i$。
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
        x_mod[:, -1] = (x_mod[:, -1] + 1) % V
<<<NEW>>>
        x_mod[:, -1] = x_mod[:, -1] % (V - 1) + 1
<<<END>>>