<<<PATCH 19>>>
<<<OLD>>>
            violation = (flipped == 0) & (cummax == 1)
            if violation.any():
                raise ValueError("PAD tokens must be at the right side of the sequence.")

        # 1. Embeddings
<<<NEW>>>
            violation = (flipped == 0) & (cummax == 1)
            if violation.any():
                raise ValueError("PAD tokens must be at the right side of the sequence.")

        # 1. Embeddings
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
            if torch.any(targets < 0) or torch.any(targets >= self.vocab_size):
                raise ValueError("Target indices out of bounds")
            
            # 有效 token 掩碼：基於 TARGETS
<<<NEW>>>
            if torch.any(targets < 0) or torch.any(targets >= self.vocab_size):
                raise ValueError("Target indices out of bounds")
            if (targets == self.PAD_ID).any():
                non_pad = (targets != self.PAD_ID).long()
                flipped = torch.flip(non_pad, dims=[1])
                cummax = torch.cummax(flipped, dim=1).values
                violation = (flipped == 0) & (cummax == 1)
                if violation.any():
                    raise ValueError("Target PAD tokens must be at the right side.")
            if torch.any(idx.eq(self.PAD_ID) & targets.ne(self.PAD_ID)):
                raise ValueError("PAD input positions must have PAD targets")
            
            # 有效 token 掩碼：基於 TARGETS
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
        -   $V_j^{(l)}$ 是第 $l-1$ 層輸出 $H_{l-1}^{(j)}$ 的線性投影。根據歸納假設，$H_{l-1}^{(j)}$ 僅依賴 $\{X_k \mid k \le j\}$。因為 $j \le i$，所以 $V_j^{(l)}$ 僅依賴 $\{X_k \mid k \le i\}$。
        -   權重 $\alpha_{ij}^{(l)}$ 依賴於 $Q_i$ 和所有參與 softmax 分母的 $K_j$ ($j \le i$)。$Q_i$ 依賴 $H_{l-1}^{(i)}$，而所有 $K_j$ 依賴 $H_{l-1}^{(j)}$。根據歸納假設，這些量均僅依賴 $\{X_k \mid k \le i\}$。
<<<NEW>>>
        -   $V_j^{(l)}$ 是 $\widetilde H_{l-1}^{(j)}=\operatorname{LN}_1(H_{l-1}^{(j)})$ 的線性投影。LayerNorm 沿特徵軸逐位置運算，故不擴大時間依賴範圍。依歸納假設，$V_j^{(l)}$ 僅依賴 $\{X_k \mid k \le j\}$；因 $j \le i$，它僅依賴 $\{X_k \mid k \le i\}$。
        -   權重 $\alpha_{ij}^{(l)}$ 依賴於 $Q_i$ 和參與 softmax 分母的 $K_j$ ($j \le i$)。它們分別由 $\widetilde H_{l-1}^{(i)}$ 與 $\widetilde H_{l-1}^{(j)}$ 投影而來，故依賴範圍均包含於 $\{X_k \mid k \le i\}$。
<<<END>>>

<<<PATCH 19>>>
<<<OLD>>>
        $$ H_l^{(i)} = \text{FFN}(\text{LN}(H_{mid}^{(i)})) + H_{mid}^{(i)} $$
<<<NEW>>>
        $$ H_l^{(i)} = \text{FFN}(\operatorname{LN}_2(H_{mid}^{(i)})) + H_{mid}^{(i)} $$
<<<END>>>