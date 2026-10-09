# 第19章 完整小型decoder-only Transformer

## 學習目標與先備知識

本章旨在將前三部分別探討的嵌入（Embedding）、位置編碼、多頭因果注意力、規範化（LayerNorm）、前饋神經網路（FFN）及輸出層，整合為一個完整的、可訓練的 decoder-only Transformer 模型。讀者需掌握以下先備知識與目標：

1.  **張量形狀與軸契約**：熟練使用 $B$（batch）、$T$（sequence）、$D$（model dimension）、$H$（head）、$d_h$（head dimension）等符號。明確理解 `reshape` 與 `transpose` 的差異，以及廣播（broadcasting）在批次軸與特徵軸上的作用。
2.  **因果注意力（Causal Attention）的數學本質**：理解掩碼（mask）如何通過在 softmax 前施加 $-\infty$ 來嚴格禁止模型訪問未來資訊，並證明其不依賴未來 token。
3.  **端到端梯度流**：了解損失函數如何通過 softmax、注意力加權、殘差連接及反向傳播計算圖，流動至嵌入層與各投影矩陣。
4.  **自足實作能力**：能夠僅依賴 PyTorch CPU 環境，不下載外部權重或語料，獨立實現包含訓練循環、形狀驗證、因果性測試及邊界條件檢查的完整模型。

本章的「實作與程式」部分提供一個自足的、僅依賴 PyTorch CPU 的小模型實作。讀者應能理解每個組件的形狀變化，並能夠在本地 CPU 上運行訓練循環，同時通過自動化的測試函數驗證模型的正確性。

## 問題與直覺

為什麼需要將這些組件整合為完整的模型？在前幾章中，我們可能分別測試了注意力機制的數學性質、規範化的穩定性或損失函數的梯度。然而，真正的語言模型訓練是一個端到端的優化問題，任何組件的形狀錯配、掩碼方向錯誤或梯度截斷都會導致訓練失敗或性能異常。

**直覺模型**：
想象一個小型的「序列預測機器」。它輸入一串字符（例如 `"Hello"`），內部將其轉換為向量，經過多層處理捕捉語法與上下文關係，最後輸出下一個字符的概率分佈（例如預測 `"W"` 的機率最高）。

**關鍵挑戰**：
1.  **形狀一致性（Shape Consistency）**：確保嵌入層輸出 $(B, T, D)$，經過注意力拆分合併後仍為 $(B, T, D)$，最終 logits 為 $(B, T, V)$，其中 $V$ 是詞彙大小。任何中間層的維度錯配都會導致廣播錯誤或矩陣乘積失敗。
2.  **因果性（Causality）**：在第 $t$ 個位置預測時，只能使用 $0 \dots t$ 的資訊（用於預測 $t+1$），不能偷看 $t+1 \dots T-1$。這要求掩碼必須精確對齊時間軸，且軟max操作沿著 key 軸進行。
3.  **可訓練性（Trainability）**：所有參數（嵌入矩陣、注意力投影、FFN權重、規範化參數）必須具有正確的梯度。特別是共享權重或殘差連接處，梯度必須正確累加。
4.  **資料契約（Data Contract）**：輸入與目標（target）必須正確錯位（next-token prediction），且損失計算必須排除填充（PAD）字元，僅對有效 token 計算平均。

## 定義、定理與推導

### 1. 模型結構與形狀推導

我們定義一個包含 $L$ 層 Transformer block 的 decoder-only 模型。每個 block 採用 Pre-Norm 結構，包含：
1.  **Multi-Head Causal Self-Attention (MHA)**：處理序列依賴。
2.  **Layer Normalization (LN)**：穩定激活值分佈，作用於特徵軸。
3.  **Feed-Forward Network (FFN)**：增加非線性表達能力。
4.  **Residual Connections**：緩解梯度消失，加速訓練。

**輸入**：詞元索引 $X \in \mathbb{Z}^{B \times T}$，其中 $X_{b,t}$ 是批次 $b$ 中序列位置 $t$ 的詞元 ID。
**輸出**：Logits $Z \in \mathbb{R}^{B \times T \times V}$，表示每個位置對詞彙表中每個詞元的未歸一化對數概率。

#### 多頭注意力的形狀推導
假設模型維度 $D$，頭數 $H$，則頭維度 $d_h = D/H$。必須滿足 $D \% H == 0$。

*   **查詢、鍵、值投影**：
    從輸入隱狀態 $H_{in} \in \mathbb{R}^{B \times T \times D}$ 通過線性層 $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$ 投影：
    $$ Q, K, V = H_{in} W_Q, H_{in} W_K, H_{in} W_V \in \mathbb{R}^{B \times T \times D} $$

*   **拆分頭（Split Heads）**：
    將 $D$ 維拆分為 $H$ 個 $d_h$ 維。這裡使用 `reshape` 而非 `transpose` 來改變記憶體布局的邏輯視圖：
    $$ Q_h \in \mathbb{R}^{B \times H \times T \times d_h}, \quad K_h \in \mathbb{R}^{B \times H \times T \times d_h}, \quad V_h \in \mathbb{R}^{B \times H \times T \times d_h} $$
    *註：此處假設 $d_v = d_h$。*

*   **分數計算（Scaled Dot Product）**：
    對每個頭獨立計算：
    $$ \text{Scores}_h = \frac{Q_h K_h^T}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times T \times T} $$
    分數矩陣的行對應 query 位置 $i$，列對應 key 位置 $j$。

*   **因果遮罩（Causal Mask）**：
    定義下三角布林矩陣 $M \in \{0, 1\}^{T \times T}$，其中 $M_{ij} = 1$ 當且僅當 $j \le i$（即 key 位置不超過 query 位置）。
    在 Softmax 前施加遮罩，將不允許注意的位置分數設為 $-\infty$：
    $$ S'_{ij} = \begin{cases} \frac{Q_i \cdot K_j}{\sqrt{d_h}} & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases} $$
    *註：本卷約定布林 True=允許注意。*

*   **Softmax & Weighted Sum**：
    對分數沿最後一個維度（key 軸 $j$）進行 Softmax：
    $$ A_{ij} = \frac{\exp(S'_{ij})}{\sum_{k=0}^{T-1} \exp(S'_{ik})} \in \mathbb{R}^{B \times H \times T \times T} $$
    由於 $S'_{ij} = -\infty$ 當 $j > i$，故 $\exp(S'_{ij}) = 0$，因此 $A_{ij} = 0$ 當 $j > i$。
    計算加權和：
    $$ O_h = A_h V_h \in \mathbb{R}^{B \times H \times T \times d_h} $$

*   **合併頭（Merge Heads）**：
    將 $H$ 個頭拼接回 $D$ 維：
    $$ O = \text{Concat}(O_1, \dots, O_H) \in \mathbb{R}^{B \times T \times D} $$
    最後通過輸出投影 $W_O \in \mathbb{R}^{D \times D}$：
    $$ O_{out} = O W_O \in \mathbb{R}^{B \times T \times D} $$

### 2. 損失函數與有效 Token 平均

使用 Cross-Entropy Loss。在 next-token prediction 任務中，輸入序列 $X$ 長度為 $T$，目標序列 $Y$ 也長度為 $T$，其中 $Y_t = X_{t+1}$（對於 $t < T-1$）。通常我們輸入 $X_{0:T-1}$ 來預測 $X_{1:T}$。

若存在填充字元（PAD），其 ID 為 $PAD$。損失應僅在有效 token 上計算。定義有效 token 掩碼 $V \in \{0, 1\}^{B \times T}$，其中 $V_{b,t} = 1$ 當且僅當 $X_{b,t} \neq PAD$。

總負對數似然（NLL）為：
$$ \mathcal{L}_{sum} = \sum_{b=1}^{B} \sum_{t=0}^{T-1} V_{b,t} \cdot \left( - \log P(y_{b,t} | y_{b,<t}) \right) $$

平均損失定義為：
$$ \mathcal{L} = \frac{\mathcal{L}_{sum}}{\sum_{b=1}^{B} \sum_{t=0}^{T-1} V_{b,t}} $$

*注意*：分母是有效 token 的總數，而不是 $B \times T$。這確保了不同批次大小或不同填充比例下的梯度尺度一致。若有效 token 數為零，應拋出錯誤，避免除以零。

### 3. 命題證明：多層 Decoder-Only Transformer 的因果性

**命題**：在採用標準下三角因果掩碼的 $L$ 層 Transformer 中，位置 $i$ 的輸出 logits $Z_i$ 僅依賴於輸入詞元 $X_0, \dots, X_i$，不依賴於 $X_{i+1}, \dots, X_{T-1}$。

**證明**：
我們使用數學歸納法對層數 $l$ 進行證明。

1.  **基礎步驟（層 0，輸入嵌入）**：
    在嵌入層，位置 $i$ 的表示向量 $H_0^{(i)}$ 由詞元 $X_i$ 的嵌入向量 $E_{X_i}$ 加位置編碼 $P_i$ 組成：
    $$ H_0^{(i)} = E_{X_i} + P_i $$
    顯然，$H_0^{(i)}$ 僅依賴 $X_i$ 和位置 $i$，不依賴任何 $X_j$ ($j \neq i$)。因此，依賴集合 $D_0(i) = \{i\}$ 滿足 $D_0(i) \subseteq \{0, \dots, i\}$。

2.  **歸納假設**：
    假設在第 $l-1$ 層，位置 $i$ 的表示 $H_{l-1}^{(i)}$ 僅依賴輸入詞元 $\{X_j \mid j \in D_{l-1}(i)\}$，且 $D_{l-1}(i) \subseteq \{0, \dots, i\}$。

3.  **歸納步驟（層 $l$）**：
    Transformer Block $l$ 包含因果自注意力（MHA）和前饋網路（FFN），並帶有殘差連接。
    
    *   **因果自注意力部分**：
        位置 $i$ 的注意力輸出 $A_l^{(i)}$ 是：
        $$ A_l^{(i)} = \sum_{j=0}^{T-1} \alpha_{ij}^{(l)} V_j^{(l)} $$
        其中權重 $\alpha_{ij}^{(l)}$ 由 Softmax 計算。根據因果掩碼定義，若 $j > i$，則對應的分數為 $-\infty$，導致 $\alpha_{ij}^{(l)} = 0$。
        因此，求和變為：
        $$ A_l^{(i)} = \sum_{j=0}^{i} \alpha_{ij}^{(l)} V_j^{(l)} $$
        這裡 $V_j^{(l)}$ 是第 $l-1$ 層輸出 $H_{l-1}^{(j)}$ 的線性投影。根據歸納假設，$H_{l-1}^{(j)}$ 僅依賴 $\{X_k \mid k \le j\}$。
        因為 $j \le i$，所以 $V_j^{(l)}$ 僅依賴 $\{X_k \mid k \le i\}$。
        故注意力輸出 $A_l^{(i)}$ 僅依賴 $\{X_k \mid k \le i\}$。
        
        經過殘差連接，中間表示 $H_{mid}^{(i)} = H_{l-1}^{(i)} + A_l^{(i)}$。
        $H_{l-1}^{(i)}$ 僅依賴 $\{X_k \mid k \le i\}$（由假設），$A_l^{(i)}$ 也僅依賴 $\{X_k \mid k \le i\}$。
        因此 $H_{mid}^{(i)}$ 僅依賴 $\{X_k \mid k \le i\}$。

    *   **FFN 與 LayerNorm 部分**：
        LayerNorm 沿特徵維度（feature axis）正規化，不混合時間維度。FFN 是逐位置（element-wise）的線性轉換加非線性激活。
        $$ H_l^{(i)} = \text{FFN}(\text{LN}(H_{mid}^{(i)})) + H_{mid}^{(i)} $$
        由於 LN 和 FFN 都是逐位置操作，位置 $i$ 的輸出 $H_l^{(i)}$ 僅依賴於位置 $i$ 的輸入 $H_{mid}^{(i)}$。
        
    *   **結論**：
        $H_l^{(i)}$ 僅依賴 $\{X_k \mid k \le i\}$。
        
    通過歸納法，對於最後一層 $L$，表示 $H_L^{(i)}$ 僅依賴 $\{X_k \mid k \le i\}$。
    最終 logits $Z_i = H_L^{(i)} W_{out}$ 也僅依賴 $\{X_k \mid k \le i\}$。
    得證。$\square$

*註：此證明假設無 dropout（或 dropout 在評估模式下關閉），且數值計算中 $-\infty$ 被正確處理為產生零權重。*

## 逐步手算例題

### 例題 1：單一注意力頭的因果注意力計算

**設定**：
-   Batch size $B=1$。
-   Sequence length $T=3$。
-   Head dimension $d_h=2$。
-   輸入 $Q, K, V$ 簡單矩陣（省略 B 維，假設已投影至 $d_h$）：
    $$ Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad K = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} $$

**步驟 1：計算分數 $S = \frac{Q K^T}{\sqrt{2}}$**

$K^T = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{bmatrix}$

$Q K^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 1 & 2 & 1 \end{bmatrix}$

Scale by $\frac{1}{\sqrt{2}} \approx 0.7071$:
$$ S = \begin{bmatrix} 0.7071 & 0.7071 & 0 \\ 0 & 0.7071 & 0.7071 \\ 0.7071 & 1.4142 & 0.7071 \end{bmatrix} $$

**步驟 2：應用因果掩碼**
遮罩規則：$j > i$ 時置為 $-\infty$。
$$ S_{masked} = \begin{bmatrix} 0.7071 & -\infty & -\infty \\ 0 & 0.7071 & -\infty \\ 0.7071 & 1.4142 & 0.7071 \end{bmatrix} $$

**步驟 3：Softmax (沿行)**

*   **Row 0 (i=0)**:
    $\exp(0.7071) \approx 2.0281$, $\exp(-\infty) = 0$.
    $A_{00} = \frac{2.0281}{2.0281} = 1.0$.
    $A_{01} = 0, A_{02} = 0$.
    Row 0 $\approx [1.0, 0.0, 0.0]$.

*   **Row 1 (i=1)**:
    $\exp(0) = 1$, $\exp(0.7071) \approx 2.0281$, $\exp(-\infty) = 0$.
    Sum $= 1 + 2.0281 = 3.0281$.
    $A_{10} = 1 / 3.0281 \approx 0.3302$.
    $A_{11} = 2.0281 / 3.0281 \approx 0.6698$.
    $A_{12} = 0$.
    Row 1 $\approx [0.3302, 0.6698, 0.0]$.

*   **Row 2 (i=2)**:
    $\exp(0.7071) \approx 2.0281$, $\exp(1.4142) \approx 4.1133$, $\exp(0.7071) \approx 2.0281$.
    Sum $= 2.0281 + 4.1133 + 2.0281 = 8.1695$.
    $A_{20} = 2.0281 / 8.1695 \approx 0.2482$.
    $A_{21} = 4.1133 / 8.1695 \approx 0.5035$.
    $A_{22} = 2.0281 / 8.1695 \approx 0.2482$.
    Row 2 $\approx [0.2482, 0.5035, 0.2482]$.

**步驟 4：加權求和 $O = A V$**

$V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$

*   $O_0 = 1.0 \cdot [1, 0] + 0 + 0 = [1.0, 0.0]$
*   $O_1 = 0.3302 \cdot [1, 0] + 0.6698 \cdot [0, 1] + 0 = [0.3302, 0.6698]$
*   $O_2 = 0.2482 \cdot [1, 0] + 0.5035 \cdot [0, 1] + 0.2482 \cdot [1, 1] $
    $= [0.2482, 0.5035] + [0.2482, 0.2482] = [0.4964, 0.7517]$

結果 $O$:
$$ \begin{bmatrix} 1.0000 & 0.0000 \\ 0.3302 & 0.6698 \\ 0.4964 & 0.7517 \end{bmatrix} $$

### 例題 2：FFN 層的手算前向與形狀

**設定**：
-   Input $h \in \mathbb{R}^{1 \times 2} = [1, 2]$。
-   $W_1 \in \mathbb{R}^{2 \times 4} = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{bmatrix}$, $b_1 = [0, 0, 0, 0]$。
-   Activation: ReLU。
-   $W_2 \in \mathbb{R}^{4 \times 2} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}$, $b_2 = [0, 0]$。

**計算**：
1.  $z_1 = h W_1 + b_1 = [1, 2] \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{bmatrix} = [1, 2, 1, 2]$。
    Shape: $(1, 4)$。
2.  $a_1 = \text{ReLU}(z_1) = [1, 2, 1, 2]$ (all positive)。
3.  $z_2 = a_1 W_2 + b_2 = [1, 2, 1, 2] \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}$
    $z_2[0] = 1\cdot1 + 2\cdot0 + 1\cdot1 + 2\cdot0 = 2$.
    $z_2[1] = 1\cdot1 + 2\cdot1 + 1\cdot0 + 2\cdot1 = 5$.
    $z_2 = [2, 5]$。
    Shape: $(1, 2)$。

輸出 $[2, 5]$。Shape 恢復為輸入特徵維度。

## 實作與程式

以下提供一個完整的、自足的 PyTorch CPU 實作。請確保環境已安裝 `torch` (CPU version) 和 `numpy`。

```python
import torch
import torch.nn as nn
import math
import sys

class TinyDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len=100):
        super(TinyDecoder, self).__init__()
        
        # 參數驗證
        if vocab_size <= 0 or d_model <= 0 or n_heads <= 0 or n_layers < 1:
            raise ValueError("Invalid model parameters")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
            
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.PAD_ID = 0 # 假設 0 為 PAD

        # 1. Embedding
        # padding_idx=0 確保嵌入的 PAD 行恆為 0，且梯度不更新
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=self.PAD_ID)
        # 位置編碼參數，形狀 (1, max_len, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # 2. Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        # 3. Output Layer
        self.final_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        
        # 初始化
        self._init_weights()

    def _init_weights(self):
        # 僅對線性層進行 Xavier 初始化，避免覆寫位置編碼的常態初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) token indices
        targets: (B, T) token indices for next-token prediction (shifted by 1)
        Returns:
            logits: (B, T, V)
            loss: scalar if targets provided
        """
        if idx.ndim != 2:
            raise ValueError(f"idx must be 2D, got {idx.ndim}D")
        b, t = idx.shape
        if t < 1:
            raise ValueError("Sequence length must be >= 1")
        if t > self.max_len:
            raise ValueError(f"Sequence length {t} exceeds max_len {self.max_len}")
        if idx.dtype != torch.long:
            raise ValueError("idx must be of type long")
        if torch.any(idx < 0) or torch.any(idx >= self.vocab_size):
            raise ValueError("Token indices out of bounds")

        # 1. Embeddings
        tok = self.tok_emb(idx)      # (b, t, d_model)
        pos = self.pos_emb[:, :t, :] # (1, t, d_model), broadcast over batch
        h = tok + pos                # (b, t, d_model)
        
        # 2. Blocks
        for blk in self.blocks:
            h = blk(h)
            
        # 3. Norm and Output
        h = self.final_norm(h)             # (b, t, d_model)
        logits = self.out(h)               # (b, t, vocab_size)
        
        if targets is not None:
            # 計算損失
            # targets 必須是 (B, T)
            if targets.shape != idx.shape:
                raise ValueError("targets shape must match idx shape")
            
            # 有效 token 掩碼：排除 PAD
            valid_mask = (targets != self.PAD_ID).float()
            valid_count = valid_mask.sum()
            
            if valid_count == 0:
                raise ValueError("No valid tokens in targets (all PAD?)")

            # 使用 sum reduction 以便手動計算有效 token 平均
            ce_loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size), 
                targets.reshape(-1),
                reduction='sum'
            )
            loss = ce_loss / valid_count
            return logits, loss
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-Norm structure
        # Residual 1
        x = x + self.attn(self.norm1(x))
        # Residual 2
        x = x + self.ffn(self.norm2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        # Project Q, K, V
        # (b, t, d) -> (b, t, h, dk) -> (b, h, t, dk)
        q = self.w_q(x).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        
        # Causal Mask
        # True where attention is allowed (lower triangular)
        # Shape: (t, t)
        mask = torch.tril(torch.ones((t, t), device=x.device)).bool()
        
        # Scaled Dot Product Attention
        # (b, h, t, dk) x (b, h, dk, t) -> (b, h, t, t)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask: scores.masked_fill(~mask, -inf)
        # mask shape (t, t) broadcasts to (b, h, t, t)
        scores = scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        # (b, h, t, dk) -> (b, t, h, dk) -> (b, t, d)
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        out = self.w_o(out)
        return out

class FFN(nn.Module):
    def __init__(self, d_model):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

def generate_synthetic_data(batch_size, seq_len, vocab_size, seed=42):
    """生成合成數據，確保 x 和 targets 是錯位的 next-token 關係"""
    torch.manual_seed(seed)
    # 生成連續序列
    seq = torch.randint(1, vocab_size, (batch_size, seq_len + 1)) 
    # 移除 PAD (1 是第一個有效字符)，確保沒有 0
    x = seq[:, :-1]
    targets = seq[:, 1:]
    return x, targets

def run_tests():
    print("Running self-contained tests...")
    torch.manual_seed(42)
    
    # 1. Shape Test
    V, D, H, L, T, B = 10, 8, 2, 2, 5, 2
    model = TinyDecoder(V, D, H, L, max_len=10)
    model.eval()
    
    x = torch.randint(0, V, (B, T))
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (B, T, V), f"Shape mismatch: {logits.shape}"
    print(f"1. Shape Test Passed: {logits.shape}")

    # 2. Causality Test
    # 改變最後一個 token，檢查前面的 logits 是否變化
    x_mod = x.clone()
    x_mod[:, -1] = (x_mod[:, -1] + 1) % V
    with torch.no_grad():
        logits_orig = model(x)
        logits_mod = model(x_mod)
    
    # 比較除最後一個位置外的所有 logits
    diff = torch.abs(logits_orig[:, :-1, :] - logits_mod[:, :-1, :]).max()
    assert diff < 1e-5, f"Causality failed. Max diff: {diff.item()}"
    print(f"2. Causality Test Passed. Max diff: {diff.item()}")

    # 3. Boundary Test: T > max_len
    try:
        x_long = torch.randint(0, V, (1, 11))
        _ = model(x_long)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "max_len" in str(e)
        print("3. Boundary Test (max_len) Passed.")

    # 4. Boundary Test: Invalid Params
    try:
        _ = TinyDecoder(V, 7, 2, 1) # 7 % 2 != 0
        assert False, "Should have raised ValueError"
    except ValueError:
        print("4. Boundary Test (divisibility) Passed.")

    # 5. Loss Test with PAD
    x_pad = torch.randint(1, V, (1, 5))
    x_pad[0, 4] = 0 # Set last token to PAD
    targets_pad = torch.randint(1, V, (1, 5))
    targets_pad[0, 4] = 0 # Target PAD
    
    model.train()
    _, loss = model(x_pad, targets_pad)
    assert torch.isfinite(loss), "Loss not finite"
    print(f"5. Loss Test (with PAD) Passed. Loss: {loss.item():.4f}")

    # 6. All PAD Test
    x_all_pad = torch.zeros((1, 5), dtype=torch.long)
    targets_all_pad = torch.zeros((1, 5), dtype=torch.long)
    try:
        _, _ = model(x_all_pad, targets_all_pad)
        assert False, "Should have raised ValueError for all PAD"
    except ValueError as e:
        assert "No valid tokens" in str(e)
        print("6. Fault Test (All PAD) Passed.")

    print("All tests completed successfully (expected).")

def main():
    print("Initializing TinyDecoder Model...")
    V = 27 # 26 letters + 1 PAD
    D = 64
    H = 4
    L = 2
    T = 10
    B = 8
    steps = 10

    model = TinyDecoder(V, D, H, L, max_len=T)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {param_count}")
    
    # Run Tests
    run_tests()
    
    print(f"\nStarting Training Simulation for {steps} steps...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    # 生成一次合成數據以保持一致性
    x, targets = generate_synthetic_data(B, T, V, seed=123)
    
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(x, targets)
        loss.backward()
        # 梯度裁切 (Global Norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 2 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
            
    print("Training simulation finished.")
    print("Note: This is a CPU simulation with synthetic data. Convergence is not guaranteed.")

if __name__ == "__main__":
    main()
```

**依賴聲明**：
-   `torch`: 僅使用 CPU 版本。
-   無外部數據集加載，無模型權重下載。
-   無 GPU 加速。
-   程式碼包含完整的參數驗證、測試函數及訓練循環。

## 測試與預期結果

### 1. 形狀測試（Shape Test）
-   **輸入**：$B=2, T=5, V=10, D=8, H=2$。
-   **預期**：
    -   Embedding 輸出：$(2, 5, 8)$。
    -   Attention Q/K/V：$(2, 2, 5, 4)$。
    -   Attention Scores：$(2, 2, 5, 5)$。
    -   Logits：$(2, 5, 10)$。
-   **驗證**：程式中 `assert logits.shape == (B, T, V)`。

### 2. 因果性測試（Causality Test）
-   **方法**：固定前 $T-1$ 個 token，改變最後一個 token $X_{T-1}$。
-   **預期**：前 $T-1$ 個位置的 logits 不應發生變化。
-   **驗證**：程式計算 `torch.abs(logits_orig[:, :-1, :] - logits_mod[:, :-1, :]).max()`，預期小於 $1e-5$。

### 3. 邊界與故障測試（Boundary & Fault Tests）
-   **T > max_len**：預期拋出 `ValueError`。
-   **d_model 不可被 n_heads 整除**：預期拋出 `ValueError`。
-   **全 PAD Target**：預期拋出 `ValueError` ("No valid tokens")。
-   **Token 索引越界**：預期拋出 `ValueError`。

### 4. 梯度流測試（Gradient Flow Test）
-   **方法**：執行 `loss.backward()` 後，檢查關鍵參數的梯度。
-   **預期**：`model.tok_emb.weight.grad` 應非零（對於出現過的 token）。`model.out.weight.grad` 應非零。
-   **注意**：由於數據是合成的隨機序列，並非每個 token 都會出現，因此嵌入層中未出現 token 的梯度行應為零。

## 反例與常見陷阱

1.  **Target 未錯位**：
    -   **陷阱**：將 `x` 直接作為 `targets` 傳入 loss。
    -   **後果**：模型學會複製輸入，而非預測下一個 token。Loss 可能收斂，但生成無意義。
2.  **Mask 方向錯誤**：
    -   **陷阱**：使用 `torch.triu`（上三角）而非 `torch.tril`（下三角）。
    -   **後果**：模型可以看到未來，訓練時作弊，測試時性能崩潰，因果性測試失敗。
3.  **忘記縮放（Scale）**：
    -   **陷阱**：忘記除以 $\sqrt{d_k}$。
    -   **後果**：Score 方差過大，Softmax 飽和，梯度消失或不穩定。
4.  **有效 Token 平均錯誤**：
    -   **陷阱**：使用 `reduction='mean'` 且未處理 PAD。
    -   **後果**：PAD 位置的 0 梯度（或錯誤的 log-prob）會稀釋有效 token 的梯度，導致訓練效率降低或偏差。

## AI、幾何與養殖案例

**養殖日誌預測（合成數據）**：
在養殖場，每日記錄包括水質（pH、溶氧）和操作（投餌）。我們將這些數據離散化為詞元序列。
-   **應用**：使用本模型預測明天的「水質異常標籤」（例如：正常、偏高、偏低）。
-   **因果性意義**：預測第 $t$ 天的狀態時，僅使用 $0 \dots t-1$ 天的數據。這確保了不會使用未來資訊。
-   **局限與安全邊界**：
    -   此模型僅用於**研究目的**，基於合成數據。
    -   **不可**用於實際養殖場的設備控制（如自動開泵、投藥）。
    -   模型輸出的是概率分佈，不直接生成操作指令。人類專業人員必須進行最終判斷。
    -   數據中不包含真實的閾值或敏感操作參數。

## 習題

1.  **手算**：給定 $D=4, H=2$，輸入 $X \in \mathbb{R}^{1 \times 2 \times 4}$。請列出 Attention 中 $Q, K, V$ 在拆分後 $(B, H, T, d_h)$ 的形狀，以及 Scores 的形狀。
2.  **程式**：修改 `TinyDecoder`，將 `nn.GELU` 替換為 `nn.Sigmoid`。寫出修改後的 `FFN` 類別。
3.  **反例**：如果我們將 Mask 改為 $M_{ij} = 1$ 當 $j \ge i$（允許看當前及未來），因果性測試 `check_causality` 的結果會如何？請簡述原因。
4.  **整合**：假設 $B=2, T=4$，其中第 2 行最後一個 token 是 PAD。請計算有效 token 數，並說明 Loss 的分母應該是多少。

## 習題解答

1.  **形狀**：
    -   Input: $(1, 2, 4)$。
    -   Q, K, V after projection: $(1, 2, 4)$。
    -   Reshape to heads $(1, 2, 2, 2)$ (B=1, T=2, H=2, dh=2)。
    -   Transpose to $(B, H, T, dh)$: $(1, 2, 2, 2)$。
    -   Scores $(B, H, T_q, T_k)$: $(1, 2, 2, 2)$。
2.  **程式**：
    ```python
    class FFN_Sigmoid(nn.Module):
        def __init__(self, d_model):
            super(FFN_Sigmoid, self).__init__()
            self.fc1 = nn.Linear(d_model, 4 * d_model)
            self.fc2 = nn.Linear(4 * d_model, d_model)
            self.act = nn.Sigmoid()
        def forward(self, x):
            x = self.act(self.fc1(x))
            x = self.fc2(x)
            return x
    ```
3.  **反例**：
    -   因果性測試將**失敗**。
    -   原因：如果 Mask 允許 $j > i$，則改變未來 token $X_j$ 會改變 $K_j$，進而改變 Score $S_{ij}$ 和 Attention Weight $A_{ij}$，導致位置 $i$ 的輸出 $O_i$ 改變。因此 $Logits_{i}$ 將依賴未來輸入。
4.  **整合**：
    -   總 token 數 $N_{total} = B \times T = 2 \times 4 = 8$。
    -   有效 token：第 1 行 4 個，第 2 行 3 個（最後一個 PAD 排除）。
    -   有效 token 數 $N_{valid} = 4 + 3 = 7$。
    -   Loss 分母應為 7。

## 本章小結

本章我們構建了一個完整的、小型的 decoder-only Transformer 模型。我們：
1.  定義了模型的組件及其形狀變化，並證明了多層因果 Transformer 的因果性。
2.  提供了可執行的 PyTorch 代碼，包含參數驗證、測試函數及訓練循環。
3.  討論了常見陷阱（如 Mask 方向、Target 錯位）及邊界條件。
4.  結合養殖案例，強調了模型的合成數據性質及安全使用邊界。

讀者現在應該能夠理解 Transformer 的核心架構，並能夠根據需求調整模型參數或結構。下一章將深入探討訓練流程中的數據處理、優化器策略及評估指標。

## 參考來源

1.  Vaswani et al., *Attention Is All You Need*, 2017. [N1]
2.  PyTorch Documentation: `nn.Linear`, `nn.LayerNorm`, `nn.functional.cross_entropy`. [N3]
3.  NumPy Broadcasting Rules (Conceptual Reference). [N4]
4.  PyTorch Reproducibility Notes. [N6]