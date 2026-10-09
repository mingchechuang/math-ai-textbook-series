<<<PATCH 21>>>
<<<OLD>>>
第三是第 20 章的優化器：SGD、動量、Adam 與 AdamW，其中 Adam 的偏差修正步數由 $1$ 開始，AdamW 的解耦權重衰減與 L2 正則化不是同一件事。
<<<NEW>>>
第三是優化器的基本概念：第 20 章的程式使用 SGD；本章另以動量、Adam 與 AdamW 說明更新狀態及 optimizer step 的邊界。其中 Adam 的偏差修正步數由 $1$ 開始，AdamW 的解耦權重衰減與 L2 正則化不是同一件事。
<<<END>>>
<<<PATCH 19>>>
<<<OLD>>>
使用 Cross-Entropy Loss。在 next-token prediction 任務中，輸入序列 $X$ 長度為 $T$，目標序列 $Y$ 也長度為 $T$，其中 $Y_t = X_{t+1}$（對於 $t < T-1$）。通常我們輸入 $X_{0:T-1}$ 來預測 $X_{1:T}$。
<<<NEW>>>
使用 Cross-Entropy Loss。對一份長度 $T+1$ 的原始序列 $(s_0,\ldots,s_T)$，next-token prediction 取長度同為 $T$ 的輸入 $X_t=s_t$ 與目標 $Y_t=s_{t+1}$，其中 $0\le t<T$。因此模型在輸入位置 $t$ 預測下一個詞元；若原始序列只有 $T$ 個詞元且沒有另外提供結尾詞元，則只能構造 $T-1$ 個這樣的輸入／目標位置。
<<<END>>>