<<<PATCH 01>>>
<<<OLD>>>
設 $\mathcal{H}=\{f_1,\dots,f_M\}$ 是**有限**假設類，$|M|=\mathcal{H}$ 為其基數。
<<<NEW>>>
設 $\mathcal{H}=\{f_1,\dots,f_M\}$ 是**有限**假設類，其基數為 $M$。
<<<END>>>
<<<PATCH 06>>>
<<<OLD>>>
若只有單一有效位置且 `y=1`、`p=1/3`，損失應為 $-\ln(1/3)=\ln 3\approx 1.099$，而非 $\log 3\approx 0.477$。
<<<NEW>>>
若只有單一有效位置且 `y=1`、`p=1/3`，損失應為 $-\ln(1/3)=\ln 3\approx 1.099$（而非 $\log_{10}3\approx 0.477$）。
<<<END>>>
<<<PATCH 06>>>
<<<OLD>>>
        raise ValueError("沒有有效 target，平均損失無定義")
<<<NEW>>>
        raise ValueError("沒有有效 target，損失無定義（對 mean 與 sum 皆然）")
<<<END>>>