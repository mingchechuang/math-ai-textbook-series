<<<PATCH 26>>>
<<<OLD>>>
**B1.** 逐列版本的 critical 行是 `max_abs = np.max(np.abs(W), axis=1, keepdims=True)`，形狀 $(D_{\mathrm{in}},1)$；`scale = np.where(max_abs==0, 1.0, max_abs/qmax)`；`q = np.clip(np.round(W/scale), -qmax, qmax)`；反量化 `Wr = q*scale`。誤差界：對第 $i$ 列，$\max_j|W_{ij}-W_{r,ij}|\le s_i/2$，證明與命題 26.1 逐列套用相同。
<<<NEW>>>
**B1.** 逐行版本的 critical 行是 `max_abs = np.max(np.abs(W), axis=1, keepdims=True)`，形狀 $(D_{\mathrm{in}},1)$；`scale = np.where(max_abs==0, 1.0, max_abs/qmax)`；`q = np.clip(np.round(W/scale), -qmax, qmax)`；反量化 `Wr = q*scale`。誤差界：對第 $i$ 行，$\max_j|W_{ij}-W_{r,ij}|\le s_i/2$，證明與命題 26.1 逐行套用相同。
<<<END>>>