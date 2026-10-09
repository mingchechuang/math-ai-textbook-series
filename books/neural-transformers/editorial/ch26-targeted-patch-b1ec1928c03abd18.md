<<<PATCH 26>>>
<<<OLD>>>
B1. 把 `quantize_symmetric` 改成逐列（沿 $D_{\mathrm{out}}$ 軸）版本，回傳尺度形狀 $(D_{\mathrm{in}},1)$，並對 $4\times 3$ 隨機矩陣檢查每列誤差界。
<<<NEW>>>
B1. 把 `quantize_symmetric` 改成逐行（對 $W\in\mathbb{R}^{D_{\mathrm{in}}\times D_{\mathrm{out}}}$ 沿第 1 軸 $D_{\mathrm{out}}$ reduction）版本，回傳尺度形狀 $(D_{\mathrm{in}},1)$，並對 $4\times 3$ 隨機矩陣檢查每行誤差界。
<<<END>>>