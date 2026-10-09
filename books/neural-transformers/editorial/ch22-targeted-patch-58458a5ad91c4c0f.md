<<<PATCH 01>>>
<<<OLD>>>
因為 $T_{\mathrm{temp}}>0$，乘回溫度不改變不等號方向，故此式等價於 $z_i>z_j$。相等情形同理。證畢。
<<<NEW>>>
因為 $T_{\mathrm{temp}}>0$，乘回溫度不改變不等號方向，故此式等價於 $z_i>z_j$。相等情形同理。證畢。此命題使用精確實數運算；程式以浮點數計算 `logits / temperature`，有限 logits 除以極小正溫度仍可能溢位。縮放結果若非有限，後續 `stable_softmax` 會拋出 `ValueError`，不能將命題理解成此 API 對所有正溫度均能成功產生機率。
<<<END>>>