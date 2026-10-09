## 複審結果

核心計算已一致：\(A\in\mathbb R^{1\times3}\)、\(B\in\mathbb R^{4\times1}\)，故 \(BA\in\mathbb R^{4\times3}\)。人工核對得

\[
\Delta W=
\begin{bmatrix}
0&0&0\\2&0&0\\-2&0&0\\4&0&0
\end{bmatrix},\quad
\operatorname{rank}(\Delta W)=1,
\]

且 Python 預期值 \(Wx=[0.4,-0.1,-0.25,-0.1]^T\)、偏移 \([0,2,-2,4]^T\)、新輸出 \([0.4,1.9,-2.25,3.9]^T\) 均正確。本次未執行程式。

但先前指出的問題尚未修正，另有一處維度說明錯誤：

1. **「形狀與乘法鏈」錯誤示範不正確。** 原文稱若 \(A\in\mathbb R^{d\times r}\)、\(B\in\mathbb R^{m\times r}\)，「則 \(BA\) 形狀為 \(m\times r\)」。實際上 \((m\times r)(d\times r)\) 的內維 \(r,d\) 通常不相等，乘法根本未定義；只有偶然 \(r=d\) 時才可乘，結果為 \(m\times r\)。應改為「一般無法相乘」。

2. **Step 4 的零空間解釋仍錯。**  
   >「只要 temperature 成分不變，LoRA 完全不改變 logit」  
   固定非零 temperature 時仍有固定 LoRA 偏移。應改成：「固定 temperature、只改變 do 或 ph 時，LoRA 偏移不變；純 do、ph 變化位於零空間。」

3. **秩等式缺少必要假設。** \(\operatorname{rank}(\Delta W)=\operatorname{rank}(BA)\) 須假設 \(\alpha\neq0\) 且 \(r>0\)。若 \(\alpha=0\)，前者為零，但秩上界仍成立。縮放「避免偏移爆炸」亦過強，應說它用於調整不同 \(r\) 下的尺度，不保證數值穩定。

4. **符號用語。** 「列空間——輸出方向位於 \(\mathbb R^m\)」實指 column space；為符合全書避免行列歧義的契約，宜寫「欄空間（column space）」。

5. **篇幅未通過。** 編輯程式計得2641字，超過2600上限41字。可刪除重複形狀提醒或完整展開的第二次人工核算。

習題答案、來源ID與養殖安全界線其餘均合格；來源清單本身不代表經驗主張已獲驗證。

VERDICT: REVISE