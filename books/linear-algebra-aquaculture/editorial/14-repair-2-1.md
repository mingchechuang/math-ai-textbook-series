## 複審結果

本次稿件與上一版實質相同，因此前次指出的問題仍未解決。

### 已核對正確

核心形狀
\[
A\in\mathbb R^{r\times d},\quad B\in\mathbb R^{m\times r},\quad
\Delta W=\frac{\alpha}{r}BA\in\mathbb R^{m\times d}
\]
相容。例題中人工計算可得
\[
\Delta W=
\begin{bmatrix}
0&0&0\\
2&0&0\\
-2&0&0\\
4&0&0
\end{bmatrix},
\quad \operatorname{rank}(\Delta W)=1.
\]
Python 預期輸出中的 \(Wx=[0.4,-0.1,-0.25,-0.1]^T\)、偏移 \([0,2,-2,4]^T\) 與新輸出 \([0.4,1.9,-2.25,3.9]^T\) 均正確。本次未執行程式。

### 仍須修正

1. **錯誤形狀示範仍錯。**「概念與推導」稱 \(A\in\mathbb R^{d\times r}\)、\(B\in\mathbb R^{m\times r}\) 時，\(BA\) 為 \(m\times r\)。其實 \((m\times r)(d\times r)\) 通常根本不能相乘。應改為「內維 \(r\) 與 \(d\) 不符，除非偶然 \(r=d\)，否則乘法未定義」。

2. **零空間解釋仍不正確。**  
   >「只要 temperature 成分不變，LoRA 完全不改變 logit」  
   固定非零 temperature 時，LoRA 仍保有固定偏移。應改為：「固定 temperature、只改變 do 或 ph 時，LoRA 偏移不變；純 do、ph 方向位於零空間。」

3. **秩等式缺必要條件。** \(\operatorname{rank}(\Delta W)=\operatorname{rank}(BA)\) 須先假設 \(r>0\)、\(\alpha\neq0\)。若 \(\alpha=0\)，\(\Delta W=0\)。此外，「避免偏移爆炸」不是縮放本身能保證，宜改為「調整不同 \(r\) 下的更新尺度，但穩定性仍取決於初始化與訓練」。

4. **行列術語有歧義。** 輸出方向所在者是 column space，建議寫成「欄空間（column space）」，不要只寫「列空間」。

5. **篇幅仍超標。** 編輯程式計得2641字，超過2600字上限41字，故未符合明示契約。可刪除重複的形狀錯誤提醒或第二次完整展開 \(Wx\)。

習題答案、先備知識、來源ID與養殖安全界線其餘均合格；來源列出不代表所有經驗主張已被證實。

VERDICT: REVISE