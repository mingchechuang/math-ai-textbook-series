"""建立教材固定前後文、原創SVG圖与合成資料；不呼叫模型。"""
import csv
import html
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

from book_spec import TITLE, CHAPTERS, PARTS, CONVENTIONS, SOURCES, MODELS

PREFACE = '''# 導讀：先理解表示，再理解決策

生成式人工智慧可以把一段描述接成下一段文字，把文字提示轉成影像，也能根據多種輸入提出下一步行動。這些能力容易讓人把「流暢」誤認成「理解」，把「能提出建議」誤認成「能安全執行」。本書選擇從線性代數開始，目的不是把每個模型都簡化成矩陣乘法，而是建立一套能逐步核對的語言：資料是什麼形狀？運算保留了什麼資訊？哪些假設沒有被檢查？當輸出有誤時，我們究竟能追查到哪一層？

全書以一座虛構的水產養殖池「池A」為共同場景。工作人員記錄水溫、溶氧與酸鹼值，攝影機提供水面畫面，值班日誌記下清潔、設備維護與人員觀察。讀者不需要有養殖背景；本書也不會替真實物種設定生存或操作閾值。場景的作用是讓抽象數學一直對應到資料、感測誤差、時間與責任，而不是給出可直接部署的養殖配方。

最初，池A只是幾個數字。我們把它們整理成向量，比較方向與距離，再學會用矩陣描述多筆量測和特徵轉換。接著，線性方程告訴我們何時能唯一解出未知數；秩與零空間則揭示資料不能告訴我們什麼。這一步十分重要：擁有更多欄位，不一定擁有更多獨立資訊。

到了投影、最小平方與矩陣分解，我們開始處理「答案不可能完全吻合資料」的情況。與其強求每筆量測都被解釋，不如明確定義誤差、比較可行模型並記錄限制。PCA與低秩近似能壓縮表示，但被丟掉的小變異不一定沒有安全意義。數值穩定性也不是程式細節；若微小輸入擾動造成巨大係數變化，再漂亮的公式都可能在現場失效。

生成式AI部分把這些觀念接到embedding、線性層、attention與低秩適配。你會看到向量與矩陣如何支撐模型，也會看到線性代數不能單獨解釋的地方：機率模型、非線性函數、資料選擇與訓練目的。讀者不需要先下載大型模型，就能用小型數值例觀察運算；理解三個token的attention，比不明所以地呼叫大型API更適合作為起點。

多模態部分把水質、影像與文字放回同一條資料路徑。不同模態具有不同單位、採樣頻率、延遲和缺值方式；把向量串接起來不代表問題已解決。模型必須說清楚如何對齊時間、如何處理缺失，以及什麼情況下應回報「資料不足」。特徵對齊與語義對齊也不是同一件事，相似分數不能直接當成健康診斷。

最後，agent利用檢索工具查找有來源的作業文件，整理觀察並提出待覆核的建議。本書刻意把建議與設備控制分開。讀者可以在模擬環境完成資料讀取、檢索、風險檢查與審批記錄，但所有真實增氧、投餌、加藥與水體管理仍需由專業人員制定規範並驗證。即使公式正確，感測器位置、物種差異或過時文件仍可能讓決策失效。

## 如何閱讀與驗證

建議依章節順序閱讀。每章先讀場景問題，自己完成手算，再對照推導與答案；程式實驗用來驗證計算，不是代替思考。遇到公式時先寫出每個物件的shape，再問乘法是否合法、假設是否成立、結果的單位是否合理。遇到圖表時則檢查它代表真實資料、合成資料，還是概念示意。

習題包括基本計算、觀念辨析與養殖應用。應用題通常不只有一個數字答案，而要求說清楚資料限制與判斷依據。若你可以寫出「不知道，因為缺少哪些資料」，而不是硬給一個結論，便已掌握本書重要的學習成果。

本書以Markdown為主，公式採LaTeX語法，原創圖表採本地SVG。`examples/linalg_lab.py`是由編輯端另行撰寫的基礎驗證程式；章節內模型生成的其他片段並不因此自動取得「已測試」資格。出版前仍應由人類教師檢查推導、例題、程式與來源。字數報告只計中文字元，不把長程式碼當作充足教學內容。

## 學習路徑

- 想補數學基礎：先完成第1～10章，能解釋解的存在性、投影、分解與數值誤差。
- 想理解生成式AI：在前述基礎上完成第11～15章，逐步核對張量形狀與非線性邊界。
- 想做多模態系統：接續第16～17章，重視時間對齊、資料切分與缺值。
- 想做agent：完成第18～20章，把來源、工具、人工覆核與可重現評估放進設計。

這些路徑不是互不相關的選修。全書的共同問題始終是：如何讓每個計算步驟都能被理解、被檢查，並在不確定時保留拒絕行動的能力？
'''
APPENDIX = r'''# 附錄：共用符號、資料與驗收清單

## 符號速查

| 符號 | 意義 | 預設形狀 |
|---|---|---|
| $x$ | 單筆特徵，column vector | $d\times1$ |
| $X$ | 每列row為一筆觀測 | $n\times d$ |
| $W$ | 線性映射 | $m\times d$ |
| $y=Wx$ | 單筆輸出 | $m\times1$ |
| $Y=XW^T$ | 批次輸出 | $n\times m$ |
| $\mu$ | 訓練資料平均向量 | $d\times1$ |
| $\Sigma$ | 共變異數矩陣 | $d\times d$ |
| $Q,K,V$ | 注意力查詢、鍵、值 | 每章另外明示 |

中文「行、列」在不同教材慣例可能不同，因此本書搭配row與column標註。任何形狀衝突以章節明確定義和矩陣乘法規則為準，不靠名詞猜測。

## 合成資料說明

`data/pond_a_synthetic.csv`是固定種子的教學合成資料，欄位包含timestamp、temperature_c、do_mg_l、ph及quality_flag。時間每十分鐘一筆；少量缺值以空欄位表示。數值只為示範資料處理，不代表某種魚蝦的適宜範圍、危險閾值或生理模型。攝影機與工作日誌在後續章節以小型示意資料呈現，不宣稱有真實現場影像。

讀取資料時先保存原始時間和品質旗標，不因為模型需要完整矩陣便悄悄補值。若使用平均值、插值或前值補齊，應標記方法與可用的歷史範圍，並防止未來資料洩漏。训练、驗證、測試的切分應依時間順序，不能把相鄰量測隨機打散後就宣稱具有跨日泛化能力。

## 專題交付驗收

1. 提供欄位字典、時間單位、物理單位和缺值規則。
2. 每個轉換都列出輸入與輸出shape，保存訓練時估計的參數。
3. 同時測試正常輸入、缺值、過期值、形狀錯誤及矛盾資料。
4. 每則agent建議附引用文件、版本、資料時間與不確定性說明。
5. 將工具分為唯讀與有副作用兩類，教材範例只使用唯讀或模擬工具。
6. 不把提示詞當作安全邊界；真實權限應由程式與現場流程控制。
7. 記錄拒絕執行與人工覆核的理由，提供可追查而非只求成功的測試報告。
8. 不把模型審稿或自動數值測試稱為完整專業驗證。

## 從教材到現場還缺什麼

真正部署仍需要物種與場域專家、設備與通訊工程、感測器校正、資安、維運、故障演練及責任界定。這些工作不能被一個向量分數或語言模型回覆取代。最安全的起點是先將系統限制在資料整理與建議展示，評估與現場人工流程的差異，再決定是否有理由進行更高風險的試驗。
'''


def figure(title, boxes):
    width=1000; height=140+len(boxes)*90
    body=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img"><title>{html.escape(title)}</title><rect width="100%" height="100%" fill="#f5f8fc"/><text x="40" y="45" font-size="26" font-family="sans-serif">{html.escape(title)}</text>']
    for i,label in enumerate(boxes):
        y=75+i*90
        body += [f'<rect x="100" y="{y}" width="800" height="60" rx="10" fill="#dfedfa" stroke="#42698b"/><text x="125" y="{y+38}" font-size="22" font-family="sans-serif">{html.escape(label)}</text>']
        if i<len(boxes)-1:body += [f'<path d="M500 {y+60}v23m-6-6 6 6 6-6" stroke="#42698b" fill="none" stroke-width="2"/>']
    return ''.join(body)+'</svg>'


def initialize(root):
    for sub in ('chapters','figures','data','editorial','agents','examples'):(root/sub).mkdir(parents=True,exist_ok=True)
    (root/'00-preface.md').write_text(PREFACE,encoding='utf-8')
    (root/'99-appendix.md').write_text(APPENDIX,encoding='utf-8')
    (root/'STYLE_GUIDE.md').write_text('# 全書編輯規格\n\n'+CONVENTIONS,encoding='utf-8')
    toc=['# '+TITLE,'\n目標：約五萬中文字。每章至少2200字，原2200～2600字為建議篇幅；依使用者指示允許超字，公式、程式不充當字數。\n','[導讀](00-preface.md)']
    for i,part in enumerate(PARTS):
        toc.append('\n## '+part)
        for n,(title,p,maths,lab) in enumerate(CHAPTERS,1):
            if p==i:toc += [f'### 第{n:02d}章｜{title}', f'- [章節檔案](chapters/{n:02d}.md)（生成後可讀）',f'- 核心：{maths}',f'- 實作：{lab}']
    toc+=['\n[附錄](99-appendix.md)','\n[參考來源](REFERENCES.md)']
    (root/'TOC.md').write_text('\n'.join(toc),encoding='utf-8')
    (root/'REFERENCES.md').write_text('# 參考來源\n\n模型應依這些來源定位概念，不可宣稱所有章節論點都已獨立查證。\n\n'+'\n'.join(f'- [{key}] [{name}]({url})' for key,name,url in SOURCES),encoding='utf-8')
    diagrams={
      'roadmap.svg':('本書的共同主軸',['線性代數：表示、映射、投影、分解','生成式AI：embedding、attention、低秩適配','多模態：文字＋影像＋水質感測','Agent：有來源的檢索與唯讀工具','養殖決策輔助：不確定性與人工批准']),
      'shapes.svg':('形狀契約',['單筆：x (d×1) → W (m×d) → y (m×1)','批次：X (n×d) × Wᵀ (d×m) → Y (n×m)','不同特徵必須處理單位、縮放與缺值']),
      'attention.svg':('注意力運算',['Q (n×dₖ), K (n×dₖ), V (n×dᵥ)','QKᵀ / √dₖ → n×n 分數矩陣','加上mask，逐row做softmax','權重矩陣 × V → n×dᵥ']),
      'fusion.svg':('多模態對齊不是簡單拼接',['時間戳與來源：感測、影像、工作日誌','各模態編碼與品質檢查','投影／對齊／融合，保留缺值旗標','評估時間洩漏、分布改變與不確定性']),
      'agent-safety.svg':('建議與執行分離',['有效且未過期的資料＋有版本的SOP','唯讀檢索與模型建議','程式限制檢查：缺值、權限、文件來源','人工審核：否決、補資料或批准模擬','只寫入稽核記錄；不連接真實設備'])}
    for name,(title,boxes) in diagrams.items():(root/'figures'/name).write_text(figure(title,boxes),encoding='utf-8')
    rng=random.Random(42);start=datetime(2026,1,1)
    with (root/'data/pond_a_synthetic.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['timestamp','temperature_c','do_mg_l','ph','quality_flag'])
        for i in range(144):
            temp=26+math.sin(i/144*2*math.pi)+rng.gauss(0,.1)
            oxygen=6-.3*math.sin(i/144*2*math.pi)+rng.gauss(0,.05)
            writer.writerow([(start+timedelta(minutes=10*i)).isoformat(),round(temp,3),'' if i in (41,42) else round(oxygen,3),round(7.5+rng.gauss(0,.02),3),'missing_do' if i in (41,42) else 'synthetic_ok'])
