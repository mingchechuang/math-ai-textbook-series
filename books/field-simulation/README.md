# Volume III｜場論、場模擬與相變

依使用者指示於2026-10-03啟動，2026-10-04 04:59完成30章／5部模型審稿，共123,889中文字、428次模型呼叫。原始Markdown是主稿，仍待人工覆核。2026-10-06已產生[離線HTML](http://localhost:8080/book3/html)及[361頁PDF](http://localhost:8080/book3/pdf)，位於`published/volume-3.html`與`published/volume-3.pdf`。6,227個數學片段、768個書籤，排版驗證見`published/validation.json`。排版未改Markdown、未執行教材程式；出版不等於人工數學審定。

入口：[進度與通知](http://localhost:8080/book3) ｜ [30章目錄](http://localhost:8080/book3/toc) ｜ [通知紀錄](http://localhost:8080/book3/notifications)。

## 章級計畫

五部各六章，初步主體目標約135,000中文字，每章目標4500、最低3000；完整卷含導讀、附錄及解答上限200,000。公式、程式、英文、標點及參考來源不充字數。

1. 場、微積分橋接與驗證。
2. 守恆離散與時間積分。
3. 橢圓問題、線性求解與變分方法。
4. 流體、多物理場與不確定性。
5. 自由能、相變與相場專題。

Volume IV尚未製作，前六章提供必要橋接並明列先備要求；不假裝已完成整套分析或熱力學課程。經典場是主體，不涵蓋完整量子場論。溶氧管理閾值不等於物理相變。

## 五模型與40個獨立角色

沿用deepseek-flash、unsloth/Qwen3.8-27B-NVFP4、gpt-5.6-sol、gpt-6-luna、gpt-6-sol，各8人。20作者、20不同模型審稿者輪派30章，ID使用`v3_editor_00`～`v3_editor_39`，與前卷分開的工作目錄及session。這是邏輯隔離，不是OS沙箱。

- 最多4個章節工作；DeepSeek／Qwen各1請求，OpenAI最多2。
- 已實際呼叫五模型作READY preflight，並由adapter核對實際provider/model、正常結束及settled事件；證據`editorial/preflight.json`。
- 採用既有無總呼叫額度上限設定，保留每次1200秒、有限重試及防無效循環保護。
- 逐章審稿、最多兩輪作者修訂、第三模型主編定點patch與複審；全章通過後做五部跨章審查。
- 自動接續最多12輪，連續3輪無進展則留下明確待辦，並由網頁通知記錄停止，不無限重播失敗。
- 共同工作流沿用Volume II已測試的實作，但以獨立模組實例注入第三卷契約／資產；不修改Volume II模組全域，也不覆寫前卷稿件。

## 資料與驗證

- `chapters/`、`book.md`：生成中的章稿與組合稿。
- `editorial/`：審稿、歷史稿、patch、錯誤、preflight及來源界線。
- `agents/`：角色輸入、事件及持久化會話。
- `examples/field_lab.py`：主編另寫的標準庫CPU實驗。
- `data/field_contract.json`、`STYLE_GUIDE.md`：索引、單位、邊界與相場慣例。
- `notifications.sqlite`：持久化通知；網頁每5秒更新，關閉頁面也繼續記錄，不作外部推播。

起步實驗9項測試已通過：常數場、質量、Fourier模態、非法dt、雙向上風、離散能量梯度及特定小步長AC／CH例子。這不證明任意步長能量下降，也不代表後續章稿程式已執行。核心後續實驗為NumPy CPU小格網；SciPy等依賴需在相應章明示，不自動安裝。

## 啟動與紀錄

```bash
python3 book3_editor.py --init-only
python3 -u continue_book3_review.py --output books/field-simulation --max-passes 12 --timeout 1200
python3 -u book2_notifications.py --output books/field-simulation --base-route /book3 --label 'Volume III' --interval 5
python3 books/field-simulation/examples/field_lab.py
```

日誌位於專案`runs/book3-editor.log`、`runs/book3-notifications.log`、`runs/book3-preflight.log`、`runs/book3-starter-tests.log`。背景nohup不保證重開機後自動恢復。出版在完成後另行安排，出版時不執行教材程式。

Qwen既有API啟動前健康正常、running／waiting／KV皆0；SSH仍未通過金鑰驗證，主機RAM／磁碟未確認。只沿用既有推論API，不安裝、訓練或重啟遠端服務。養殖情境皆為合成，不連接真實設備。
