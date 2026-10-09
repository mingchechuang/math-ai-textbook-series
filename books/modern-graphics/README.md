# Volume II｜現代電腦圖學

狀態：2026-10-03完成30章逐章與5部跨章模型審查，並已產生HTML／PDF；仍待人工覆核與完整程式驗證。入口：`http://localhost:8080/book2`。

出版版：[HTML閱讀](http://localhost:8080/book2/html) ｜ [PDF（384頁）](http://localhost:8080/book2/pdf) ｜ [下載離線HTML](http://localhost:8080/book2/download/html)。出版工具及驗證方式見[出版說明](../../publishing/README.md)。

## 範圍與篇幅

五部、30章，每部6章：幾何／成像、網格／建模、材質／著色、光線追蹤／光傳輸、動畫／數位分身。完整章級契約見`TOC.md`。以每章約4500中文字編排，主體約13.5萬字，含導讀及附錄的完整卷不得超過20萬中文字；公式、程式、英文、標點與參考來源不充字數。

## 框架改進紀錄

已保存[40角色協作架構觀察與改進參考](../../docs/MULTI_AGENT_FRAMEWORK_REVIEW.md)。後續依續跑指示，已局部實作[格式回收修正](editorial/RECOVERY_20261002.md)及[定點修補模式](editorial/TARGETED_REPAIR_20261002.md)，並非全面重構。

## 協作

沿用原五模型，各8個角色，共40個獨立session角色：20作者＋20不同模型審稿者。30章輪派給這40個角色，不代表同時發出40個請求。每個角色工作有鎖；DeepSeek與Qwen各最多1個請求，OpenAI最多2個，最多4個章節工作。

- DeepSeek：deepseek-flash
- GB10-2 vLLM：unsloth/Qwen3.8-27B-NVFP4
- OpenAI Codex：gpt-5.6-sol、gpt-6-luna、gpt-6-sol

每章：初稿→獨立複審→最多兩輪有針對性的作者修訂與複審。仍有具體問題時，轉交第三模型主編精準patch，再交原審稿者核對。

全部章節通過後，五部各由指定不同模型做跨章審查；拒稿可由主編自動提出唯一匹配的patch，驗證不重疊與結構後保存歷史，重新審查受影響章節與該部。最多兩輪跨章修補，避免無限改稿。未通過不得假稱完成。

原先全卷240次上限已於2026-10-02依使用者「繼續，不要管預算」指示取消；本次以 `--max-calls 0` 表示不設總呼叫上限。仍保留單次1200秒、供應者並行限制及有限重試。`continue_book2_review.py` 自動接續未通過工作，已通過且內容／審稿契約未變的跨章批准可重用；連續3輪無進展或12輪仍未完成時留下明確待辦，不陷入無限循環。這些是失敗保護，不是預算停止。

## 檔案與執行

- `chapters/`：當前稿件；`editorial/`：審稿、patch、錯誤及舊稿。
- `agents/`：各角色輸入、Pi事件與持久化session。
- `status.json`、`word_count.json`：實際進度；`usage.json`：Pi模型目錄等值費用，非訂閱帳單。
- `examples/graphics_lab.py`：主編另寫的標準庫CPU起步實驗；不等於全書程式已驗證。
- `figures/`：五張原創SVG流程圖；`data/scene.json`：合成場景契約。
- `book.md`：隨進度組合的主稿，未產生章節明確標示。

```bash
python3 book2_editor.py --output books/modern-graphics --max-calls 0 --timeout 1200
python3 books/modern-graphics/examples/graphics_lab.py
# 對有既存稿件的未通過章節開新修補世代；保留已通過章節與累計預算
python3 book2_editor.py --output books/modern-graphics --targeted-repair --max-calls 0 --timeout 1200
# 本次自動續跑：無總呼叫上限，仍有防循環保護
python3 continue_book2_review.py --output books/modern-graphics --max-passes 12 --timeout 1200
```

背景執行紀錄：專案`runs/book2-editor.log`；起步實驗紀錄：`runs/book2-starter-tests.log`。程序以nohup執行，可跨終端關閉，不保證重開機後自動恢復。

## 網頁進度通知

`/book2` 已內嵌通知紀錄；也可獨立開啟 `/book2/notifications`，最近200則JSON在 `/book2/notifications.json`。每5秒更新，記錄章節／部份通過、全卷模型審查完成、逾時／重試、工作中断與恢復。關閉網頁仍由獨立監測程序記錄，但不發桌面或外部推播。

- 完整事件保存在 `notifications.sqlite`；重啟監測不重複通知。
- 時間是觀測時間，啟用前事件不逐項回填；啟用時已完成的全卷狀態會註明原進度更新時間。
- 逾時依據實際錯誤檔，不由等待時長猜測；未偵測到工作程序超過15秒才報停止，不把單次失敗誤說成全卷停止。
- 通知程序 `book2_notifications.py` 唯讀觀察原進度，不呼叫模型、不修改主稿或status，也不算作編輯程序。
- 日誌：專案 `runs/book2-notifications.log`；PID：`runs/book2-notifications.pid`。
- 啟動：`python3 -u book2_notifications.py --output books/modern-graphics --interval 5`。nohup只跨終端關閉，不保證重開機後自動恢復。
- 網頁由HTTP路由注入通知面板，編輯器重建index.html不會覆蓋入口。若監測停止，頁面的最近監測時間也會停止更新。

## 驗證與資源界線

作者／審稿模型不使用工具、不安裝套件、不執行生成程式。稿件中的Python先做AST語法檢查；實際執行須另經檢視、記錄環境與結果。模型審稿完成仍待人工及程式驗證；HTML／PDF直接由現有定稿排版；連結僅在出版紀錄、原稿／各章雜湊與PDF排版驗證一致時展示，不代表已人工審定。

來源選擇與查閱界線見`REFERENCES.md`。未下載模型、不更改远端服務。啟動前GB10-2既有API健康正常，running／waiting與KV使用為0；SSH金鑰驗證仍失敗，主機RAM／磁碟未確認。因此僅串行沿用既有推論API，不新增遠端GPU分配、訓練、安裝或重啟服務。
