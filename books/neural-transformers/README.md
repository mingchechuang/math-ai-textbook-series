# Volume V｜神經網路、深度學習與Transformer的數學及實作

依使用者指示於2026-10-06啟動。五個原模型已各自通過實際模型識別及READY測試，23:11起開始首批四章写作。最新狀態請看網頁／`status.json`，不要把本README的啟動紀錄當即時進度。

[進度與通知](http://localhost:8080/book5) ｜ [30章契約](http://localhost:8080/book5/toc) ｜ [通知紀錄](http://localhost:8080/book5/notifications) ｜ [組合Markdown](http://localhost:8080/book5.md)

## 完稿與出版（2026-10-08）

17:21完成30／30章、5／5部模型審稿，130,923中文字、570次模型呼叫，仍待人工覆核。HTML／PDF已按最終稿出版：**474頁PDF、5,325個數學片段**。公式、字型、頁面結構、手機版與來源／成品SHA-256已檢查；未執行章內程式。

[閱讀HTML](http://localhost:8080/book5/html) ｜ [下載PDF](http://localhost:8080/book5/pdf) ｜ [五卷總索引](http://localhost:8080/books/)

成品在`published/`，版本核對見`../publication-status.json`，渲染層公式拼字修復記錄在`published/publication.json`。Canonical Markdown及完稿狀態未修改。

## 五部、三階段驗收

1. 張量、機率與學習問題。
2. 反向傳播與可驗證訓練：NumPy兩層網路、全部參數梯度核對。
3. 注意力與Transformer構件：多頭shape、mask、梯度與未來洩漏測試。
4. 小型語言模型訓練與推論：自足CPU decoder、合成日誌、next-token訓練、cache等價與保留集評估。
5. 適配、多模態與可稽核Agent：LoRA、量化、資料對齊、檢索與唯讀mock工具。

每部6章；每章正文最低3000、目標4500中文字；含導讀、附錄、解答總計不超過200000。公式、程式、英文及來源不充字數。每章至少一個完整小命題證明、兩個手算、自足CPU程式、正常／邊界／故障測試與四類習題解答。

## 工作架構

- 保留DeepSeek Flash、Qwen3.8 27B、gpt-5.6-sol、gpt-6-luna、gpt-6-sol五模型。
- 40個`v5_editor_*`邏輯角色，並非40個並行程序或OS沙箱；最多4章並行，DeepSeek／Qwen各1請求，OpenAI 2請求。
- 按角色×章／部隔離session；作者與原審稿者為不同模型。
- 精準修補必須唯一匹配、不重疊、通過结构檢查，並保留原輸出、失敗、備份、雜湊與用量。
- 不設總呼叫額度上限，仍保留1200秒timeout、12輪續跑及連續3輪無進展保護。
- 網頁通知持續寫SQLite，不會喚醒parent聊天或發桌面推播。

```bash
python3 -u continue_book5_review.py --output books/neural-transformers --max-passes 12 --timeout 1200
```

主流程日誌`runs/book5-editor.log`，監測日誌`runs/book5-notifications.log`。重啟前檢查程序與鎖，不重複啟動。

## 2026-10-07 停滯修復

第29章因長patch格式失敗而跑完12輪保護停止後，已完成主編定點修正並恢復原審稿者複審。其他29章保持不動。修後4454中文字，已完整閱讀且雜湊固定的mock快照斷言及8項獨立CPU核對通過；安全限制與證據見[修復紀錄](editorial/RECOVERY_20261007.md)。本次日誌`runs/book5-recovery.log`，最新進度仍以狀態檔／網頁為準。

## 2026-10-08 跨章修復

跨章審稿達3／5部後再次停滯，已定點修正15／16／17／26／30章並恢復原審稿者複審。第16章補完整多頭反向傳播；固定CPU快照的輸入／W／bias最大梯度差約1.37e-10，8項獨立核對通過，未執行PyTorch對照或訓練。其餘25章未動。詳見[跨章修復證據](editorial/CROSS_RECOVERY_20261008.md)，本次日誌`runs/book5-cross-recovery.log`。本機網頁及通知監測亦已恢復。

### 第16章參數量補註

同日16:49再次續跑，已補清dv=dh與一般dv的參數量公式，只改兩段正文，所有Python區塊未動。此次僅複審第16章，再自動接續跨章流程，日誌`runs/book5-parameter-note.log`；詳見[補註與測試證據](editorial/PARAMETER_NOTE_20261008.md)。

## 已驗證與尚未驗證

- `editorial/preflight.json`及`agents/*--preflight/`保留5個真實模型READY及provider/model事件證據。
- 主編獨立撰寫的`examples/neural_lab.py`在Python 3.10.12／NumPy 2.2.6、x86_64 CPU通過12項起步測試；紀錄`runs/book5-starter-tests.log`。僅測primitive及微小兩層網路，不代表完整Transformer完成、泛化或真實養殖有效。
- 全專案62項unit tests通過，`/tmp/book5-all-tests.log`；工作流測試使用mock，不冒充真實模型審稿。
- 前四卷211個章稿／主稿／狀態／圖片／出版檔案的SHA256基準保留在`editorial/prior-volumes-sha256.json`，啟動後已核對未改動。
- 來源取得範圍見`REFERENCES.md`：兩論文摘要與指定PyTorch API頁，不宣稱已讀完論文或所有網站。
- 只用既有模型API；SSH仍無法驗證遠端RAM／磁碟／GPU整體狀態，不視API閒置為硬體全空閒。未安裝、重啟遠端服務、安排GPU訓練或執行生成章稿程式。

本卷尚未完成，也尚無HTML／PDF出版成品。所有養殖資料為合成、Agent只讀mock，不控制實體設備。模型審稿、程式驗證與人工覆核分開記錄。
