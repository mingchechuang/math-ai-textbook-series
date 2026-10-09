# Volume IV｜微積分與數學分析的線性代數方法

2026-10-06 18:39完成30章／5部模型審稿，共117,879中文字、487次模型呼叫。已產生[離線HTML](http://localhost:8080/book4/html)及[373頁PDF](http://localhost:8080/book4/pdf)，位於`published/volume-4.html`與`published/volume-4.pdf`。8,642個數學片段、786個書籤，排版驗證見`published/validation.json`。Markdown仍為主稿，排版未修改原稿或執行教材程式；模型審稿不能取代人工數學覆核。

[進度與通知](http://localhost:8080/book4) ｜ [30章目錄](http://localhost:8080/book4/toc) ｜ [通知紀錄](http://localhost:8080/book4/notifications)。

## 五部30章

1. 極限、範數與分析基礎。
2. 導數作為線性映射。
3. 二階近似與最佳化。
4. 積分、幾何與極限交換。
5. 微分方程、算子與整合專題。

每章目標4500、最低3000中文字，主體約135,000字；全卷含導讀、附錄、解答最多200,000字。公式、程式、英文、標點及來源不充字數。每章至少一個完整小命題證明、兩個手算例、自足CPU實作、正常／故障測試、四類習題與解答。有限取樣不是定理證明，明列域、正則性、必要／充分及局部／全域的界線。

## 模型、會話及修補

沿用deepseek-flash、unsloth/Qwen3.8-27B-NVFP4、gpt-5.6-sol、gpt-6-luna、gpt-6-sol，各8個角色。20作者、20不同模型審稿者輪派30章。五個模型實際READY連線測試已通過，adapter核對實際provider/model及正常結束；記錄`editorial/preflight.json`及`runs/book4-preflight.log`。

- 40是邏輯角色數，不是40個並行工作或OS沙箱。
- 最多4個章節工作，DeepSeek／Qwen各1請求，OpenAI最多2。
- 本卷角色ID為`v4_editor_00`～`v4_editor_39`；session與工作目錄依任務另分，例如`v4_editor_00--ch01`和`v4_editor_00--ch21`。同一章修訂可延續上下文，不把前一章歷史帶入另一章。跨部審稿及preflight亦分開。
- 主編patch使用純文字OLD／NEW分隔區塊，不再要求長LaTeX／Python以JSON跳脫。仍驗證原文唯一、片段互不重疊、長度及修後結構／Python語法；最多5個patch一批，失敗有限重試，不盲目採用模型修補。
- 原審稿者複審，之後做五部跨章審查。批准仍是模型結果，不代表已執行所有教材程式。
- 沿用不設總呼叫額度上限設定；每次1200秒、最多12輪自動續跑，連續3輪無內容／審查進展停止並留待辦，不無限重播。

## 驗證與來源

`examples/analysis_lab.py`是主編另写的標準庫CPU實驗，9項測試通過：Jacobian、JVP/VJP對偶、餘項、偏導不足的反例、隱函數、Gram面積、行列式取向、Jordan半群及非法輸入。這不等於所有章稿已執行或數學定理已由實驗證明。全專案53項unit tests於啟動前通過。

來源見REFERENCES.md。已取得作者入口、課程概要與API頁面，不聲稱完整教材PDF及所有定理均已閱讀／查證。核心採NumPy CPU小型例子，SciPy可選、JAX僅延伸；不自動安裝、訓練或配置GPU。資料均合成，agent不操作真實設備。

Qwen既有API健康正常，啟動前running／waiting／KV使用量皆0；SSH仍拒絕金鑰登入，遠端RAM、磁碟及整體GPU資源未確認。只使用既有推論API，不改動遠端主機或服務。

## 2026-10-06 停滯修復

第09、19、29章在原精準patch流程中反覆失敗後，已啟動定點／分段修復與原審稿者複審，不只重啟相同嘗試。第09章另有已閱讀及SHA256固定的CPU測試快照；詳見[修復證據與範圍](editorial/RECOVERY_20261006.md)。本次日誌`runs/book4-recovery.log`，仍以`status.json`及網頁為最新狀態。

### 跨章複審續修

同日第二次停滯後，已定點修正16／25／26章；獨立重算確認Picard部分舊審稿指控有誤，並保留CPU測試的真實失敗與修正證據。詳見[跨章修復紀錄](editorial/CROSS_RECOVERY_20261006.md)。目前續跑日誌為`runs/book4-cross-recovery.log`；最新進度仍以狀態檔與網頁為準。

## 執行與通知

```bash
python3 book4_editor.py --init-only
python3 -u continue_book4_review.py --output books/calculus-analysis --max-passes 12 --timeout 1200
python3 -u book2_notifications.py --output books/calculus-analysis --base-route /book4 --label 'Volume IV' --interval 5
python3 books/calculus-analysis/examples/analysis_lab.py
```

日誌：`runs/book4-editor.log`、`runs/book4-notifications.log`、`runs/book4-preflight.log`、`runs/book4-starter-tests.log`。通知持久化到`notifications.sqlite`，網頁每5秒更新；不主動通知或喚醒parent聊天session。nohup不保證主機重開後自動恢復。

前三卷章稿、組合稿、狀態與既有出版物的SHA256快照位於`editorial/prior-volumes-sha256.json`，啟動後已核對未變。本卷工作目錄独立，不順帶出版第三卷或開始第五卷。
