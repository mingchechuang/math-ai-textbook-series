# 數學、AI與模擬教材｜Volumes I–V

五卷繁體中文教材，涵蓋線性代數、電腦圖學、場模擬、微積分與分析、神經網路與Transformer。保存最終Markdown、可離線閱讀的HTML／PDF，以及多模型寫作、審稿與修訂紀錄。

> 五卷已通過模型逐章及跨章審稿；這不等於人工數學審定或全面程式執行。所有養殖案例為合成教學資料，沒有實際設備控制或作業閾值保證。

## 教材

| 卷 | 主題 | 章數 | 中文字數 | HTML | PDF |
|---|---|---:|---:|---|---|
| I | 線性代數與生成式AI應用 | 20 | 50,062 | [HTML](books/linear-algebra-aquaculture/published/volume-1.html) | [116頁](books/linear-algebra-aquaculture/published/volume-1.pdf) |
| II | 現代電腦圖學 | 30 | 111,551 | [HTML](books/modern-graphics/published/volume-2.html) | [384頁](books/modern-graphics/published/volume-2.pdf) |
| III | 場論、場模擬與相變 | 30 | 123,889 | [HTML](books/field-simulation/published/volume-3.html) | [361頁](books/field-simulation/published/volume-3.pdf) |
| IV | 微積分與數學分析 | 30 | 117,879 | [HTML](books/calculus-analysis/published/volume-4.html) | [373頁](books/calculus-analysis/published/volume-4.pdf) |
| V | 神經網路、深度學習與Transformer | 30 | 130,923 | [HTML](books/neural-transformers/published/volume-5.html) | [474頁](books/neural-transformers/published/volume-5.pdf) |

GitHub檔案頁不會直接執行HTML：請下載／clone後開啟[總索引](books/index.html)，或在本repo根目錄執行：

```bash
python3 -m http.server 8080
```

再開啟`http://localhost:8080/books/`。HTML內嵌中文字型、公式及SVG，不依賴CDN；PDF置於同資料夾即可使用內頁下載連結。未自動啟用GitHub Pages。

## 討論與修訂過程

- [`discussions/`](discussions/README.md)：五卷共3,925筆已完成的user／assistant文字訊息，依卷、角色／任務及原始事件檔名保留。含寫作提示、草稿、審稿與修訂討論，並非完整原始session備份。
- `books/*/editorial/`：原審稿輸出、patch候選、核對及修復說明；被拒絕或後來取代的內容也保留，不能將每個候選都當作最終教材。
- `books/*/status.json`、`usage.json`：完稿狀態及用量紀錄。用量估算不等於實際OAuth帳單。
- `books/*/examples/`：獨立檢查器及SHA固定快照驗證；不代表所有章稿程式都已執行。

排除原始session、thinking／system／工具事件、私人設定、憑證、備份副本、暫存失敗出版目錄、虛擬環境及套件目錄。未收錄交易模擬或台灣收入討論資料。教材目錄沒有此助理主對話的完整逐字歷史，故不宣稱包含。完整匯出範圍見[`EXPORT_SUMMARY.json`](EXPORT_SUMMARY.json)。

## 最新版本及重建

2026-10-08重建全部五卷，原稿未改動。來源清單、原稿及HTML／PDF的SHA-256、版本時間與頁數見[`books/publication-status.json`](books/publication-status.json)；每卷`published/publication.json`保存渲染層排版修復紀錄，`validation.json`保存PDF結構檢查。這是帶時間的驗證快照，不是保證日後修改仍同步。

出版工具及依賴版本見[`publishing/README.md`](publishing/README.md)。套件與Chromium不隨repo複製；依文件自行準備環境，再執行：

```bash
for v in 1 2 3 4 5; do
  node publishing/publish.mjs --volume "$v"
  publishing/.venv/bin/python publishing/validate.py --volume "$v"
done
python3 publishing/catalog.py
```

編輯／續跑程式和共用基礎設施原始碼也收錄，但不附模型登入資料、API服務設定或交易資料庫。不要將出版指令當作執行章內程式、開啟推論或控制遠端服務的授權。

## 授權與公開範圍

初始GitHub上傳預設為私人repo；尚未選定本教材／專案的對外開源授權，不應自行假定可再散布。Noto Sans CJK字型依SIL OFL 1.1（`publishing/fonts/OFL.txt`），HTML含字型及KaTeX授權文字；第三方來源另依原授權。對外公開前應再次人工檢視讨论內容、來源使用及授權。
