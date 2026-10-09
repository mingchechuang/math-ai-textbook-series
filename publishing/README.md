# Volume I–V：HTML／PDF出版工具

以既有Markdown為來源，不修改章稿、不呼叫模型、不執行教材內Python。使用markdown-it、KaTeX與既有Chromium產生圖文版，避免為此安裝完整LaTeX工具鏈。這是輕量出版流程，不依賴Quarto。

## 五卷索引及最新版本核對（2026-10-08）

**總索引：`http://localhost:8080/books/`**，離線檔`books/index.html`。五卷各有HTML及PDF直連；索引內嵌既有Noto CJK子集與授權，不依賴本機中文字型或網路。既有首頁亦增加總索引連結。

本次已重新產生全部五卷，PDF頁數依序為116、384、361、373、474。先備份I–IV原出版目錄到各卷`backups/publication-20261008-190123/`；原稿、章稿及狀態雜湊未變，基準為`runs/all-volumes-prepublication-hashes.json`。

```bash
for v in 1 2 3 4 5; do
  node publishing/publish.mjs --volume "$v"
  publishing/.venv/bin/python publishing/validate.py --volume "$v"
done
python3 publishing/catalog.py
```

`catalog.py`核對完整來源清單、組合來源雜湊、主稿、最新status／word_count／封面版本、HTML與PDF的完整SHA-256、位元組數、PDF驗證紀錄及排版檢查。五卷全部通過才更新索引與`books/publication-status.json`；索引明示檢查時間，不將永久不變的新鮮度當保證。網站出版路由設定`Cache-Control: no-store`，20個索引／既有入口皆以HTTP取得檔案後再次核對完整SHA。離線索引相對連結、390px無水平溢出與內嵌中文字型均已檢查。

Volume V初次排版發現TeX拼字問題，只在渲染層修復並記入manifest：第11章`g^\\*`上標星號、第14章`G_A`之後的索引分組、第26章log與變數間空白，以及第11章星號方程標籤內的巢狀美元符號。失敗紀錄與暫存目錄保留，不修改Markdown、不忽略公式錯誤，也不執行教材程式。PDF抽查不是逐頁人工校對。

## 網站入口

- `/book`：原有教材頁，增加出版版連結。
- `/book/html`：HTML閱讀版。
- `/book/pdf`：PDF列印版。
- `/book/download/html`：下載可離線閱讀的單一HTML。
- `/book/volume-1.html`、`/book/volume-1.pdf`：同檔別名，支援HTML內的相對PDF連結。

HTML內嵌字型、數學排版CSS及SVG，不需CDN、外部字型或JavaScript。下載後可直接用瀏覽器開啟；若要從離線HTML的PDF連結開啟PDF，將`volume-1.pdf`放在同一資料夾。參考來源的外部連結當然仍需要網路。

PDF為A4，含章節目錄連結、PDF書籤與頁碼。版面長度會隨原稿變更，不固定宣稱頁數。生成結果目前為116頁。

## Volume II

同一出版工具可直接使用第二卷既有稿件：

```bash
node publishing/publish.mjs --volume 2
publishing/.venv/bin/python publishing/validate.py --volume 2
```

成品位於`books/modern-graphics/published/volume-2.html`及`volume-2.pdf`；網站入口為`/book2/html`、`/book2/pdf`與`/book2/download/html`，並有`/book2/volume-2.html`、`/book2/volume-2.pdf`別名。`/book2`自動顯示通過驗證且仍符合原稿／各章／圖片雜湊的出版連結。

第二卷本版為30章、111,551中文字、384頁PDF、775個PDF書籤、5,351個數學片段。支援美元及LaTeX括號式數學界定符；第3章一處跨行行內公式只在排版時合併空白，不改Markdown。圖文、公式與字型離線內嵌，原稿未變，也未執行教材程式。第二卷生成不重建或覆寫第一卷成品。

## Volume III／IV

```bash
node publishing/publish.mjs --volume 3
publishing/.venv/bin/python publishing/validate.py --volume 3
node publishing/publish.mjs --volume 4
publishing/.venv/bin/python publishing/validate.py --volume 4
```

2026-10-06出版結果：

| 卷 | 成品目錄 | PDF頁數 | 數學片段 | PDF書籤 |
|---|---|---:|---:|---:|
| III | `books/field-simulation/published/` | 361 | 6,227 | 768 |
| IV | `books/calculus-analysis/published/` | 373 | 8,642 | 786 |

檔名分別為`volume-3.html/pdf`與`volume-4.html/pdf`。網站支援`/book3/html`、`/book3/pdf`、`/book3/download/html`及相同的`/book4/…`入口，並提供相對PDF連結所需的同名別名。兩卷主頁有新鮮度檢查後的出版連結；通知SQLite亦記錄publication_ready。

沿用既有依賴、字型及停用GPU的Chromium，未呼叫模型或執行教材程式。所有章稿、組合Markdown及狀態檔、第一／二卷原出版檔案的SHA256均保持不變，基準為`runs/book34-prepublication-hashes.json`。排版驗證、390px螢幕、無外部資源請求、內部連結、公式解析、PDF空白頁／缺字標記及章節書籤均通過；已抽看公式PDF頁與窄螢幕截圖，不宣稱逐頁人工校對。全專案57項unit tests通過，出版連結測試涵蓋II／III／IV。

只在渲染層修正並記錄於各卷`publication.json.renderer_normalizations`：inline TeX首尾空白及跨行空白；第三卷第19章一處誤用backtick的公式結尾；第三卷第26章被退格控制字元破壞的boldsymbol命令（同章小結有完整原式可核對）；第四卷第26章定義編號的prime。圖片說明亦經數學渲染。原稿不改動、不默默忽略公式錯誤；失敗暫存目錄保留供稽核。

## 重建

本專案已建立本機依賴，不需重新安裝。由專案根目錄執行：

```bash
python3 book_editor.py --build-only
node publishing/publish.mjs
publishing/.venv/bin/python publishing/validate.py
python3 book_editor.py --build-only
```

第一步重組主稿；最後一步在`/book`加入與本稿雜湊一致的出版版連結。內容修改後，舊出版版連結會自動隱藏，需重建。發布時先在暫存目錄產生並檢查，再更新成品，最後更新manifest；失敗時暫存診斷保留。

在新環境才需要：

```bash
npm ci --prefix publishing
python3 -m venv publishing/.venv
publishing/.venv/bin/pip install -r publishing/requirements.txt
```

另需Chromium，預設路徑`/snap/bin/chromium`，可用環境變數`BOOK_CHROMIUM`指定。不得把瀏覽器安裝、GPU運算或遠端主機變更視為出版步驟的隱含授權；本次沿用本機既有瀏覽器，停用GPU渲染。

## 字型與授權

`fonts/NotoSansCJKtc-Regular.otf`來自：
https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf

授權：`fonts/OFL.txt`（SIL OFL 1.1）。出版時依全文、圖中文字及UI做WOFF2子集化並內嵌。未複製Windows商業字型。KaTeX採MIT授權；HTML包含兩份授權文字。

## 成品與驗證

`books/linear-algebra-aquaculture/published/`：

- `volume-1.html`、`volume-1.pdf`：交付成品。
- `publication.json`：原稿及成品雜湊、字數、公式數、字型、連結與版面檢查。
- `validation.json`：PDF頁數、章節書籤頁碼、空白頁及缺字標記檢查。
- `math-errors.json`：必須為空清單才發布。
- `preview-cover.png`、`preview-mobile.png`：預覽截圖。

目前2199個公式片段由KaTeX預先排版，包含行內公式。印刷時只縮放超出可用寬度的公式；窄螢幕的長公式可以在公式區水平捲動，長網址自動換行。原稿中`\\@`僅在出版轉換時排成字面的矩陣乘法運算符`@`，不改原稿。

驗證包含桌面／390px窄螢幕、中文內嵌字型、20章、章節連結、公式解析、PDF書籤與缺字標記。抽看封面、公式及習題頁；不宣稱逐頁人工校對。這些是排版檢查，不是重新證明教材的數學或養殖內容。
