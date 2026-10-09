# 從生成式 AI 到智慧水產養殖：線性代數、 多模態與 Agent 實作

- 本輪修訂與模型審稿已完成：20／20章、5／5部通過，共50,062中文字；仍待人工審定。詳見[結案紀錄](editorial/VOLUME_I_MODEL_REVIEW_COMPLETE.md)。
- 本教材定位為系列Volume I；[四大正式主題與後續Volume II～V規劃](../SERIES_ROADMAP.md)。
- 主稿：`book.md`；分章：`chapters/01.md`～`20.md`。
- 圖文版：[HTML](published/volume-1.html)／[PDF](published/volume-1.pdf)。網站原有`/book`已加入閱讀與下載連結；[重建及驗證說明](../../publishing/README.md)。
- 目錄與章節契約：`TOC.md`；共用符號：`STYLE_GUIDE.md`。
- 原創SVG：`figures/`；合成資料：`data/`；基礎驗證程式：`examples/`。
- 40位模型編輯的session及事件：`agents/`；審稿與修訂：`editorial/`。
- 完成20章後做5部跨章一致性審查，仍需人工審定。
- 字數規則：正文中文字元，不含標點、英文、公式、程式及章末參考來源；目標45,000～55,000。依使用者最新指示允許超字，不以章節2600字或全書55000字上限拒稿。
- 定點修訂與驗證界線：`editorial/TARGETED_REPAIR_NOTES.md`、`editorial/CROSS_REPAIR_NOTES.md`；數值核對：`examples/check_chapter_repairs.py`、`examples/check_cross_chapter_repairs.py`。
- 定點修訂後只審不重寫：`python3 book_editor.py --repair --review-only --wait-discussion`；拒稿保留意見，全部章節通過後重跑五部跨章審查。
- 合成資料不能作真實養殖閾值；範例不連接現場設備。
- 執行：`python3 book_editor.py --output /home/mingche/ai_workspace/marketsim/books/linear-algebra-aquaculture --wait-discussion`；已存的成功階段不重跑。
- `python3 book_editor.py --output /home/mingche/ai_workspace/marketsim/books/linear-algebra-aquaculture --build-only`只重組既有稿件，不呼叫模型。
- 失敗可用相同命令恢復；每次重試保留之前事件與token紀錄，不自行換模型。
