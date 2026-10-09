# 2026-10-08 第16章參數量條件補註

第27世代、558次呼叫後，僅第16章待逐章通過。原審稿者已核對完整反向、程式及解答，阻擋項是參數量公式沒有明示dv=dh；長patch再次被安全檢查拒絕。

## 修正

先備份到`backups/parameter-note-20261008-164740/`。只改第16章兩段正文：

- 基本shape與手算採dv=dh預設。
- 預設參數量3D²+3D+D*Dout+Dout；一般dv為2D²+2D+(D+1)H*dv+(H*dv+1)Dout。只有總投影寬度固定時，改變頭數不改變參數總數。

已比對所有Python區塊與先前已閱讀的SHA固定快照，完全相同；其餘29章雜湊亦未變。新增參數逐項計數對照，連同原NumPy檢查共9項通過，紀錄`runs/book5-parameter-note-tests.log`。未執行PyTorch、未啟動訓練。專案測試另見`/tmp/book5-parameter-note-all-tests.log`。

## 只複審必要章節

續跑入口新增`--review-chapters`，限於已修訂章節，不重審另外四個已通過的修改章。16:49以原審稿者開始複審：

```bash
python3 -u continue_book5_review.py --output books/neural-transformers --recover-cross-stalled --review-chapters 16 --max-passes 12 --timeout 1200
```

日誌`runs/book5-parameter-note.log`，通過後自動續跑跨章審查。仍保留原模型、timeout、並行、12輪及無進展保護。既有通知監測與網頁正常；只用既有推論API，健康檢查見`runs/book5-parameter-note-health.log`，SSH無法驗證的主機整體資源維持未知，沒有新增GPU配置或修改遠端服務。
