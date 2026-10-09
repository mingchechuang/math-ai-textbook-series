# 2026-10-08 跨章停滯修復

起點為542次呼叫、第22修補世代、29／30章及3／5部通過；10月7日14:06連續3輪無進展而停止。使用者要求繼續。當次檢查編輯、通知及本機網頁服務皆未執行；只恢復本專案本機程序，未改動遠端服務。

## 備份及定點修正

`backups/cross-manual-20261008-105431/`保留15／16／17／26／30章、主稿、狀態及全章雜湊。其餘25章及前四卷211個保護檔案經SHA256核對未變。

- 第15章：allowed已有嚴格布林檢查，保留。補key_is_valid、query_is_valid、valid_targets的dtype驗證及四入口故障測試，避免把數值bias靜默當bool。
- 第16章：保留既有正確拆合證明、手算、章號、mask前導軸與參數總數。新增完整forward快照及backward，包含三路dX、四個權重與全部bias、固定mask的softmax VJP；value每頭寬dv及總寬H*dv一致。補非bool正整數、有限值、溢位、空輸入與mask錯誤處理。修正未訓練例子的能力措辭、columns投影方向、H=1等價、TypeError文字及PyTorch的dtype／權重轉置／mask轉接。
- 第17章：只修來源尾註的證據範圍，不把題目附註當作者自行查閱證據，保留RoPE公式程式。
- 第26章：量化「可能」有誤差，格點值可精確還原；不改證明或程式。
- 第30章：步長context的輸入窗口不重疊，位移標籤可共用邊界token；仍先按文件切分。不改模型、訓練loop或建窗介面。

## CPU驗證範圍

完整閱讀後，僅執行以下SHA固定快照的選定NumPy片段：

- 第15章：`5904991a57ca1ceac956919260923a49028be8fa178b92d49eecb0f39d4863a5`
- 第16章：`62565894565056ac6bf70042506d5c08811eb47e27ae6213b10ac7af947da8ee`

快照在`editorial/cross-recovery-20261008/`，檢查器為`examples/check_cross_recovery_20261008.py`。若快照改變則拒絕執行，不讀後續模型稿件。刻意不執行反例片段及選用PyTorch對照，也不執行17／26／30章或啟動訓練。

Python3.10.12／NumPy2.2.6／x86_64 CPU下，8項獨立核對通過：完整輸入／W／bias梯度、mask各入口、拆分伴隨、未來洩漏、forward快照、單頭對照、量化格點及窗口界線。含無mask、causal及dv!=dh的有限差分最大誤差為1.365083046600546e-10。紀錄`runs/book5-cross-recovery-tests.log`。這不是全域數學證明、全書程式驗證或PyTorch實測。

工作流unit tests另見`/tmp/book5-cross-recovery-all-tests.log`，使用mock模型／暫存稿件，不冒充實際審稿。

## 原審稿者複審與接續

`recover_book5_cross.py`先將五個修改章節標為待審，逐一交原審稿者複審，不由修稿者自證。複審阶段不改任何章稿；API失敗時不沿用修改前批准。完成後自動接續逐章與跨章流程，保留timeout、並行限制、12輪與無進展保護。

```bash
python3 -u continue_book5_review.py --output books/neural-transformers --recover-cross-stalled --max-passes 12 --timeout 1200
```

11:04恢復執行，從第15章原審稿者開始。日誌`runs/book5-cross-recovery.log`；網頁`/book5`與通知已恢复、HTTP200。通知只寫web／SQLite，不喚醒parent聊天。

再次讀compute profile並做只讀健康檢查，見`runs/book5-cross-recovery-health.log`。僅用既有推論API；SSH未能驗證的整體RAM／磁碟／GPU狀態仍標未知。沒有安裝、重啟遠端服務或分派新增GPU工作。
