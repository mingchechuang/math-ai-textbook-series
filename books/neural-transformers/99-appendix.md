# 附錄：形狀、驗收與證據

- 批次X=(B,Din)，權重W=(Din,Dout)，輸出Y=XW+b。
- 多頭Q=(B,H,Tq,dh)、K=(B,H,Tk,dh)、V=(B,H,Tk,dv)，softmax沿key軸。
- 本卷布林mask的True表示允許；框架轉接不可假設同義。
- 因果遮罩以絕對位置判斷，cache的矩形遮罩不能隨意套用左上三角。
- 有效token平均只除一次，微批累積依有效token數加權。
- 先分資料，再擬合詞表與標準化，再切窗口；test不參與調參。
- LoRA採W+alpha/r AB，A=(Din,r)、B=(r,Dout)。

`examples/neural_lab.py`是獨立NumPy CPU起步驗證，只測小型primitive與兩層網路，不代表完整Transformer已完成或已訓練。逐章程式、後續選定快照的驗證及人工審定必須分開記錄。

保留章稿、審稿、原始輸出、呼叫用量、修補備份及雜湊。四十個邏輯角色不是四十個並行程序，也不是作業系統隔離沙箱；最多四章並行，同一角色的不同章使用分開session。
