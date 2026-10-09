# 啟動時來源取得紀錄（2026-10-03）

透過公開頁面取得FiPy有限體積、FEniCSx Poisson、PETSc KSP、SciPy sparse.linalg、NumPy FFT、FiPy Cahn–Hilliard及phase.simple文件。來源列表及URL見REFERENCES.md。

- 搜尋工具回傳紀錄：`murdtmapm98nd1`、`murdu25frqkscx`。
- 局部核對：FiPy cell-centered FVM的面通量／自然無通量邊界；PETSc對預條件與真殘差的區分；FiPy Cahn–Hilliard的守恆序參量與雙井自由能。
- FiPy Cahn–Hilliard例使用0至1的序參量與不同能量係數。本卷預設-1至1雙井；換變數須同步推導，不直接複製其參數。
- PFHub benchmark1網址取得的是初始條件／繪圖輔助內容，不足以證明已讀完整基準規格，故未加入核心已核對來源。
- 未執行來源程式，未逐頁閱讀157KB的PETSc全文；不聲稱所有章節公式已獨立查證。
- 來源頁面可能對應比本機更新的NumPy／SciPy；章稿需標示所用API條件，不以文件最新版號冒充本機環境。
