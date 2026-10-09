# 共同契約

本卷是Volume V：神經網路、深度學習與Transformer的數學及實作。Volume I與IV為矩陣／微分先備，第一部另橋接機率及學習評估；不得虛構前四卷章號或已有可匯入模組。30章，每章最低3000、目標4500中文字，全卷含前言、解答、附錄最多200000；公式、程式、英文與參考來源不充字數。
每章至少完整證明一個小命題、兩個逐步手算、自足CPU程式、正常／邊界／故障測試、手算／程式／反例／整合四類習題及完整解答。未證明的進階定理標引用與条件；數值測試不代替證明。無執行紀錄只能寫預期，不能寫已通過或已訓練；不得捏造時間、收斂率、設備與實測指標。
矩陣橫列row、縱行column；column向量是直向量。批次X採(B,Din)，W採(Din,Dout)，Y=XW+b；這是batch中每筆樣本橫向儲存，與微分的column表示不矛盾。梯度與參數同shape；NumPy一維.T不會改shape。矩陣微分df=tr(G^T dX)，廣播的反向需sum到原shape，共享參數及重複索引的梯度必須累加。
張量固定B=batch、T=sequence、D=model feature、H=head、dh=D/H；Q,K形狀(B,H,Tq,dh)/(B,H,Tk,dh)，V=(B,H,Tk,dv)，scores=(B,H,Tq,Tk)，softmax沿最後key軸。reshape不是transpose。每次reduction寫明軸、sum／mean及mask；有效token平均只除一次，微批梯度累積按有效token數加權。
穩定softmax先減max，CE直接由logits與logsumexp計算，不以先取softmax再log或任意加epsilon假裝精確。概率支撐、零機率及非有限logits需定義策略。全遮罩query列核心實作明確拒絕；若採零輸出另列規則與測試，不讓NaN靜默流入。遮罩在softmax前施加。本卷布林True=允許注意；框架API可能相反，轉接時按具體版本核對。padding的key遮罩不等於loss遮罩。
causal mask以key絕對位置<=query絕對位置判斷；KV cache非方形不能無條件tril(Tq,Tk)。prefill／incremental需位置與cache長度一致、同權重及相同模式；eval關閉dropout，SDPA即使外層eval也須傳dropout_p=0。不得把attention權重直接解讀為因果解釋。
LayerNorm/RMSNorm沿feature軸；RMSNorm不減均值，epsilon位置寫清。pre-norm／post-norm分明，殘差兩路梯度累加。RoPE固定相鄰偶奇配對或明示另一約定，dh偶數；測試相對位置內積及cache位置偏移。Adam偏差修正步數從1開始，AdamW與L2不可混同；裁切為所有參數拼接的全域範數時，不能每層各自clip冒充。
資料切分先於擬合詞表／標準化及窗口建立；同文件或同序列的重疊窗口不得跨集合。合成資料保留group／time／seed與生成规则；不得用test調參。next-token輸入target位移與PAD忽略清楚，perplexity=exp(有效token總NLL/有效token數)，不平均不同token數批次的perplexity。有限seed實驗不代表統計顯著或跨設備逐位重現。
前三部核心標準庫／NumPy；完整Transformer可用PyTorch CPU，但章內自足列模型、資料、loss、optimizer及loop，不下载權重、語料或tokenizer，不安裝套件、不啟用GPU。不載入pickle或不可信checkpoint，不執行shell及網路工具。大型預訓練、分散式及CUDA效率只列範圍與理論，不作實際資源安排。
至少三階段驗收：NumPy兩層網路全部參數梯度核對與小資料訓練；多頭attention的形狀／mask／梯度；小型decoder-only模型在合成日誌上的訓練／生成／評估。模型生成的程式不由寫作流程自動執行，獨立主編起步測試另存；框架版本、CPU與dtype若未檢查就明示未知。
LoRA在本卷W(Din,Dout)約定採A(Din,r)、B(r,Dout)、DeltaW=(alpha/r)AB；其他文献转置表示可用但須轉接，不把數學rank上界當品質保證。多模態對齊與來源支持、模型流暢度與正確性、安全性分開。
養殖、感測、文件及操作日誌全部合成，無真實操作閾值；Agent只讀mock資料，工具schema與allowlist由程式控制，檢索文本、模型輸出都不是權限授予。不得控制泵浦、曝氣、投餌、加藥、財務或外部設備。引用須可定位，不存在來源就拒答；不用agent取代現場專業判斷。
公式用$...$及$$...$$，行內不跨行且內側不留多餘空白；LaTeX命令保留實際反斜線，不把JSON换行誤讀成nu。審稿先重算再提最小修法，不盲從前次錯誤審稿。程式不得留pass／TODO／omitted、未定義模組或虛構測試成功；圖片只用提供的本機SVG。
