"""Volume II正式章級契約；40個角色沿用五模型，不改Volume I。"""
from book_spec import MODELS

TITLE = 'Volume II｜現代電腦圖學：2D／3D繪圖、建模、材質、光線追蹤與動畫'
ROOT = 'books/modern-graphics'
MAX_CHARACTERS = 200_000
TARGET_PER_CHAPTER = 4500
MIN_CHARACTERS = 3000
PARTS = ['第一部｜幾何、座標與成像', '第二部｜網格、曲線與建模',
         '第三部｜材質、貼圖與著色', '第四部｜光線追蹤與光傳輸', '第五部｜動畫與養殖數位分身']
CHAPTERS = [
 ('從像素到養殖場數位分身',0,'圖學管線、離散影像、解析度、可見性與合成資料限制','用標準庫輸出16×16 PPM；像素索引與RGB範圍測試'),
 ('向量、外積與幾何判定',0,'內積、外積、面積、法線、平面、退化三角形與容差','實作dot/cross/normalize及點到平面距離，測零長向量'),
 ('2D／3D變換與齊次座標',0,'旋轉、縮放、平移、點w=1與方向w=0、非交換與反變換','實作4×4 TRS；手算次序差異與逆轉換測試'),
 ('相機、座標系與視圖矩陣',0,'右手系、look-at、外參、主動被動轉換、近平面','把池體頂點轉到相機座標，檢查平行up向量與朝向'),
 ('透視投影、裁切與深度',0,'透視／正交、clip與NDC、透視除法、近平面裁切、深度精度','推導OpenGL式右手投影並測near/far；勿混用WebGPU深度'),
 ('三角形光柵化與插值',0,'邊函數、重心座標、top-left規則、z-buffer、透視正確插值','CPU畫三角形，測邊界共享、遮擋與斜面UV'),
 ('三角網格與拓撲資料結構',1,'頂點索引、邊鄰接、繞序、流形與邊界、退化面','建立低面数池體網格並檢查非法索引、法線方向與邊界'),
 ('法線、切線與逆轉置',1,'幾何／著色法線、非均勻縮放、逆轉置、TBN與鏡射','驗證變換後法線仍垂直切向；奇異縮放拒絕'),
 ('Bézier曲線與樣條',1,'仿射組合、Bernstein基底、de Casteljau、導數、連續性','用控制點做魚體輪廓，端點、凸包與切線測試'),
 ('參數曲面與曲面離散化',1,'曲面偏導、切平面、法線、網格採樣與接縫','旋轉曲面生成簡化魚體；極點退化與接縫測試'),
 ('程序化建模與幾何品質',1,'掃掠、局部框架、LOD、包圍體、自交與尺度','建立魚身／魚鰭與池體的參數化模型，輸出OBJ並檢查'),
 ('資產交換與可重現場景',1,'場景圖、材質指派、單位、座標轉換、OBJ與glTF概念','建立JSON場景清單及簡單OBJ讀寫；Blender僅作可選檢視'),
 ('色彩、線性光與影像取樣',2,'線性RGB、sRGB轉換、alpha、雙線性取樣、混疊、mipmap','實作sRGB往返與checker貼圖取樣；區分色彩與法線資料'),
 ('UV參數化與貼圖座標',2,'三角形UV、重心插值、接縫、wrap／clamp、微分足跡','把checker貼上魚體；測接縫、上下翻轉及透視插值'),
 ('局部照明與BRDF基礎',2,'Lambert、餘弦、立體角、輻射度量、Phong與物理模型差別','CPU計算Lambert球面，測背面光源與能量尺度'),
 ('物理材質與微表面模型',2,'反照率、粗糙度、金屬度、Fresnel、NDF／幾何項與能量','手算簡化微表面BRDF；明示粗糙度映射及掠射限制'),
 ('法線貼圖與表面細節',2,'切線空間、法線解碼、TBN正交化、法線圖非sRGB、凹凸與位移','測中性法線圖、UV鏡射手性與非均勻縮放'),
 ('著色器與GPU管線橋接',2,'頂點／片段階段、插值、uniform、精度、buffer布局','用CPU參考輸出對照GLSL偽最小流程；不要求GPU安裝或實測'),
 ('射線、交點與數值穩健性',3,'射線參數、球／平面／三角形求交、t範圍、自相交','建立小型CPU ray caster，測相切、平行、內部出射與epsilon'),
 ('包圍盒與BVH加速',3,'slab測試、零方向、AABB、樹遍歷、分割策略與成本','比較暴力求交與小型BVH結果；效能只報實測或明標預期'),
 ('光傳輸積分與Monte Carlo',3,'積分／期望橋接、PDF、無偏估計、變異數、半球取樣','估計Lambert半球積分；多seed誤差與零PDF處理'),
 ('路徑追蹤與重要性取樣',3,'渲染方程、throughput、光源取樣、俄羅斯輪盤、MIS入門','可重現的小型CPU path tracer設計；低解析度固定預算'),
 ('水面反射、折射與介質',3,'Snell、全反射、Fresnel、法線朝向、吸收與Beer–Lambert','手算入射角與臨界角；合成水面示例非完整水下光學模型'),
 ('渲染誤差、降噪與效能驗證',3,'參考圖、MSE／偏差、取樣噪聲、降噪假細節、時間記錄','制定跨seed對照與計時實驗，未執行不得虛構FPS或品質'),
 ('場景階層與關鍵影格',4,'局部／世界TRS、父子矩陣、時間參數、插值與尺度','魚體與魚鰭階層動畫；固定時間步及座標往返測試'),
 ('四元數、旋轉與SLERP',4,'Euler角局限、單位四元數、乘法、雙覆蓋、短弧插值','使用(w,x,y,z)手算90度旋轉，測q與-q及近共線SLERP'),
 ('骨架、綁定姿勢與蒙皮',4,'骨骼空間、inverse bind、LBS、權重和、糖紙效應','兩骨骼魚尾蒙皮，bind pose還原測試；雙四元數僅延伸'),
 ('逆向運動學與限制',4,'FK、位置Jacobian、阻尼最小平方、關節界限與不可達目標','兩連桿IK，有限差分Jacobian與停止準則，勿宣稱全域解'),
 ('魚群動畫與合成資料標註',4,'局部規則、時間積分、碰撞近似、相機／深度／ID標註','合成魚群短序列與資料清單；動畫行為不等於真實生態'),
 ('整合專題：可稽核養殖數位分身',4,'場景→渲染→動畫→標註→多模態觀測→agent唯讀查詢','完整目錄、執行順序、驗收與失效清單；區分模擬、實測與控制'),
]
SOURCES = [
 ('G1','PBRT 4：Transformations','https://pbr-book.org/4ed/Geometry_and_Transformations/Transformations'),
 ('G2','PBRT 4：Reflection Models','https://pbr-book.org/4ed/Reflection_Models'),
 ('G3','PBRT 4：The Light Transport Equation','https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/The_Light_Transport_Equation'),
 ('G4','Ray Tracing in One Weekend','https://raytracing.github.io/books/RayTracingInOneWeekend.html'),
 ('G5','LearnOpenGL：Transformations','https://learnopengl.com/Getting-started/Transformations'),
 ('G6','Blender Manual：Skinning Introduction','https://docs.blender.org/manual/en/latest/animation/armatures/skinning/introduction.html'),
 ('G7','NumPy線性代數參考','https://numpy.org/doc/stable/reference/routines.linalg.html'),
 ('G8','Khronos glTF 2.0規格','https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html'),
]
SOURCE_NOTES = '''G1～G6已取得頁面內容作為候選參考，不表示逐條論點已外部驗證。G7沿用第一卷來源，G8本次尚未逐節查核。SciPy SLERP頁面本次取用失敗，不列為已查閱來源。讀者可按URL回查；不得仿抄來源的長段文字、杜撰頁碼或聲稱已跑官方範例。'''
CONVENTIONS = '''繁體中文；先備為Volume I向量、矩陣、內積與基本Python。微積分、機率、光度學必須在用到時橋接。
本卷專注圖學能力，不用重複AI宣傳或安全口號湊字；養殖案例貫穿但不能替代數學與工程內容。
全書右手世界系：+X向右、+Y向上、+Z由畫面向觀者；相機局部看向-Z，up為+Y。
數學向量採column vector（直向），位置p_h=(x,y,z,1)^T，方向v_h=(x,y,z,0)^T。
矩陣位置用row/column或橫列/縱行，避免中文行列歧義。M_world=M_parent M_local；p_world=M_world p_local。
主動變換、右手正角，角度運算用弧度；三角形前面按外向法線看為逆時針。T R S由右先作用，NumPy陣列儲存順序不等於數學乘法慣例。
相機管線p_clip=P V M p；主線採OpenGL式NDC z∈[-1,1]、near>0、far>near；深度buffer=(z_ndc+1)/2。其他API要明說轉換，不能混用。
像素原點左上，中心(u+0.5,v+0.5)，寬W高H；影像Y向下與世界Y向上分開。UV約定v向上，讀影像時明寫翻轉。
法線用線性部分A的A^(-T)並重新正規化，要求A可逆；鏡射需交代繞序／手性。
色彩計算在線性RGB，輸出sRGB要做分段轉換；normal/depth等資料貼圖不套sRGB。所有數值場景使用合成資料。
長度公尺、時間秒；數值epsilon按場景尺度設定，不能充當物理安全閾值。
四元數固定(w,x,y,z)、Hamilton乘法、單位四元數主動旋轉；q v q*；角度半角，插值檢查q/-q與最短弧。
蒙皮需明示mesh與骨架空間、inverse bind及權重和；不能把動畫或照片真實感當作生物／物理驗證。
核心程式Python 3.10+標準庫與NumPy 2.2.6相容寫法；NumPy需讀者既有環境，禁止作者安裝套件或執行工具。圖像優先PPM/SVG；不依賴GPU、Blender或網路才能完成核心實驗。
Blender與GLSL作可選橋接，不宣稱特定版本UI或硬體已測。若需新函式，當章提供完整定義；不能引用尚未存在的模組。
每章目標約4500中文字、至少3000；一般不超過6000以預留附錄及修訂，全卷硬上限200000中文字。公式、英文、標點、程式、參考來源不充字數。
至少兩個逐步手算／可重現數值案例，一段完整可獨立閱讀的小程式與明確測試。未執行的輸出標「預期」，不得虛構驗證、GPU效能、物理準確度或來源查核。
禁止HTML與遠端圖片；數學$...$或$$...$$，$$區塊前後留空行；不得在Markdown外再套程式圍欄。
末章agent僅查詢合成場景與整理證據，不連真實設備、不自行加藥投餌或控制養殖。
'''
SECTIONS = ['學習目標與先備知識','問題與直覺','數學與幾何推導','逐步手算例題','實作與程式','測試與預期結果','除錯與常見陷阱','養殖數位分身案例','習題','習題解答','本章小結','參考來源']
AUTHOR_RULES = '''你是Volume II圖學教材作者。直接輸出指定章的完整繁體中文Markdown，不是大綱、計畫或工具呼叫。
依共同契約與章級任務，寫到足以讓讀者動手做；不得抄來源長段文字。定義符號、維度、單位、必要條件；先推導再實作。
所有指定小節都要有。至少4題習題：手算、程式測試、反例／除錯、整合應用，附完整答案。不以長程式或重複條列充正文。
程式只提供給讀者，不執行、不讀檔、不用網路或工具。不得聲稱預期輸出已跑過、模型審稿等於人工驗證。
若收到修訂意見，僅修錯誤與必要缺漏，保留正確推導與可重現例子，不為文風偏好重寫。'''
REVIEW_RULES = '''你是不同模型的獨立圖學／數學審稿者。輸出繁體中文Markdown，核對維度、座標系、法線、色彩、索引、邊界、手算、程式與預期結果、習題及引用界線。
逐項給出可定位的原句／公式、錯誤原因與修法；重算具體例題。真錯誤與可選文風建議分開，不因風格偏好拒稿。
模型沒有執行工具或程式，不能聲稱實際跑過；來源清單不等於獨立查證。數學正確、教學內容完整可批准，仍保留人工與程式驗證待辦。
只審輸入稿件，不使用任何工具。最後一行必須為VERDICT: APPROVE或VERDICT: REVISE。'''
PATCH_RULES = '''你是主編，收到跨章審稿後只做精準修正。輸出JSON物件{"patches":[{"chapter":1,"old":"原稿中唯一且完整匹配的文字","new":"修正文字"}],"reason":"修訂說明"}。
每個old必須逐字出自輸入章稿且只出現一次，不重疊；不要整章重寫。限20個patch，每個old不超過5000字元。只處理有證據的錯誤，不自行移除重要內容或改全書契約。不呼叫工具、不執行程式；不聲稱修稿已通過驗證。'''


def roster():
    people=[]
    for i in range(40):
        provider,model=MODELS[i%5]
        people.append({'id':f'v2_editor_{i:02d}','provider':provider,'model':model,
                       'role':'author' if i%2==0 else 'reviewer'})
    return people


def chapter_people(number):
    people=roster();i=((number-1)%20)*2
    return people[i],people[i+1]
