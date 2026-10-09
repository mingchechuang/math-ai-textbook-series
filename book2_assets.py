"""Volume II的本機原創前言、圖解與可測試起步程式，不執行模型程式。"""
import json
from pathlib import Path
from book_assets import figure
import book2_spec as spec

PREFACE = '''# 導讀：從公式到可檢查的圖學實作

本卷是Volume II，主題為現代2D／3D圖學。Volume I的線性代數是先備工具，不代表讀者已學過相機、材質、渲染或動畫。每章先建立幾何與數學，再提供手算、程式、測試、除錯與習題。

全書五部、三十章，初步目標約十三萬五千中文字，含附錄及修訂仍不得超過二十萬字。篇幅用來補足推導和實作，不要求湊滿。核心實驗以CPU、小型合成資料與Python／NumPy為主；Blender、GPU著色器是延伸，不是完成所有基礎練習的必要條件。

共同案例是一座虛構養殖場數位分身：幾何池體、簡化魚體、相機、光源、水面與動畫。合成影像可用於理解多模態觀測與agent的唯讀查詢，但不能憑圖像真實感推論物理精確或真實魚群行為，也不連接現場設備。

## 閱讀路線

第一部先處理向量、座標、相機與像素。第二部建立可交換的幾何資產。第三部處理顏色、材質與著色。第四部從求交走到光傳輸、取樣與誤差。第五部建立動畫、蒙皮、逆運動學及整合專題。

微積分、機率、輻射度量在首次使用時橋接；更完整的理論可銜接後續Volume III～V。學習目標不是只會調用軟體，而是能說清座標、推導、數值結果與失效條件。

## 驗證標籤

- 「预期輸出」：依推導預測，尚未實際執行。
- 「模型審稿」：不同模型核對稿件，不等於人工審定。
- 「已執行」：只限明列腳本、環境及紀錄的測試，不可推廣成全書已驗證。

起步程式`examples/graphics_lab.py`由主編另行撰寫，與模型生成片段分開。後續章內程式經靜態檢查及人工檢視後，才另行安排執行，避免自動執行未審查程式。
'''
APPENDIX = '''# 附錄：實作契約與驗收清單

## 建議目錄

- `chapters/`：章稿；`editorial/`：逐章及跨章意見、精準修訂與版本紀錄。
- `figures/`：本機SVG圖解；`data/scene.json`：合成場景契約。
- `examples/`：可獨立執行的起步程式與測試。
- `agents/`：40個角色各自的session、輸入及事件，不是作業系統安全沙箱。

## 座標與顏色

使用右手世界系、column向量、相機朝局部負Z、OpenGL式NDC深度[-1,1]。其他API需明列轉換。像素原點左上，UV的v向上，兩者不可直接混用。材質與光照在線性RGB計算；顏色輸出才做sRGB轉換，法線與深度不是顏色。

## 實驗紀錄

每個實作應記錄Python與NumPy版本、種子、輸入檔雜湊、尺寸、輸出及容差。效能紀錄還需硬體、解析度、樣本數及計時方法。單一seed的好圖不構成統計可靠性；未執行就寫預期，不捏造時間或FPS。

## 整合驗收

座標往返、退化幾何、透視深度、共邊光柵化、法線正交、sRGB往返、射線最近交點、BVH與暴力法一致、抽樣PDF、全反射、四元數雙覆蓋、bind pose還原、IK不可達與資料時間切分，都應各有正例及反例。不是每章模型批准就等於這些程式已跑完。

## 安全與範圍

模型只生成文件，不安裝套件、不執行程式、不更動遠端服务。核心CPU實驗不需要真實設備或敏感資料。真實養殖操作、物理場標定與專業判斷不由圖學模型代替。
'''
LAB = '''"""Volume II起步實驗：標準庫、CPU、合成數值；無外部檔案／設備。"""
import math
import unittest


def dot(a,b):
    if len(a)!=len(b): raise ValueError('dimension mismatch')
    return sum(x*y for x,y in zip(a,b))


def cross(a,b):
    if len(a)!=3 or len(b)!=3: raise ValueError('3D required')
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def normalize(v):
    norm=math.sqrt(dot(v,v))
    if not math.isfinite(norm) or norm<=1e-12: raise ValueError('degenerate vector')
    return tuple(x/norm for x in v)


def apply4(m,v):
    if len(m)!=4 or any(len(row)!=4 for row in m) or len(v)!=4: raise ValueError('4D required')
    return tuple(dot(row,v) for row in m)


def srgb_encode(x):
    if not 0<=x<=1: raise ValueError('reference example uses [0,1]')
    return 12.92*x if x<=0.0031308 else 1.055*x**(1/2.4)-0.055


def ray_sphere(origin,direction,center,radius):
    if radius<=0: raise ValueError('positive radius required')
    oc=tuple(a-b for a,b in zip(origin,center));a=dot(direction,direction)
    if a<=1e-24: raise ValueError('zero ray direction')
    half_b=dot(oc,direction);c=dot(oc,oc)-radius*radius
    disc=half_b*half_b-a*c
    if disc<0:return None
    roots=sorted(((-half_b-math.sqrt(disc))/a,(-half_b+math.sqrt(disc))/a))
    return next((t for t in roots if t>=1e-8),None)


class LabChecks(unittest.TestCase):
    def test_cross(self):self.assertEqual(cross((1,0,0),(0,1,0)),(0,0,1))
    def test_normalize(self):self.assertEqual(normalize((3,0,0)),(1.,0.,0.))
    def test_zero(self):
        with self.assertRaises(ValueError):normalize((0,0,0))
    def test_point_direction(self):
        m=((1,0,0,2),(0,1,0,3),(0,0,1,4),(0,0,0,1))
        self.assertEqual(apply4(m,(1,2,3,1)),(3,5,7,1))
        self.assertEqual(apply4(m,(1,2,3,0)),(1,2,3,0))
    def test_color(self):
        self.assertAlmostEqual(srgb_encode(0),0);self.assertAlmostEqual(srgb_encode(1),1)
        self.assertAlmostEqual(srgb_encode(.18),.4613561295)
    def test_ray(self):
        self.assertAlmostEqual(ray_sphere((0,0,3),(0,0,-1),(0,0,0),1),2)
        self.assertAlmostEqual(ray_sphere((0,0,0),(0,0,1),(0,0,0),1),1)
        self.assertIsNone(ray_sphere((0,0,3),(1,0,0),(0,0,0),1))


if __name__=='__main__':unittest.main(verbosity=2)
'''
FIGURES = {
 'pipeline.svg':('Volume II：從幾何到數位分身',['幾何與座標 → 相機與光柵化','網格、曲線與程序化建模','線性光、材質與貼圖','射線、光傳輸與抽樣','動畫、合成資料與唯讀agent']),
 'coordinates.svg':('座標與成像契約',['局部點 p，齊次 w=1；方向 w=0','世界：M_parent × M_local × p','相機：V × p_world，向前是 -Z','裁切：P × p_camera；先裁切後除w','NDC → 像素；影像Y軸向下']),
 'surface.svg':('表面表示的檢查鏈',['控制點 → 曲線／曲面 → 三角網格','頂點索引、繞序、邊界與退化檢查','法線逆轉置；UV、切線與手性','材質資料分清線性RGB及非色彩資料']),
 'light.svg':('光傳輸不是只把顏色相加',['射線與最近交點，含t範圍','法線朝向、BRDF與光源','餘弦、取樣PDF與throughput','多seed估計誤差，不虛構FPS','水面折射是簡化示範，非完整現場光學']),
 'animation.svg':('動畫與資料驗收',['時間 → 局部TRS → 父子階層','單位四元數與最短弧SLERP','骨骼、inverse bind與蒙皮權重','相機／深度／ID標註及資料來源','人工覆核；不連接真實養殖設備'])}


def initialize(root):
    root=Path(root)
    for sub in ('chapters','editorial','agents','figures','examples','data'):(root/sub).mkdir(parents=True,exist_ok=True)
    for file,text in [('00-preface.md',PREFACE),('99-appendix.md',APPENDIX),('STYLE_GUIDE.md','# 共同契約\n\n'+spec.CONVENTIONS),('examples/graphics_lab.py',LAB)]:
        p=root/file
        if not p.exists():p.write_text(text,encoding='utf-8')
    toc=['# '+spec.TITLE,'\n30章；每章約4500字，全卷20萬字以內。\n']
    for part,title in enumerate(spec.PARTS):
        toc.append('## '+title)
        for n,(name,p,maths,lab) in enumerate(spec.CHAPTERS,1):
            if p==part:toc.extend([f'### 第{n:02d}章 {name}',f'- 核心：{maths}',f'- 實作：{lab}',f'- 稿件：`chapters/{n:02d}.md`'])
    (root/'TOC.md').write_text('\n\n'.join(toc))
    (root/'REFERENCES.md').write_text('# 參考來源\n\n'+spec.SOURCE_NOTES+'\n\n'+'\n'.join(f'- [{i}] [{t}]({url})' for i,t,url in spec.SOURCES))
    for name,(title,boxes) in FIGURES.items():
        p=root/'figures'/name
        if not p.exists():p.write_text(figure(title,boxes))
    (root/'data/scene.json').write_text(json.dumps({'synthetic':True,'length_unit':'metre','handedness':'right','camera_forward':'-Z','world_up':'+Y','ndc_depth':[-1,1],'seed':42,'pond_size':[8,2,4],'note':'教學合成幾何，不提供現場控制閾值'},ensure_ascii=False,indent=2))
