"""Volume III原創導讀、SVG及標準庫CPU起步測試。"""
import json
from pathlib import Path
from book_assets import figure
import book3_spec as spec

PREFACE='''# 導讀：从場方程到可驗證的科學計算

本卷研究經典場、偏微分方程、守恆离散與相場，不是量子場論。Volume I提供矩陣工具，Volume II提供幾何與可視化背景；Volume IV尚未寫成，因此前六章補上必要的微積分、積分守恆與誤差橋接，不假設讀者已讀不存在的章節。

五部三十章，主體初步目標十三萬五千中文字，全卷含導讀、附錄與解答不超過二十萬字。核心實作以小型CPU格網與Python／NumPy為主；SciPy是特定章節的明示依賴，FiPy、FEniCS與PETSc作進階參考，不要求安裝大型求解器才能理解入門例題。

第一部建立場與驗證語言；第二部處理有限差分、有限體積及時間方法；第三部連接稀疏線性系統、弱形式、有限元素與Fourier算子；第四部討論流體、多物理場及不確定性；第五部從自由能、變分走到Allen–Cahn、Cahn–Hilliard與固液相變。

## 先備知識與閱讀界線

需要基本單變量微積分、向量矩陣及Python。橋接章不等於完整數學分析或熱力學課程；每個推導必須交代成立條件。相分離與固液相變另用簡化材料案例，不硬套到所有池水現象。溶氧跨過管理閾值不是物理相變。

## 什麼才算驗證

圖像平滑不代表守恆，時間積分不爆炸不代表準確，殘差小不代表解誤差小，模型吻合校準資料不代表已通過獨立現場驗證。每個例子須明列單位、初始／邊界條件、離散格式、容差、來源項及對照證據。

本卷區分預期輸出、模型審稿、實際執行與人工領域覆核。起步腳本`examples/field_lab.py`是主編另寫的標準庫實驗；它通過不代表後續所有模型程式都已執行。模型產生的程式先做靜態檢查，實際執行須另經閱讀與範圍限制。

## 合成養殖案例

池域、溫度、流速與濃度都是教學合成資料，不提供現場操作閾值。agent只讀取有時間與設定版本的模擬紀錄，不連接真實泵浦、曝氣、投餌或加藥設備。渲染真實感、數值一致性與物理可信度分別記錄。
'''
APPENDIX='''# 附錄：共同資料與驗收契約

## 網格與單位

cell資料以q[j,i]存放，j沿物理Y向上，i沿X向右；影像顯示要明示origin或翻轉。node與cell中心不可默默互換。MAC速度分別位於X、Y方向的面；共享面通量只計一次，邊界流出取外法向為正。

有因次傳輸先用SI，濃度以kg/m³，轉換mg/L時另列因子。相場章預設無因次phi與雙井自由能；若改序參量範圍、界面係數或流動耦合，須同步修改能量與化學勢。

## 驗收清單

- 製造解的來源項、邊界與網格節點位置一致。
- 分別細化空間和時間，報告加權L2／L∞誤差與觀測階。
- 零通量／週期情況的質量收支，含反應及外部來源。
- 線性系統檢查零空間、SPD假設、真殘差與容差。
- AC／CH分清序參量守恆；連續與離散能量耗散分開。
- 顯式擴散、平流與四階方程使用各自的穩定條件。
- 固液相變核對顯熱、潛熱與邊界能量收支。
- 保存程式／設定雜湊、單位、seed、版本、時間、失敗與拒絕紀錄。

## 軟體及安全

核心為標準庫與NumPy小格網。稀疏套件只作明示選項；不自動安裝求解器、不訓練模型、不執行未審查生成程式、不修改遠端推論服務。模型審查不是人工審定，數位分身不是現場控制系統。
'''
LAB='''"""主編原創CPU起步實驗；標準庫、合成無因次週期場，不讀寫外部資料。"""
import math
import unittest


def field(q,h):
    q=list(q)
    if len(q)<3 or not math.isfinite(h) or h<=0 or not all(math.isfinite(x) for x in q):
        raise ValueError('finite field, N>=3 and h>0 required')
    return q


def laplacian(q,h):
    q=field(q,h);n=len(q)
    return [(q[(j-1)%n]-2*q[j]+q[(j+1)%n])/(h*h) for j in range(n)]


def diffuse(q,h,dt,D=1):
    q=field(q,h)
    if not all(math.isfinite(x) for x in (dt,D)) or dt<0 or D<0:raise ValueError('invalid coefficients')
    if D*dt/(h*h)>.5:raise ValueError('1D periodic FTCS positivity/stability bound exceeded')
    return [a+dt*D*b for a,b in zip(q,laplacian(q,h))]


def upwind(q,h,dt,velocity):
    q=field(q,h)
    if not math.isfinite(dt) or dt<0 or not math.isfinite(velocity):raise ValueError('invalid step')
    c=abs(velocity)*dt/h
    if c>1:raise ValueError('upwind CFL exceeded')
    shift=-1 if velocity>=0 else 1
    return [(1-c)*q[j]+c*q[(j+shift)%len(q)] for j in range(len(q))]


def energy(q,h,kappa=.01):
    q=field(q,h)
    if not math.isfinite(kappa) or kappa<=0:raise ValueError('kappa must be positive')
    return h*sum((x*x-1)**2/4 for x in q)+kappa/(2*h)*sum((q[(j+1)%len(q)]-q[j])**2 for j in range(len(q)))


def chemical_potential(q,h,kappa=.01):
    q=field(q,h)
    if not math.isfinite(kappa) or kappa<=0:raise ValueError('kappa must be positive')
    return [x**3-x-kappa*l for x,l in zip(q,laplacian(q,h))]


class Checks(unittest.TestCase):
    def setUp(self):
        self.n=16;self.h=1/self.n
        self.q=[.2+.02*math.sin(2*math.pi*(j+.5)/self.n) for j in range(self.n)]
    def test_constant(self):self.assertEqual(laplacian([2]*self.n,self.h),[0]*self.n)
    def test_mass(self):
        updated=diffuse(self.q,self.h,.2*self.h**2)
        self.assertAlmostEqual(sum(updated)*self.h,sum(self.q)*self.h)
    def test_fourier_mode(self):
        q=[math.sin(2*math.pi*(j+.5)/self.n) for j in range(self.n)]
        updated=diffuse(q,self.h,.2*self.h**2);factor=1-.8*math.sin(math.pi/self.n)**2
        self.assertLess(max(abs(a-factor*b) for a,b in zip(updated,q)),1e-12)
    def test_reject_unstable(self):
        with self.assertRaises(ValueError):diffuse(self.q,self.h,self.h**2)
        with self.assertRaises(ValueError):upwind(self.q,self.h,2*self.h,1)
    def test_upwind_both_signs(self):
        for velocity in [-1,1]:
            q=upwind(self.q,self.h,.5*self.h,velocity)
            self.assertAlmostEqual(sum(q),sum(self.q));self.assertGreaterEqual(min(q),min(self.q));self.assertLessEqual(max(q),max(self.q))
    def test_energy_gradient(self):
        direction=[math.cos(j+.3) for j in range(self.n)];eps=1e-6
        plus=[x+eps*d for x,d in zip(self.q,direction)];minus=[x-eps*d for x,d in zip(self.q,direction)]
        fd=(energy(plus,self.h)-energy(minus,self.h))/(2*eps)
        analytic=self.h*sum(m*d for m,d in zip(chemical_potential(self.q,self.h),direction))
        self.assertAlmostEqual(fd,analytic,places=8)
    def test_ac_specific_small_step(self):
        mu=chemical_potential(self.q,self.h);q=[x-1e-4*m for x,m in zip(self.q,mu)]
        self.assertLess(energy(q,self.h),energy(self.q,self.h))
        self.assertGreater(abs(sum(q)-sum(self.q)),1e-6)
    def test_ch_specific_small_step(self):
        rhs=laplacian(chemical_potential(self.q,self.h),self.h);q=[x+1e-6*d for x,d in zip(self.q,rhs)]
        self.assertAlmostEqual(sum(q),sum(self.q),places=12)
        self.assertLess(energy(q,self.h),energy(self.q,self.h))
    def test_invalid_field(self):
        with self.assertRaises(ValueError):laplacian([0,float('nan'),0],1)


if __name__=='__main__':unittest.main(verbosity=2)
'''
FIGURES={
 'pipeline.svg':('Volume III：從場方程到驗證',['物理假設、場、單位與邊界','守恆律 → 空間離散 → 時間方法','線性／非線性求解與殘差','解析／製造解、網格細化、收支','自由能、相場與可稽核數位分身']),
 'operators.svg':('網格與算子契約',['cell中心q[j,i]：i沿X，j沿物理Y','面速度與外法向通量','L近似Delta；A=-L','零模態、邊界與相容條件']),
 'conservation.svg':('守恆不等於穩定或準確',['初始總量＋邊界流入＋來源','共享面通量成對抵消','時間步长與空間誤差分開','負值與裁切造成的總量改變要記錄']),
 'solvers.svg':('求解器不能取代模型驗證',['檢查形狀、單位、SPD／非對稱','零空間與右端相容性','真殘差b-Ax、預條件與停止規則','用解析／製造解檢查真正誤差']),
 'phase-field.svg':('相場的守恆與能量',['F(phi) → 第一變分mu','Allen–Cahn：非守恆梯度流','Cahn–Hilliard：守恆梯度流','連續耗散不保證任意離散步長耗散','材料相變不等於溶氧管理閾值'])}


def initialize(root):
    root=Path(root)
    for sub in ('chapters','editorial','agents','figures','examples','data'):(root/sub).mkdir(parents=True,exist_ok=True)
    for name,text in [('00-preface.md',PREFACE),('99-appendix.md',APPENDIX),('STYLE_GUIDE.md','# 共同契約\n\n'+spec.CONVENTIONS),('examples/field_lab.py',LAB)]:
        p=root/name
        if not p.exists():p.write_text(text)
    toc=['# '+spec.TITLE,'30章；每章目標4500、至少3000中文字，全卷20萬字以內。']
    for part,title in enumerate(spec.PARTS):
        toc.append('## '+title)
        for n,(name,p,core,lab) in enumerate(spec.CHAPTERS,1):
            if p==part:toc.extend([f'### 第{n:02d}章 {name}',f'- 核心：{core}',f'- 實作：{lab}'])
    (root/'TOC.md').write_text('\n\n'.join(toc))
    (root/'REFERENCES.md').write_text('# 參考來源\n\n'+spec.SOURCE_NOTES+'\n\n'+'\n'.join(f'- [{key}] [{title}]({url})' for key,title,url in spec.SOURCES))
    for name,(title,items) in FIGURES.items():
        p=root/'figures'/name
        if not p.exists():p.write_text(figure(title,items))
    p=root/'data/field_contract.json'
    if not p.exists():p.write_text(json.dumps({'synthetic':True,'array_order':'q[j,i], i:+X, j:+Y','flatten':'j*Nx+i','cell_shape':[16,16],'domain_m':[1,1],'time_unit':'s','concentration_unit':'kg/m^3','seed':42,'phase_field':'separate nondimensional example, phi convention -1/+1','note':'不是現場物性、設備參數或安全閾值'},ensure_ascii=False,indent=2))
