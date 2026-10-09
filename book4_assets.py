"""Volume IV導讀、原創圖解與獨立CPU起步測試；不執行生成章稿。"""
import json
from pathlib import Path
from book_assets import figure
import book4_spec as spec

PREFACE='''# 導讀：線性近似之外，還需要什麼？

第四卷以線性代數為入口，建立微積分與數學分析的條件意識。把偏導數排成矩陣還不足以說明可微性；真正的核心是：線性近似留下的誤差，除以擾動尺度後是否趨於零？本卷不只給公式，也給定義、量詞、證明、反例與數值證據的界線。

全卷五部三十章，正文初步目標十三萬五千中文字，每章至少三千、目標四千五百，含導讀、附錄及解答不超過二十萬字。公式、程式與英文不充字數。不因篇幅限制跳過必要假設，也不以重複敘述湊數。

## 閱讀路線

第一部建立極限、範數、緊緻性及單變量橋接；第二部將導數理解成線性映射，接到鏈式法則、反函數與隱函數；第三部由Hessian走到凸性、最佳化及约束；第四部連接積分、幾何與極限交換；第五部討論ODE、算子、變分及整合專題。

Volume I提供矩陣基礎；Volume II的幾何與Volume III的場模擬作應用對照。第三卷已完成模型審稿，仍不是人工審定；本卷不把它當成所有分析定理都已證明的先備課。Volume V尚未製作，反向傳播先從小型計算圖介紹，不需要先學完整Transformer。

## 教學與驗證

每章至少含一個完整小命題證明、兩個手算例、自足CPU程式、正常與故障測試，以及四類習題與解答。重要進階結果若未完整證明，明示引用與條件，不把證明省略藏在圖像或程式後面。

核心採Python標準庫／NumPy；SciPy可選，JAX僅為自動微分延伸參考，不要求GPU。有限差分可發現梯度錯誤，但不能憑幾個方向就證明Fréchet可微；有限格點不能證明緊緻或一致收斂。尚未執行的輸出明標預期。

## 合成案例与安全

感測映射、幾何域及校準資料都是合成教學資料。分量單位、縮放、Jacobian形狀、容差和適用範圍必須列出；局部近似不能當作全域模型。agent只讀證據，不控制真實泵浦、投餌、加藥或其他設備。
'''
APPENDIX='''# 附錄：符號、證據與驗收索引

## 共同符號

- x、h：輸入及擾動；數學上為n×1 column向量。
- f：從R^n到R^m的映射；Df是線性導數，Jf是其m×n矩陣。
- 梯度：標量函數的Euclidean直向量；其轉置表示標量導數。
- H：Hessian；C2鄰域是常用的對稱充分條件。
- Jv及J^T w：分別為JVP與VJP的column表示。
- c=0、g<=0：等式與不等式約束；L=f+λ^T c或加入λ^T g，後者λ>=0。
- 普通換變數用Jacobian行列式的絕對值；取向積分另處理符號。

## 驗收問題

公式在哪個域成立？需要幾階連續可微？條件是必要還是充分？例題位於內點還是邊界？是否有秩退化或零空間？局部結論能否推到全域？使用哪個範數？是否混合不同單位？有限維结論是否被誤用到函數空間？

數值測試至少包括解析對照、維度、有限輸入、退化情況與步長變化。觀測收斂階要有實際紀錄，未執行只能列預期。校準成功不等於物理Validation；模型審稿不代替人工數學審查。

## 檔案與限制

章稿、審稿、修補、原始事件及版本備份分開保存。`examples/analysis_lab.py`是主編另寫的標準庫起步測試，不載入模型生成程式。40角色使用按章／部區分的session，角色數不是同時執行程序數，也不是作業系統沙箱。
'''
LAB='''"""主編原創標準庫實驗：數值核對不是分析定理的證明。"""
import math
import unittest


def mapping(x):
    if len(x)!=2 or not all(math.isfinite(v) for v in x):raise ValueError('finite 2-vector required')
    a,b=x
    return [a*a+2*b,a*b]


def jacobian(x):
    mapping(x);a,b=x
    return [[2*a,2],[b,a]]


def matvec(a,v):
    if not a or any(len(row)!=len(v) for row in a):raise ValueError('dimension mismatch')
    return [sum(x*y for x,y in zip(row,v)) for row in a]


def dot(a,b):
    if len(a)!=len(b):raise ValueError('dimension mismatch')
    return sum(x*y for x,y in zip(a,b))


def norm(x):return math.sqrt(dot(x,x))


def circle_y(x):
    if not math.isfinite(x) or not -1<x<1:raise ValueError('smooth upper branch requires -1<x<1')
    return math.sqrt(1-x*x)


def non_frechet(x,y):
    return 0. if x==0 and y==0 else x**3/(x*x+y*y)


def jordan_exp(t):
    e=math.exp(-t)
    return [[e,t*e],[0.,e]]


class Checks(unittest.TestCase):
    def test_jacobian(self):
        x=[.7,-.2];h=1e-5;j=jacobian(x)
        for k in range(2):
            plus=x.copy();minus=x.copy();plus[k]+=h;minus[k]-=h
            fd=[(a-b)/(2*h) for a,b in zip(mapping(plus),mapping(minus))]
            for i in range(2):self.assertAlmostEqual(fd[i],j[i][k],places=8)
    def test_jvp_vjp(self):
        j=jacobian([.7,-.2]);v=[.3,-.4];w=[.8,.2]
        jt=[list(row) for row in zip(*j)]
        self.assertAlmostEqual(dot(w,matvec(j,v)),dot(matvec(jt,w),v))
    def test_normalized_remainder(self):
        x=[.7,-.2];ratios=[]
        for scale in [.02,.01,.005]:
            h=[scale,2*scale]
            actual=mapping([a+b for a,b in zip(x,h)])
            predicted=[a+b for a,b in zip(mapping(x),matvec(jacobian(x),h))]
            ratios.append(norm([a-b for a,b in zip(actual,predicted)])/norm(h))
        self.assertAlmostEqual(ratios[0]/ratios[1],2,places=8)
        self.assertAlmostEqual(ratios[1]/ratios[2],2,places=8)
    def test_partials_not_sufficient(self):
        for h in [.1,.01,.001]:
            self.assertAlmostEqual(non_frechet(h,0)/h,1)
            self.assertEqual(non_frechet(0,h)/h,0)
            ratio=abs(non_frechet(h,h)-h)/math.hypot(h,h)
            self.assertAlmostEqual(ratio,1/(2*math.sqrt(2)))
    def test_implicit_derivative(self):
        x=.3;h=1e-5
        self.assertAlmostEqual((circle_y(x+h)-circle_y(x-h))/(2*h),-x/circle_y(x),places=8)
    def test_gram_area(self):
        a=[1,0,1];b=[0,2,0]
        self.assertAlmostEqual(math.sqrt(dot(a,a)*dot(b,b)-dot(a,b)**2),math.sqrt(8))
    def test_jacobian_orientation(self):
        determinant=2*(-3)
        self.assertEqual(determinant,-6);self.assertEqual(abs(determinant),6)
    def test_jordan_semigroup(self):
        a=jordan_exp(.3);b=jordan_exp(.4);c=jordan_exp(.7)
        for i in range(2):
            for j in range(2):self.assertAlmostEqual(sum(a[i][k]*b[k][j] for k in range(2)),c[i][j])
    def test_invalid_input(self):
        with self.assertRaises(ValueError):mapping([float('nan'),0])
        with self.assertRaises(ValueError):matvec([[1,2]],[1])
        with self.assertRaises(ValueError):circle_y(1)


if __name__=='__main__':unittest.main(verbosity=2)
'''
FIGURES={
 'pipeline.svg':('Volume IV：從線性近似到分析',['極限、範數、緊緻與條件','導數：線性映射＋小於一階的餘項','Hessian、約束與最佳化','積分換變數、幾何與極限交換','ODE、算子、變分與可稽核專題']),
 'derivative.svg':('導數不只是偏導表格',['f: R^n → R^m，J是m×n','擾動h → Jh → 餘項r(h)','可微需 norm(r)/norm(h) → 0','方向取樣不是全方向極限證明']),
 'chain.svg':('JVP與VJP的形狀契約',['J: m×n；輸入v: n×1','Jv: m×1，推送切向量','輸出協向量w: m×1','J^T w: n×1，拉回協向量','w^T(Jv) = (J^T w)^T v']),
 'geometry.svg':('積分的幾何與取向',['普通體積：abs(det DT)','曲面面積：sqrt(det(J^T J))','有向通量另含法向與符號','局部可逆不保證全域一對一']),
 'conditions.svg':('每個結論都要帶條件',['定義域、單位、範數與正則性','必要／充分；內點／邊界','局部／全域；有限／無限維','證明、反例與數值證據分開'])}


def initialize(root):
    root=Path(root)
    for sub in ('chapters','editorial','agents','figures','examples','data'):(root/sub).mkdir(parents=True,exist_ok=True)
    for name,text in [('00-preface.md',PREFACE),('99-appendix.md',APPENDIX),('STYLE_GUIDE.md','# 共同契約\n\n'+spec.CONVENTIONS),('examples/analysis_lab.py',LAB)]:
        p=root/name
        if not p.exists():p.write_text(text)
    toc=['# '+spec.TITLE,'五部30章；每章目標4500、最低3000中文字；全卷含解答上限200000。']
    for part,title in enumerate(spec.PARTS):
        toc.append('## '+title)
        for n,(name,p,core,lab) in enumerate(spec.CHAPTERS,1):
            if p==part:toc.extend([f'### 第{n:02d}章 {name}',f'- 核心：{core}',f'- 實作：{lab}'])
    (root/'TOC.md').write_text('\n\n'.join(toc))
    (root/'REFERENCES.md').write_text('# 參考來源\n\n'+spec.SOURCE_NOTES+'\n\n'+'\n'.join(f'- [{key}] [{title}]({url})' for key,title,url in spec.SOURCES))
    for name,(title,items) in FIGURES.items():
        p=root/'figures'/name
        if not p.exists():p.write_text(figure(title,items))
    p=root/'data/analysis_contract.json'
    if not p.exists():p.write_text(json.dumps({'synthetic':True,'dimensionless_demo':True,'input_shape':[2],'output_shape':[2],'jacobian_shape':[2,2],'mapping':['x0**2+2*x1','x0*x1'],'seed':42,'note':'只作合成分析與形狀檢查；非實際感測器或操作閾值'},ensure_ascii=False,indent=2))
