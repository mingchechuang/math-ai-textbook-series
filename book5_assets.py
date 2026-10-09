"""Volume V導讀、原創SVG及獨立CPU起步驗證，不載入生成章稿。"""
import json
from pathlib import Path
from book_assets import figure
import book5_spec as spec

PREFACE='''# 導讀：從可核對的梯度到可稽核的模型

第五卷以矩陣與微分為先備，走向張量、機率、反向傳播、多頭注意力，以及可在CPU上理解與測試的小型Transformer。重點不是呼叫現成大模型API，而是說清楚每個形狀、遮罩、損失平均及資料切分如何影響結果。

第一部橋接張量與機率；第二部建立NumPy兩層網路與梯度驗證；第三部逐一組合embedding、注意力、位置與正規化；第四部完成小型decoder的訓練、生成與cache對照；第五部延伸低秩適配、多模態、檢索與只讀Agent。前四卷已有出版成品，但模型審稿不是人工數學審定，不能以引用前卷代替明確條件。

## 三階段驗收

1. NumPy兩層網路：逐項有限差分、可重現合成資料、訓練與保留集分開。
2. 多頭注意力：形狀、梯度、遮罩與未來資訊洩漏測試。
3. 小型decoder-only Transformer：合成日誌訓練、生成、有效token評估、cache等價與失敗案例。

全卷五部三十章，每章至少三千、目標四千五百中文字，含導讀、附錄及解答總計不超過二十萬字。公式、程式與英文不充字數；不省略關鍵假設或以重複句子湊數。每章有證明、手算、完整程式、正常／邊界／故障測試及四類習題解答。

## 執行與證據

核心採Python／NumPy，小型完整Transformer可使用PyTorch CPU，清楚列出版本與依賴，不下載模型、語料、tokenizer或套件，不安排GPU訓練。稿件中的程式不由模型工作流自動執行；起步驗證由主編獨立撰寫，與生成程式分開。未執行的結果只標預期，不宣稱模型已訓練或測試全部通過。

## 安全界線

所有養殖日誌、感測與多模態配對都是合成資料。分布外評估、可靠來源與操作安全分開；流暢輸出不是能力或現場有效性的證明。Agent只用記憶體中的唯讀mock工具，不執行shell、網路請求或設備操作。檢索文件中的指令不授予任何權限。
'''
APPENDIX='''# 附錄：形狀、驗收與證據

- 批次X=(B,Din)，權重W=(Din,Dout)，輸出Y=XW+b。
- 多頭Q=(B,H,Tq,dh)、K=(B,H,Tk,dh)、V=(B,H,Tk,dv)，softmax沿key軸。
- 本卷布林mask的True表示允許；框架轉接不可假設同義。
- 因果遮罩以絕對位置判斷，cache的矩形遮罩不能隨意套用左上三角。
- 有效token平均只除一次，微批累積依有效token數加權。
- 先分資料，再擬合詞表與標準化，再切窗口；test不參與調參。
- LoRA採W+alpha/r AB，A=(Din,r)、B=(r,Dout)。

`examples/neural_lab.py`是獨立NumPy CPU起步驗證，只測小型primitive與兩層網路，不代表完整Transformer已完成或已訓練。逐章程式、後續選定快照的驗證及人工審定必須分開記錄。

保留章稿、審稿、原始輸出、呼叫用量、修補備份及雜湊。四十個邏輯角色不是四十個並行程序，也不是作業系統隔離沙箱；最多四章並行，同一角色的不同章使用分開session。
'''
LAB=r'''"""獨立主編起步測試：CPU NumPy，不載入或執行生成章稿。"""
import math
import unittest
import numpy as np


def softmax(z, allowed=None):
    z=np.asarray(z,dtype=float)
    if z.ndim<1 or z.size==0 or not np.isfinite(z).all():raise ValueError('nonempty finite logits required')
    if allowed is not None:
        mask=np.asarray(allowed)
        if mask.dtype!=np.bool_:raise ValueError('boolean allow mask required')
        mask=np.broadcast_to(mask,z.shape)
        if not mask.any(axis=-1).all():raise ValueError('all-masked query')
        z=np.where(mask,z,-np.inf)
    e=np.exp(z-z.max(axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)


def cross_entropy(z,y):
    z=np.asarray(z,dtype=float);y=np.asarray(y)
    p=softmax(z)
    if z.ndim!=2 or y.shape!=(z.shape[0],) or not np.issubdtype(y.dtype,np.integer):raise ValueError('B,C logits and integer labels required')
    if (y<0).any() or (y>=z.shape[1]).any():raise ValueError('label out of range')
    shifted=z-z.max(axis=1,keepdims=True)
    loss=np.mean(np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(y)),y])
    g=p.copy();g[np.arange(len(y)),y]-=1
    return float(loss),g/len(y)


def mlp(x,y,params):
    w,b,v,c=params;h=np.tanh(x@w+b);loss,g=cross_entropy(h@v+c,y)
    dh=(g@v.T)*(1-h*h)
    return loss,[x.T@dh,dh.sum(axis=0),h.T@g,g.sum(axis=0)]


def attention(q,k,v,allowed):
    q,k,v=[np.asarray(a,dtype=float) for a in (q,k,v)]
    if any(a.ndim!=2 or not np.isfinite(a).all() for a in (q,k,v)):raise ValueError('finite matrices required')
    if q.shape[1]==0 or q.shape[1]!=k.shape[1] or k.shape[0]!=v.shape[0]:raise ValueError('shape mismatch')
    p=softmax(q@k.T/math.sqrt(q.shape[1]),allowed)
    return p@v,p


def attention_backward(q,k,v,p,g):
    dp=g@v.T;ds=p*(dp-(dp*p).sum(axis=-1,keepdims=True));scale=math.sqrt(q.shape[1])
    return ds@k/scale,ds.T@q/scale,p.T@g


def finite_difference(func,a,eps=1e-6):
    result=np.zeros_like(a)
    for idx in np.ndindex(a.shape):
        old=a[idx]
        try:
            a[idx]=old+eps;plus=func()
            a[idx]=old-eps;minus=func()
        finally:a[idx]=old
        result[idx]=(plus-minus)/(2*eps)
    return result


class Checks(unittest.TestCase):
    def test_softmax_extreme_shift(self):
        z=np.array([[1000.,999.,-1000.]])
        np.testing.assert_allclose(softmax(z),softmax(z-900),atol=1e-14)
        self.assertAlmostEqual(softmax(z).sum(),1)
        loss,_=cross_entropy(np.array([[1000.,-1000.]]),np.array([1]))
        self.assertEqual(loss,2000.)
    def test_ce_gradient(self):
        z=np.array([[.2,.7,-.4],[-.3,.2,.5]]);y=np.array([1,2]);_,g=cross_entropy(z,y)
        np.testing.assert_allclose(g,finite_difference(lambda:cross_entropy(z,y)[0],z),atol=1e-8)
    def test_mlp_all_parameters(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(3,2));y=np.array([0,1,0])
        ps=[rng.normal(size=s)*.2 for s in ((2,3),(3,),(3,2),(2,))]
        _,grad=mlp(x,y,ps)
        for a,g in zip(ps,grad):np.testing.assert_allclose(g,finite_difference(lambda:mlp(x,y,ps)[0],a),atol=1e-8)
    def test_small_training(self):
        rng=np.random.default_rng(7);x=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]]);y=np.array([0,1,1,0])
        ps=[rng.normal(size=s)*.4 for s in ((2,4),(4,),(4,2),(2,))];initial=mlp(x,y,ps)[0]
        for _ in range(500):
            _,grads=mlp(x,y,ps)
            for p,g in zip(ps,grads):p-=.2*g
        self.assertLess(mlp(x,y,ps)[0],initial*.3) # only training-fit smoke test, not generalization
    def test_attention_gradients(self):
        rng=np.random.default_rng(8);q=rng.normal(size=(3,2));k=rng.normal(size=(3,2));v=rng.normal(size=(3,2));g=rng.normal(size=(3,2));mask=np.tril(np.ones((3,3),dtype=bool))
        _,p=attention(q,k,v,mask)
        for a,expected in zip((q,k,v),attention_backward(q,k,v,p,g)):
            actual=finite_difference(lambda:float((attention(q,k,v,mask)[0]*g).sum()),a)
            np.testing.assert_allclose(actual,expected,atol=1e-8)
    def test_no_future_leak(self):
        rng=np.random.default_rng(2);q=rng.normal(size=(4,2));k=rng.normal(size=(4,2));v=rng.normal(size=(4,2));mask=np.tril(np.ones((4,4),dtype=bool))
        original=attention(q,k,v,mask)[0];k[2:]+=100;v[2:]-=100
        np.testing.assert_allclose(original[:2],attention(q,k,v,mask)[0][:2],atol=1e-13)
    def test_incremental_attention(self):
        rng=np.random.default_rng(3);q=rng.normal(size=(4,2));k=rng.normal(size=(4,2));v=rng.normal(size=(4,2))
        full=attention(q,k,v,np.tril(np.ones((4,4),dtype=bool)))[0]
        for t in range(4):
            allow=np.arange(t+1)[None,:]<=t
            one=attention(q[t:t+1],k[:t+1],v[:t+1],allow)[0]
            np.testing.assert_allclose(one[0],full[t],atol=1e-13)
    def test_heads_roundtrip(self):
        x=np.arange(24).reshape(2,3,4);heads=x.reshape(2,3,2,2).transpose(0,2,1,3)
        self.assertEqual(heads[1,1,2,1],x[1,2,3])
        np.testing.assert_array_equal(heads.transpose(0,2,1,3).reshape(2,3,4),x)
    def test_embedding_accumulation(self):
        ids=np.array([1,1,0]);g=np.array([[1.,2.],[3.,4.],[5.,6.]]);table=np.zeros((3,2));np.add.at(table,ids,g)
        np.testing.assert_array_equal(table,[[5,6],[4,6],[0,0]])
    def test_rope_identity(self):
        def r(t):return np.array([[math.cos(t),-math.sin(t)],[math.sin(t),math.cos(t)]])
        x=np.array([.2,.7]);y=np.array([-.1,.4]);a=.3;b=.8
        self.assertAlmostEqual(np.linalg.norm(r(a)@x),np.linalg.norm(x))
        self.assertAlmostEqual((r(a)@x)@(r(b)@y),x@(r(b-a)@y))
    def test_token_weighting(self):
        means=np.array([1.,3.]);counts=np.array([2,6])
        self.assertEqual(float(means@counts/counts.sum()),2.5)
        self.assertNotEqual(float(means.mean()),2.5)
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):softmax([1.,float('nan')])
        with self.assertRaises(ValueError):softmax([[1.,2.]],[[False,False]])
        with self.assertRaises(ValueError):softmax([[1.,2.]],[[1,0]])
        with self.assertRaises(ValueError):cross_entropy([[1.,2.]],[2])
        with self.assertRaises(ValueError):attention(np.ones((1,2)),np.ones((2,3)),np.ones((2,1)),True)


if __name__=='__main__':
    import platform
    print('Python',platform.python_version(),'NumPy',np.__version__,'CPU',platform.machine(),flush=True)
    unittest.main(verbosity=2)
'''
FIGURES={
 'pipeline.svg':('Volume V：可核對的模型與證據',['張量、概率與資料切分','NumPy前向、反傳與梯度檢查','注意力、位置與正規化','小型decoder訓練、生成與cache','適配、多模態、來源與唯讀Agent']),
 'shapes.svg':('每條軸都是介面契約',['X: B×T×D','Q/K/V: B×H×T×dh','scores: B×H×Tq×Tk','softmax沿key軸；梯度回原shape']),
 'backprop.svg':('反向傳播與獨立檢查',['前向：輸入、參數、loss','反向：局部VJP與分支累加','有限差分：獨立方向對照','小資料擬合不是泛化證明']),
 'mask.svg':('因果遮罩與cache',['True代表允許，不是遮住','key絕對位置不可超過query','矩形cache遮罩不能盲用tril','改未來token不影響過去輸出']),
 'evidence.svg':('模型不是操作授權',['先分資料，再擬合與切窗','合成數據、保留集、分布外','引用來源與指令分離','工具allowlist、唯讀mock與稽核'])}


def initialize(root):
    root=Path(root)
    for sub in ('chapters','editorial','agents','figures','examples','data'):(root/sub).mkdir(parents=True,exist_ok=True)
    for name,text in [('00-preface.md',PREFACE),('99-appendix.md',APPENDIX),('STYLE_GUIDE.md','# 共同契約\n\n'+spec.CONVENTIONS),('examples/neural_lab.py',LAB)]:
        p=root/name
        if not p.exists():p.write_text(text)
    toc=['# '+spec.TITLE,'五部30章；正文每章最低3000、目標4500字；全卷上限200000中文字。']
    for part,title in enumerate(spec.PARTS):
        toc.append('## '+title)
        for n,(name,p,core,lab) in enumerate(spec.CHAPTERS,1):
            if p==part:toc.extend([f'### 第{n:02d}章 {name}',f'- 核心：{core}',f'- 實作：{lab}'])
    (root/'TOC.md').write_text('\n\n'.join(toc))
    (root/'REFERENCES.md').write_text('# 參考來源\n\n'+spec.SOURCE_NOTES+'\n\n'+'\n'.join(f'- [{key}] [{title}]({url})' for key,title,url in spec.SOURCES))
    for name,(title,items) in FIGURES.items():
        p=root/'figures'/name
        if not p.exists():p.write_text(figure(title,items))
    p=root/'data/learning_contract.json'
    if not p.exists():p.write_text(json.dumps({'synthetic':True,'seed':42,'split_before_windowing':True,'fit_preprocessing':'train_only','mask_true_means':'allowed','device':'cpu','agent_permissions':'read_only_mock','operational_thresholds':None},ensure_ascii=False,indent=2))
