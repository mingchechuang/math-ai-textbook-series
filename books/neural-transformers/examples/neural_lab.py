"""獨立主編起步測試：CPU NumPy，不載入或執行生成章稿。"""
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
