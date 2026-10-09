"""只執行已完整閱讀且SHA固定的15章及16章NumPy片段；不執行PyTorch或訓練新模型。"""
import hashlib
from pathlib import Path
import platform
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from book5_editor import ENGINE as E
PINS={15:'5904991a57ca1ceac956919260923a49028be8fa178b92d49eecb0f39d4863a5',
      16:'62565894565056ac6bf70042506d5c08811eb47e27ae6213b10ac7af947da8ee'}


def load(n,indices,count):
    raw=(ROOT/'books/neural-transformers/editorial/cross-recovery-20261008'/f'{n:02d}.md').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PINS[n]:raise RuntimeError('Snapshot changed; refusing execution')
    blocks=[f['content'] for f in E.markdown_structure(raw.decode())['fences'] if f['language']=='python']
    if len(blocks)!=count:raise RuntimeError('Unexpected code block count')
    scope={'__name__':'reviewed_snapshot'}
    for i in indices:exec(compile(blocks[i],f'reviewed-ch{n}-block{i}','exec'),scope)
    return scope


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.masks=load(15,[0],1);cls.mha=load(16,[0,1,2,4],6)

    def test_full_input_weight_bias_gradients(self):
        error=self.mha['run_mha_checks']()
        print('MHA maximum gradient error:',error,flush=True)
        self.assertLess(error,2e-7)

    def test_all_mask_interfaces(self):self.masks['run_checks']()

    def test_split_adjoint(self):
        m=self.mha['MultiHeadAttention'](4,2)
        rng=np.random.default_rng(1);x=rng.normal(size=(2,3,4));g=rng.normal(size=(2,2,3,2))
        self.assertAlmostEqual(float((m.split_heads(x)*g).sum()),float((x*m.merge_heads(g)).sum()),places=13)

    def test_future_token_does_not_change_past(self):
        m=self.mha['MultiHeadAttention'](4,2);rng=np.random.default_rng(8)
        x=rng.normal(size=(1,3,4));mask=np.tril(np.ones((3,3),dtype=bool))
        original=m.forward(x,mask)[0];x[:,2,:]+=50
        np.testing.assert_allclose(original[:,:2],m.forward(x,mask)[0][:,:2],atol=1e-13)

    def test_backward_uses_forward_snapshot(self):
        m=self.mha['MultiHeadAttention'](4,2);x=np.arange(8,dtype=float).reshape(1,2,4)/10
        m.forward(x);up=np.arange(8,dtype=float).reshape(1,2,4)
        dx,g=m.backward(up);m.W_Q+=2;dx2,g2=m.backward(up)
        np.testing.assert_array_equal(dx,dx2)
        for name in g:np.testing.assert_array_equal(g[name],g2[name])

    def test_single_head_reference(self):
        m=self.mha['MultiHeadAttention'](4,1,D_out=3);x=np.arange(8,dtype=float).reshape(1,2,4)/10
        q=x@m.W_Q+m.b_Q;k=x@m.W_K+m.b_K;v=x@m.W_V+m.b_V
        s=q@k.transpose(0,2,1)/2;s-=s.max(axis=-1,keepdims=True)
        p=np.exp(s);p/=p.sum(axis=-1,keepdims=True)
        y,a=m.forward(x)
        np.testing.assert_allclose(y,(p@v)@m.W_O+m.b_O,atol=1e-13)
        np.testing.assert_allclose(a[:,0],p,atol=1e-13)

    def test_parameter_count_general_and_default(self):
        for D,H,dv,out in ((4,2,2,4),(4,2,3,3),(8,4,1,5),(4,1,4,4)):
            model=self.mha['MultiHeadAttention'](D,H,D_out=out,dv=dv)
            actual=sum(getattr(model,n).size for n in model.parameter_names)
            self.assertEqual(actual,2*D*D+2*D+(D+1)*H*dv+(H*dv+1)*out)
            if dv==D//H:self.assertEqual(actual,3*D*D+3*D+D*out+out)

    def test_quantization_can_be_exact(self):
        for qmax in (7,127):
            scale=.5/qmax;code=round(.5/scale)
            self.assertAlmostEqual(scale*code,.5,places=15)

    def test_windows_not_overlapping(self):
        context=4;starts=list(range(0,12-1,context));self.assertEqual(starts,[0,4,8])
        inputs=[set(range(s,s+context)) for s in starts]
        self.assertFalse(inputs[0]&inputs[1])
        targets=set(range(1,5));self.assertEqual(targets&inputs[1],{4})


if __name__=='__main__':
    print('Python',platform.python_version(),'NumPy',np.__version__,'CPU',platform.machine(),PINS,flush=True)
    unittest.main(verbosity=2)
