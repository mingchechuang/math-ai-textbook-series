"""只執行已完整閱讀、雜湊固定的16章兩段及26章習題程式；非通用稿件執行器。"""
import hashlib
import math
from pathlib import Path
import platform
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from book4_editor import ENGINE as E
PINS={16:'c1a3fd96c328295644169a4c990a2ad5f1b846b0e8c351ea712306d216580201',
      26:'66264926c5507fedf54c04b9a2cafb653173ac969f95717b2d9c596eeda83c37'}


def load_reviewed(n,indices):
    filename='16-v2.md' if n==16 else '26.md'
    p=ROOT/'books/calculus-analysis/editorial/cross-recovery-20261006'/filename
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PINS[n]:raise RuntimeError('Snapshot changed; refuse execution')
    blocks=[f['content'] for f in E.markdown_structure(raw.decode())['fences'] if f['language']=='python']
    if len(blocks)!=2:raise RuntimeError('Unexpected code block count')
    scope={'__name__':'reviewed_snapshot'}
    for i in indices:exec(compile(blocks[i],f'reviewed-{n}-block-{i}','exec'),scope)
    return scope


class IndependentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optim=load_reviewed(16,[0,1]);cls.picard=load_reviewed(26,[1])

    def test_newton_non_descent(self):
        g=np.array([-1.5,1.]);H=np.diag([-1.,2.]);d=np.linalg.solve(H,-g)
        np.testing.assert_array_equal(d,[-1.5,-.5]);self.assertEqual(g@d,1.75)
        f=lambda z:(z[0,0]**2-1)**2+z[1,0]**2
        grad=lambda z:np.array([[4*z[0,0]*(z[0,0]**2-1)],[2*z[1,0]]])
        hess=lambda z:np.diag([12*z[0,0]**2-4,2.])
        _,log=self.optim['newton']([.5,.5],f,grad,hess)
        self.assertEqual(log[-1][4],'newton_not_descent')
        point,log=self.optim['newton']([0,.5],f,grad,hess)
        np.testing.assert_allclose(point,0,atol=1e-14)
        self.assertEqual(log[-1][4],'grad_tol');self.assertLess(np.linalg.eigvalsh(hess(point))[0],0)

    def test_saddle_one_step(self):
        f,g,H=self.optim['quad_factory'](np.diag([2.,-2.]),np.zeros(2))
        point,log=self.optim['newton']([1,.1],f,g,H)
        np.testing.assert_allclose(point,0,atol=1e-14)
        self.assertEqual(sum(row[4]=='ok' for row in log),1)
        self.assertEqual(log[-1][4],'grad_tol')

    def test_near_minimizer(self):
        A=np.array([[4.,1.],[1.,3.]]);f,g,H=self.optim['quad_factory'](A,np.ones(2))
        x=np.linalg.solve(A,np.ones(2))+1e-12
        self.assertLess(np.linalg.norm(g(x)),1e-10)
        for solver in ('newton','gradient_descent'):
            args={'hess':H} if solver=='newton' else {}
            _,log=self.optim[solver](x,f,g,**args)
            self.assertEqual(log[0][0],0);self.assertEqual(log[0][4],'grad_tol')

    def test_fixed_step_count(self):
        self.assertGreater(9*.99**2051,1e-8)
        self.assertLess(9*.99**2052,1e-8)
        f,_,_=self.optim['quad_factory'](np.diag([1.,100.]),np.array([1.,100.]))
        self.assertEqual(f(np.ones(2)),-50.5)

    def test_picard_recurrence(self):
        coefficients=self.picard['picard_cos_coefficients'];evaluate=self.picard['evaluate_picard']
        for n in range(1,11):
            a,b,p=coefficients(n);pa,pb,pp=coefficients(n-1)
            self.assertEqual(a,1+pb);self.assertEqual(-b,pa)
            self.assertEqual([p[j+1]*(j+1) for j in range(len(p)-1)],pp)
            for t in (0,.25,.5,1):
                derivative=float(a)*math.cos(t)-float(b)*math.sin(t)+sum(float(p[j])*j*t**(j-1) for j in range(1,len(p)))
                self.assertAlmostEqual(derivative,math.cos(t)+evaluate(n-1,t),places=13)
        for t in (.1,.5,1.):
            self.assertAlmostEqual(evaluate(3,t),t+1-math.cos(t),places=14)
            self.assertAlmostEqual(evaluate(4,t),t+t*t/2,places=14)
        self.assertAlmostEqual(evaluate(10,1),1.5097252265588046,places=14)

    def test_guaranteed_radius(self):
        self.assertEqual(min(1,1/17,1/(2*math.sqrt(21))),1/17)
        self.assertLess(1/17,.059)
        self.assertEqual(min(.01,1/17,1/(2*math.sqrt(21))),.01)


if __name__=='__main__':
    print('Python',platform.python_version(),'NumPy',np.__version__,'CPU',platform.machine(),flush=True)
    print('Snapshot pins',PINS,flush=True)
    unittest.main()
