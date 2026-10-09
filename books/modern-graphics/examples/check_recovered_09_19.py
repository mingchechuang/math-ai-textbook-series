"""僅驗證已逐段檢視、雜湊固定的回收候選稿；不自動執行後續模型修改。
CPU/NumPy測試，不呼叫run_tests、不寫PPM、不存取網路。
"""
import hashlib
import sys
import unittest
from pathlib import Path
import numpy as np

BOOK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BOOK.parents[1]))
from book2_editor import markdown_structure

HASHES={9:'de69c779f888ad5f536668e6e1199ca1e89d29e6c8c2325336ec5fa764e1b84c',
        19:'d0c33f564501a27024aad97b8077ca8a0b6eea0782d10a3bd23c0de47bc339b2'}

def load_reviewed(n):
    source=BOOK/'editorial'/f'recovered-{n:02d}-candidate.md'
    code='\n\n'.join(f['content'] for f in markdown_structure(source.read_text())['fences'] if f['language']=='python')
    if hashlib.sha256(code.encode()).hexdigest()!=HASHES[n]:
        raise RuntimeError('程式已變更，必須先重新檢視，不執行未知版本')
    namespace={'__name__':'reviewed_snippet'}
    exec(compile(code,str(source),'exec'),namespace)
    return namespace


class RecoveryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.curve=load_reviewed(9);cls.ray=load_reviewed(19)

    def test_line_and_zero_length(self):
        build=self.curve['build_arc_length_table'];lookup=self.curve['parameter_at_length']
        ts,arc,total=build([(0,0),(2,0)],20)
        self.assertAlmostEqual(total,2)
        for s,t in [(0,0),(.5,.25),(2,1)]:self.assertAlmostEqual(lookup(ts,arc,s),t)
        ts,arc,total=build([(1,1),(1,1)],20)
        self.assertEqual(total,0);self.assertEqual(lookup(ts,arc,0),0)
        with self.assertRaises(ValueError):lookup(ts,arc,.1)

    def test_repeated_lengths_and_invalid_inputs(self):
        f=self.curve['parameter_at_length'];build=self.curve['build_arc_length_table']
        self.assertAlmostEqual(f([0,.25,.5,1],[0,0,1,1],.5),.375)
        self.assertEqual(f([0,.25,.5,1],[0,0,1,1],1),1)
        for m in [0,-1,True,2.5]:
            with self.assertRaises(ValueError):build([(0,0),(1,0)],m)
        for s in [-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError):f([0,1],[0,1],s)
        for arc in [[0,-1],[0,float('inf')]]:
            with self.assertRaises(ValueError):f([0,1],arc,0)

    def test_g1_counterexamples(self):
        f=self.curve['check_g1_continuity'];a=[(0,0),(1,0)]
        self.assertTrue(f(a,[(1,0),(2,0)])[0])
        for b in [[(1,0),(0,0)],[(1,0),(1,0)],[(2,0),(3,0)]]:self.assertFalse(f(a,b)[0])
        with self.assertRaises(ValueError):f(a,[(1,0,0),(2,0,0)])
        # 局部導數相同，但時長不同時全域導數不同。
        da=np.array([1.,0.]);db=np.array([1.,0.])
        self.assertFalse(np.allclose(da/1,db/2))

    def test_sphere_nearest_and_near_surface(self):
        R=self.ray['Ray'];S=self.ray['Sphere'];hit=self.ray['intersect_sphere'];s=S(np.zeros(3),1)
        self.assertAlmostEqual(hit(R(np.array([0,0,-5.]),np.array([0,0,1.])),s),4)
        self.assertAlmostEqual(hit(R(np.array([1,0,-5.]),np.array([0,0,1.])),s),5)
        origin=np.array([0,0,1-1e-8]);direction=np.array([0,0,1.])
        self.assertAlmostEqual(hit(R(origin,direction,t_min=0),s),1e-8,places=12)
        self.assertIsNone(hit(R(origin,direction,t_min=1e-6),s))

    def test_triangle_and_plane(self):
        R=self.ray['Ray'];T=self.ray['Triangle'];P=self.ray['Plane']
        tri=T(np.array([0.,0,0]),np.array([1.,0,0]),np.array([0.,1,0]))
        hit=self.ray['intersect_triangle'](R(np.array([.2,.2,-1]),np.array([0.,0,1])),tri)
        self.assertAlmostEqual(hit[0],1);np.testing.assert_allclose(hit[2],[.6,.2,.2])
        self.assertIsNone(self.ray['intersect_plane'](R(np.zeros(3),np.array([1.,0,0])),P(np.array([0.,1,0]),0)))

    def test_shadow_endpoints_and_occluders(self):
        f=self.ray['shadow_test'];S=self.ray['Sphere'];P=self.ray['Plane']
        p=np.zeros(3);n=np.array([0.,0,1]);light=np.array([0.,0,10])
        self.assertTrue(f(p,n,light,[S(np.array([0.,0,5]),1)],0))
        self.assertFalse(f(p,n,light,[S(np.array([0.,0,12]),1)],0))
        self.assertFalse(f(p,n,light,[P(n,-10)],0))
        # 近根在排除端點，但球的遠根在線段內，不能漏掉。
        self.assertTrue(f(p,n,light,[S(np.array([0.,0,1]),1)],0))

    def test_render_and_invalid_geometry(self):
        r=self.ray;S=r['Sphere'];P=r['Plane'];T=r['Triangle']
        objects=[S(np.array([0.,0,-2]),.5),P(np.array([0.,1,0]),0),T(np.array([-1.,.1,-3]),np.array([1.,.1,-3]),np.array([0.,1.5,-3]))]
        img=r['render'](16,16,objects,np.array([0.,1,0]),np.array([0.,0,-1]))
        self.assertEqual(img.shape,(16,16,3));self.assertEqual(img.dtype,np.uint8)
        for color in ([255,0,0],[0,255,0],[0,0,255]):self.assertTrue(np.any(np.all(img==color,axis=-1)))
        with self.assertRaises(ValueError):S(np.zeros(3),float('nan'))
        with self.assertRaises(ValueError):r['Ray'](np.zeros(2),np.ones(3))
        with self.assertRaises(ValueError):r['render'](16,0,objects,np.zeros(3),np.array([0.,0,-1]))


if __name__=='__main__':unittest.main(verbosity=2)
