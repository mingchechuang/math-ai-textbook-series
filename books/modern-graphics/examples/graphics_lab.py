"""Volume II起步實驗：標準庫、CPU、合成數值；無外部檔案／設備。"""
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
