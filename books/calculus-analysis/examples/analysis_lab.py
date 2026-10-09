"""主編原創標準庫實驗：數值核對不是分析定理的證明。"""
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
