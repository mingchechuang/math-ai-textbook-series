"""主編原創CPU起步實驗；標準庫、合成無因次週期場，不讀寫外部資料。"""
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
