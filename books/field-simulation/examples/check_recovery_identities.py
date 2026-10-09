"""主編獨立核對第19章修復契約；不載入或執行模型章稿。標準庫、CPU。"""
import math
import unittest

L=4.0
K=2*math.pi/L
RHO=998.0
NU=.03
ALPHA=.2
AMPLITUDE=.7*math.exp(-ALPHA*.4)


def values(x,y):
    a=AMPLITUDE;k=K
    return (a*math.sin(k*x)*math.cos(k*y),
            -a*math.cos(k*x)*math.sin(k*y),
            RHO*a*a*(math.cos(2*k*x)+math.cos(2*k*y))/4)


def spatial_residual(n):
    h=L/n;total=0.
    for j in range(n):
        for i in range(n):
            x=(i+.5)*h;y=(j+.5)*h
            center=values(x,y);east=values(x+h,y);west=values(x-h,y)
            north=values(x,y+h);south=values(x,y-h)
            for component in (0,1):
                q=center[component]
                dx=(east[component]-west[component])/(2*h)
                dy=(north[component]-south[component])/(2*h)
                lap=(east[component]+west[component]+north[component]+south[component]-4*q)/(h*h)
                pressure=(east[2]-west[2])/(2*h) if component==0 else (north[2]-south[2])/(2*h)
                force=(2*NU*K*K-ALPHA)*q
                r=-ALPHA*q+center[0]*dx+center[1]*dy+pressure/RHO-NU*lap-force
                total+=r*r
    return math.sqrt(total/(n*n))


class Checks(unittest.TestCase):
    def projection(self,u,p,h,dt,rho):
        n=len(u);div=[(u[(i+1)%n]-u[i])/h for i in range(n)]
        b=[-rho*d/dt for d in div]
        ap=[(2*p[i]-p[(i-1)%n]-p[(i+1)%n])/(h*h) for i in range(n)]
        for a,c in zip(ap,b):self.assertAlmostEqual(a,c)
        return [u[i]-dt/rho*(p[i]-p[(i-1)%n])/h for i in range(n)]
    def test_four_cell_projection(self):
        self.assertEqual(self.projection([1,3,1,1],[-1.875,1.875,.625,-.625],.25,.1,1),[1.5]*4)
    def test_two_cell_duplicate_neighbors(self):
        self.assertEqual(self.projection([1,-1],[.5,-.5],1,1,1),[0,0])
    def test_manufactured_both_momenta(self):
        for x,y in ((.33,.51),(1.1,2.3),(2.7,.8)):
            a=AMPLITUDE;k=K;u,v,_=values(x,y)
            ux=a*k*math.cos(k*x)*math.cos(k*y)
            uy=-a*k*math.sin(k*x)*math.sin(k*y)
            vx=a*k*math.sin(k*x)*math.sin(k*y)
            vy=-a*k*math.cos(k*x)*math.cos(k*y)
            self.assertAlmostEqual(ux+vy,0)
            gradients=(-RHO*a*a*k*math.sin(2*k*x)/2,-RHO*a*a*k*math.sin(2*k*y)/2)
            for q,advection,gp in zip((u,v),(u*ux+v*uy,u*vx+v*vy),gradients):
                lhs=-ALPHA*q+advection
                rhs=-gp/RHO-2*NU*k*k*q+(2*NU*k*k-ALPHA)*q
                self.assertAlmostEqual(lhs,rhs,places=12)
    def test_nonconstant_pressure_is_not_unforced_stokes(self):
        gradient=-AMPLITUDE**2*K*math.sin(2*K*.33)/2
        self.assertGreater(abs(gradient),.01)
    def test_specific_spatial_refinement(self):
        errors=[spatial_residual(n) for n in (16,32,64)]
        print('manufactured momentum RMS for N=16/32/64:',errors)
        for a,b in zip(errors,errors[1:]):
            self.assertGreater(a/b,3.5);self.assertLess(a/b,4.5)


if __name__=='__main__':unittest.main(verbosity=2)
