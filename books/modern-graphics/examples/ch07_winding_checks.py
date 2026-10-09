"""主編獨立重算第7章的池殼／頂蓋繞序；不執行章稿內程式。"""
import unittest
from collections import defaultdict

V=[(-2,0,-1.5),(2,0,-1.5),(2,0,1.5),(-2,0,1.5),
   (-2,1,-1.5),(2,1,-1.5),(2,1,1.5),(-2,1,1.5)]
F=[(0,2,1),(0,3,2),(0,1,5),(0,5,4),(1,2,6),(1,6,5),
   (2,3,7),(2,7,6),(3,0,4),(3,4,7)]


def normal(face):
    p,q,r=(V[i] for i in face)
    a=[q[i]-p[i] for i in range(3)];b=[r[i]-p[i] for i in range(3)]
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def edges(faces):
    result=defaultdict(list)
    for a,b,c in faces:
        for edge in [(a,b),(b,c),(c,a)]:result[tuple(sorted(edge))].append(edge)
    return result


class WindingTests(unittest.TestCase):
    def test_open_tank(self):
        e=edges(F)
        self.assertEqual((len(V),len(e),len(F)),(8,17,10))
        self.assertEqual(sum(len(v)==1 for v in e.values()),4)
        self.assertTrue(all(len(v)!=2 or v[0]==v[1][::-1] for v in e.values()))

    def test_consistent_inward_lid(self):
        lid=[(4,5,6),(4,6,7)];e=edges(F+lid)
        self.assertEqual(normal(lid[0]),(0,-12,0))
        self.assertEqual(normal(lid[1]),(0,-12,0))
        self.assertEqual((len(e),len(F+lid)),(18,12))
        self.assertTrue(all(len(v)==2 and v[0]==v[1][::-1] for v in e.values()))
        self.assertEqual(len(V)-len(e)+len(F+lid),2)

    def test_upward_water_is_not_same_orientation(self):
        lid=[(4,6,5),(4,7,6)];e=edges(F+lid)
        self.assertEqual(normal(lid[0]),(0,12,0))
        self.assertEqual(sum(len(v)==2 and v[0]==v[1] for v in e.values()),4)
        self.assertEqual(sum(len(v)==1 for v in e.values()),0)


if __name__=='__main__':unittest.main(verbosity=2)
