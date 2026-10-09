"""跨章修補驗證：標準庫CPU；第30章只執行已檢視、雜湊固定的兩個函式。"""
import ast
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

BOOK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BOOK.parents[1]))
from book2_editor import markdown_structure


def edge(a,b,p):return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])

def covered(triangle,p,top_left):
    a,b,c=triangle
    if edge(a,b,c)<0:a,b=b,a
    for start,end in [(a,b),(b,c),(c,a)]:
        e=edge(start,end,p)
        if e<0:return False
        if top_left and e==0 and not(end[1]<start[1] or (end[1]==start[1] and end[0]>start[0])):return False
    return True


class CrossChecks(unittest.TestCase):
    def test_viewport(self):
        u,v=(.1+1)*8/2,(1+.1)*8/2
        self.assertEqual((math.floor(u),math.floor(v)),(4,4))
        self.assertEqual((4*8+4)*3,108)
        self.assertFalse(0<=(1+1)*8/2<8) # 右邊界不是索引8

    def test_overlap_is_not_removed_by_top_left(self):
        triangles=[[(0,0),(32,0),(0,32)],[(0,0),(0,32),(32,32)],[(16,0),(32,16),(16,32)]]
        for rule,expected in [(True,{0:144,1:528,2:352}),(False,{0:112,1:496,2:416})]:
            actual=Counter(sum(covered(t,(u+.5,v+.5),rule) for t in triangles) for v in range(32) for u in range(32))
            self.assertEqual(dict(actual),expected)

    def test_nlerp_midpoint(self):
        values=[]
        for t in [.25,.5,.75]:
            w=1-t+t*math.cos(math.pi/4);z=t*math.sin(math.pi/4)
            angle=2*math.atan2(z,w)*180/math.pi
            values.append(angle)
        self.assertLess(values[0],22.5);self.assertAlmostEqual(values[1],45);self.assertGreater(values[2],67.5)
        self.assertAlmostEqual(values[0]+values[2],90)

    def test_config_snapshot_rejection(self):
        source=(BOOK/'editorial/cross-repair-20261002/30.md').read_text()
        nodes=[]
        for fence in markdown_structure(source)['fences']:
            if fence['language']=='python':nodes.extend(n for n in ast.parse(fence['content']).body if isinstance(n,ast.FunctionDef) and n.name in ('config_id','validate_frame_record'))
        tree=ast.Module(body=nodes,type_ignores=[])
        self.assertEqual(hashlib.sha256(ast.dump(tree).encode()).hexdigest(),'434eb0c9f1abd3eae10407eff80dcba4f15059a3ceff5c3896da067c9e96c4fe')
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)
            namespace={'json':json,'hashlib':hashlib,'OUTPUT':output}
            exec(compile(tree,'reviewed_ch30_functions','exec'),namespace)
            config={'width':16};identifier=namespace['config_id'](config)
            (output/'frame.json').write_text(json.dumps({'frame_index':0,'time_s':0,'objects':[]}))
            manifest={'data_status':'simulated','config_snapshot':config,'config_id':identifier,'dataset_id':'test','generator_version':'test','frames':[{'frame_index':0,'time_s':0,'annotation':'frame.json'}]}
            validate=namespace['validate_frame_record']
            self.assertEqual(validate(manifest,0)['config_id'],identifier)
            missing=copy.deepcopy(manifest);del missing['config_snapshot']
            with self.assertRaises(ValueError):validate(missing,0)
            changed=copy.deepcopy(manifest);changed['config_snapshot']['width']=32
            with self.assertRaises(ValueError):validate(changed,0)


if __name__=='__main__':unittest.main(verbosity=2)
