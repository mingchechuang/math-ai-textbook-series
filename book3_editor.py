"""Volume III入口；隔離載入已測試工作流，不改Volume II模組全域或稿件。"""
import argparse
import importlib.util
from pathlib import Path
import sys
import book3_spec
import book3_assets


def load_engine():
    name='_volume_three_editor_engine'
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('book2_editor.py'))
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    module.S=book3_spec
    module.A=book3_assets
    return module


ENGINE=load_engine()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=book3_spec.ROOT)
    parser.add_argument('--init-only',action='store_true')
    parser.add_argument('--timeout',type=int,default=1200)
    parser.add_argument('--max-calls',type=int,default=0)
    parser.add_argument('--targeted-repair',action='store_true')
    ENGINE.run(parser.parse_args())
