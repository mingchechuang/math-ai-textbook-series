"""第五卷隔離引擎：保留五模型、任務分session與精準patch安全檢查。"""
import argparse
import importlib.util
from pathlib import Path
import sys
import book5_spec
import book5_assets
from book4_editor import parse_patches


def load_engine():
    name='_volume_five_editor_engine'
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('book2_editor.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    module.S=book5_spec;module.A=book5_assets
    base=module.Editor
    class FifthEditor(base):
        def patch(self,person,phase,context,chapters):
            for attempt in range(2):
                raw=self.invoke(person,phase,context,module.S.PATCH_RULES,'markdown')
                try:return module.checked_patches(chapters,parse_patches(raw))
                except (ValueError,KeyError,TypeError) as exc:
                    if attempt==1:raise
                    context={**context,'previous_invalid_patch':raw,'patch_error':str(exc),
                             'task':'縮小到真正需要修改的唯一原文區塊，不重播失敗的巨大patch；同一原稿上互不重疊，保留縮排及反斜線。'}
    module.Editor=FifthEditor
    return module


ENGINE=load_engine()
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=book5_spec.ROOT)
    parser.add_argument('--init-only',action='store_true')
    parser.add_argument('--timeout',type=int,default=1200)
    parser.add_argument('--max-calls',type=int,default=0)
    parser.add_argument('--targeted-repair',action='store_true')
    ENGINE.run(parser.parse_args())
