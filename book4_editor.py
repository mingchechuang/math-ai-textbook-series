"""Volume IV隔離工作流；按任務分session，使用免JSON跳脫的精準patch。"""
import argparse
import importlib.util
from pathlib import Path
import re
import sys
import book4_spec
import book4_assets


def parse_patches(text):
    if not isinstance(text,str):raise ValueError('patch必須是純文字')
    text=text.replace('\r\n','\n')
    pattern=re.compile(r'(?m)^<<<PATCH ([0-9]{1,2})>>>\n<<<OLD>>>\n(.*?)\n<<<NEW>>>\n(.*?)\n<<<END>>>[ \t]*(?=\n|$)',re.S)
    patches=[];end=0
    for match in pattern.finditer(text):
        if text[end:match.start()].strip():raise ValueError('patch區塊外有額外文字')
        patches.append({'chapter':int(match[1]),'old':match[2],'new':match[3]});end=match.end()
    if text[end:].strip() or not 1<=len(patches)<=5:raise ValueError('需1至5個完整patch區塊，不可加包裝或說明')
    return {'patches':patches}


def load_engine():
    name='_volume_four_editor_engine'
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('book2_editor.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    module.S=book4_spec;module.A=book4_assets
    base=module.Editor

    class FourthEditor(base):
        def patch(self,person,phase,context,chapters):
            for attempt in range(2):
                raw=self.invoke(person,phase,context,module.S.PATCH_RULES,'markdown')
                try:return module.checked_patches(chapters,parse_patches(raw))
                except (ValueError,KeyError,TypeError) as exc:
                    if attempt==1:raise
                    context={**context,'previous_invalid_patch':raw,'patch_error':str(exc),
                             'task':'對照同一原稿重給逐字唯一且不重疊的patch。不要JSON或整篇包裝；保留原縮排及數學反斜線。'}
    module.Editor=FourthEditor
    return module


ENGINE=load_engine()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=book4_spec.ROOT)
    parser.add_argument('--init-only',action='store_true')
    parser.add_argument('--timeout',type=int,default=1200)
    parser.add_argument('--max-calls',type=int,default=0)
    parser.add_argument('--targeted-repair',action='store_true')
    ENGINE.run(parser.parse_args())
