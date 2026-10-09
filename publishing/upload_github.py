"""Create a PRIVATE GitHub repository and push the curated export. Token stays in memory.
Run interactively with a newly issued token; never put a token in a command argument.
"""
import argparse
import base64
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request
try:
    from .export_repository import audit
except ImportError:
    from export_repository import audit


def upload(root,owner,name):
    root=Path(root).resolve()
    if not (root/'EXPORT_SUMMARY.json').is_file():raise RuntimeError('Only a curated export may be uploaded')
    if not all(c.isalnum() or c in '-_' for c in owner+name):raise ValueError('Invalid owner/repository name')
    audit(root)
    token=os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN') or getpass.getpass('New GitHub token (hidden; do not reuse the exposed token): ')
    if not token or any(c.isspace() for c in token):raise ValueError('Invalid token input')
    def api(path,data=None,missing_ok=False):
        req=urllib.request.Request('https://api.github.com'+path,data=None if data is None else json.dumps(data).encode(),headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'math-ai-textbook-export','Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)
        except urllib.error.HTTPError as e:
            if missing_ok and e.code==404:return None
            raise RuntimeError(f'GitHub API request failed (HTTP {e.code}); response omitted') from None
        except urllib.error.URLError:
            raise RuntimeError('GitHub connection failed; details omitted') from None
    user=api('/user')
    if user.get('login','').lower()!=owner.lower():raise RuntimeError('Authenticated account does not match requested owner')
    repository=api(f'/repos/{owner}/{name}',missing_ok=True)
    if repository is None:
        repository=api('/user/repos',{'name':name,'description':'Five-volume mathematics, AI and simulation textbooks, publications and curated editorial discussions','private':True,'auto_init':False})
    if not repository.get('private'):raise RuntimeError('Refusing automatic upload to a public repository')
    remote=f'https://github.com/{owner}/{name}.git'
    env={k:v for k,v in os.environ.items() if not k.startswith('GIT_') and k not in ('GH_TOKEN','GITHUB_TOKEN')}
    env.update(GIT_TERMINAL_PROMPT='0',GIT_CONFIG_COUNT='3',GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',GIT_CONFIG_VALUE_0='Authorization: Basic '+base64.b64encode(('x-access-token:'+token).encode()).decode(),GIT_CONFIG_KEY_1='credential.helper',GIT_CONFIG_VALUE_1='',GIT_CONFIG_KEY_2='http.followRedirects',GIT_CONFIG_VALUE_2='false')
    def git(*args):
        r=subprocess.run(['git',*args],cwd=root,env=env,capture_output=True,text=True)
        if r.returncode:raise RuntimeError('Git operation failed; output suppressed to protect credentials: '+args[0])
        return r.stdout.strip()
    git('init','-b','main')
    git('config','user.name',owner);git('config','user.email',f'{user["id"]}+{owner}@users.noreply.github.com')
    git('add','.')
    if git('status','--porcelain'):git('commit','-m','Publish five textbook volumes and curated editorial discussions')
    # An existing nonmatching history is rejected by ordinary push. Never force-push.
    git('push',remote,'HEAD:refs/heads/main')
    local=git('rev-parse','HEAD');remote_head=git('ls-remote',remote,'refs/heads/main').split()[0]
    if local!=remote_head:raise RuntimeError('Remote commit verification failed')
    receipt={'url':f'https://github.com/{owner}/{name}','private':True,'branch':'main','commit':local,'verified_remote_commit':remote_head}
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--owner',default='mingchechuang');p.add_argument('--name',default='math-ai-textbook-series');a=p.parse_args()
    try:upload(a.directory,a.owner,a.name)
    except Exception as error:
        if isinstance(error,(RuntimeError,ValueError)):print(str(error),file=sys.stderr)
        else:print('Upload failed; exception details suppressed to protect credentials',file=sys.stderr)
        sys.exit(1)
