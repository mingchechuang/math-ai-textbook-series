import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from types import SimpleNamespace
from publishing.export_repository import audit,clean
from publishing.upload_github import upload


class ExportUploadTests(unittest.TestCase):
    def test_secret_patterns_redacted_and_rejected(self):
        secret='ghp_'+'x'*36
        self.assertNotIn(secret,clean('token='+secret))
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'example.txt').write_text(secret)
            with self.assertRaises(ValueError):audit(p)

    def test_upload_auth_stays_out_of_arguments_and_remote_verified(self):
        sha='a'*40;token='synthetic-test-token'
        responses=[io.StringIO(json.dumps(x)) for x in ({'login':'mingchechuang','id':1},{'private':True})]
        def git(args,**kwargs):
            self.assertNotIn(token,' '.join(args))
            self.assertEqual(kwargs['env']['GIT_CONFIG_KEY_0'],'http.https://github.com/.extraheader')
            result={'status':' M README.md','rev-parse':sha,'ls-remote':sha+'\trefs/heads/main'}.get(args[1],'')
            return SimpleNamespace(returncode=0,stdout=result,stderr='')
        with tempfile.TemporaryDirectory() as t,patch.dict('os.environ',{},clear=True),patch('getpass.getpass',return_value=token),patch('urllib.request.urlopen',side_effect=responses),patch('subprocess.run',side_effect=git) as run,redirect_stdout(io.StringIO()) as output:
            p=Path(t);(p/'EXPORT_SUMMARY.json').write_text('{}')
            upload(p,'mingchechuang','math-ai-textbook-series')
            self.assertEqual(json.loads(output.getvalue())['verified_remote_commit'],sha)
            self.assertNotIn(token,output.getvalue())
            self.assertFalse(any('--force' in call.args[0] for call in run.call_args_list))

    def test_public_repository_refused(self):
        responses=[io.StringIO(json.dumps(x)) for x in ({'login':'mingchechuang','id':1},{'private':False})]
        with tempfile.TemporaryDirectory() as t,patch.dict('os.environ',{},clear=True),patch('getpass.getpass',return_value='test-only'),patch('urllib.request.urlopen',side_effect=responses),patch('subprocess.run') as run:
            p=Path(t);(p/'EXPORT_SUMMARY.json').write_text('{}')
            with self.assertRaises(RuntimeError):upload(p,'mingchechuang','math-ai-textbook-series')
            run.assert_not_called()


if __name__=='__main__':unittest.main()
