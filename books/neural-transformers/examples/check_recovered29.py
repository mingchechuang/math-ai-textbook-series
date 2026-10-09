"""只執行已完整閱讀且SHA256固定的第29章mock程式；不讀後續模型改稿。"""
import hashlib
import json
from pathlib import Path
import platform
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from book5_editor import ENGINE as E
PIN='a72da16d0169b3c8337e2e8f21fe71558499288f51c01c823afbafef1d4e1c1d'


def load_reviewed():
    p=ROOT/'books/neural-transformers/editorial/recovery-20261007/29-reviewed.md'
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PIN:raise RuntimeError('Reviewed snapshot changed; refuse execution')
    blocks=[f['content'] for f in E.markdown_structure(raw.decode())['fences'] if f['language']=='python']
    if len(blocks)!=22:raise RuntimeError('Unexpected code block count')
    scope={'__name__':'reviewed_mock_chapter'}
    for i,code in enumerate(blocks):exec(compile(code,f'reviewed29-block-{i}','exec'),scope)
    return scope


class IndependentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.code=load_reviewed()

    def test_all_chapter_assertions_and_chain(self):
        self.assertTrue(self.code['gateway'].logger.verify_chain())
        self.assertEqual(self.code['flaky_count'],2)

    def test_transient_failures_have_events(self):
        g=self.code['ReadOnlyAgentGateway']();calls=[]
        def fail(p):
            calls.append(1)
            raise self.code['TransientReadError']('not logged private detail')
        g.register_tool('fail',{},['monitor'],fail)
        r=g.execute_tool_call({'tool':'fail','params':{}},max_retries=2)
        self.assertEqual(r['code'],'RETRY_EXHAUSTED');self.assertEqual(len(calls),3)
        failed=[e['event']['details'] for e in g.logger.events if e['event']['status']=='ATTEMPT_FAILED']
        self.assertEqual([e['attempt'] for e in failed],[1,2,3])
        self.assertEqual([e['will_retry'] for e in failed],[True,True,False])
        self.assertNotIn('private detail',json.dumps(g.logger.events))

    def test_permanent_exception_not_retried(self):
        g=self.code['ReadOnlyAgentGateway']();calls=[]
        def fail(p):
            calls.append(1);raise RuntimeError('do not expose me')
        g.register_tool('fail',{},['monitor'],fail)
        result=g.execute_tool_call({'tool':'fail','params':{}},max_retries=3)
        self.assertEqual(result['code'],'EXECUTION_FAILED');self.assertEqual(len(calls),1)
        self.assertEqual(g.logger.events[-1]['event']['details']['reason_code'],'HANDLER_EXCEPTION')
        self.assertNotIn('do not expose me',json.dumps(g.logger.events))

    def test_empty_string_boundary_and_huge_number(self):
        g=self.code['ReadOnlyAgentGateway']()
        g.register_tool('empty',{'s':{'type':'str','max_length':0},'required':['s']},['monitor'],lambda p:p)
        self.assertEqual(g.execute_tool_call({'tool':'empty','params':{'s':''}})['status'],'success')
        self.assertEqual(g.execute_tool_call({'tool':'empty','params':{'s':'a'}})['code'],'SCHEMA_VIOLATION')
        g.register_tool('number',{'x':{'type':'float'},'required':['x']},['monitor'],lambda p:p)
        self.assertEqual(g.execute_tool_call({'tool':'number','params':{'x':10**400}})['code'],'SCHEMA_VIOLATION')
        with self.assertRaises(ValueError):g.register_tool('bounds',{'x':{'type':'float','min':10**400}},['monitor'],lambda p:p)

    def test_invalid_top_keys_are_not_logged_raw(self):
        g=self.code['ReadOnlyAgentGateway']()
        response=g.execute_tool_call({'tool':'x','params':{},object():{1}})
        self.assertEqual(response['code'],'INVALID_KEYS');self.assertTrue(g.logger.verify_chain())

    def test_registration_rejects_typos_and_is_atomic(self):
        g=self.code['ReadOnlyAgentGateway']()
        for schema in ({'x':{'type':'int','max_lenght':3}}, {'required':['required']}, {'x':{'type':'str','max_length':5,'pattern':3}}):
            with self.assertRaises(ValueError):g.register_tool('bad',schema,['monitor'],lambda p:p)
            self.assertNotIn('bad',g.tools)

    def test_parameter_and_result_digests_separately(self):
        g=self.code['ReadOnlyAgentGateway']();g.register_tool('count',{},['monitor'],lambda p:{'count':3})
        result=g.execute_tool_call({'tool':'count','params':{}})
        canon=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        events=[e['event'] for e in g.logger.events if e['event']['call_id']==result['call_id']]
        self.assertEqual(events[0]['details']['params_hash'],hashlib.sha256(canon({})).hexdigest())
        self.assertEqual(events[-1]['details']['result_hash'],hashlib.sha256(canon(result['result'])).hexdigest())
        self.assertNotEqual(events[0]['details']['params_hash'],events[-1]['details']['result_hash'])

    def test_hash_chain_is_not_authenticated(self):
        # Consistency-only verification cannot detect an attacker rewriting the whole chain.
        logger=self.code['AuditLogger']();status=self.code['EventStatus']
        logger.log(status.SUCCEEDED,'mock',{},1)
        logger.events[0]['event']['status']='DENIED'
        self.assertFalse(logger.verify_chain())
        payload=logger._serialize_event(logger.events[0]['event'])
        rewritten=hashlib.sha256(b'GENESIS'+payload).hexdigest()
        logger.events[0]['hash']=rewritten;logger.prev_hash=rewritten
        self.assertTrue(logger.verify_chain())


if __name__=='__main__':
    print('Python',platform.python_version(),'CPU',platform.machine(),'snapshot SHA256',PIN,flush=True)
    unittest.main(verbosity=2)
