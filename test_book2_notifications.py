import json
from pathlib import Path
import tempfile
import unittest
import book2_notifications as N


class NotificationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'editorial').mkdir()
        self.state={'status':'writing','updated':'v1','calls':1,'repair_generation':1,'chapters':{},'part_reviews':{},'active':{},'attention':[]}
        self.save()

    def tearDown(self):self.temp.cleanup()
    def save(self):(self.root/'status.json').write_text(json.dumps(self.state))
    def poll(self,t=1,pids=None):return N.collect(self.root,now=t,pids=[42] if pids is None else pids)

    def test_stage_progress_persists_without_duplicates(self):
        self.poll()
        self.state.update(status='cross_chapter_review',updated='v2')
        self.state['chapters']['1']={'status':'model_reviewed','sha256':'abc'}
        self.state['part_reviews']['0']={'approved':True,'fingerprint':'xyz'}
        self.save();data=self.poll(2)
        self.assertEqual({e['kind'] for e in data['events']},{'monitor_started','chapter_approved','part_approved','cross_review_started'})
        self.assertEqual(len(self.poll(3)['events']),len(data['events']))
        self.state['chapters']['1']['status']='needs_review';self.state['updated']='v3';self.save()
        self.assertEqual(self.poll(4)['events'][0]['kind'],'chapter_reopened')

    def test_timeout_is_not_global_stop_and_hides_raw_command(self):
        old=self.root/'editorial/error-old-1-0.json';old.write_text(json.dumps({'type':'TimeoutExpired'}))
        self.poll()
        p=self.root/'editorial/error-ch19-review-300-0.json'
        p.write_text(json.dumps({'type':'TimeoutExpired','message':'secret-command-token timed out'}))
        data=self.poll(2)
        event=data['events'][0]
        self.assertEqual(event['kind'],'timeout');self.assertEqual(event['details']['attempt'],1)
        self.assertTrue(data['worker_alive'])
        self.assertNotIn('secret-command-token',(self.root/'notifications.json').read_text())
        self.assertNotIn('secret-command-token',(self.root/'notifications.html').read_text())
        self.assertEqual(len(self.poll(3)['events']),2)
        (self.root/'editorial/error-ch19-review-301-1.json').write_text(json.dumps({'type':'TimeoutExpired'}))
        self.assertIn('重試已耗盡',self.poll(4)['events'][0]['message'])

    def test_stop_grace_and_resume(self):
        self.poll()
        self.poll(10,[])
        self.assertNotIn('worker_stopped',[e['kind'] for e in self.poll(24,[])['events']])
        self.assertEqual(self.poll(25,[])['events'][0]['kind'],'worker_stopped')
        length=len(self.poll(26,[])['events'])
        self.assertEqual(len(self.poll(27,[])['events']),length)
        self.assertEqual(self.poll(28,[43])['events'][0]['kind'],'worker_resumed')

    def test_completed_baseline_not_fake_historical_stage_events(self):
        self.state['status']='completed_model_review_pending_human';self.save()
        data=self.poll(1,[])
        self.assertEqual({e['kind'] for e in data['events']},{'monitor_started','book_completed'})
        self.assertTrue(data['events'][0]['details']['baseline_snapshot'])
        self.assertEqual(len(self.poll(30,[])['events']),2)

    def test_bad_error_file_can_be_retried_and_render_is_escaped(self):
        self.poll();p=self.root/'editorial/error-probe-2-0.json';p.write_text('{')
        self.assertEqual(len(self.poll(2)['events']),1)
        p.write_text(json.dumps({'type':'ValueError'}))
        self.assertEqual(self.poll(3)['events'][0]['kind'],'attempt_failed')
        N.render(self.root,{'worker_alive':True,'status':'writing','checked_at':'now','events':[{'level':'warning','observed_at':'now','message':'<script>alert(1)</script>'}]})
        page=(self.root/'notifications.html').read_text()
        self.assertNotIn('<script>',page);self.assertIn('&lt;script&gt;',page)

    def test_panel_survives_page_regeneration_and_is_idempotent(self):
        source='<html><p>初步目標：教材</p></html>'
        page=N.inject_panel(source)
        self.assertIn('src="/book2/notifications"',page)
        self.assertEqual(N.inject_panel(page),page)
        self.assertIn('id="book2-notifications"',N.inject_panel(source))


if __name__=='__main__':unittest.main()
