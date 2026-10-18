"""CLI rejection contracts for the precompiled workshop step scripts."""
from pathlib import Path
import json,os,shutil,subprocess,tempfile,unittest
APP=Path(__file__).resolve().parents[2]/'examples/arm_cpu/workshop1018'
@unittest.skipIf(Path('/.dockerenv').exists(),'These rejection contracts exercise a host shell')
class WorkshopSteps(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='workshop-step-contract-')
  self.root=Path(self.temp.name)/'workshop'
  shutil.copytree(APP,self.root)
 def tearDown(self):self.temp.cleanup()
 def run_step(self,name,*args):
  env=dict(os.environ);env.pop('WORK_ROOT',None)
  return subprocess.run(['bash',str(self.root/name),*args],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=20)
 def test_container_step_rejects_host_without_creating_install(self):
  q=self.run_step('step06.sh')
  self.assertNotEqual(q.returncode,0)
  self.assertIn('must run in the development container',q.stdout)
  self.assertFalse((self.root/'envs').exists())
 def test_invalid_run_cannot_escape_workshop(self):
  q=self.run_step('step01.sh','--run','../../outside')
  self.assertNotEqual(q.returncode,0)
  self.assertIn('use --run',q.stdout)
  self.assertFalse((self.root.parent/'outside').exists())
 def test_modified_entrypoint_rejected_before_it_runs(self):
  p=self.root/'step03.sh';marker=self.root/'should-not-exist'
  p.write_text(p.read_text()+'\ntouch "'+str(marker)+'"\n')
  q=self.run_step('step03.sh')
  self.assertNotEqual(q.returncode,0)
  self.assertIn('Missing/modified step file',q.stdout)
  self.assertFalse(marker.exists())
 def test_existing_http_evidence_is_preserved(self):
  p=self.root/'evidence/http-run01/text-response.json';p.parent.mkdir(parents=True)
  p.write_text('keep-original-response')
  q=self.run_step('step09.sh')
  self.assertNotEqual(q.returncode,0)
  self.assertIn('Existing response/request',q.stdout)
  self.assertEqual(p.read_text(),'keep-original-response')
  self.assertFalse((p.parent/'text-response-request.json').exists())
 def test_run_choice_survives_separate_processes(self):
  tool=self.root/'_steps/select_run.py'
  first=subprocess.check_output(['python3',str(tool),str(self.root),'run02'],text=True).strip()
  second=subprocess.check_output(['python3',str(tool),str(self.root),''],text=True).strip()
  self.assertEqual((first,second),('run02','run02'))
if __name__=='__main__':unittest.main(verbosity=2)
