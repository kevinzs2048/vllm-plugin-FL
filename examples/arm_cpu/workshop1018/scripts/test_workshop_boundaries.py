"""Local r1: 17 black-box rejection contracts; fixtures never touch live inputs."""
import argparse,hashlib,json,os,pathlib,shutil,socket,subprocess,sys,tempfile,time
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
S=pathlib.Path(__file__).resolve().parent
cases=[]
def sha(b):return hashlib.sha256(b).hexdigest()
with tempfile.TemporaryDirectory(prefix='workshop-boundaries-') as td:
 T=pathlib.Path(td)
 def fixture():
  r=T/str(len(list(T.iterdir())));r.mkdir();shutil.copytree(S,r/'scripts');m=r/'models/MiniCPM-V-4.7-1B-W8A8-FlagOS';m.mkdir(parents=True)
  (m/'config.json').write_text('{}');(m/'model.safetensors.index.json').write_text('{"weight_map":{}}')
  lock={'helper_files':{'scripts/http_request.py':sha((r/'scripts/http_request.py').read_bytes())},'model_runtime_files':{'config.json':sha(b'{}')},'sources':{}}
  (r/'ONLINE-INPUTS.lock.json').write_text(json.dumps(lock));return r,m,lock
 def reject(name,cmd,needle,env=None,preserve=None):
  before={str(f):f.read_bytes() for f in (preserve or [])}
  q=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,timeout=40)
  ok=q.returncode!=0 and needle in q.stdout and all(pathlib.Path(f).read_bytes()==v for f,v in before.items())
  cases.append({'name':name,'passed':ok,'exit_code':q.returncode,'expected_message':needle,'output':q.stdout[-3000:]})
 def verify(r):return [sys.executable,str(r/'scripts/verify_online_inputs.py')]
 def locksave(r,l):(r/'ONLINE-INPUTS.lock.json').write_text(json.dumps(l))
 # Run the actual platform checker with only the machine probe replaced.
 r,m,l=fixture();code="import runpy;from unittest.mock import patch;\nwith patch('platform.machine',return_value='x86_64'):runpy.run_path("+repr(str(r/'scripts/check_platform.py'))+",run_name='__main__')"
 reject('01_wrong_architecture',[sys.executable,'-c',code],'requires Linux aarch64')
 e=dict(os.environ,WORK_ROOT=str(T/'wrong'))
 reject('02_work_root_mismatch',[sys.executable,str(r/'scripts/check_platform.py')],'WORK_ROOT does not match',e)
 for name,mutation,needle in [('03_missing_helper',lambda r,m:(r/'scripts/http_request.py').unlink(),'Workshop helper'),('04_modified_helper',lambda r,m:(r/'scripts/http_request.py').write_text('#changed'),'Workshop helper'),('05_missing_model_file',lambda r,m:(m/'config.json').unlink(),'checkpoint file'),('06_corrupt_model_file',lambda r,m:(m/'config.json').write_text('corrupt'),'checkpoint file')]:
  r,m,l=fixture();mutation(r,m);reject(name,verify(r),needle)
 r,m,l=fixture();(m/'model.safetensors.index.json').write_text('{"weight_map":{"x":"missing.safetensors"}}');reject('07_missing_weight_shard',verify(r),'missing.safetensors')
 def repo(r,l):
  d=r/'source/Test';d.mkdir(parents=True);subprocess.run(['git','init','-q',str(d)],check=True);subprocess.run(['git','-C',str(d),'-c','user.name=Fixture','-c','user.email=fixture@localhost','commit','-q','--allow-empty','-m','fixture'],check=True);l['sources']['Test']=subprocess.check_output(['git','-C',str(d),'rev-parse','HEAD'],text=True).strip();return d
 r,m,l=fixture();repo(r,l);l['sources']['Test']='0'*40;locksave(r,l);reject('08_wrong_source_pin',verify(r),'Test')
 r,m,l=fixture();d=repo(r,l);(d/'dirty').write_text('x');locksave(r,l);reject('09_dirty_source',verify(r),'Dirty source Test')
 # The installed helper is supplied alongside this script by the prebuilt archive.
 tool=S/'prebuilt_tools.py'
 for name,corrupt in [('10_missing_prebuilt_wheel',False),('11_corrupt_prebuilt_wheel',True)]:
  r,m,l=fixture();b=r/'prebuilt';b.mkdir();manifest={'files':{'prebuilt/test.whl':{'bytes':4,'sha256':sha(b'good')}}};(b/'manifest.json').write_text(json.dumps(manifest))
  if corrupt:(b/'test.whl').write_bytes(b'evil')
  reject(name,[sys.executable,str(tool),'verify',str(r)],'test.whl')
 def fake(r,existing=False,badimage=False):
  b=r/'fake-bin';b.mkdir();d=b/'docker';d.write_text('#!/bin/sh\ncase "$1 $2" in\n"image inspect") echo "'+('wrong' if badimage else 'sha256:72d655d6bf308074aa53374ae32e7f1596d0493b81c1960df0209994bc54816b')+'";;\n"container inspect") exit '+('0' if existing else '1')+';;\n*) echo FORBIDDEN_DOCKER_MUTATION;exit 98;;\nesac\n');d.chmod(0o700);u=b/'sudo';u.write_text('#!/bin/sh\n[ "$1" != "-n" ] || shift\nexec "$@"\n');u.chmod(0o700)
  return dict(os.environ,PATH=str(b)+os.pathsep+os.environ['PATH'],WORK_ROOT=str(r),MODEL_DIR=str(r/'models/MiniCPM-V-4.7-1B-W8A8-FlagOS'),CONTAINER='fixture-only')
 r,m,l=fixture();reject('12_wrong_image_identity',['bash',str(r/'scripts/init_container.sh')],'',fake(r,badimage=True))
 r,m,l=fixture();reject('13_existing_dev_container',['bash',str(r/'scripts/init_container.sh')],'already exists',fake(r,existing=True))
 r,m,l=fixture();s=socket.socket();s.bind(('0.0.0.0',8000));s.listen();reject('14_occupied_http_port',['bash',str(r/'scripts/start_service.sh')],'port 8000 is occupied',fake(r));s.close()
 r,m,l=fixture();reject('15_existing_http_container',['bash',str(r/'scripts/start_service.sh')],'already exists',fake(r,existing=True))
 r,m,l=fixture();out=r/'response.json';out.write_text('preserve');reject('16_existing_response_and_request',[sys.executable,str(r/'scripts/http_request.py'),'--output',str(out)],'FileExistsError',preserve=[out]);out.unlink();req=r/'response-request.json';req.write_text('preserve');prev=cases.pop();reject('16_existing_response_and_request',[sys.executable,str(r/'scripts/http_request.py'),'--output',str(out)],'FileExistsError',preserve=[req]);cases[-1]['passed'] &= prev['passed'];cases[-1]['response_subcase']=prev
 r,m,l=fixture();x=r/'partial.json';x.write_text('{"complete":false,"passed":false}');reject('17_partial_benchmark_comparison',[sys.executable,str(r/'scripts/compare_bench.py'),str(x),str(x),'--output',str(r/'compare.json')],'AssertionError')
result={'revision':'local-r1-20261017','qualification':'New 17-case regression suite; original historical suite was absent from online release. Tests actual entrypoints with isolated fixtures. No inference accuracy claim.','passed':len(cases)==17 and all(c['passed'] for c in cases),'cases':cases}
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print('BOUNDARIES',sum(c['passed'] for c in cases),'/',len(cases));sys.exit(0 if result['passed'] else 1)
