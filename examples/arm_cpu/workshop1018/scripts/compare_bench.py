"""Independently recompute the per-request metrics; refuse partial runs."""
import argparse,json,statistics,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('minimal',type=Path);p.add_argument('flagos',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
x=json.loads(a.minimal.read_text());y=json.loads(a.flagos.read_text())
results={}
for name,d in [('minimal',x),('flagos',y)]:
 assert d['complete'] and d['passed'],name
 assert len(d['runs'])==6 and d['runs'][0]['warmup'] and all(not r['warmup'] for r in d['runs'][1:]),name
 for r in d['runs']:
  assert len(r['token_ids'])==128
  for k,v in [('pp512',512/r['prefill_s']),('tg128',127/r['decode_s']),('total',640/r['execution_s'])]:
   assert abs(v-r[k])<1e-8,(name,k)
 m={k:statistics.median(r[k] for r in d['runs'][1:]) for k in ('pp512','tg128','total')}
 assert m==d['medians'],name
 results[name]={'medians':m,'ranges':{k:[min(r[k] for r in d['runs'][1:]),max(r[k] for r in d['runs'][1:])] for k in m}}
assert x['checkpoint']==y['checkpoint'] and x['versions']==y['versions']
assert x['flag_gems_modules_loaded']==0
assert x['w8_apply_module_after_load'].startswith('vllm.model_executor.')
result={'passed':True,'results':results,'gain_pct':{k:100*(results['flagos']['medians'][k]/results['minimal']['medians'][k]-1) for k in ('pp512','tg128','total')},'matching_token_sequences':sum(rx['token_ids']==ry['token_ids'] for rx,ry in zip(x['runs'],y['runs'])),'protocol':'fixed 512 input / 128 output; one excluded warmup plus five measured resident requests; batch1; DSpark off; no profiler','qualification':'Minimal baseline includes architecture/CPU compatibility only and BF16 head. FlagOS uses W8A8 head and phase threads. This comparison includes all these changes; six token sequences do not establish benchmark accuracy.'}
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.with_suffix('.tmp').open('w') as f:
 json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
a.output.with_suffix('.tmp').replace(a.output)
for name,d in results.items():print(name, ' '.join(k+'='+format(v,'.2f') for k,v in d['medians'].items()))
print('COMPARISON_PASS',a.output)
