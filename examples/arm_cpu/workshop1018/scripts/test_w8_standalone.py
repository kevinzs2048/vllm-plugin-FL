"""Independent quantized reference, KAI control, and exact BF16 comparison."""
import argparse,json,time,os
from pathlib import Path
import torch
from flag_gems.runtime.backend._arm.fused.native_ops import arm_native_ops
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
p.add_argument('--golden',type=Path);p.add_argument('--save-golden',type=Path)
p.add_argument('--triton',action='store_true')
a=p.parse_args();assert not a.output.exists()
torch.set_num_threads(4);torch.manual_seed(781);ops=arm_native_ops('w8a8',raw=True)
report={'complete':False,'passed':False,'cases':[],'library_override':os.getenv('FLAGGEMS_ARM_NATIVE_LIBRARY')}
def save():
 t=a.output.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2));t.replace(a.output)
save();golden=torch.load(a.golden,weights_only=True) if a.golden else {};outputs={}
try:
 for m,n,k,kind in [(1,64,128,'random'),(3,64,128,'random'),(16,384,1024,'random'),
                    (512,128,1024,'random'),(1,64,128,'zero'),(1,64,128,'constant'),
                    (1,64,128,'negative'),(1,64,128,'halfway')]:
  x=torch.randn(m,k).bfloat16().float();w=torch.randint(-127,128,(n,k),dtype=torch.int8)
  scales=torch.rand(n)*0.01+0.001
  if kind=='zero':x.zero_()
  if kind=='constant':x.fill_(3.5)
  if kind=='negative':x=-x.abs()
  if kind=='halfway':x[0]=torch.linspace(-128,127,k)+0.5
  key=f'{m}x{n}x{k}-{kind}';packed=ops.pack(w,scales,False)
  out=ops.linear(x,packed,n,False,32,128)
  lo=torch.minimum(x.amin(1),torch.zeros(m));hi=torch.maximum(x.amax(1),torch.zeros(m))
  mult=torch.where(lo==hi,torch.ones(m),torch.div(torch.full_like(lo,255),hi-lo))
  sx=torch.where(mult!=0,torch.div(torch.ones_like(mult),mult),torch.zeros(m))
  dmin=lo*mult;dmax=hi*mult
  z=torch.where((-128+dmin)+(127+dmax)>0,-128-dmin,127-dmax).clamp(-128,127).round().to(torch.int32)
  value=x*mult[:,None]
  q=(torch.sign(value)*torch.floor(value.abs()+0.5)).to(torch.int32)+z[:,None]
  q=q.clamp(-128,127)
  dot=(q-z[:,None]).double()@w.double().T
  ref=dot.float()*(sx[:,None]*scales[None,:])
  torch.testing.assert_close(out,ref,atol=2e-5,rtol=2e-5)
  row={'shape':[m,n,k],'kind':kind,'max_ref_error':float((out-ref).abs().max()),'passed':True}
  if a.triton:
   from flag_gems.runtime.backend._arm.fused.w8a8_standalone_triton import linear_triton
   tl_out=linear_triton(x,packed,n)
   torch.testing.assert_close(tl_out,ref,atol=2e-5,rtol=2e-5)
   row['triton_max_error']=float((tl_out-ref).abs().max())
   for name,fn in [('raw',lambda:ops.linear(x,packed,n,False,32,128)),('triton',lambda:linear_triton(x,packed,n))]:
    fn();t=time.perf_counter()
    for _ in range(10):fn()
    row[name+'_mean_us']=(time.perf_counter()-t)*1e5
  if a.golden:
   old=golden[key];row['max_kai_error']=float((out-old).abs().max())
   row['bf16_equal']=bool(torch.equal(out.bfloat16(),old.bfloat16()))
   torch.testing.assert_close(out,old,atol=2e-5,rtol=2e-5)
  outputs[key]=out;report['cases'].append(row);save()
 if a.save_golden:torch.save(outputs,a.save_golden)
 report.update(complete=True,passed=True);save();print(json.dumps(report,indent=2))
except BaseException as e:report['error']=repr(e);save();raise
