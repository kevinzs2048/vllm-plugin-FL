"""Source-built CPU Triton smoke with saved compiler/module provenance."""
import json
import argparse,time
from pathlib import Path
import torch
import triton
import triton.language as tl
@triton.jit
def add(a,b,c,N:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    tl.store(c+index,tl.load(a+index,index<N,0)+tl.load(b+index,index<N,0),index<N)
torch.manual_seed(42)
a=torch.randn(4096);b=torch.randn_like(a);c=torch.empty_like(a)
add[(triton.cdiv(a.numel(),256),)](a,b,c,a.numel(),256)
torch.testing.assert_close(c,a+b)
report={'passed':True,'triton_version':triton.__version__,'triton_module':triton.__file__,
        'max_abs_error':float((c-a-b).abs().max())}
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path(f'/work/evidence/tree-smoke-{time.time_ns()}.json'))
args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(report)
