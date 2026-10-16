"""End-to-end text/image client with durable request and response records."""
import argparse, base64, json, os, time, urllib.request
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--url',default='http://127.0.0.1:8000')
p.add_argument('--model',default='minicpm-v47-1b')
p.add_argument('--text',default='只输出结果：2+3等于多少？')
p.add_argument('--image',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
a.output.parent.mkdir(parents=True,exist_ok=True)
content=a.text
if a.image:
    import mimetypes
    mime=mimetypes.guess_type(a.image.name)[0]
    assert mime in ('image/jpeg','image/png')
    url='data:'+mime+';base64,'+base64.b64encode(a.image.read_bytes()).decode()
    content=[{'type':'image_url','image_url':{'url':url}},{'type':'text','text':a.text}]
body={'model':a.model,'messages':[{'role':'user','content':content}],
      'chat_template_kwargs':{'enable_thinking':False},'temperature':0,'max_tokens':128}
request=a.output.with_name(a.output.stem+'-request.json')
# Exclusive create keeps the request left by an interrupted/failed call.
with request.open('x') as f:
    f.write(json.dumps(body,ensure_ascii=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
start=time.perf_counter()
req=urllib.request.Request(a.url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
with opener.open(req,timeout=600) as response:result=json.load(response)
assert result['choices'][0]['message']['content'].strip()
result['_http_seconds']=time.perf_counter()-start
temp=a.output.with_suffix('.tmp')
with temp.open('w') as f:
    json.dump(result,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
temp.replace(a.output)
print(result['choices'][0]['message']['content'])
print(result['usage'],result['_http_seconds'])
