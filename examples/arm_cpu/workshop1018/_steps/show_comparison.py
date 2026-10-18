from pathlib import Path
import json,sys
d=json.loads(Path(sys.argv[1]).read_text())
assert d['passed']
print('\nPerformance (tok/s), median of 5 measured requests after 1 warmup')
print('{:<16} {:>10} {:>10} {:>10}'.format('Route','PP512','TG128','Total'))
for name in ('minimal','flagos'):
 m=d['results'][name]['medians']
 print('{:<16} {:>10.2f} {:>10.2f} {:>10.2f}'.format(name,m['pp512'],m['tg128'],m['total']))
g=d['gain_pct']
print('{:<16} {:>9.2f}% {:>9.2f}% {:>9.2f}%'.format('Gain',g['pp512'],g['tg128'],g['total']))
print('PP512=512/prefill_s; TG128=127/decode_s; Total=640/execution_s')
print('Stock vLLM architecture unsupported: N/A. Minimal includes registration/CPU compatibility.')
print('Comparison includes phase threads and W8 output-head execution.')
print('Matching token sequences:',d['matching_token_sequences'],'/ 6; not an accuracy benchmark.')
