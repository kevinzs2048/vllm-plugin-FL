"""Only accept the documented unregistered-architecture result as expected."""
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
if d.get('stock_architecture_registered') is not False or 'not supported' not in d.get('error','').lower() or 'MiniCPMV4_7ForConditionalGeneration' not in d.get('error',''):
    raise SystemExit('ERROR: unexpected stock result, inspect '+str(p))
print('STOCK_EXPECTED_UNSUPPORTED: performance N/A')
