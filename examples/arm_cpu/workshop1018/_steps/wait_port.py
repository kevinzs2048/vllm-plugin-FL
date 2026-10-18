"""Wait for TIME_WAIT without terminating processes or changing TCP settings."""
import socket,time
deadline=time.monotonic()+75
while True:
 s=socket.socket()
 try:s.bind(('0.0.0.0',8000));s.close();print('PORT8000_FREE');break
 except OSError:s.close()
 p=socket.socket();p.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
 try:p.bind(('0.0.0.0',8000))
 except OSError as e:raise SystemExit('ERROR: port8000 has an active/conflicting binding. Stop only your workshop service via Step11: '+str(e))
 finally:p.close()
 if time.monotonic()>deadline:raise SystemExit('ERROR: port8000 not released within 75 seconds')
 print('Waiting for TCP TIME_WAIT on port8000...',flush=True);time.sleep(3)
