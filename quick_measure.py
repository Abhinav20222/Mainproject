"""Measure startup time with new optimization"""
import time
import subprocess
import requests
import sys
import os

os.chdir(r"c:\Users\MY PC\phishing_detection")

print("=" * 50)
print("MEASURING OPTIMIZED FLASK STARTUP TIME")
print("=" * 50)

start = time.time()
print(f"\n[START] {time.strftime('%H:%M:%S')}")

# Start server
proc = subprocess.Popen([sys.executable, "src/api.py"], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.STDOUT)

# Wait for first health response (even "loading" counts)
for i in range(120):
    try:
        r = requests.get("http://localhost:5000/api/health", timeout=0.5)
        if r.status_code == 200:
            elapsed = time.time() - start
            status = r.json().get('status', 'unknown')
            print(f"[FIRST RESPONSE] {time.strftime('%H:%M:%S')}")
            print(f"\n{'='*50}")
            print(f"⏱️  TIME TO FIRST RESPONSE: {elapsed:.2f} seconds")
            print(f"📊 STATUS: {status}")
            print(f"{'='*50}")
            
            # Now wait for online status
            if status == 'loading':
                print("\n⏳ Waiting for 'online' status...")
                for j in range(60):
                    try:
                        r2 = requests.get("http://localhost:5000/api/health", timeout=0.5)
                        if r2.json().get('status') == 'online':
                            full_elapsed = time.time() - start
                            print(f"\n✅ AI ONLINE after {full_elapsed:.2f} seconds")
                            break
                    except:
                        pass
                    time.sleep(0.25)
            
            with open("startup_time_new.txt", "w") as f:
                f.write(f"Time to first response: {elapsed:.2f} seconds\n")
                f.write(f"Status: {status}\n")
            
            proc.terminate()
            break
    except:
        pass
    time.sleep(0.25)
else:
    print("TIMEOUT!")
    proc.terminate()
