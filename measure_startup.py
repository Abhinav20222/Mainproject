"""
Flask API Startup Time Measurement Script
Measures the time taken for the Flask server to become AI Online
"""
import time
import subprocess
import requests
import sys

API_URL = "http://localhost:5000"

def measure_startup_time():
    """Measure the Flask server startup time"""
    
    print("="*60)
    print("FLASK SERVER STARTUP TIME MEASUREMENT")
    print("="*60)
    
    # Start timing
    start_time = time.time()
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting Flask server...")
    
    # Start Flask server in background
    process = subprocess.Popen(
        [sys.executable, "-m", "src.api_fast"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for server to become ready
    print(f"[{time.strftime('%H:%M:%S')}] Waiting for server to become online...")
    
    max_wait = 60  # Maximum 60 seconds
    check_interval = 0.5  # Check every 0.5 seconds
    elapsed = 0
    
    while elapsed < max_wait:
        try:
            response = requests.get(f"{API_URL}/api/health", timeout=1)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'online':
                    end_time = time.time()
                    startup_time = end_time - start_time
                    
                    print(f"\n{'='*60}")
                    print("RESULTS")
                    print("="*60)
                    print(f"\n✓ AI ONLINE!")
                    print(f"\n⏱️  STARTUP TIME: {startup_time:.2f} seconds")
                    print(f"\n📊 Breakdown:")
                    print(f"   - Model loading + initialization")
                    print(f"   - Flask server startup")
                    print(f"   - Health check success")
                    print(f"\n{'='*60}")
                    
                    # Let the server keep running for the user
                    print(f"\n🚀 Server is now running at {API_URL}")
                    print("Press Ctrl+C to stop the server")
                    
                    # Wait for user to stop
                    try:
                        process.wait()
                    except KeyboardInterrupt:
                        process.terminate()
                        print("\nServer stopped.")
                    
                    return startup_time
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(check_interval)
        elapsed += check_interval
        
        # Print progress dots
        if int(elapsed) % 2 == 0 and elapsed == int(elapsed):
            print(f"   Checking... ({elapsed:.0f}s elapsed)")
    
    # Timeout
    print(f"\n✗ Server failed to start within {max_wait} seconds")
    process.terminate()
    return None

if __name__ == "__main__":
    measure_startup_time()
