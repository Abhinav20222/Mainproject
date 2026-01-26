"""Test API performance"""
import requests
import time

msgs = [
    "Hi how are you?",
    "URGENT! Your bank account suspended! Click bit.ly/xyz123",
    "Meeting at 3pm tomorrow"
]

print("="*60)
print("PERFORMANCE TEST - OPTIMIZED API")
print("="*60)

for i, msg in enumerate(msgs, 1):
    start = time.time()
    r = requests.post("http://localhost:5000/api/analyze", json={"message": msg})
    total = (time.time() - start) * 1000
    
    result = r.json()
    server_time = result.get("processing_time_ms", "N/A")
    prediction = result.get("prediction", "N/A")
    score = result.get("threat_score", "N/A")
    
    print(f"\nTest {i}: {msg[:40]}...")
    print(f"  Result: {prediction.upper()} (score: {score})")
    print(f"  Server time: {server_time}ms")
    print(f"  Total time (with network): {total:.2f}ms")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
