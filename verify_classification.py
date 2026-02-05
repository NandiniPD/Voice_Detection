#!/usr/bin/env python
"""
Verify audio classification with Flask API
"""

import requests
import json
import time
from pathlib import Path

API_URL = "http://localhost:5000/api/detect"

# Audio files with expected classification
test_files = {
    "data/human/11.wav": "Human-generated",
    "data/human/Hasan_hum.wav": "Human-generated",
    "data/ai/Hindi_Ai.mp3": "AI-generated",
    "data/ai/hi_ai_6.mp3": "AI-generated",
    "data/ai/ml_ai_11.mp3": "AI-generated",
    "data/ai/ta_ai_7.mp3": "AI-generated",
    "data/ai/telhum.mp4": "AI-generated",
    "data/ai/te_ai_15.mp3": "AI-generated",
}

print("=" * 70)
print("AUDIO CLASSIFICATION VERIFICATION")
print("=" * 70)
print()

passed = 0
failed = 0

for file_path, expected in test_files.items():
    if not Path(file_path).exists():
        print(f"SKIP: {Path(file_path).name:30} (File not found)")
        continue
    
    try:
        with open(file_path, "rb") as f:
            response = requests.post(API_URL, files={"audio": f}, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            classification = result.get("classification", "Unknown")
            confidence = result.get("confidence", 0)
            
            status = "PASS" if classification == expected else "FAIL"
            
            if classification == expected:
                passed += 1
            else:
                failed += 1
            
            print(f"{status} {Path(file_path).name:30} | Expected: {expected:20} | Got: {classification:20} | Confidence: {confidence:.2f}")
        else:
            print(f"FAIL {Path(file_path).name:30} | API Error: {response.status_code}")
            failed += 1
    except Exception as e:
        print(f"FAIL {Path(file_path).name:30} | Error: {str(e)}")
        failed += 1

print()
print("=" * 70)
print(f"RESULTS: {passed} Passed, {failed} Failed")
print("=" * 70)
