import requests
from pathlib import Path

files = [
    'data/human/11.wav',
    'data/human/Hasan_hum.wav',
    'data/ai/Hindi_Ai.mp3',
    'data/ai/hi_ai_6.mp3',
    'data/ai/ml_ai_11.mp3',
    'data/ai/ta_ai_7.mp3',
    'data/ai/telhum.mp4',
    'data/ai/te_ai_15.mp3',
]

for p in files:
    print('\n---', p)
    try:
        with open(p,'rb') as f:
            r = requests.post('http://localhost:5000/api/detect?debug=true', files={'audio': f}, timeout=180)
        print('STATUS', r.status_code)
        try:
            print(r.json())
        except Exception:
            print(r.text)
    except Exception as e:
        print('ERROR', e)
