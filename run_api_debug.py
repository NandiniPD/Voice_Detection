import requests
from pathlib import Path
p = Path('data/human/11.wav')
url = 'http://localhost:5000/api/detect?debug=true'
with open(p,'rb') as f:
    r = requests.post(url, files={'audio': f}, timeout=120)
    print('STATUS', r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text)
