# AI-Generated Voice Detection API

This API classifies a voice sample as AI-generated or Human-generated and returns confidence plus a short explanation. It supports uploads, Base64 audio, and direct audio URLs.

**Core endpoints**
- `POST /api/detect`
- `GET /api/health`
- Web tester at `/`

**API key**
- Default key: `hackathon-2026-demo-key`
- You can override with the `API_KEY` environment variable.

## Step-by-step: Run locally (Windows)

1. Open PowerShell in the project folder.
2. (Optional) Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. (Optional) Set your own API key for this session:
   ```powershell
   $env:API_KEY="your-secret-key"
   ```
   If you skip this, the default key `hackathon-2026-demo-key` is used.
5. Run the server:
   ```powershell
   python app.py
   ```
6. Open the tester:
   - `http://localhost:5000/`
7. API endpoint:
   - `http://localhost:5000/api/detect`

## Step-by-step: Use the API

**Headers (required)**
```
Authorization: Bearer hackathon-2026-demo-key
```
You can also send `X-API-Key: hackathon-2026-demo-key`.

### Option A: Upload a file (FormData)
```bash
curl -X POST http://localhost:5000/api/detect \
  -H "Authorization: Bearer hackathon-2026-demo-key" \
  -F "audio=@path/to/audio.wav"
```

### Option B: Send Base64 audio (JSON)
```bash
curl -X POST http://localhost:5000/api/detect \
  -H "Authorization: Bearer hackathon-2026-demo-key" \
  -H "Content-Type: application/json" \
  -d "{\"audio\":\"<base64-mp3>\"}"
```

### Option C: Send a direct MP3 URL (JSON)
```bash
curl -X POST http://localhost:5000/api/detect \
  -H "Authorization: Bearer hackathon-2026-demo-key" \
  -H "Content-Type: application/json" \
  -d "{\"audio_url\":\"https://example.com/sample.mp3\"}"
```

## Step-by-step: Deploy to Render

1. Push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<yourname>/<repo>.git
   git push -u origin main
   ```
2. Create a Render Web Service:
   - Connect your GitHub repo.
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
3. Set environment variable:
   - `API_KEY = your-secret-key` (or keep the default)
4. Deploy and copy the public URL:
   - Example: `https://your-app.onrender.com`

Your public endpoint will be:
```
https://your-app.onrender.com/api/detect
```

## Hackathon Submission Template

```
Public API Endpoint:
https://your-app.onrender.com/api/detect

Method:
POST

Header (API Key):
Authorization: Bearer hackathon-2026-demo-key
```

## Troubleshooting

- `401 Unauthorized`: You forgot the `Authorization` header or the key is wrong.
- `Audio file too large`: Audio URL downloads are limited to 25 MB.
- `Audio too short`: Minimum length is 0.4 seconds.

## License

This project is created for hackathon purposes.
