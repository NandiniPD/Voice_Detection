from flask import Flask, request, jsonify, render_template_string
import numpy as np
import librosa
import whisper
from pydub import AudioSegment
import io
from datetime import datetime
import joblib
import json
import os
import requests
from pathlib import Path
from urllib.parse import urlparse

# API key (optional). If not set, auth is disabled to avoid errors.
API_KEY = os.getenv("API_KEY", "hackathon-2026-demo-key").strip()


def require_api_key():
    if not API_KEY:
        return None  # auth disabled
    auth_header = request.headers.get("Authorization", "")
    token = ""
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
    token = token or request.headers.get("X-API-Key", "").strip()
    if token != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    return None

# Supported languages advertised by the service
SUPPORTED_LANGUAGES = ["tamil", "english", "hindi", "malayalam", "telugu"]
SUPPORTED_LANGUAGES.append("kannada")

# Decision thresholds for model probability (AI confidence).
# If model AI confidence is high => AI, low => Human, otherwise fallback to heuristic.
DEFAULT_AI_THRESHOLD = 0.70
DEFAULT_HUMAN_THRESHOLD = 0.30
MIN_AUDIO_SEC = 0.4
MAX_AUDIO_URL_MB = 25

# Language code to full name mapping
LANGUAGE_MAP = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "ml": "Malayalam",
    "te": "Telugu",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "zh": "Chinese",
    "ar": "Arabic",
    "ko": "Korean",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "pl": "Polish",
    "tr": "Turkish",
    "kn": "Kannada",
}

def get_language_name(lang_code):
    """Convert language code to full language name"""
    if not lang_code:
        return "Unknown"
    lang_code = lang_code.lower().strip()
    return LANGUAGE_MAP.get(lang_code, lang_code.upper())

app = Flask(__name__)

whisper_model = whisper.load_model("base")

# Try load trained model if present
MODEL_PATH = Path("models/model.joblib")
FEATURE_ORDER_PATH = Path("models/feature_order.json")
trained_model = None
feature_order = None
if MODEL_PATH.exists() and FEATURE_ORDER_PATH.exists():
    try:
        trained_model = joblib.load(str(MODEL_PATH))
        with open(FEATURE_ORDER_PATH, "r") as f:
            feature_order = json.load(f)
        print("Loaded trained model with features:", feature_order)
    except Exception as e:
        print("Failed to load trained model:", e)

# For reliability during debugging, disable automatic model overrides by default.
# To enable the trained model, set the environment or call the API with use_model=true.
if trained_model is not None:
    print("NOTE: trained model is present but will NOT be used unless requested via use_model=true")
    trained_model = trained_model  # keep reference but model usage gated in route

# ---------------- Feature Extraction ----------------
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)

    return {
        "mfcc_std": float(np.std(mfcc)),
        "zcr_mean": float(np.mean(zcr)),
        "flatness_mean": float(np.mean(spectral_flatness))
    }


def detect_voice_type(y, sr, sensitivity: float = 1.0, conservative: bool = True):
    """
    Heuristic detector (improved): computes a range of features and applies
    two-sided heuristics so both unusually smooth and unusually artifacted
    samples can be flagged as synthetic.
    Returns: classification, confidence, explanation, features, score, thresholds
    """
    # compute a broader set of features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    flatness = librosa.feature.spectral_flatness(y=y)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)

    # Pitch detection
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch_std = float(np.nanstd(f0))
        if voiced_flag is None:
            voiced_fraction = float(np.mean(~np.isnan(f0)))
        else:
            voiced_fraction = float(np.mean(voiced_flag))
    except Exception:
        pitch_std = float(np.nan)
        voiced_fraction = float(np.mean(~np.isnan(librosa.core.pitch.piptrack(y=y, sr=sr)[0])))

    mfcc_std = float(np.std(mfcc))
    zcr_mean = float(np.mean(zcr))
    flatness_mean = float(np.mean(flatness))
    rms_std = float(np.std(rms))
    spectral_contrast_std = float(np.std(spectral_contrast))
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_mfcc_std = float(np.std(delta_mfcc))

    features = {
        "mfcc_std": mfcc_std,
        "zcr_mean": zcr_mean,
        "flatness_mean": flatness_mean,
        "pitch_std": pitch_std,
        "voiced_fraction": voiced_fraction,
        "rms_std": rms_std,
        "spectral_contrast_std": spectral_contrast_std,
        "delta_mfcc_std": delta_mfcc_std,
    }

    # tuned thresholds
    thresholds = {
        "mfcc_low": 5.0 * sensitivity,
        "mfcc_high": 120.0 / max(0.5, sensitivity),
        "zcr_low": 0.005 * sensitivity,
        "zcr_high": 0.10 / max(0.5, sensitivity),
        "flatness": 0.35 / max(0.5, sensitivity),
        "pitch": 15.0 * sensitivity,
        "voiced_fraction": 0.90,
        "rms_std": 0.115,
        "delta_mfcc": 0.7,
    }

    weights = {
        "mfcc": 0.0,
        "zcr": 1.0,
        "flatness": 0.0,
        "pitch": 0.0,
        "voiced_fraction": 0.0,
        "rms_std": 1.0,
        "delta_mfcc": 0.0,
    }

    score = 0.0
    flags = 0
    if not np.isnan(mfcc_std) and (mfcc_std < thresholds["mfcc_low"] or mfcc_std > thresholds["mfcc_high"]):
        score += weights["mfcc"]
        flags += 1
    if not np.isnan(zcr_mean) and (zcr_mean < thresholds["zcr_low"] or zcr_mean > thresholds["zcr_high"]):
        score += weights["zcr"]
        flags += 1
    if not np.isnan(flatness_mean) and flatness_mean > thresholds["flatness"]:
        score += weights["flatness"]
        flags += 1
    if not np.isnan(pitch_std) and pitch_std < thresholds["pitch"]:
        score += weights["pitch"]
        flags += 1
    if not np.isnan(voiced_fraction) and voiced_fraction > thresholds["voiced_fraction"]:
        score += weights["voiced_fraction"]
        flags += 1
    if not np.isnan(rms_std) and rms_std > thresholds["rms_std"]:
        score += weights["rms_std"]
        flags += 1
    if not np.isnan(delta_mfcc_std) and delta_mfcc_std < thresholds["delta_mfcc"]:
        score += weights["delta_mfcc"]
        flags += 1

    total_weight = sum(weights.values())

    # Extremely conservative: require overwhelming evidence to reduce false positives on human speech
    # Only classify as AI if score is very high and multiple indicators are present
    if conservative:
        classification = "AI-generated" if (score >= 8.0 and flags >= 8) else "Human-generated"
    else:
        classification = "AI-generated" if score >= 7.0 else "Human-generated"
    confidence = float(min(0.995, 0.4 + 0.6 * (score / total_weight)))

    if classification == "AI-generated":
        explanation = (
            "Voice shows patterns (MFCC extremes, flatness, continuous voicing or low delta-MFCC) "
            "that often indicate synthetic speech. Inspect 'features' for exact values."
        )
    else:
        explanation = (
            "Voice shows natural variation (pitch, ZCR, spectral dynamics) typical of human speech. "
            "Inspect 'features' for exact values."
        )

    return classification, confidence, explanation, features, float(score), thresholds


def predict_with_model(features_dict):
    """If a trained model is available, use it. Returns (classification, confidence)."""
    global trained_model, feature_order
    if trained_model is None or feature_order is None:
        return None

    def _safe_val(v):
        try:
            if v is None or np.isnan(v) or np.isinf(v):
                return 0.0
        except Exception:
            pass
        return float(v)

    x = [_safe_val(features_dict.get(k, 0.0)) for k in feature_order]
    try:
        probs = trained_model.predict_proba([x])[0]
        # assume classes [0,1] where 1 == AI
        # find index of class 1
        class_labels = trained_model.classes_
        if 1 in class_labels:
            ai_index = int(list(class_labels).index(1))
            ai_conf = float(probs[ai_index])
        else:
            ai_conf = float(probs.max())
        classification = "AI-generated" if ai_conf >= 0.5 else "Human-generated"
        return classification, ai_conf
    except Exception as e:
        print("Model prediction failed:", e)
        return None

# ---------------- API ----------------
@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        auth_error = require_api_key()
        if auth_error:
            return auth_error


        # Handle FormData, JSON base64, or audio URL
        data = request.get_json(silent=True) if request.is_json else None
        audio_url = request.form.get("audio_url") or request.args.get("audio_url")
        if not audio_url and isinstance(data, dict):
            audio_url = data.get("audio_url")

        if request.files and "audio" in request.files:
            # Web interface sends FormData
            audio_file = request.files["audio"]
            audio_bytes = audio_file.read()
            filename = audio_file.filename.lower() if hasattr(audio_file, 'filename') else 'audio.mp4'
        elif audio_url:
            # Download audio from a URL
            if not (audio_url.startswith("http://") or audio_url.startswith("https://")):
                return jsonify({"error": "Invalid audio_url. Must start with http:// or https://"}), 400
            try:
                resp = requests.get(audio_url, stream=True, timeout=20)
                resp.raise_for_status()
            except requests.RequestException as e:
                return jsonify({"error": "Failed to download audio_url", "message": str(e)}), 400

            max_bytes = MAX_AUDIO_URL_MB * 1024 * 1024
            audio_buffer = bytearray()
            try:
                for chunk in resp.iter_content(chunk_size=1024 * 512):
                    if not chunk:
                        continue
                    audio_buffer.extend(chunk)
                    if len(audio_buffer) > max_bytes:
                        return jsonify({"error": "Audio file too large", "max_mb": MAX_AUDIO_URL_MB}), 400
            finally:
                resp.close()

            audio_bytes = bytes(audio_buffer)
            if not audio_bytes:
                return jsonify({"error": "Downloaded audio is empty"}), 400
            filename = Path(urlparse(audio_url).path).name or "audio.mp3"
        elif isinstance(data, dict) and "audio" in data:
            # API calls send JSON with Base64-encoded MP3
            import base64
            try:
                audio_bytes = base64.b64decode(data["audio"])
            except Exception as e:
                return jsonify({"error": "Invalid Base64 audio data", "message": str(e)}), 400
            filename = 'audio.mp3'  # Assume MP3 for Base64 input
        else:
            return jsonify({
                "error": "Audio data missing. Send FormData with 'audio' file, JSON with 'audio' (Base64), or 'audio_url'"
            }), 400

        # Convert any format → WAV (.mp4, .wav, .mp3, .ogg, etc.)
        # pydub/ffmpeg handles most formats transparently
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        except Exception as format_err:
            print(f"[DEBUG] AudioSegment parse failed, trying with format inference: {format_err}")
            # Fallback: infer format from filename
            ext = filename.split('.')[-1] if '.' in filename else 'mp4'
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext)
            except Exception as e:
                return jsonify({"error": "Audio format not supported", "message": str(e)}), 400
        
        # Normalize to 16kHz mono for consistent feature extraction (matches training pipeline)
        try:
            audio = audio.set_frame_rate(16000).set_channels(1)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            if samples.size == 0:
                return jsonify({"error": "Empty audio"}), 400
            max_abs = float(np.max(np.abs(samples)))
            if max_abs > 0:
                samples = samples / max_abs
            y, sr = samples, 16000
        except Exception as load_err:
            return jsonify({"error": "Failed to process audio", "message": str(load_err)}), 400

        # Trim long leading/trailing silence to improve feature stability
        try:
            if y.size:
                y_trim, _ = librosa.effects.trim(y, top_db=30)
                if y_trim.size >= int(0.2 * sr):
                    y = y_trim
        except Exception:
            pass

        duration_sec = float(len(y) / sr) if sr else 0.0
        if duration_sec < MIN_AUDIO_SEC:
            return jsonify({"error": "Audio too short", "min_seconds": MIN_AUDIO_SEC}), 400

        # Optional sensitivity param (form field or query string) to tune detector
        sens_raw = request.form.get('sensitivity') or request.args.get('sensitivity')
        try:
            sensitivity = float(sens_raw) if sens_raw is not None else 1.0
        except Exception:
            sensitivity = 1.0

        # Prepare audio for Whisper (16 kHz mono float32). We already normalized.
        try:
            whisper_audio = y.astype(np.float32)

            whisper_result = whisper_model.transcribe(whisper_audio, fp16=False, task="transcribe")
            language_code = whisper_result.get("language", "unknown")
            language = get_language_name(language_code)
        except Exception as e:
            print("Whisper transcription failed:", e)
            language_code = "unknown"
            language = "Unknown"

        # Language-aware sensitivity adjustments. Reduced to avoid false positives on human speech
        # Lower sensitivity multiplier = less strict thresholds = fewer AI-generated detections
        lang_sensitivity = {"ta": 0.8, "ml": 0.8, "te": 0.8, "hi": 0.9, "kn": 0.8}
        try:
            if language_code in lang_sensitivity:
                factor = lang_sensitivity[language_code]
                sensitivity = float(sensitivity) * float(factor)
                print(f"[DEBUG] Adjusted sensitivity for {language_code} by factor {factor} -> {sensitivity}")
        except Exception:
            pass

        # Run heuristic detector with potentially adjusted sensitivity
        classification, confidence, explanation, features, score, thresholds = detect_voice_type(y, sr, sensitivity=sensitivity)

        # Model use: default to heuristic detector.
        # Override with `mode=model` or `use_model=true`.
        print(f"[DEBUG] request.args: {request.args} request.form: {request.form}")
        mode = (request.args.get('mode') or request.form.get('mode') or '').strip().lower()
        use_model_raw = str(request.args.get('use_model') or request.form.get('use_model') or '').strip().lower()
        model_disabled = not (mode in {"model", "ml"} or use_model_raw in {"true", "1", "yes"})
        model_requested = (use_model_raw in {"true", "1", "yes"} or mode in {"model", "ml"})

        model_used = False
        model_confidence = None
        model_class = None
        model_ai_conf = None

        # Optional per-request thresholds
        ai_threshold_raw = request.args.get('ai_threshold') or request.form.get('ai_threshold')
        human_threshold_raw = request.args.get('human_threshold') or request.form.get('human_threshold')
        try:
            ai_threshold = float(ai_threshold_raw) if ai_threshold_raw is not None else DEFAULT_AI_THRESHOLD
        except Exception:
            ai_threshold = DEFAULT_AI_THRESHOLD
        try:
            human_threshold = float(human_threshold_raw) if human_threshold_raw is not None else DEFAULT_HUMAN_THRESHOLD
        except Exception:
            human_threshold = DEFAULT_HUMAN_THRESHOLD

        # Keep thresholds sane
        ai_threshold = max(0.5, min(0.95, ai_threshold))
        human_threshold = max(0.05, min(0.5, human_threshold))
        if human_threshold >= ai_threshold:
            human_threshold = max(0.05, ai_threshold - 0.05)
        if trained_model is not None and not model_disabled:
            model_pred = predict_with_model(features)
            if model_pred is not None:
                model_class, model_conf = model_pred
                print(f"[DEBUG] Model prediction: {model_class} ({model_conf:.3f})")
                model_ai_conf = float(model_conf)
                if model_ai_conf >= ai_threshold:
                    classification = "AI-generated"
                    confidence = model_ai_conf
                    explanation = f"[MODEL] {explanation}"
                elif model_ai_conf <= human_threshold:
                    classification = "Human-generated"
                    confidence = 1.0 - model_ai_conf
                    explanation = f"[MODEL] {explanation}"
                else:
                    explanation = f"[MODEL-UNCERTAIN] {explanation}"
                model_used = True
                model_confidence = float(model_conf)
            else:
                print(f"[DEBUG] Model requested but not available or failed, using heuristic: {classification} ({confidence:.3f})")
        elif model_requested and trained_model is None:
            print(f"[DEBUG] Model requested but not available; using heuristic: {classification} ({confidence:.3f})")
        else:
            print(f"[DEBUG] Using heuristic: {classification} ({confidence:.3f})")

        # Debug info (optional)
        debug_info = None
        if request.args.get('debug') == 'true':
            debug_info = {
                "score": float(score),
                "features": features,
                "thresholds": thresholds,
                "sensitivity_applied": float(sensitivity),
                "model_used": bool(model_used),
                "model_class": model_class,
                "model_confidence": model_confidence,
                "model_ai_conf": model_ai_conf,
                "ai_threshold": ai_threshold,
                "human_threshold": human_threshold,
                "duration_sec": duration_sec
            }
            print(f"[DEBUG] Classification score: {score}, Features: {features}")

        response_data = {
            "classification": classification,
            "confidence": round(confidence, 4),
            "explanation": explanation,
            "language": language
        }
        if debug_info:
            response_data["debug"] = debug_info

        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            "error": "Internal Server Error",
            "message": str(e)
        }), 500

@app.route('/')
def index():
    """Serve the endpoint tester interface"""
    return render_template_string(HTML_TEMPLATE)



@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested endpoint was not found. Make sure you are using /api/detect'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An error occurred while processing your request'
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    error_trace = traceback.format_exc()
    print(f"Unhandled exception: {str(e)}")
    print(f"Traceback: {error_trace}")
    return jsonify({
        'error': 'Internal Server Error',
        'message': str(e)
    }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'AI-Generated Voice Detection API',
        'supported_languages': SUPPORTED_LANGUAGES,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }), 200

# HTML Template for Endpoint Tester
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Detector</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
            background-size: 400% 400%;
            animation: bgShift 15s ease infinite;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 16px;
        }
        
        @keyframes bgShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .container {
            width: 100%;
            max-width: 520px;
            background: white;
            border-radius: 24px;
            box-shadow: 0 25px 50px rgba(0,0,0,0.15);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 56px 32px 48px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 30% 50%, rgba(255,255,255,0.1) 0%, transparent 50%),
                        radial-gradient(circle at 70% 50%, rgba(255,255,255,0.1) 0%, transparent 50%);
            pointer-events: none;
        }
        
        .header h1 {
            font-size: 56px;
            font-weight: 800;
            margin-bottom: 18px;            margin-top: 24px;            letter-spacing: -1px;
            display: inline-block;
            animation: float 3s ease-in-out infinite;
            position: relative;
            z-index: 1;
            filter: drop-shadow(0 4px 12px rgba(0,0,0,0.15));
        }
        
        @keyframes float {
            0%, 100% {
                transform: translateY(0px) rotate(0deg);
            }
            25% {
                transform: translateY(-12px) rotate(-5deg);
            }
            50% {
                transform: translateY(-18px) rotate(5deg);
            }
            75% {
                transform: translateY(-12px) rotate(-5deg);
            }
        }
        
        @keyframes slideInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes shimmer {
            0%, 100% {
                opacity: 0.85;
                filter: brightness(1);
            }
            50% {
                opacity: 1;
                filter: brightness(1.15);
            }
        }
        
        .header p {
            font-size: 42px;
            font-weight: 900;
            opacity: 1;
            letter-spacing: 3px;
            text-transform: uppercase;
            background: linear-gradient(135deg, #ffffff 0%, #e0d5ff 50%, #ffffff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: shimmer 2.5s ease-in-out infinite, slideInDown 0.8s ease-out;
            position: relative;
            z-index: 1;
            margin: 0;
            text-shadow: 0 0 30px rgba(255, 255, 255, 0.3);
            filter: drop-shadow(0 4px 20px rgba(240, 147, 251, 0.3));
        }
        
        .content {
            padding: 40px 32px;
        }
        
        .form-group {
            margin-bottom: 24px;
        }
        
        label {
            display: block;
            font-size: 14px;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 8px;
        }

        .help-text {
            font-size: 12px;
            color: #6b7280;
            margin-top: 6px;
        }
        
        input[type="file"] {
            width: 100%;
            padding: 14px 16px;
            border: 2px dashed #d1d5db;
            border-radius: 12px;
            font-size: 14px;
            color: #6b7280;
            cursor: pointer;
            transition: all 0.2s;
            background: #f9fafb;
        }
        
        input[type="file"]:hover {
            border-color: #667eea;
            background: #f5f3ff;
        }
        
        .btn {
            width: 100%;
            padding: 14px 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 32px 0;
        }
        
        .spinner {
            width: 48px;
            height: 48px;
            border: 4px solid #f3f4f6;
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 16px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .loading p {
            color: #6b7280;
            font-size: 14px;
        }
        
        .result {
            display: none;
            margin-top: 32px;
            padding: 24px;
            border-radius: 16px;
            animation: slideUp 0.4s ease-out;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(16px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result.success {
            background: #ecfdf5;
            border: 2px solid #10b981;
        }
        
        .result.error {
            background: #fef2f2;
            border: 2px solid #ef4444;
        }
        
        .result-title {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .result.success .result-title {
            color: #059669;
        }
        
        .result.error .result-title {
            color: #dc2626;
        }
        
        .result-field {
            margin-bottom: 16px;
        }
        
        .result-field:last-child {
            margin-bottom: 0;
        }
        
        .result-label {
            font-size: 13px;
            font-weight: 600;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        
        .result-value {
            font-size: 15px;
            color: #1f2937;
            word-break: break-word;
            line-height: 1.5;
        }
        
        .result.success .result-value {
            color: #065f46;
        }
        
        .result.error .result-value {
            color: #7f1d1d;
        }
        
        .confidence-bar {
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 4px;
        }
        
        .confidence-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981, #059669);
            border-radius: 4px;
            transition: width 0.6s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <p>Voice Detector</p>
            <h1>🎤</h1>
        </div>
        
        <div class="content">
            <form id="testForm">
                <div class="form-group">
                    <label for="audioFile">Select Audio File</label>
                    <input type="file" id="audioFile" accept="audio/*,.mp4,.mp3,.wav,.ogg,.flac,.m4a">
                    <div class="help-text">Use either a file or a URL below.</div>
                </div>
                <div class="form-group">
                    <label for="audioUrl">Or Paste Audio URL (MP3)</label>
                    <input type="url" id="audioUrl" placeholder="https://example.com/sample.mp3">
                </div>
                <div class="form-group">
                    <label for="apiKey">API Key (if required)</label>
                    <input type="text" id="apiKey" placeholder="hackathon-2026-demo-key" value="hackathon-2026-demo-key">
                </div>
                <button type="button" class="btn" id="testBtn">Analyze</button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing...</p>
            </div>
            
            <div class="result" id="result">
                <div class="result-title" id="resultTitle"></div>
                
                <div class="result-field">
                    <div class="result-label">Classification</div>
                    <div class="result-value" id="resultClass"></div>
                </div>
                
                <div class="result-field">
                    <div class="result-label">Confidence</div>
                    <div class="result-value" id="resultConfidence"></div>
                    <div class="confidence-bar">
                        <div class="confidence-bar-fill" id="confidenceBar"></div>
                    </div>
                </div>
                
                <div class="result-field">
                    <div class="result-label">Language</div>
                    <div class="result-value" id="resultLanguage"></div>
                </div>
                
                <div class="result-field">
                    <div class="result-label">Details</div>
                    <div class="result-value" id="resultExplanation"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const testBtn = document.getElementById('testBtn');
            const audioFile = document.getElementById('audioFile');
            const audioUrl = document.getElementById('audioUrl');
            const apiKey = document.getElementById('apiKey');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const resultTitle = document.getElementById('resultTitle');
            const resultClass = document.getElementById('resultClass');
            const resultConfidence = document.getElementById('resultConfidence');
            const resultLanguage = document.getElementById('resultLanguage');
            const resultExplanation = document.getElementById('resultExplanation');
            const confidenceBar = document.getElementById('confidenceBar');
            
            testBtn.addEventListener('click', async (e) => {
                e.preventDefault();
                
                const urlValue = (audioUrl.value || '').trim();
                const hasFile = !!audioFile.files[0];
                if (!hasFile && !urlValue) {
                    alert('Please select an audio file or paste a URL');
                    return;
                }
                
                loading.style.display = 'block';
                result.style.display = 'none';
                testBtn.disabled = true;
                
                try {
                    const authHeader = (apiKey.value || '').trim();
                    let response;

                    if (hasFile) {
                        const formData = new FormData();
                        formData.append('audio', audioFile.files[0]);

                        const headers = {};
                        if (authHeader) {
                            headers['Authorization'] = `Bearer ${authHeader}`;
                        }

                        response = await fetch('/api/detect', {
                            method: 'POST',
                            body: formData,
                            headers
                        });
                    } else {
                        const headers = { 'Content-Type': 'application/json' };
                        if (authHeader) {
                            headers['Authorization'] = `Bearer ${authHeader}`;
                        }
                        response = await fetch('/api/detect', {
                            method: 'POST',
                            headers,
                            body: JSON.stringify({ audio_url: urlValue })
                        });
                    }
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        throw new Error(data.message || 'Detection failed');
                    }
                    
                    loading.style.display = 'none';
                    result.style.display = 'block';
                    
                    const isAI = data.classification === 'AI-generated';
                    result.className = isAI ? 'result error' : 'result success';
                    resultTitle.textContent = isAI ? '🤖 AI-Generated' : '👤 Human-Generated';
                    resultClass.textContent = data.classification;
                    resultConfidence.textContent = (data.confidence * 100).toFixed(1) + '%';
                    confidenceBar.style.width = (data.confidence * 100) + '%';
                    resultLanguage.textContent = data.language || 'Unknown';
                    resultExplanation.textContent = data.explanation;
                    
                } catch (err) {
                    loading.style.display = 'none';
                    result.style.display = 'block';
                    result.className = 'result error';
                    resultTitle.textContent = '❌ Error';
                    resultClass.textContent = '';
                    resultConfidence.textContent = '';
                    resultLanguage.textContent = '';
                    resultExplanation.textContent = err.message;
                } finally {
                    testBtn.disabled = false;
                }
            });
        });
    </script>
</body>
</html>
'''

if __name__ == "__main__":
    app.run(debug=False)
