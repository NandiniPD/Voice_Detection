import sys
sys.path.insert(0,'.')
import app
import librosa

p = 'data/ai/telhum.mp4'
try:
    y, sr = librosa.load(p, sr=None)
except Exception as e:
    print('LOAD_ERROR', e)
    sys.exit(1)

res = app.detect_voice_type(y, sr)
classification, confidence, explanation, features, score, thresholds = res
print('CLASSIFICATION:', classification)
print('SCORE:', score)
print('CONFIDENCE:', confidence)
print('\nFEATURES:')
for k,v in features.items():
    print(f'  {k}: {v}')
print('\nTHRESHOLDS:')
for k,v in thresholds.items():
    print(f'  {k}: {v}')
print('\nEXPLANATION:', explanation)
