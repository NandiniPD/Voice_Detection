import requests
import os

API_URL = "http://localhost:5000/api/detect"

def test_api_with_audio_file(audio_file_path):
    try:
        files = {
            'audio': open(audio_file_path, 'rb')
        }

        print(f"Testing API with file: {audio_file_path}")

        response = requests.post(API_URL, files=files)

        print("Status Code:", response.status_code)
        print("Response:")
        print(response.json())

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_api_with_audio_file("11.wav")  # put your audio file here
