import requests

res = requests.get(
    "http://127.0.0.1:8000/predict",
    params={
        "lat": 37.589372,
        "lng": 127.016745
    }
)

print("status code:", res.status_code)
print("response text:", res.text)

if res.headers.get("content-type", "").startswith("application/json"):
    print(res.json())
else:
    print("JSON 응답이 아닙니다.")