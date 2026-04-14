import requests
import os
import time
import json

dialect_starts = {
    "ALG": 0, "EGY": 30000, "IRA": 175000, "JOR": 455000,
    "KSA": 460000, "KUW": 525000, "LEB": 555000, "LIB": 590000,
    "MAU": 625000, "MOR": 755000, "OMA": 770000, "PAL": 795000,
    "QAT": 835000, "SUD": 860000, "SYR": 880000, "UAE": 925000,
    "YEM": 975000,
}

audio_dir = "adi17_audio"
os.makedirs(audio_dir, exist_ok=True)

metadata = []

for dialect, offset in dialect_starts.items():
    target = 35 if dialect == "JOR" else 200
    batches = 1 if dialect == "JOR" else 2
    count = 0

    for batch in range(batches):
        batch_offset = offset + (batch * 100)
        url = f"https://datasets-server.huggingface.co/rows?dataset=ArabicSpeech/ADI17&config=default&split=train&offset={batch_offset}&length=100"

        while True:
            r = requests.get(url)
            if r.status_code == 200:
                try:
                    data = r.json()
                    break
                except:
                    pass
            print(f"  Rate limited, waiting 10s...")
            time.sleep(10)

        for row in data["rows"]:
            if count >= target:
                break
            rd = row["row"]
            if rd["dialect"] != dialect:
                continue

            filename = f"{rd['dialect']}_{rd['id']}.wav"
            filepath = os.path.join(audio_dir, filename)

            if not os.path.exists(filepath):
                audio_url = rd["audio"][0]["src"]
                try:
                    ar = requests.get(audio_url, timeout=30)
                    if ar.status_code == 200:
                        with open(filepath, "wb") as f:
                            f.write(ar.content)
                        count += 1
                    else:
                        print(f"  Failed {rd['id']}: status {ar.status_code}")
                except Exception as e:
                    print(f"  Failed {rd['id']}: {e}")
            else:
                count += 1

        time.sleep(1)

    metadata.append({"dialect": dialect, "downloaded": count})
    print(f"{dialect}: downloaded {count}")

with open("adi17_metadata.json", "w") as f:
    json.dump(metadata, f)

print("\nDone!")