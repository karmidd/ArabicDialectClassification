import requests
import time
import json

dialect_starts = {
    "ALG": 0, "EGY": 30000, "IRA": 175000, "JOR": 455000,
    "KSA": 460000, "KUW": 525000, "LEB": 555000, "LIB": 590000,
    "MAU": 625000, "MOR": 755000, "OMA": 770000, "PAL": 795000,
    "QAT": 835000, "SUD": 860000, "SYR": 880000, "UAE": 925000,
    "YEM": 975000,
}

all_rows = []

for dialect, offset in dialect_starts.items():
    dialect_rows = []

    for batch in range(2):  # 2 requests of 100 = 200
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

        rows = [row["row"] for row in data["rows"] if row["row"]["dialect"] == dialect]
        dialect_rows.extend(rows)
        time.sleep(1)

    all_rows.extend(dialect_rows[:200])
    print(f"{dialect}: got {len(dialect_rows[:200])} rows")

metadata = [{"id": r["id"], "dialect": r["dialect"]} for r in all_rows]
with open("adi17_sample_ids.json", "w") as f:
    json.dump(metadata, f)

print(f"\nTotal: {len(metadata)} sample IDs saved")