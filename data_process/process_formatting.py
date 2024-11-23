import pandas as pd


# CSV 파일 읽기
df = pd.read_csv("external_raw_v1.csv", encoding="utf-8")

# 새로운 형식으로 변환
new_records = []
for idx, row in df.iterrows():
    new_record = {
        "id": f"external-data-{idx + 1}",
        "paragraph": row["paragraph"],
        "problems": {"question": row["question"].strip(), "choices": eval(row["choices"]), "answer": row["answer"]},
    }
    new_records.append(new_record)

# 새로운 DataFrame 생성
new_df = pd.DataFrame(new_records)
new_df.to_csv("external_v1.csv", index=False)
