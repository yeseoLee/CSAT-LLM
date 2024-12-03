import pandas as pd


def formatting(suffix, input_filename, output_filename):
    # CSV 파일 읽기
    df = pd.read_csv(input_filename, encoding="utf-8")

    # 새로운 형식으로 변환
    new_records = []
    for idx, row in df.iterrows():
        new_record = {
            "id": f"external-data-{suffix}{idx + 1}",
            "paragraph": "" if pd.isna(row["paragraph"]) else str(row["paragraph"]),
            "problems": {"question": row["question"].strip(), "choices": eval(row["choices"]), "answer": row["answer"]},
        }
        new_records.append(new_record)

    # 새로운 DataFrame 생성
    new_df = pd.DataFrame(new_records)
    new_df.to_csv(output_filename, index=False)


if __name__ == "__main__":
    formatting("gichulpass20-", "gichulpass_20_107_raw.csv", "gichulpass_20_107.csv")
    formatting("gichulpass26-", "gichulpass_26_1319_raw.csv", "gichulpass_24_1319.csv")
    formatting("gichulpass34-", "gichulpass_34_1352_raw.csv", "gichulpass_34_1352.csv")
    formatting("gichulpass35-", "gichulpass_35_568_raw.csv", "gichulpass_35_568.csv")
    formatting("SAT", "sat_gaokao_ko_raw.csv", "sat_gaokao_ko.csv")
    formatting("MuSR", "MuSR_ko_raw.csv", "MuSR_ko.csv")
