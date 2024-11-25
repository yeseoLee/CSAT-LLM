import pandas as pd


def formatting(suffix, input_filename, output_filename):
    # CSV 파일 읽기
    df = pd.read_csv(input_filename, encoding="utf-8")

    # 새로운 형식으로 변환
    new_records = []
    for idx, row in df.iterrows():
        new_record = {
            "id": f"external-data-{suffix}{idx + 1}",
            "paragraph": row["paragraph"],
            "problems": {"question": row["question"].strip(), "choices": eval(row["choices"]), "answer": row["answer"]},
        }
        new_records.append(new_record)

    # 새로운 DataFrame 생성
    new_df = pd.DataFrame(new_records)
    new_df.to_csv(output_filename, index=False)


if __name__ == "__main__":
    # formatting("external_raw.csv", "external.csv")
    formatting("SAT", "sat_gaokao_ko_raw.csv", "sat_gaokao_ko.csv")
    formatting("MuSR", "MuSR_ko_raw.csv", "MuSR_ko.csv")
