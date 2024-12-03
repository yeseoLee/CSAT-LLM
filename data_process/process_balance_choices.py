from ast import literal_eval
import random

import pandas as pd


def balance_choices_dataset(file_path):
    # CSV 파일 읽기
    df = pd.read_csv(file_path)
    df["problems"] = df["problems"].apply(literal_eval)

    # 선택지와 답 교체 함수
    def swap_choices_and_answer(df):
        for index, row in df.iterrows():
            problems = row["problems"]
            choices = problems["choices"]
            answer = problems["answer"]
            # 선택지 랜덤 섞기
            shuffled_choices = choices[:]
            random.shuffle(shuffled_choices)
            # 새로운 답 인덱스 계산
            new_answer = shuffled_choices.index(choices[answer - 1])
            # 교체된 선택지와 답으로 업데이트
            problems["choices"] = shuffled_choices
            problems["answer"] = new_answer + 1
            df.at[index, "problems"] = problems
        return df

    # 선택지와 답 교체 적용
    return swap_choices_and_answer(df)


def answer_counts(file_path):
    df = pd.read_csv(file_path)
    df["problems"] = df["problems"].apply(literal_eval)

    records = []
    for idx, row in df.iterrows():
        problems = row["problems"]
        record = {
            "id": row["id"],
            "paragraph": row["paragraph"],
            "question": problems["question"],
            "choices": problems["choices"],
            "answer": problems.get("answer", None),
            "question_plus": problems.get("question_plus", None),
        }
        records.append(record)

    processed_df = pd.DataFrame(records)
    print(len(processed_df))

    print(processed_df["choices"].apply(len).value_counts())
    print(processed_df["answer"].value_counts())


if __name__ == "__main__":
    train_balanced = balance_choices_dataset("../data/train.csv")
    train_balanced.to_csv("../data/train_balanced.csv", index=False)
    answer_counts("../data/train_balanced.csv")
