import csv
import json
import re

from rank_bm25 import BM25Okapi


def safe_parse(s):
    try:
        s = s.replace("'", '"')
        s = re.sub(r'(?<!\\)"', '\\"', s)
        s = f'"{s}"'
        return json.loads(s), None
    except json.JSONDecodeError as e:
        return None, str(e)

def simple_tokenize(text):
    # 간단한 토큰화: 공백을 기준으로 단어 분리
    return text.split()

def simple_sent_tokenize(text):
    # 간단한 문장 분리: 마침표, 물음표, 느낌표를 기준으로 문장 분리
    return re.split('[.!?]+', text)

def find_answer_in_paragraph(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row_num, row in enumerate(reader, start=2):
            id = row.get('id', f'Row {row_num}')
            paragraph = row.get('paragraph', '')
            question_plus_raw = row.get('question_plus', '')
            answer = row.get('answer', '')

            question_plus, error = safe_parse(question_plus_raw)
            if error:
                print(f"Parsing error at Row {row_num} (ID: {id}):")
                print(f"  Error: {error}")
                print(f"  Raw question_plus data: {question_plus_raw[:200]}")
                print(f"  Paragraph preview: {paragraph[:100]}...")
                print("\nFull row data:")
                for key, value in row.items():
                    print(f"  {key}: {value[:100]}...")
                continue  # 오류가 있는 행은 건너뛰고 다음 행으로 진행

            # 문단을 문장으로 분리
            sentences = simple_sent_tokenize(paragraph)

            # BM25 모델 생성
            tokenized_sentences = [simple_tokenize(sent) for sent in sentences]
            bm25 = BM25Okapi(tokenized_sentences)

            # 질문을 토큰화
            query = simple_tokenize(question_plus['question'])

            # BM25 점수 계산
            scores = bm25.get_scores(query)

            # 가장 높은 점수를 가진 문장 찾기
            best_match_index = scores.argmax()
            best_match_sentence = sentences[best_match_index]

            print(f"Row {row_num} (ID: {id}):")
            print(f"Question: {question_plus['question']}")
            print(f"Answer: {answer}")
            print(f"Best matching sentence: {best_match_sentence}")
            print(f"BM25 Score: {scores[best_match_index]}")
            print("---")

    print("Processing completed.")

# CSV 파일 경로
csv_file = '../data/train.csv'

# 실행
find_answer_in_paragraph(csv_file)
