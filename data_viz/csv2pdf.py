import argparse
import ast
import os
import sys
import textwrap

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


def draw_wrapped_text(c, text, x, y, max_width, max_height, line_height=14):
    """텍스트를 페이지 너비에 맞게 줄바꿈하여 출력하며, 페이지 높이를 초과하면 페이지를 넘김."""
    wrapped_text = textwrap.fill(text, width=70)
    text_obj = c.beginText(x, y)
    text_obj.setFont("NanumGothic", 10)
    text_obj.setLeading(line_height)

    for line in wrapped_text.splitlines():
        if text_obj.getY() < max_height:
            c.drawText(text_obj)
            c.showPage()
            text_obj = c.beginText(x, A4[1] - 3 * cm)
            text_obj.setFont("NanumGothic", 10)
            text_obj.setLeading(line_height)
        text_obj.textLine(line)

    c.drawText(text_obj)


def create_csat_style_pdf(data, filename):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    for index, row in data.iterrows():
        try:
            # 문제 ID 출력
            c.setFont("NanumGothic", 12)
            c.drawString(1 * cm, height - 1 * cm, f"문제 ID: {row['id']}")

            # 본문 출력
            paragraph = row["paragraph"]
            draw_wrapped_text(c, paragraph, 1 * cm, height - 2 * cm, max_width=85, max_height=14 * cm)

            # 문제 출력
            problem_data = ast.literal_eval(row["problems"])
            question = problem_data["question"]
            draw_wrapped_text(
                c,
                f"문제: {question}",
                1 * cm,
                height - 16 * cm,
                max_width=85,
                max_height=8 * cm,
            )

            # 선택지 출력
            choices = problem_data["choices"]
            choice_y = height - 19 * cm
            for i, choice in enumerate(choices, 1):
                draw_wrapped_text(
                    c,
                    f"{i}. {choice}",
                    1 * cm,
                    choice_y,
                    max_width=85,
                    max_height=3 * cm,
                )
                choice_y -= 1.2 * cm

            # 정답 표시
            answer = problem_data["answer"]
            draw_wrapped_text(
                c,
                f"정답: {answer}",
                1 * cm,
                choice_y - 1 * cm,
                max_width=85,
                max_height=3 * cm,
            )

            c.showPage()
        except KeyError as ke:
            print(f"Data format error: 필수 키가 없습니다. {ke}")

    # PDF 저장
    c.save()


if __name__ == "__main__":
    # 인자 파서 설정
    parser = argparse.ArgumentParser(description="Generate a CSAT style PDF from CSV data.")
    parser.add_argument(
        "--csv_path",
        default="../data/train.csv",
        help="Path to the CSV file containing the data.",
    )
    args = parser.parse_args()

    # CSV 파일 읽기 및 컬럼 확인
    try:
        df = pd.read_csv(args.csv_path)
        required_columns = {"id", "paragraph", "problems"}
        if not required_columns.issubset(df.columns):
            missing_columns = required_columns - set(df.columns)
            raise ValueError(f"CSV 파일에 필요한 컬럼이 없습니다: {', '.join(missing_columns)}")
    except FileNotFoundError:
        print(f"CSV 파일을 찾을 수 없습니다: {args.csv_path}")
        sys.exit(1)
    except ValueError as ve:
        print(ve)
        sys.exit(1)

    # 한글 폰트 등록
    font_path = os.path.abspath("../data/NanumGothic.ttf")
    if not os.path.isfile(font_path):
        print(
            f"폰트 파일을 찾을 수 없습니다: {font_path}"
            f"https://hangeul.naver.com/fonts/search?f=nanum 에서 폰트를 다운받아 data폴더에 넣어주세요."
        )
        sys.exit(1)

    pdfmetrics.registerFont(TTFont("NanumGothic", font_path))

    # PDF 파일명 설정 (입력 파일과 동일 경로 및 이름으로 설정, 확장자만 .pdf로 변경)
    pdf_filename = os.path.splitext(args.csv_path)[0] + ".pdf"

    # PDF 생성 함수 호출
    create_csat_style_pdf(df, pdf_filename)
