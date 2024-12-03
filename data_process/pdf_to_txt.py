import os
import re

from pdfminer.high_level import extract_text


def split_text_by_keyword(text, keyword):
    sections = re.split(rf"{keyword}", text)
    sections = [section.strip() + keyword for section in sections[:-1]] + [sections[-1].strip()]
    return sections


def save_sections_to_files(sections, output_dir="sections"):
    os.makedirs(output_dir, exist_ok=True)
    for i, section in enumerate(sections):
        file_name = os.path.join(output_dir, f"section_{i+1}.txt")
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(section)
    print(f"Sections saved to '{output_dir}' directory.")


if __name__ == "__main__":
    pdf_file_path = "./data/test/2025.pdf"
    output_dir = "./data/test/sections"
    keyword = "답하시오"

    text = extract_text(pdf_file_path)
    split_text = split_text_by_keyword(text, keyword)
    save_sections_to_files(split_text, output_dir=output_dir)
