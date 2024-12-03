import json
import os
import re
import urllib.request

from loguru import logger


def dump_wiki(data_path: str = "../data"):
    """
    위키피디아 덤프를 다운로드하고 추출하는 함수
    """
    dump_filename = "kowiki-latest-pages-articles.xml.bz2"
    dump_path = os.path.join(data_path, dump_filename)
    wiki_url = f"https://dumps.wikimedia.org/kowiki/latest/{dump_filename}"

    # wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2
    if not os.path.exists(dump_path):
        logger.debug(f"위키피디아 덤프를 다운로드합니다: {wiki_url}")
        urllib.request.urlretrieve(wiki_url, dump_path)
        logger.debug(f"다운로드 완료: {dump_path}")

    # python -m wikiextractor.WikiExtractor kowiki-latest-pages-articles.xml.bz2
    extract_dir = os.path.join(data_path, "text")
    if not os.path.exists(extract_dir):
        logger.debug("WikiExtractor로 덤프 파일을 추출합니다...")
        os.system(f"python -m wikiextractor.WikiExtractor {dump_path} -o {extract_dir}")
        logger.debug("추출 완료")

    def _get_filename_list(dirname):
        filepaths = []
        for root, dirs, files in os.walk(dirname):
            for file in files:
                filepath = os.path.join(root, file)
                if re.match(r"wiki_[0-9][0-9]", file):
                    filepaths.append(filepath)
        return sorted(filepaths)

    filepaths = _get_filename_list(extract_dir)
    output_path = os.path.join(data_path, "wiki_dump.txt")

    # 파일 내용 읽기
    all_text = ""
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            all_text += f.read() + "\n"

    # 전체 텍스트를 하나의 파일로 저장
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(all_text)

    logger.debug(f"총 {len(filepaths)}개의 파일을 처리했습니다.")
    logger.debug(f"모든 내용이 {output_path} 파일에 저장되었습니다.")


def parse_wiki_dump(file_path: str = "../data/wiki_dump.txt"):
    """
    위키피디아 덤프 파일을 JSON 형식으로 변환하는 함수
    """
    documents = []
    current_doc = ""
    doc_id = None
    title = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 새로운 문서 시작
            if line.startswith("<doc"):
                doc_id = re.search(r'id="([^"]+)"', line).group(1)
                title = re.search(r'title="([^"]+)"', line).group(1)
                current_doc = ""
            # 문서 끝
            elif line.startswith("</doc>"):
                if current_doc.strip():  # 빈 문서가 아닌 경우만 추가
                    documents.append({"id": doc_id, "title": title, "text": current_doc.strip()})
            # 문서 내용
            else:
                current_doc += line

    # JSON 파일로 저장
    output_path = file_path.replace(".txt", ".json")
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(documents, json_file, ensure_ascii=False, indent=4)

    logger.debug(f"JSON 파일이 생성되었습니다: {output_path}")
    logger.debug(f"총 {len(documents)}개의 문서가 처리되었습니다.")
    return documents


if __name__ == "__main__":
    os.chdir("../../")
    dump_wiki()
    parse_wiki_dump()
