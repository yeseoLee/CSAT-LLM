import json
import os
from pathlib import Path
import re
import urllib.request

from loguru import logger


def preprocess_text(text):
    # 한글, 숫자, 특수문자, 공백만 남기고 나머지 제거
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r'[^ㄱ-ㅎ가-힣0-9!"#%&\'(),-./:;<=>?@[\]^_`{|}~\s]', "", text)

    # 내용이 빈 괄호 제거
    pattern = r"\(\s*\)"
    while re.search(pattern, text):
        text = re.sub(pattern, "", text)

    return text


def process_json_array(json_data):
    # text 필드 전처리
    if "text" in json_data:
        json_data["text"] = preprocess_text(json_data["text"])

    # title 필드 전처리
    if "title" in json_data:
        json_data["title"] = preprocess_text(json_data["title"])

    return json_data


def process_json_file(json_filename):
    with open(json_filename, "r", encoding="utf-8") as f:
        docs = json.load(f)

    processed_docs = [process_json_array(item) for item in docs]

    # 디렉토리와 파일명 분리 후 파일명에만 'processed_' 추가
    directory, filename = os.path.split(json_filename)
    output_path = os.path.join(directory, "processed_" + filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, ensure_ascii=False, indent=2)


def _dump_wiki(data_path: str = "../data"):
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


def _parse_wiki_dump(file_path: str = "../data/wiki_dump.txt"):
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


def wikipedia():
    """
    위키피디아 한국어 덤프 문서를 가져오고 파싱하여 하나의 JSON 파일 생성
    """
    _dump_wiki()
    _parse_wiki_dump()


def ai_hub_news_corpus(input_dir: str, output_file: str):
    """
    대규모 웹데이터 기반 한국어 말뭉치 데이터
    \n https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=624
    \n 지정된 디렉토리의 모든 JSON 파일을 처리하여 하나의 JSON 파일로 통합
    Args:
        input_dir: 입력 JSON 파일들이 있는 디렉토리 경로
        output_file: 출력될 통합 JSON 파일 경로
    """
    all_documents = []
    input_path = Path(input_dir)

    try:
        # 입력 디렉토리 내의 모든 JSON 파일 처리
        for json_file in input_path.glob("**/*.json"):
            logger.info(f"처리 중인 파일: {json_file}")

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # SJML 구조 확인 및 데이터 추출
                if "SJML" in data and "text" in data["SJML"]:
                    for doc in data["SJML"]["text"]:
                        processed_doc = {"title": doc["title"], "text": doc["content"]}
                        all_documents.append(processed_doc)
                else:
                    logger.warning(f"잘못된 JSON 구조: {json_file}")

            except json.JSONDecodeError:
                logger.error(f"JSON 파싱 오류: {json_file}")
            except Exception as e:
                logger.error(f"파일 처리 중 오류 발생: {json_file}, 오류: {str(e)}")

        # 최종 결과를 단일 JSON 파일로 저장
        if all_documents:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_documents, f, ensure_ascii=False, indent=2)
            logger.info(f"처리 완료: 총 {len(all_documents)}개 문서가 {output_file}에 저장됨")
        else:
            logger.warning("처리된 문서가 없습니다.")

    except Exception as e:
        logger.error(f"전체 처리 과정 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    os.chdir("../../")

    PROCESS_JSON_FILE = False
    WIKIPEDIA = False
    AI_HUB_NEWS_CORPUS = False

    if PROCESS_JSON_FILE:
        process_json_file("../data/documents.json")
    if WIKIPEDIA:
        wikipedia()
    if AI_HUB_NEWS_CORPUS:
        ai_hub_news_corpus("../data/ai_hub_news_corpus", "../data/ai_hub_news_corpus.json")
