from collections import defaultdict
from glob import glob
import logging
import os
import pickle

from tqdm import tqdm
from transformers import AutoTokenizer


os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()


class DataChunk:
    """인풋 text를 tokenizing한 뒤에 주어진 길이로 chunking 해서 반환합니다.
    이때 하나의 chunk(context, index 단위)는 하나의 article에만 속해있어야 합니다."""

    def __init__(self, chunk_size=100):
        self.chunk_size = chunk_size
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

    def chunk(self, input_file):
        logger.info(f"Processing file: {input_file}")
        with open(input_file, "rt", encoding="utf8") as f:
            input_txt = f.read().strip()
        input_txt = input_txt.split("</doc>")
        chunk_list = []
        orig_text = []
        for art in input_txt:
            art = art.strip()
            if not art:
                logger.debug("Article is empty, passing")
                continue
            title = art.split("\n")[0].strip(">").split("title=")[1].strip('"')
            text = "\n".join(art.split("\n")[2:]).strip()

            logger.debug(f"Processing article: {title}")

            encoded_title = self.tokenizer.encode(title, add_special_tokens=True)
            encoded_txt = self.tokenizer.encode(text, add_special_tokens=True)
            if len(encoded_txt) < 5:
                logger.debug(f"Title {title} has <5 subwords in its article, passing")
                continue

            for start_idx in range(0, len(encoded_txt), self.chunk_size):
                end_idx = min(len(encoded_txt), start_idx + self.chunk_size)
                chunk = encoded_title + encoded_txt[start_idx:end_idx]
                orig_text.append(self.tokenizer.decode(chunk))
                chunk_list.append(chunk)

        logger.info(f"Processed {len(orig_text)} chunks from {input_file}.")
        return orig_text, chunk_list


def save_orig_passage(input_path="text", passage_path="processed_passages", chunk_size=100):
    os.makedirs(passage_path, exist_ok=True)
    app = DataChunk(chunk_size=chunk_size)
    idx = 0
    for path in tqdm(glob(f"{input_path}/*/wiki_*")):
        ret, _ = app.chunk(path)
        logger.info(f"Processed {len(ret)} chunks from {path}.")  # 추가된 로그
        if len(ret) > 0:  # 청크가 있는 경우에만 저장
            to_save = {idx + i: ret[i] for i in range(len(ret))}
            with open(f"{passage_path}/{idx}-{idx+len(ret)-1}.p", "wb") as f:
                pickle.dump(to_save, f)
            idx += len(ret)


def save_title_index_map(index_path="title_passage_map.p", source_passage_path="processed_passages"):
    logging.getLogger()
    logger.debug(f"Looking for files in {source_passage_path}")
    files = glob(f"{source_passage_path}/*")
    logger.debug(f"Found {len(files)} files")

    title_id_map = defaultdict(list)
    for f in tqdm(files):
        logger.debug(f"Processing file: {f}")
        with open(f, "rb") as _f:
            id_passage_map = pickle.load(_f)

        # 로그 추가: id_passage_map의 형식 및 내용 확인
        logger.debug(f"Loaded {len(id_passage_map)} passages from {f}")
        logger.debug(f"Sample passage: {list(id_passage_map.items())[:5]}")  # 첫 5개 항목 출력

        for id, passage in id_passage_map.items():
            parts = passage.split("[SEP]")
            if len(parts) > 1:
                title = parts[0].split("[CLS]")[1].strip()
                title_id_map[title].append(id)
            else:
                logger.debug(f"Unexpected passage format in file {f}, id {id}")

        logger.debug(f"Processed {len(id_passage_map)} passages from {f}...")

        logger.debug(f"Total unique titles: {len(title_id_map)}")

    with open(index_path, "wb") as f:
        pickle.dump(title_id_map, f)

    logger.debug(f"Finished saving title_index_mapping at {index_path}!")


# if __name__ == "__main__":
#     save_orig_passage()
#     save_title_index_map()
