from ast import literal_eval
import json
import os
import time

from googletrans import Translator
from loguru import logger
from tqdm import tqdm


class TranslationCache:
    def __init__(self, cache_file="translation_cache.json"):
        self.cache_file = cache_file
        self.cache = {}

    def __enter__(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
        self.cache.clear()  # 메모리 해제

    def get_translation(self, text):
        return self.cache.get(text)

    def add_translation(self, text, translation):
        self.cache[text] = translation


def translate_with_retry(translator, text, cache, max_retries=3):
    # 캐시에서 번역 확인
    cached_translation = cache.get_translation(text)
    if cached_translation:
        return cached_translation

    # 캐시에 없는 경우 번역 수행
    for attempt in range(max_retries):
        try:
            translated = translator.translate(text, src="en", dest="ko").text
            time.sleep(0.01)
            cache.add_translation(text, translated)
            return translated
        except Exception as e:
            if attempt == max_retries - 1:
                logger.info(f"번역 실패: {str(e)}")
                return text
            time.sleep(1)


def translate_list_column(text, cache):
    items = literal_eval(text)
    translator = Translator()
    translated_items = []

    for item in items:
        translated = translate_with_retry(translator, item, cache)
        translated_items.append(translated)
    return translated_items


def translate_column(texts, cache):
    translator = Translator()
    translated_texts = []

    # 중복 제거를 위해 유니크한 텍스트만 추출
    unique_texts = list(set(texts))

    for text in tqdm(unique_texts, desc="고유 텍스트 번역 중"):
        translated = translate_with_retry(translator, text, cache)

    # 원본 순서대로 캐시에서 번역 가져오기
    for text in texts:
        translated = cache.get_translation(text)
        translated_texts.append(translated)

    return translated_texts
