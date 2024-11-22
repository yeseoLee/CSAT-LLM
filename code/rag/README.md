# Dense Retriever 사용 가이드

이 가이드는 Dense Retriever를 설정하고 사용하는 방법을 설명합니다.

## 준비 단계

1. **위키피디아 덤프 파일 준비**
   - `rag` 폴더 내에 위키피디아 덤프 파일을 다운로드하고 압축을 해제합니다.
   - `text` 폴더 내에 `AA`, `AB`, `AC` 폴더가 존재해야 합니다.
   - 각 폴더 안에 `wiki_`로 시작하는 파일들이 있어야 합니다.
2. **KorQuAD_v1.0 데이터셋 준비**
    - `data` 폴더 내에 KorQuAD_v1.0_dev, KorQuAD_v1.0_train 파일을 준비해야합니다.
    - https://korquad.github.io/category/1.0_KOR.html
3. **데이터 전처리**
   - `prepare_dense.py` 스크립트를 실행합니다.
   - 실행 후 다음 파일들이 생성되어야 합니다:
     - `preproccessed_passages/0-XXXX.p,XXXX-XXXX.p...`
     - `titled_passage_map.p`
     - `2050iter_flat/index_meta.dpr,index.dpr`
   - 생성된 파일을 사용하여 예시 쿼리에 대한 적절한 문서 검색 결과를 확인합니다.
## 사용 방법

1. **설정 파일 수정**
   - `config` 파일에서 `retriever_type`을 `"DPR"`로 설정합니다.

2. **실행**
   - 다음 명령어를 실행하여 Dense Retriever를 사용합니다:
     ```bash
     python main.py
     ```

## 주의사항
- 이 코드는 https://github.com/TmaxEdu/KorDPR를 참고하여 작성되었습니다.
- 위의 단계들을 순서대로 진행해야 Dense Retriever가 정상적으로 작동합니다.
