# Credit : facebookresearch/DPR

"""
FAISS-based index components for dense retriever
"""

import logging
import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np


logger = logging.getLogger()


class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        """
        인덱서를 초기화합니다.
        - buffer_size: 한 번에 처리할 벡터의 최대 개수 (기본값은 50000).
        - index_id_to_db_id: FAISS에서 인덱스 ID와 실제 데이터베이스 ID를 매핑하기 위한 리스트.
        - index: FAISS 인덱스 객체.
        """
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []  # 인덱스 ID를 실제 데이터베이스 ID와 매핑하는 리스트.
        self.index = None  # FAISS 인덱스 객체.

    def init_index(self, vector_sz: int):
        raise NotImplementedError

    def index_data(self, data: List[Tuple[object, np.array]]):
        raise NotImplementedError

    def get_index_name(self):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        raise NotImplementedError

    def serialize(self, file: str):
        logger.info("Serializing index to %s", file)

        if os.path.isdir(file):
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + ".index.dpr"
            meta_file = file + ".index_meta.dpr"

        faiss.write_index(self.index, index_file)  # FAISS 인덱스를 파일에 저장.
        with open(meta_file, mode="wb") as f:
            pickle.dump(self.index_id_to_db_id, f)  # ID 매핑 정보를 저장.

    def get_files(self, path: str):
        if os.path.isdir(path):
            index_file = os.path.join(path, "index.dpr")  # FAISS 인덱스를 파일에서 로드.
            meta_file = os.path.join(path, "index_meta.dpr")
        else:
            index_file = path + ".{}.dpr".format(self.get_index_name())
            meta_file = path + ".{}_meta.dpr".format(self.get_index_name())
        return index_file, meta_file

    def index_exists(self, path: str):
        index_file, meta_file = self.get_files(path)
        return os.path.isfile(index_file) and os.path.isfile(meta_file)

    def deserialize(self, path: str):
        logger.info("Loading index from %s", path)
        index_file, meta_file = self.get_files(path)

        self.index = faiss.read_index(index_file)
        logger.info("Loaded index of type %s and size %d", type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert (
            len(self.index_id_to_db_id) == self.index.ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"

    def _update_id_mapping(self, db_ids: List) -> int:
        self.index_id_to_db_id.extend(db_ids)
        return len(self.index_id_to_db_id)


class DenseFlatIndexer(DenseIndexer):
    def __init__(self, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)

    def init_index(self, vector_sz: int):
        self.index = faiss.IndexFlatIP(vector_sz)  # Inner Product를 사용하는 기본 인덱스 초기화.

    def index_data(self, data: List[Tuple[object, np.array]]):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            db_ids = [t[0] for t in data[i : i + self.buffer_size]]
            vectors = [np.reshape(t[1], (1, -1)) for t in data[i : i + self.buffer_size]]
            vectors = np.concatenate(vectors, axis=0)
            total_data = self._update_id_mapping(db_ids)
            self.index.add(vectors)
            logger.info("data indexed %d", total_data)

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info("Total data indexed %d", indexed_cnt)

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        scores, indexes = self.index.search(query_vectors, top_docs)  # 쿼리 벡터와 가장 유사한 벡터 검색.
        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

    def get_index_name(self):
        return "flat_index"


class DenseHNSWFlatIndexer(DenseIndexer):
    """
    Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
    """

    def __init__(
        self,
        buffer_size: int = 1e9,
        store_n: int = 512,
        ef_search: int = 128,
        ef_construction: int = 200,
    ):
        super(DenseHNSWFlatIndexer, self).__init__(buffer_size=buffer_size)
        self.store_n = store_n
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.phi = 0

    def init_index(self, vector_sz: int):
        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = faiss.IndexHNSWFlat(vector_sz + 1, self.store_n)  # L2 거리 기반 HNSW 인덱스 초기화.
        index.hnsw.efSearch = self.ef_search
        index.hnsw.efConstruction = self.ef_construction
        self.index = index

    def index_data(self, data: List[Tuple[object, np.array]]):
        n = len(data)

        # max norm is required before putting all vectors in the index to convert inner product similarity to L2
        if self.phi > 0:
            raise RuntimeError(
                "DPR HNSWF index needs to index all data at once," "results will be unpredictable otherwise."
            )
        phi = 0
        for i, item in enumerate(data):
            id, doc_vector = item[0:2]
            norms = (doc_vector**2).sum()
            phi = max(phi, norms)
        logger.info("HNSWF DotProduct -> L2 space phi={}".format(phi))
        self.phi = phi

        # indexing in batches is beneficial for many faiss index types
        bs = int(self.buffer_size)
        for i in range(0, n, bs):
            db_ids = [t[0] for t in data[i : i + bs]]
            vectors = [np.reshape(t[1], (1, -1)) for t in data[i : i + bs]]

            norms = [(doc_vector**2).sum() for doc_vector in vectors]
            aux_dims = [np.sqrt(phi - norm) for norm in norms]
            hnsw_vectors = [np.hstack((doc_vector, aux_dims[i].reshape(-1, 1))) for i, doc_vector in enumerate(vectors)]
            hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)
            self.train(hnsw_vectors)

            self._update_id_mapping(db_ids)
            self.index.add(hnsw_vectors)
            logger.info("data indexed %d", len(self.index_id_to_db_id))
        indexed_cnt = len(self.index_id_to_db_id)
        logger.info("Total data indexed %d", indexed_cnt)

    def train(self, vectors: np.array):
        pass

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        aux_dim = np.zeros(len(query_vectors), dtype="float32")
        query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        logger.info("query_hnsw_vectors %s", query_nhsw_vectors.shape)
        scores, indexes = self.index.search(query_nhsw_vectors, top_docs)
        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

    def deserialize(self, file: str):
        super(DenseHNSWFlatIndexer, self).deserialize(file)
        # to trigger exception on subsequent indexing
        self.phi = 1

    def get_index_name(self):
        return "hnsw_index"


class DenseHNSWSQIndexer(DenseHNSWFlatIndexer):
    """
    Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
    """

    def __init__(
        self,
        buffer_size: int = 1e10,
        store_n: int = 128,
        ef_search: int = 128,
        ef_construction: int = 200,
    ):
        super(DenseHNSWSQIndexer, self).__init__(
            buffer_size=buffer_size,
            store_n=store_n,
            ef_search=ef_search,
            ef_construction=ef_construction,
        )

    def init_index(self, vector_sz: int):
        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = faiss.IndexHNSWSQ(vector_sz + 1, faiss.ScalarQuantizer.QT_8bit, self.store_n)
        index.hnsw.efSearch = self.ef_search
        index.hnsw.efConstruction = self.ef_construction
        self.index = index

    def train(self, vectors: np.array):
        self.index.train(vectors)

    def get_index_name(self):
        return "hnswsq_index"
