import chromadb
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from utils import get_env, simple_tokenize, normalize_scores

# ─────────────────────────────
# 1. 환경변수 읽기
# ─────────────────────────────
# 예: C:/Users/playdata/Desktop/(Chroma)vector_DB
VECTOR_DB_DIR = get_env("VECTOR_DB_DIR")
# 예: gemma_full_collection
COLLECTION_NAME = get_env("COLLECTION_NAME")
# 예: C:/Users/playdata/Desktop/embeddinggemma-300m
EMBEDDING_MODEL = get_env("EMBEDDING_MODEL")
# 하이브리드 가중치
W_VEC = float(get_env("HYBRID_WEIGHT_VECTOR", "0.8"))
W_BM25 = float(get_env("HYBRID_WEIGHT_BM25", "0.2"))

# ─────────────────────────────
# 2. 임베딩 모델 (로컬 경로도 OK)
# ─────────────────────────────
# 코랩에서 만들었든, 로컬에서 만들었든 .env에 있는 경로를 그대로 쓴다.
_st_model = SentenceTransformer(EMBEDDING_MODEL)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """텍스트 리스트를 벡터 리스트로 임베딩"""
    embs = _st_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return [e.tolist() for e in embs]

def embed_query(q: str) -> List[float]:
    """단일 쿼리 임베딩"""
    return embed_texts([q])[0]

# ─────────────────────────────
# 3. Chroma client / collection
# ─────────────────────────────
_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    # 코랩에서 만들 때 cosine으로 했으면 여기서도 맞춰야 한다
    metadata={"hnsw:space": "cosine"}
)

# ─────────────────────────────
# 4. BM25 (후보군 위에서만 수행)
# ─────────────────────────────
def bm25_on_candidates(query: str, candidates: List[str]) -> np.ndarray:
    """
    벡터로 먼저 후보를 뽑은 뒤,
    그 '후보들'에 대해서만 BM25를 돌려서 가볍게 하이브리드한다.
    """
    if not candidates:
        return np.array([])
    tokenized_cands = [simple_tokenize(t) for t in candidates]
    bm25 = BM25Okapi(tokenized_cands)
    return np.array(
        bm25.get_scores(simple_tokenize(query)),
        dtype=float,
    )

# ─────────────────────────────
# 5. 하이브리드 검색
# ─────────────────────────────
def hybrid_search(
    query: str,
    candidate_k: int = int(get_env("CANDIDATE_K", "50")),
    top_k: int = int(get_env("TOP_K", "5")),
) -> Tuple[List[str], List[str], List[float]]:
    """
    1) 벡터 검색으로 candidate_k 개수만큼 크게 뽑고
    2) 그 위에 BM25를 얹어서
    3) 가중합으로 상위 top_k만 남긴다.
    """
    # 1) 쿼리 임베딩
    q_emb = embed_query(query)

    # 2) Chroma에서 후보 뽑기
    # ⚠ chromadb 0.5.x에서는 include에 "ids" 넣으면 에러 나므로 빼야 한다.
    res = _collection.query(
        query_embeddings=[q_emb],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],  # ← 여기!
    )

    # 3) 응답 파싱
    docs = res["documents"][0] if res["documents"] else []
    # ids는 include에 안 넣어도 기본으로 온다
    ids = res["ids"][0] if "ids" in res and res["ids"] else []
    dists = np.array(res.get("distances", [[0] * len(docs)])[0], dtype=float)

    # 4) 결과가 없을 때 방어
    if len(docs) == 0:
        return [], [], []

    # 5) 벡터 점수 (distance → similarity)
    # cosine distance 기준: 1 - d
    vec_scores = 1.0 - dists
    vec_scores = normalize_scores(vec_scores)

    # 6) BM25 점수
    bm25_scores = bm25_on_candidates(query, docs)
    bm25_scores = normalize_scores(bm25_scores)

    # 7) 가중 합산
    hybrid_scores = W_VEC * vec_scores + W_BM25 * bm25_scores

    # 8) 상위 top_k만 정렬해서 반환
    order = np.argsort(-hybrid_scores)[:top_k]

    top_docs = [docs[i] for i in order]
    top_ids = [ids[i] for i in order] if ids else []
    top_scores = hybrid_scores[order].tolist()

    return top_docs, top_ids, top_scores
