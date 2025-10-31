from typing import List, Tuple
from utils import get_env
from sentence_transformers import CrossEncoder

# ─────────────────────────────
# 1️⃣ .env 에서 경로/모델명 읽기
# 예시: C:/Users/playdata/Desktop/multilingual-e5-large
# ─────────────────────────────
RERANKER_MODEL = get_env("RERANKER_MODEL", "intfloat/multilingual-e5-large")

# ─────────────────────────────
# 2️⃣ 모델 로드
# ─────────────────────────────
try:
    reranker = CrossEncoder(RERANKER_MODEL)
except Exception as e:
    raise RuntimeError(
        f"Reranker 모델을 로드하지 못했습니다. "
        f"RERANKER_MODEL={RERANKER_MODEL} 값을 확인하세요."
    ) from e


# ─────────────────────────────
# 3️⃣ 리랭크 함수 정의
# ─────────────────────────────
def rerank(query: str, docs: List[str], top_k: int = 5) -> Tuple[List[str], List[float]]:
    """
    multilingual-e5-large 기반 CrossEncoder로 재정렬 수행.
    실패 시 에러를 바로 던짐.
    """
    if not docs:
        return [], []

    pairs = [[query, d] for d in docs]
    scores = reranker.predict(pairs)  # list[float]

    # 점수 내림차순 정렬
    order = sorted(range(len(docs)), key=lambda i: -scores[i])[:top_k]

    ranked_docs = [docs[i] for i in order]
    ranked_scores = [float(scores[i]) for i in order]

    return ranked_docs, ranked_scores
