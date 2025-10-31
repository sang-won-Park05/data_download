import gradio as gr
from typing import List
import os

from utils import get_env
import math
from retrieval import hybrid_search
from reranker import rerank
from rag import build_prompt, post_edit_instruction, SYSTEM_TONE
from openai import OpenAI

# ─────────────────────────────
# 1. 환경변수 로드
# ─────────────────────────────
TOP_K = int(get_env("TOP_K", "5"))              # 리랭크 후 최종 개수
CANDIDATE_K = int(get_env("CANDIDATE_K", "50")) # 리랭크 전 벡터 후보
GPT_MODEL = get_env("GPT_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = get_env("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ─────────────────────────────
# 2. 답변 생성 함수
# ─────────────────────────────
def generate_answer(message, history, top_k=TOP_K, candidate_k=CANDIDATE_K):
    query = message.strip()

    # 1) 하이브리드 검색
    cand_docs, cand_ids, hybrid_scores = hybrid_search(
        query,
        candidate_k=candidate_k,
        top_k=top_k
    )

    # 2) 리랭크
    reranked_docs, rerank_scores = rerank(query, cand_docs, top_k=top_k)

    # 3) 프롬프트 구성
    user_prompt = build_prompt(query, reranked_docs)

    # 4) 1차 초안
    draft = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_TONE},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    ).choices[0].message.content

    # 5) 후처리(말투 보정)
    polished = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_TONE},
            {"role": "user", "content": post_edit_instruction()},
            {"role": "user", "content": draft},
        ],
        temperature=0.2,
    ).choices[0].message.content

    debug = {
        "candidate_ids": cand_ids[:top_k],
        "hybrid_scores(top)": hybrid_scores[:top_k],
        "rerank_scores": rerank_scores,
        "contexts": reranked_docs,
    }

    return polished, debug

# ─────────────────────────────
# 3. Gradio UI 구성
# ─────────────────────────────
with gr.Blocks(title="Hybrid RAG Chat (Chroma + BM25 + Rerank)") as demo:
    gr.Markdown("## 🔎 Hybrid RAG Chat\nChroma + 하이브리드 검색 + 리랭커 + GPT 후처리")

    with gr.Row():
        # ─── 챗봇 영역 ───
        with gr.Column(scale=3):
            chat = gr.ChatInterface(
                fn=lambda msg, hist, tk, ck: generate_answer(msg, hist, tk, ck)[0],
                additional_inputs=[
                    gr.Slider(3, 10, value=TOP_K, step=1, label="최종 Top-K"),
                    gr.Slider(10, 100, value=CANDIDATE_K, step=5, label="벡터 후보 풀(Candidate-K)"),
                ],
                type="messages",  # ← 이거 추가해서 경고 없앰
                chatbot=gr.Chatbot(
                    height=480,
                    show_copy_button=True,
                    type="messages",  # ← Chatbot도 맞춰줌
                ),
            )

        # ─── 디버그 / 검색 확인 영역 ───
        with gr.Column(scale=2):
            gr.Markdown("### 🧪 Debug & Retrieval View")
            query_in = gr.Textbox(label="테스트 쿼리", placeholder="예) 머리가 아픈데 어떤 병원 가면 좋을까요?")
            tk_in = gr.Slider(3, 10, value=TOP_K, step=1, label="최종 Top-K")
            ck_in = gr.Slider(10, 100, value=CANDIDATE_K, step=5, label="벡터 후보 풀(Candidate-K)")
            btn = gr.Button("검색 & 리랭크 보기")

            # ⚠ 너 그라디오 버전에서는 height / visible_rows 안 됨 → row_count로
            ctx_out = gr.Dataframe(
                headers=["#", "Score", "Snippet"],
                value=[],              # 초기값 빈 표
                row_count=10,          # 표시할 줄 수
                col_count=3,
                interactive=False,
            )
            metrics_json = gr.JSON(label="Reranker Metrics (vs Hybrid): @5")
            raw_json = gr.JSON(label="Raw debug")

            def inspect(query, tk, ck):
                # 하이브리드로 먼저 '후보 풀(ck)'까지 크게 뽑고 (점수 포함)
                docs, ids, hybrid_scores = hybrid_search(
                    query,
                    candidate_k=int(ck),
                    top_k=int(ck)
                )
                # 그 위에 리랭크하여 최종 tk만큼 선택
                rer_docs, r_scores = rerank(query, docs, top_k=int(tk))

                # 표에 넣을 데이터 만들기
                rows = []
                for i, (d, s) in enumerate(zip(rer_docs, r_scores), start=1):
                    rows.append([
                        i,
                        round(float(s), 4),
                        (d[:250] + "...") if len(d) > 250 else d,
                    ])

                # ── 리랭커 평가 지표(@5): 하이브리드 상위와의 합치도 기준 ──
                def compute_metrics_at5(candidates, cand_scores, reranked, k=5):
                    k = min(k, len(candidates))
                    if k == 0:
                        return {
                            "MRR@5": 0.0,
                            "nDCG@5": 0.0,
                            "Recall@5": 0.0,
                            "Precision@5": 0.0,
                        }

                    # 하이브리드 정렬은 이미 점수 내림차순 → 상위 k를 이상(ideal)으로 사용
                    ideal_docs = candidates[:k]

                    # 점수 맵 (nDCG용, 점수는 0~1 정규화된 값)
                    score_map = {doc: float(s) for doc, s in zip(candidates, cand_scores)}

                    # rerank 상위 k
                    rr_k = reranked[:k]

                    # Precision/Recall@k (ideal@k과의 교집합 기반)
                    ideal_set = set(ideal_docs)
                    rr_set = set(rr_k)
                    inter = len(ideal_set.intersection(rr_set))
                    precision = inter / max(1, len(rr_k))
                    recall = inter / max(1, len(ideal_docs))

                    # MRR@k: 첫 관련문서(ideal@k에 속함)의 역순위
                    mrr = 0.0
                    for rank, d in enumerate(rr_k, start=1):
                        if d in ideal_set:
                            mrr = 1.0 / rank
                            break

                    # nDCG@k: 하이브리드 점수를 graded relevance로 사용
                    def dcg(docs_seq):
                        total = 0.0
                        for i, d in enumerate(docs_seq[:k], start=1):
                            rel = score_map.get(d, 0.0)
                            total += (2 ** rel - 1.0) / math.log2(i + 1)
                        return total

                    idcg = dcg(ideal_docs)
                    ndcg = (dcg(rr_k) / idcg) if idcg > 0 else 0.0

                    return {
                        "MRR@5": round(mrr, 4),
                        "nDCG@5": round(ndcg, 4),
                        "Recall@5": round(recall, 4),
                        "Precision@5": round(precision, 4),
                    }

                metrics = compute_metrics_at5(docs, hybrid_scores, rer_docs, k=5)

                # generate_answer 의 debug도 같이
                _, debug = generate_answer(query, [], int(tk), int(ck))
                return rows, metrics, debug

            btn.click(fn=inspect, inputs=[query_in, tk_in, ck_in], outputs=[ctx_out, metrics_json, raw_json])

# ─────────────────────────────
# 4. 실행
# ─────────────────────────────
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
