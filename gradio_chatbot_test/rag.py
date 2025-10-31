from typing import List

SYSTEM_TONE = (
    "너는 한국어 의료/일반 지식 어시스턴트야. "
    "친절하고 차분한 의사 말투로, 과장 없이 핵심만 설명해줘. "
    "모호하면 가정 말고 불확실성을 밝혀줘."
)

def build_prompt(query: str, passages: List[str]) -> str:
    context = "\n\n".join([f"- {p}" for p in passages])
    user = f"""[사용자 질문]
{query}

[검색 컨텍스트(요약·인용 가능)]
{context}

[요구사항]
1) 한국어 답변, 2) 지나친 확신 금지, 3) 필요 시 근거를 간단히 요약."""
    return user

def post_edit_instruction() -> str:
    return (
        "다음 초안을 자연스럽고 간결한 문장으로 다듬어줘. "
        "반말 대신 존댓말, 전문 용어는 풀어 설명:"
    )
