"""Shared structured-answer schema and citation validation for RAG and model checks."""


def answer_schema(ids):
    return {"type": "object", "additionalProperties": False, "required": ["status", "claims"], "properties": {
        "status": {"type": "string", "enum": ["answered", "insufficient_evidence"]},
        "claims": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["text", "citations"], "properties": {"text": {"type": "string"},
            "citations": {"type": "array", "minItems": 1, "items": {"type": "integer", "enum": ids}}}}}}}


def validate_answer(result, ids):
    if not isinstance(result, dict) or set(result) != {"status", "claims"}:
        raise ValueError("Invalid answer object.")
    if result["status"] not in {"answered", "insufficient_evidence"} or not isinstance(result["claims"], list):
        raise ValueError("Invalid status or claims.")
    if result["status"] == "insufficient_evidence":
        if result["claims"]:
            raise ValueError("Abstention cannot include claims.")
        return
    if not result["claims"]:
        raise ValueError("Answered output requires claims.")
    for claim in result["claims"]:
        if not isinstance(claim, dict) or set(claim) != {"text", "citations"}:
            raise ValueError("Invalid claim object.")
        if not isinstance(claim["text"], str) or not claim["text"].strip():
            raise ValueError("Empty claim.")
        if not isinstance(claim["citations"], list) or not claim["citations"]:
            raise ValueError("Missing citations.")
        if any(type(i) is not int or i not in ids for i in claim["citations"]):
            raise ValueError("Citation ID is not supplied evidence.")
