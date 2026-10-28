"""Conservative UTF-8 byte budgets, avoiding dependence on a model-specific tokenizer."""
import re


def chunks(units, target=400, overlap=60):
    target, overlap = int(target), int(overlap)
    if target < 64 or overlap < 0 or overlap >= target:
        raise ValueError("Chunk size must be >=64 and overlap must be between zero and size.")
    result = []
    for unit in units:
        text = re.sub(r"\s+", " ", unit["text"]).strip()
        # UTF-8 bytes conservatively bound tokens for the intended byte-token models.
        position = 0
        while position < len(text):
            end, used = position, 0
            while end < len(text) and used + len(text[end].encode("utf-8")) <= target:
                used += len(text[end].encode("utf-8"))
                end += 1
            if end < len(text):
                boundary = text.rfind(" ", position, end)
                if boundary > position + (end - position) // 2:
                    end = boundary
            passage = text[position:end].strip()
            if passage:
                result.append({**unit, "text": passage})
            if end == len(text):
                break
            next_position, used = end, 0
            while next_position > position and used + len(text[next_position - 1].encode("utf-8")) <= overlap:
                next_position -= 1
                used += len(text[next_position].encode("utf-8"))
            position = max(position + 1, next_position)
    return result
