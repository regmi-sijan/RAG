"""Curated local model choices. Sizes are snapshots, memory ranges are estimates.

Published parameter/disk sizes checked against linked Ollama pages, 2026-10-28.
Memory ranges are planning estimates for one loaded model at about 8K context,
not vendor requirements or measurements. They exclude Python/browser/OS/index.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModelChoice:
    tag: str
    role: str
    parameters: str
    download_bytes: int
    memory_gb: tuple[float, float]
    quantization: str
    note: str
    embedding_context: int | None = None
    disable_thinking: bool = False

    @property
    def url(self):
        return f"https://ollama.com/library/{self.tag}"


CATALOG = (
    ModelChoice("qwen2.5:0.5b", "generation", "494 million", 398_000_000, (1, 2.5), "Q4_K_M",
                "Smallest generation option here; test answer and citation quality."),
    ModelChoice("qwen3:0.6b", "generation", "752 million", 523_000_000, (1.2, 3), "Q4_K_M",
                "Small Qwen 3 candidate; thinking is disabled to reserve output for the cited answer.", disable_thinking=True),
    ModelChoice("gemma3:1b", "generation", "1.00 billion", 806_000_000, (1.5, 3), "Q4_K_M",
                "Text-only Gemma alternative. Requires Ollama 0.6 or later; check the model license."),
    ModelChoice("qwen2.5:1.5b", "generation", "1.54 billion", 986_000_000, (2, 4), "Q4_K_M",
                "A larger lightweight candidate; quality is not benchmarked in this app."),
    ModelChoice("llama3.2:1b", "generation", "1.24 billion", 1_300_000_000, (2.5, 4), "Q8_0",
                "This tag uses 8-bit weights; fewer parameters need not mean a smaller file."),
    ModelChoice("qwen3:1.7b", "generation", "2.03 billion", 1_400_000_000, (2.5, 4.5), "Q4_K_M",
                "Larger Qwen 3 candidate with thinking disabled. Published parameter count differs from its tag name.", disable_thinking=True),
    ModelChoice("qwen2.5:3b", "generation", "3.09 billion", 1_900_000_000, (3, 5.5), "Q4_K_M",
                "Larger answer-model candidate for 16 GB Macs. Review its research license before use."),
    ModelChoice("llama3.2:3b", "generation", "3.21 billion", 2_000_000_000, (3, 5.5), "Q4_K_M",
                "Original application default; uses more memory than the smallest choices."),
    ModelChoice("nomic-embed-text:latest", "embedding", "137 million", 274_000_000, (0.3, 1), "F16",
                "Original retrieval model. Task-specific prefixes are applied automatically.", embedding_context=8192),
    ModelChoice("all-minilm:22m", "embedding", "22.6 million", 46_000_000, (0.15, 0.6), "F16",
                "Lowest-memory embedding option here. Uses shorter passages; long search questions are shortened with a warning.", embedding_context=256),
)


def canonical_tag(tag):
    return tag if ":" in tag.rsplit("/", 1)[-1] else tag + ":latest"


def choice_for(tag):
    tag = canonical_tag(tag)
    if tag == "all-minilm:latest":
        tag = "all-minilm:22m"
    return next((choice for choice in CATALOG if choice.tag == tag), None)


def embedding_byte_budget(model):
    choice = choice_for(model)
    return choice.embedding_context - 32 if choice and choice.embedding_context else None


def shorten_embedding_query(model, text):
    budget = embedding_byte_budget(model)
    return text.encode("utf-8")[:budget].decode("utf-8", errors="ignore") if budget else text


def format_size(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        return "Unknown"
    return f"{value / 1_000_000_000:.2f} GB" if value >= 1_000_000_000 else f"{value / 1_000_000:.0f} MB"


def memory_label(choice):
    return f"~{choice.memory_gb[0]:g}–{choice.memory_gb[1]:g} GB (estimate)"
