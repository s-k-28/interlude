"""Token accounting for the Paritok integration.

Every description call sends the same style guide as a prefix. Across a library
of thousands of scenes that prefix is retransmitted thousands of times, and it
dominates the bill. Paritok compresses it.

To make the saving *measurable* rather than asserted, this module records both
arms of the comparison: what the prompt would have cost uncompressed, and what
it actually cost. Raw counts are kept so a reader can recompute the ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class TokenCounter(Protocol):
    """Anything that can count tokens in a string."""

    def __call__(self, text: str) -> int: ...


def approximate_tokens(text: str) -> int:
    """Fallback counter: ~4 characters per token.

    Used only when tiktoken is unavailable. Clearly labelled as approximate so
    it is never mistaken for a measured figure in reporting.
    """
    return max(0, (len(text) + 3) // 4)


def tiktoken_counter(encoding_name: str = "cl100k_base") -> TokenCounter:
    """Real token counter, backed by tiktoken (a Paritok dependency)."""
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)

    def count(text: str) -> int:
        return len(encoding.encode(text))

    return count


@dataclass(slots=True)
class CallRecord:
    """One model call, measured in both arms."""

    uncompressed_tokens: int
    compressed_tokens: int
    completion_tokens: int = 0

    @property
    def saved(self) -> int:
        return max(0, self.uncompressed_tokens - self.compressed_tokens)


@dataclass(slots=True)
class TokenLedger:
    """Accumulates per-call measurements across a run.

    Deliberately reports totals and counts rather than a single headline
    percentage — the methodology has to be checkable.
    """

    calls: list[CallRecord] = field(default_factory=list)

    def record(
        self,
        *,
        uncompressed_tokens: int,
        compressed_tokens: int,
        completion_tokens: int = 0,
    ) -> CallRecord:
        if uncompressed_tokens < 0 or compressed_tokens < 0:
            raise ValueError("token counts must be non-negative")
        record = CallRecord(
            uncompressed_tokens=uncompressed_tokens,
            compressed_tokens=compressed_tokens,
            completion_tokens=completion_tokens,
        )
        self.calls.append(record)
        return record

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def total_uncompressed(self) -> int:
        return sum(c.uncompressed_tokens for c in self.calls)

    @property
    def total_compressed(self) -> int:
        return sum(c.compressed_tokens for c in self.calls)

    @property
    def total_completion(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_saved(self) -> int:
        return max(0, self.total_uncompressed - self.total_compressed)

    @property
    def reduction_ratio(self) -> float:
        """Fraction of prompt tokens eliminated. Zero if nothing was measured."""
        if self.total_uncompressed <= 0:
            return 0.0
        return self.total_saved / self.total_uncompressed

    def summary(self) -> dict[str, float | int]:
        """A reportable summary. Ratio is rounded; raw counts are not."""
        return {
            "calls": self.call_count,
            "prompt_tokens_uncompressed": self.total_uncompressed,
            "prompt_tokens_compressed": self.total_compressed,
            "completion_tokens": self.total_completion,
            "tokens_saved": self.total_saved,
            "reduction_ratio": round(self.reduction_ratio, 4),
        }
