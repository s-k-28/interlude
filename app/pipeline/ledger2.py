"""Dual-arm token accounting for Interlude's Paritok integration.

Interlude generates audio description for university lecture videos. It applies
Paritok token compression on two *independent* paths:

    arm "prefix"  - a ~180-word style guide that is prepended to every model
                    prompt. It is compressed once per call, before the request
                    leaves the pipeline.

    arm "client"  - Paritok's SDK wrapper installed on the model client itself.
                    It compresses conversation history and tool output as they
                    flow through the client, with no involvement from the
                    prompt-building code.

These are two separate optimizations with separate failure modes. Blending them
into one savings number would make it impossible to tell which one actually
worked: a strong prefix result could hide a client wrapper that is doing
nothing at all (or inflating payloads). Two measured numbers are also more
auditable - a reader can check each arm's arithmetic on its own.

This module is ADDITIVE. ``app.pipeline.tokens.TokenLedger`` remains the
single-arm ledger and is unchanged. :meth:`DualLedger.to_legacy_summary`
emits exactly the six keys that ``TokenLedger.summary()`` emits, computed
across both arms, so existing callers and the provenance manifest keep
working without modification.

Desk-checked against the worked example in :func:`_self_check`
(prefix 1000->250, client 400->300); untested at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

Arm = Literal["prefix", "client"]

ARMS: tuple[Arm, ...] = get_args(Arm)


@dataclass(frozen=True, slots=True)
class ArmRecord:
    """A single measured model call, attributed to one compression arm."""

    arm: Arm
    uncompressed_tokens: int
    compressed_tokens: int
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        for name in (
            "uncompressed_tokens",
            "compressed_tokens",
            "completion_tokens",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

    @property
    def saved(self) -> int:
        """Tokens removed by compression. Never negative.

        If compression inflated the payload we report zero saved rather than a
        negative number: the cost is real but it is not a "saving", and a
        negative entry would silently offset a genuine saving in another call.
        """
        return max(0, self.uncompressed_tokens - self.compressed_tokens)

    @property
    def ratio(self) -> float:
        """Fraction of the uncompressed prompt removed. 0.0 when there is nothing to compress."""
        if self.uncompressed_tokens == 0:
            return 0.0
        return self.saved / self.uncompressed_tokens


@dataclass(slots=True)
class DualLedger:
    """Records model calls under two independent Paritok arms and reports each separately."""

    records: list[ArmRecord] = field(default_factory=list)

    def record(
        self,
        arm: Arm,
        *,
        uncompressed_tokens: int,
        compressed_tokens: int,
        completion_tokens: int = 0,
    ) -> ArmRecord:
        if arm not in ARMS:
            valid = ", ".join(repr(a) for a in ARMS)
            raise ValueError(f"unknown arm {arm!r}; valid arms are: {valid}")
        entry = ArmRecord(
            arm=arm,
            uncompressed_tokens=uncompressed_tokens,
            compressed_tokens=compressed_tokens,
            completion_tokens=completion_tokens,
        )
        self.records.append(entry)
        return entry

    def records_for(self, arm: Arm) -> list[ArmRecord]:
        return [r for r in self.records if r.arm == arm]

    def arm_totals(self, arm: Arm) -> dict[str, int | float]:
        rows = self.records_for(arm)
        uncompressed = sum(r.uncompressed_tokens for r in rows)
        compressed = sum(r.compressed_tokens for r in rows)
        completion = sum(r.completion_tokens for r in rows)
        saved = sum(r.saved for r in rows)
        ratio = round(saved / uncompressed, 4) if uncompressed else 0.0
        return {
            "calls": len(rows),
            "uncompressed_tokens": uncompressed,
            "compressed_tokens": compressed,
            "completion_tokens": completion,
            "tokens_saved": saved,
            "reduction_ratio": ratio,
        }

    @property
    def total_saved(self) -> int:
        return sum(r.saved for r in self.records)

    @property
    def total_uncompressed(self) -> int:
        return sum(r.uncompressed_tokens for r in self.records)

    @property
    def total_compressed(self) -> int:
        return sum(r.compressed_tokens for r in self.records)

    @property
    def total_completion(self) -> int:
        return sum(r.completion_tokens for r in self.records)

    @property
    def combined_ratio(self) -> float:
        if self.total_uncompressed == 0:
            return 0.0
        return self.total_saved / self.total_uncompressed

    def summary(self) -> dict[str, dict[str, int | float]]:
        return {
            "prefix": self.arm_totals("prefix"),
            "client": self.arm_totals("client"),
            "combined": {
                "calls": len(self.records),
                "uncompressed_tokens": self.total_uncompressed,
                "compressed_tokens": self.total_compressed,
                "completion_tokens": self.total_completion,
                "tokens_saved": self.total_saved,
                "reduction_ratio": round(self.combined_ratio, 4),
            },
        }

    def to_legacy_summary(self) -> dict[str, int | float]:
        """Both arms flattened into the six keys emitted by ``TokenLedger.summary()``."""
        return {
            "calls": len(self.records),
            "prompt_tokens_uncompressed": self.total_uncompressed,
            "prompt_tokens_compressed": self.total_compressed,
            "completion_tokens": self.total_completion,
            "tokens_saved": self.total_saved,
            "reduction_ratio": round(self.combined_ratio, 4),
        }

    def render_table(self) -> str:
        """Fixed-width table for terminal / demo-video display.

        An arm with no calls shows "not measured" in the reduction cell: we
        never print a percentage for a measurement that did not happen.
        """
        widths = (8, 7, 12, 12, 12, 13)
        header = (
            f"{'Arm':<{widths[0]}}"
            f"{'Calls':>{widths[1]}}"
            f"{'Before':>{widths[2]}}"
            f"{'After':>{widths[3]}}"
            f"{'Saved':>{widths[4]}}"
            f"{'Reduction':>{widths[5]}}"
        )
        rule = "-" * sum(widths)

        def row(
            label: str,
            calls: int,
            before: int,
            after: int,
            saved: int,
            ratio: float,
        ) -> str:
            reduction = f"{ratio * 100:.1f}%" if calls else "not measured"
            return (
                f"{label:<{widths[0]}}"
                f"{calls:>{widths[1]},}"
                f"{before:>{widths[2]},}"
                f"{after:>{widths[3]},}"
                f"{saved:>{widths[4]},}"
                f"{reduction:>{widths[5]}}"
            )

        lines = [header, rule]
        for arm in ARMS:
            totals = self.arm_totals(arm)
            lines.append(
                row(
                    arm,
                    int(totals["calls"]),
                    int(totals["uncompressed_tokens"]),
                    int(totals["compressed_tokens"]),
                    int(totals["tokens_saved"]),
                    float(totals["reduction_ratio"]),
                )
            )
        lines.append(rule)
        lines.append(
            row(
                "TOTAL",
                len(self.records),
                self.total_uncompressed,
                self.total_compressed,
                self.total_saved,
                self.combined_ratio,
            )
        )
        return "\n".join(lines)


def _self_check() -> None:
    # --- validation ------------------------------------------------------
    empty = DualLedger()
    try:
        empty.record("preflix", uncompressed_tokens=10, compressed_tokens=5)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "prefix" in str(exc) and "client" in str(exc), str(exc)
    else:
        raise AssertionError("bad arm name was accepted")

    for bad in (
        {"uncompressed_tokens": -1, "compressed_tokens": 0},
        {"uncompressed_tokens": 0, "compressed_tokens": -1},
        {"uncompressed_tokens": 0, "compressed_tokens": 0, "completion_tokens": -1},
    ):
        try:
            ArmRecord(arm="prefix", **bad)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError(f"negative count accepted: {bad}")

    # --- inflation clamps to zero ----------------------------------------
    inflated = ArmRecord(arm="client", uncompressed_tokens=50, compressed_tokens=80)
    assert inflated.saved == 0, inflated.saved
    assert inflated.ratio == 0.0, inflated.ratio

    # --- empty arm: zeros, no ZeroDivisionError --------------------------
    zeros = empty.arm_totals("prefix")
    assert zeros == {
        "calls": 0,
        "uncompressed_tokens": 0,
        "compressed_tokens": 0,
        "completion_tokens": 0,
        "tokens_saved": 0,
        "reduction_ratio": 0.0,
    }, zeros
    assert empty.combined_ratio == 0.0

    # --- worked example ---------------------------------------------------
    # prefix: 1000 -> 250  => saved 750, ratio 750/1000 = 0.75
    # client:  400 -> 300  => saved 100, ratio 100/400  = 0.25
    # combined: saved 850 over 1000+400 = 1400 => 0.6071428... -> 0.6071
    ledger = DualLedger()
    ledger.record("prefix", uncompressed_tokens=1000, compressed_tokens=250, completion_tokens=120)
    ledger.record("client", uncompressed_tokens=400, compressed_tokens=300, completion_tokens=80)

    prefix = ledger.arm_totals("prefix")
    client = ledger.arm_totals("client")
    assert prefix["tokens_saved"] == 750, prefix
    assert prefix["reduction_ratio"] == 0.75, prefix
    assert client["tokens_saved"] == 100, client
    assert client["reduction_ratio"] == 0.25, client

    assert ledger.total_saved == 850, ledger.total_saved
    assert ledger.total_uncompressed == 1400, ledger.total_uncompressed
    assert ledger.total_compressed == 550, ledger.total_compressed
    assert round(ledger.combined_ratio, 4) == 0.6071, ledger.combined_ratio

    summary = ledger.summary()
    assert set(summary) == {"prefix", "client", "combined"}, summary
    assert summary["combined"]["reduction_ratio"] == 0.6071, summary["combined"]
    assert summary["combined"]["calls"] == 2, summary["combined"]
    assert summary["combined"]["completion_tokens"] == 200, summary["combined"]

    legacy = ledger.to_legacy_summary()
    assert set(legacy) == {
        "calls",
        "prompt_tokens_uncompressed",
        "prompt_tokens_compressed",
        "completion_tokens",
        "tokens_saved",
        "reduction_ratio",
    }, legacy
    assert legacy["prompt_tokens_uncompressed"] == 1400, legacy
    assert legacy["tokens_saved"] == 850, legacy

    # --- rendering --------------------------------------------------------
    table = ledger.render_table()
    assert "75.0%" in table, table
    assert "25.0%" in table, table
    assert "60.7%" in table, table
    assert "1,000" in table, table
    assert "not measured" not in table, table

    one_arm = DualLedger()
    one_arm.record("prefix", uncompressed_tokens=1000, compressed_tokens=259)
    partial = one_arm.render_table()
    assert "not measured" in partial, partial
    assert "74.1%" in partial, partial  # 741/1000

    print("ledger2 self-check passed")


if __name__ == "__main__":
    _self_check()
