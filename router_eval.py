"""Dataset + kerangka evaluasi akurasi semantic router.

Tanpa dataset ini, menyetel ``ROUTER_MAX_DISTANCE`` atau menambah contoh rute
hanya berdasarkan tebakan. Jalankan sebagai skrip::

    python router_eval.py

atau sebagai test::

    pytest test_router_eval.py

Dua dataset dipisah supaya hasilnya jujur:

* ``DEV_CASES`` — dipakai saat kalibrasi ambang batas / menambah contoh rute.
* ``HELDOUT_CASES`` — tidak pernah dipakai saat menyetel, jadi akurasinya
  membuktikan router menggeneralisasi, bukan menghafal.

Rute ``"out"`` berarti pertanyaan di luar ketiga kategori dan seharusnya
jatuh ke rute fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import router

DEV_CASES: tuple[tuple[str, str], ...] = (
    # ---------------- coding / SQL ----------------
    ("How do I write a SQL JOIN between orders and customers?", "coding"),
    ("Select the top 10 revenue rows grouped by month in Postgres", "coding"),
    ("fix this python syntax error please", "coding"),
    ("write a function that reverses a linked list", "coding"),
    ("why does my javascript code throw undefined is not a function", "coding"),
    ("how do I create an index to speed up a slow query", "coding"),
    ("debug this stack trace from my Django app", "coding"),
    ("write a regex to validate an email address", "coding"),
    ("how can I read a CSV file into a pandas dataframe", "coding"),
    ("optimize this nested loop that runs too slowly", "coding"),
    # ---------------- AI / ML ----------------
    ("what is attention in a transformer model", "ml"),
    ("explain vector databases and semantic search", "ml"),
    ("how does backpropagation update weights", "ml"),
    ("what is the difference between precision and recall", "ml"),
    ("how do I fine-tune a language model on my own dataset", "ml"),
    ("explain gradient descent and learning rate", "ml"),
    ("what is retrieval augmented generation", "ml"),
    ("how does tokenization work in an LLM", "ml"),
    ("what is overfitting and how do I prevent it", "ml"),
    ("explain embedding similarity and cosine distance", "ml"),
    # ---------------- general technology ----------------
    ("how does the internet work for a beginner", "general"),
    ("what is cloud computing", "general"),
    ("explain what an operating system actually does", "general"),
    ("how do DNS and IP addresses work together", "general"),
    ("what happens when I type a URL into a browser", "general"),
    ("what is a virtual machine and why use one", "general"),
    ("explain the difference between RAM and storage", "general"),
    ("how does HTTPS keep a website connection secure", "general"),
    ("what is the history of the CPU", "general"),
    ("how do containers differ from virtual machines", "general"),
    # ---------------- di luar kategori ----------------
    ("siapa presiden pertama indonesia", "out"),
    ("resep nasi goreng paling enak", "out"),
    ("how do I fix a leaking kitchen sink faucet", "out"),
    ("what is the capital of France", "out"),
    ("tips trading saham yang menguntungkan", "out"),
    ("how to train a puppy to stop barking at night", "out"),
    ("harga beras hari ini di pasar", "out"),
    ("best hiking trails in new zealand", "out"),
)

# Dataset ini ditulis tanpa melihat hasil router, untuk menguji generalisasi.
HELDOUT_CASES: tuple[tuple[str, str], ...] = (
    ("my code throws a null pointer exception on line 42", "coding"),
    ("how should I structure a REST endpoint for user signup", "coding"),
    ("convert this nested dictionary loop into something readable", "coding"),
    ("query postgres for daily active users over the last week", "coding"),
    ("write a merge sort implementation in go", "coding"),
    ("what is the context window of a large language model", "ml"),
    ("how do I measure whether my model is actually good", "ml"),
    ("explain the difference between classification and regression", "ml"),
    ("why does my neural network loss not go down", "ml"),
    ("what are latent representations in an embedding space", "ml"),
    ("what is the difference between a router and a switch", "general"),
    ("how is a solid state drive different from a hard disk", "general"),
    ("explain what an API is to someone non technical", "general"),
    ("how do push notifications reach my phone", "general"),
    ("what does a compiler actually produce", "general"),
    ("cuaca besok di bandung", "out"),
    ("cara memperbaiki ban bocor di jalan", "out"),
    ("how do I get rid of ants in my kitchen", "out"),
    ("who won the world cup in 2010", "out"),
    ("tips memilih restoran untuk makan malam", "out"),
)

# Seluruh kasus, untuk laporan gabungan.
EVAL_CASES: tuple[tuple[str, str], ...] = DEV_CASES + HELDOUT_CASES

MIN_ACCURACY = float(os.getenv("ROUTER_MIN_ACCURACY", "0.85"))


@dataclass(frozen=True)
class EvalReport:
    """Ringkasan hasil evaluasi router."""

    in_scope_correct: int
    in_scope_total: int
    fallback_correct: int
    fallback_total: int
    failures: tuple[str, ...]

    @property
    def accuracy(self) -> float:
        if not self.in_scope_total:
            return 1.0
        return self.in_scope_correct / self.in_scope_total

    @property
    def fallback_accuracy(self) -> float:
        if not self.fallback_total:
            return 1.0
        return self.fallback_correct / self.fallback_total

    @property
    def passed(self) -> bool:
        return (
            self.accuracy >= MIN_ACCURACY
            and self.fallback_accuracy >= MIN_ACCURACY
        )

    def summary(self) -> str:
        return (
            f"in-scope {self.in_scope_correct}/{self.in_scope_total} "
            f"({self.accuracy:.1%}), out-of-scope caught "
            f"{self.fallback_correct}/{self.fallback_total} "
            f"({self.fallback_accuracy:.1%})"
        )


def run_evaluation(
    cases: tuple[tuple[str, str], ...],
    *,
    max_distance: float | None = None,
    rescue_distance: float | None = None,
) -> EvalReport:
    """Jalankan dataset dan kumpulkan akurasi rute + penangkapan out-of-scope."""
    in_scope_correct = 0
    in_scope_total = 0
    fallback_correct = 0
    fallback_total = 0
    failures: list[str] = []

    for question, expected in cases:
        kwargs: dict[str, float] = {}

        if max_distance is not None:
            kwargs["max_distance"] = max_distance

        if rescue_distance is not None:
            kwargs["rescue_distance"] = rescue_distance

        decision = router.semantic_route(question, **kwargs)

        if expected == "out":
            fallback_total += 1
            ok = decision.route == router.FALLBACK_ROUTE
            if ok:
                fallback_correct += 1
        else:
            in_scope_total += 1
            ok = decision.route == expected
            if ok:
                in_scope_correct += 1

        if not ok:
            failures.append(
                f"expected={expected:8s} got={decision.route:8s} "
                f"dist={decision.distance:.4f} "
                f"neighbors={decision.neighbors} "
                f"rescued={decision.rescued} | {question}"
            )

    return EvalReport(
        in_scope_correct=in_scope_correct,
        in_scope_total=in_scope_total,
        fallback_correct=fallback_correct,
        fallback_total=fallback_total,
        failures=tuple(failures),
    )


def main() -> int:
    print("=== Router evaluation ===")
    print(f"Max distance : {router.MAX_ROUTE_DISTANCE}")
    print(f"Rescue limit : {router.RESCUE_DISTANCE} (unanimous neighbours only)")
    print(f"Samples      : {len(router.ROUTING_SAMPLES)} routing examples\n")

    reports = {
        "DEV": run_evaluation(DEV_CASES),
        "HELDOUT": run_evaluation(HELDOUT_CASES),
    }

    for label, report in reports.items():
        print(f"{label:8s}: {report.summary()}")

        for failure in report.failures:
            print(f"    - {failure}")

        print()

    passed = all(report.passed for report in reports.values())
    print(f"Threshold {MIN_ACCURACY:.0%}: {'PASS' if passed else 'FAIL'}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
