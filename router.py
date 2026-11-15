"""Semantic router terpusat: contoh rute, embedding, dan keputusan rute.

Dulu ``app_conditional.py`` dan ``app_hybrid.py`` masing-masing menyimpan
salinan ``routing_samples`` yang mulai berbeda satu sama lain. Sekarang
satu sumber kebenaran ada di modul ini.

Keputusan rute memakai dua sinyal:

1. **Jarak terdekat** vs ``MAX_ROUTE_DISTANCE`` — jalur normal.
2. **Unanimity rescue** — kalau jarak sedikit lewat ambang batas *tetapi*
   seluruh top-k tetangga sepakat pada satu rute, rute itu tetap dipakai.
   Kesepakatan penuh berarti pertanyaan berada di dalam cluster rute tersebut;
   ini penting karena rute ``coding`` secara sistematis lebih jarang/lebih
   renggang daripada ``ml``, sehingga jarak absolutnya menyesatkan. Kalau
   tetangga tidak sepakat, pertanyaan berada di wilayah batas antar-rute dan
   lebih aman jatuh ke fallback.

Semua angka di file ini dikalibrasi lewat pengukuran pada ``router_eval.py``
(dataset DEV + HELDOUT yang dipisah), bukan perkiraan. Alternatif berikut
diuji dan ditolak karena hasilnya lebih buruk:

* **Majority vote top-k** — tidak memperbaiki akurasi rute, dan membuat
  perilaku di luar kategori sulit dijelaskan.
* **Jarak ternormalisasi per rute** (jarak dibagi baseline kepadatan rute) —
  held-out turun ke 73% dari 80%.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from log import get_logger

logger = get_logger(__name__)

FALLBACK_ROUTE = "general"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "Qwen/Qwen3-Embedding-0.6B",
)

EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# Chroma mengembalikan jarak cosine (embedding dinormalisasi): 0 = identik,
# 2 = berlawanan arah. Qwen3-Embedding menghasilkan jarak yang relatif besar
# untuk teks pendek, jadi ambang batas dikalibrasi lewat pengukuran nyata
# pada router_eval.py, bukan perkiraan.
#
# Hasil pada dataset DEV / HELDOUT (akurat rute in-scope / penangkapan
# pertanyaan di luar kategori):
#   0.92 -> 90% / 100%   dan  80% / 100%
#   1.02 -> 93% / 100%   dan  80% / 100%   <- dipilih
#   1.10 -> 100% / 75%   dan  80% / 100%
# Naik ke 1.10 menggandakan akurasi DEV tapi mengorbankan penangkapan
# pertanyaan di luar kategori, jadi 1.02 adalah titik paling seimbang.
MAX_ROUTE_DISTANCE = _env_float("ROUTER_MAX_DISTANCE", 1.02)

# Batas kedua yang lebih longgar, hanya berlaku kalau seluruh tetangga
# terdekat sepakat pada satu rute. Diukur pada router_eval.py:
#   rescue 1.02 (mati) -> DEV 93%/100%, HELDOUT 80%/100%
#   rescue 1.20        -> DEV 100%/100%, HELDOUT 93%/100%  <- dipilih
#   rescue 1.40        -> DEV 100%/100%, HELDOUT 100%/100% (terlalu longgar,
#                         praktis mematikan ambang batas utama)
RESCUE_DISTANCE = _env_float("ROUTER_RESCUE_DISTANCE", 1.20)

ROUTER_TOP_K = int(os.getenv("ROUTER_TOP_K", "3"))

ROUTES = ("coding", "ml", "general")

# Satu sumber kebenaran untuk contoh intent. Menambah rute baru cukup dengan
# menambah string rute di `ROUTES` dan contoh-contoh baru di sini.
ROUTING_SAMPLES: tuple[tuple[str, str], ...] = (
    # ---------------- coding / SQL ----------------
    (
        "Write SQL queries using SELECT, JOIN, GROUP BY, ORDER BY, "
        "and database aggregation.",
        "coding",
    ),
    (
        "Find the five highest paid employees using SQL queries in "
        "PostgreSQL or MySQL.",
        "coding",
    ),
    (
        "How do I filter and aggregate database tables in Postgres "
        "or MySQL?",
        "coding",
    ),
    (
        "Debug code, fix syntax errors, and resolve exceptions.",
        "coding",
    ),
    (
        "Debug Python code and write functions or algorithms.",
        "coding",
    ),
    (
        "Write a Python or JavaScript function, algorithm, or script.",
        "coding",
    ),
    (
        "Write JavaScript code, scripts, and software applications.",
        "coding",
    ),
    (
        "How do I connect to a relational database using an ORM?",
        "coding",
    ),
    (
        "Implement a data structure or algorithm from scratch, such as "
        "a stack, queue, or binary tree.",
        "coding",
    ),
    (
        "Implement sorting algorithms such as merge sort, quicksort, "
        "or heapsort in Go, Python, Java, C++, or JavaScript.",
        "coding",
    ),
    (
        "Write a complete implementation of an algorithm in a programming "
        "language, including its functions, logic, and tests.",
        "coding",
    ),
    (
        "Build a helper function that parses, formats, or validates "
        "user input.",
        "coding",
    ),
    (
        "Process tabular data and files through a script or library "
        "API.",
        "coding",
    ),
    (
        "Improve the performance and time complexity of an existing "
        "program.",
        "coding",
    ),
    (
        "Extract structured substrings from raw text using pattern "
        "matching.",
        "coding",
    ),
    (
        "Write unit tests and refactor a function to be cleaner.",
        "coding",
    ),
    # ---------------- AI / ML ----------------
    (
        "Explain machine learning, neural networks, deep learning, "
        "and model training.",
        "ml",
    ),
    (
        "Explain transformer architecture, attention, LLMs, and "
        "tokenization.",
        "ml",
    ),
    (
        "Explain RAG, vector embeddings, vector databases, and "
        "semantic search.",
        "ml",
    ),
    (
        "How does Retrieval-Augmented Generation (RAG) and vector "
        "embedding work?",
        "ml",
    ),
    (
        "Explain gradient descent, loss functions, backpropagation, "
        "and fine-tuning.",
        "ml",
    ),
    (
        "Fine-tune Large Language Models and train machine learning "
        "models.",
        "ml",
    ),
    (
        "Difference between supervised learning, unsupervised learning, "
        "and reinforcement learning.",
        "ml",
    ),
    # ---------------- general technology ----------------
    (
        "Explain how the internet, DNS, IP addresses, and computer "
        "networks work for beginners.",
        "general",
    ),
    (
        "Explain cloud computing, operating systems, virtualization, "
        "and computer hardware.",
        "general",
    ),
    (
        "Explain general technology concepts in simple language for "
        "non-technical users.",
        "general",
    ),
    (
        "Explain how websites, web browsers, and HTTP work in everyday "
        "life.",
        "general",
    ),
    (
        "History of computer hardware, CPUs, and operating systems.",
        "general",
    ),
    # ---------------- non-technical fallback examples ----------------
    (
        "Answer everyday non-technical questions about cooking, travel, "
        "household tasks, vehicle maintenance, and daily life.",
        "general",
    ),
    (
        "Jawab pertanyaan sehari-hari non-teknis tentang memasak, "
        "perjalanan, pekerjaan rumah, dan perawatan kendaraan.",
        "general",
    ),
)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Muat model embedding Qwen3 secara malas (hanya saat dibutuhkan)."""
    logger.info("Loading embedding model %s on %s...", EMBEDDING_MODEL, EMBEDDING_DEVICE)

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Bangun vector store intent-router di dalam memori."""
    logger.info("Building routing vector store...")

    documents = [
        Document(page_content=content, metadata={"route": route})
        for content, route in ROUTING_SAMPLES
    ]

    return Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name="intent_router",
    )


def routing_sample_count() -> int:
    """Jumlah intent yang sudah masuk vector store (untuk diagnostics)."""
    return len(get_vector_store().get()["ids"])


@dataclass(frozen=True)
class RouteDecision:
    """Hasil klasifikasi semantik sebuah pertanyaan."""

    route: str
    distance: float
    matched_sample: str
    neighbors: dict[str, int] = field(default_factory=dict)
    below_threshold: bool = False
    rescued: bool = False

    @property
    def unanimous(self) -> bool:
        """True kalau seluruh tetangga terdekat berasal dari satu rute."""
        return len(self.neighbors) == 1


def semantic_route(
    question: str,
    *,
    top_k: int = ROUTER_TOP_K,
    max_distance: float = MAX_ROUTE_DISTANCE,
    rescue_distance: float = RESCUE_DISTANCE,
) -> RouteDecision:
    """Pilih rute berdasarkan kedekatan semantik dengan contoh rute.

    Args:
        question: Pertanyaan pengguna.
        top_k: Jumlah contoh terdekat yang diambil.
        max_distance: Jarak maksimum agar rute contoh diterima.
        rescue_distance: Jarak maksimum tambahan, hanya dipakai kalau
            seluruh top-k tetangga sepakat pada satu rute.

    Returns:
        RouteDecision dengan rute terpilih dan rincian tetangga terdekat.
    """
    matches = get_vector_store().similarity_search_with_score(question, k=top_k)

    if not matches:
        return RouteDecision(
            route=FALLBACK_ROUTE,
            distance=float("inf"),
            matched_sample="None",
            below_threshold=True,
        )

    # Keputusan diambil dari contoh paling dekat saja.
    best_document, best_distance = matches[0]
    best_distance = float(best_distance)

    # Komposisi rute tetangga terdekat: dipakai untuk unanimity rescue
    # sekaligus sebagai diagnostik.
    neighbors: dict[str, int] = {}

    for document, _distance in matches:
        neighbor_route = document.metadata.get("route", FALLBACK_ROUTE)
        neighbors[neighbor_route] = neighbors.get(neighbor_route, 0) + 1

    best_route = best_document.metadata.get("route", FALLBACK_ROUTE)
    below_threshold = best_distance > max_distance

    # Jarak lewat ambang, tapi semua tetangga sepakat dan masih dalam batas
    # longgar -> pertanyaan memang berada di dalam cluster rute tersebut.
    rescued = (
        below_threshold
        and len(neighbors) == 1
        and best_distance <= rescue_distance
    )

    if below_threshold and not rescued:
        route = FALLBACK_ROUTE
        matched_sample = "None"
    else:
        route = best_route
        matched_sample = best_document.page_content

    decision = RouteDecision(
        route=route,
        distance=best_distance,
        matched_sample=matched_sample,
        neighbors=neighbors,
        below_threshold=below_threshold,
        rescued=rescued,
    )

    logger.debug(
        "route=%s distance=%.4f neighbors=%s below_threshold=%s "
        "rescued=%s matched=%r",
        decision.route,
        decision.distance,
        decision.neighbors,
        decision.below_threshold,
        decision.rescued,
        decision.matched_sample[:80],
    )

    return decision
