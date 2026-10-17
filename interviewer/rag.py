"""Retrieval-Augmented Generation (RAG) module for Project-Dragon.

Provides:
1. Section-aware document chunking (preserves doc_type, section, interview_id, index).
2. Embeddings service with fastembed (BAAI/bge-small-en-v1.5) and offline deterministic fallback.
3. Cosine similarity calculation.
4. Persistent SQLite storage and retrieval strictly isolated by candidate/interview_id.
5. Context-grounded evidence retrieval for interview questions and follow-ups.
"""
from __future__ import annotations

import hashlib
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DocumentChunk:
    interview_id: str
    doc_type: str                  # "resume" | "job_description"
    section: str                   # "experience" | "skills" | "requirements" | "education" | "summary" | "general"
    chunk_index: int
    text: str
    embedding: Optional[List[float]] = None
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Section-Aware Document Chunking
# ---------------------------------------------------------------------------

_RESUME_SECTION_PATTERNS = [
    (r"(?im)^(?:#+\s*)?(?:work\s+experience|professional\s+experience|experience|employment(?:\s+history)?)\b", "experience"),
    (r"(?im)^(?:#+\s*)?(?:technical\s+skills|skills\s+&\s+proficiencies|skills|technologies|core\s+competencies)\b", "skills"),
    (r"(?im)^(?:#+\s*)?(?:projects|key\s+projects|selected\s+projects)\b", "projects"),
    (r"(?im)^(?:#+\s*)?(?:education|academic\s+background|qualifications)\b", "education"),
    (r"(?im)^(?:#+\s*)?(?:certifications|licenses|credentials)\b", "certifications"),
    (r"(?im)^(?:#+\s*)?(?:summary|professional\s+summary|about\s+me|profile|objective)\b", "summary"),
    (r"(?im)^(?:#+\s*)?(?:achievements|awards|publications|patents)\b", "achievements"),
]

_JD_SECTION_PATTERNS = [
    (r"(?im)^(?:#+\s*)?(?:responsibilities|key\s+responsibilities|what\s+you(?:'ll|\swill)\s+do|duties)\b", "responsibilities"),
    (r"(?im)^(?:#+\s*)?(?:requirements|qualifications|basic\s+qualifications|must\s+haves?|what\s+we(?:'re|\sare)\s+looking\s+for)\b", "requirements"),
    (r"(?im)^(?:#+\s*)?(?:preferred\s+qualifications|nice\s+to\s+haves?|bonus\s+points?)\b", "preferred_qualifications"),
    (r"(?im)^(?:#+\s*)?(?:about\s+the\s+role|role\s+overview|job\s+summary|about\s+us|company\s+overview)\b", "role_overview"),
    (r"(?im)^(?:#+\s*)?(?:benefits|perks|compensation|what\s+we\s+offer)\b", "benefits"),
]


def _detect_section(heading: str, doc_type: str) -> Optional[str]:
    patterns = _RESUME_SECTION_PATTERNS if doc_type == "resume" else _JD_SECTION_PATTERNS
    # Also check the other patterns if no match found
    all_patterns = patterns + (_JD_SECTION_PATTERNS if doc_type == "resume" else _RESUME_SECTION_PATTERNS)
    for pattern, name in all_patterns:
        if re.search(pattern, heading.strip()):
            return name
    return None


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into raw paragraphs, handling newlines and bullet points."""
    paras = re.split(r"\n\s*\n", text)
    result = []
    for p in paras:
        p_clean = p.strip()
        if p_clean:
            result.append(p_clean)
    return result


def chunk_document(
    text: str,
    doc_type: str,
    interview_id: str = "",
    target_words: int = 100,
    overlap_words: int = 20,
) -> List[DocumentChunk]:
    """Split extracted text into section-aware, overlapping chunks.

    Guarantees:
    - Retains section labels where headings exist.
    - Preserves document type ('resume' vs 'job_description') and interview_id.
    - Generates coherent chunks of reasonable size with contextual overlap.
    """
    if not text or not text.strip():
        return []

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return []

    sections: List[Tuple[str, List[str]]] = []
    current_section = "summary" if doc_type == "resume" else "role_overview"
    current_lines: List[str] = []

    for line in lines:
        # Check if line looks like a section header (short line, matches heading pattern)
        if len(line.split()) <= 6:
            detected = _detect_section(line, doc_type)
            if detected:
                if current_lines:
                    sections.append((current_section, current_lines))
                    current_lines = []
                current_section = detected
                continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_section, current_lines))

    # If no headings matched, treat whole text under "general"
    if not sections:
        sections = [("general", lines)]

    chunks: List[DocumentChunk] = []
    chunk_index = 0
    now = time.time()

    for sec_name, sec_lines in sections:
        sec_text = "\n".join(sec_lines).strip()
        words = sec_text.split()
        if not words:
            continue

        if len(words) <= target_words:
            chunks.append(DocumentChunk(
                interview_id=interview_id,
                doc_type=doc_type,
                section=sec_name,
                chunk_index=chunk_index,
                text=sec_text,
                created_at=now,
            ))
            chunk_index += 1
        else:
            # Slide a window over the words with overlap
            step = max(1, target_words - overlap_words)
            for i in range(0, len(words), step):
                chunk_words = words[i : i + target_words]
                if not chunk_words:
                    break
                chunk_str = " ".join(chunk_words)
                chunks.append(DocumentChunk(
                    interview_id=interview_id,
                    doc_type=doc_type,
                    section=sec_name,
                    chunk_index=chunk_index,
                    text=chunk_str,
                    created_at=now,
                ))
                chunk_index += 1
                if i + target_words >= len(words):
                    break

    return chunks


# ---------------------------------------------------------------------------
# Embedding Generation & Semantic Cosine Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class DeterministicNgramEmbedder:
    """Zero-dependency, offline deterministic semantic vectorizer.

    Projects text into a fixed 256-dimensional space using character n-grams
    and word tokens. Used as a rock-solid, zero-latency fallback and for test isolation.
    """
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return vec

        # Unigrams & Bigrams
        features = list(tokens)
        for i in range(len(tokens) - 1):
            features.append(f"{tokens[i]}_{tokens[i+1]}")

        for feat in features:
            # Deterministic hash bucket
            h = int(hashlib.md5(feat.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % self.dim
            weight = 1.0 + (len(feat) * 0.1)
            vec[idx] += weight

        # L2 Normalization
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class EmbeddingService:
    """Unified embedding provider.

    Tries FastEmbed (BAAI/bge-small-en-v1.5) first.
    If unavailable or fails, gracefully falls back to DeterministicNgramEmbedder.
    """
    _instance: Optional["EmbeddingService"] = None

    def __init__(self, force_fallback: bool = False):
        self.force_fallback = force_fallback
        self._fastembed_model = None
        self._fallback_embedder = DeterministicNgramEmbedder()
        self.provider_name = "fallback"

        if not force_fallback:
            try:
                from fastembed import TextEmbedding
                # Use bge-small-en-v1.5 (compact, fast 384-dim ONNX model)
                self._fastembed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
                self.provider_name = "fastembed (BAAI/bge-small-en-v1.5)"
            except Exception as err:
                sys.stderr.write(f"[RAG] FastEmbed unavailable ({err}); using deterministic vectorizer.\n")
                self._fastembed_model = None
                self.provider_name = "deterministic_ngram"

    @classmethod
    def get_instance(cls, force_fallback: bool = False) -> "EmbeddingService":
        if cls._instance is None or cls._instance.force_fallback != force_fallback:
            cls._instance = cls(force_fallback=force_fallback)
        return cls._instance

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self._fastembed_model is not None and not self.force_fallback:
            try:
                # FastEmbed returns an iterator of numpy ndarrays
                results = list(self._fastembed_model.embed(texts))
                return [arr.tolist() if hasattr(arr, "tolist") else list(arr) for arr in results]
            except Exception as exc:
                sys.stderr.write(f"[RAG] FastEmbed error during inference: {exc}; falling back.\n")

        return self._fallback_embedder.embed(texts)

    def embed_query(self, query: str) -> List[float]:
        res = self.embed_texts([query])
        return res[0] if res else [0.0] * 256


# ---------------------------------------------------------------------------
# High-Level RAG Indexing & Retrieval
# ---------------------------------------------------------------------------

def index_interview_documents(
    store: Any,
    interview_id: str,
    resume_text: str = "",
    job_description: str = "",
    embedding_service: Optional[EmbeddingService] = None,
) -> Dict[str, Any]:
    """Chunk, embed, and store document chunks in SQLite for an interview session.

    Handles updates idempotently: replaces previous chunks for the session.
    """
    if not interview_id:
        return {"error": "interview_id required", "total_indexed": 0}

    svc = embedding_service or EmbeddingService.get_instance()

    resume_chunks = chunk_document(resume_text, "resume", interview_id=interview_id)
    jd_chunks = chunk_document(job_description, "job_description", interview_id=interview_id)
    all_chunks = resume_chunks + jd_chunks

    if not all_chunks:
        # Clear out any previous chunks if both are now empty
        store.delete_document_chunks(interview_id)
        return {"resume_chunks": 0, "jd_chunks": 0, "total_indexed": 0}

    # Batch embed texts for maximum throughput
    texts = [c.text for c in all_chunks]
    try:
        vectors = svc.embed_texts(texts)
        for chunk, vec in zip(all_chunks, vectors):
            chunk.embedding = vec
    except Exception as exc:
        sys.stderr.write(f"[RAG] Failed to generate embeddings: {exc}\n")
        # Ensure non-null fallback embeddings
        fallback_vecs = DeterministicNgramEmbedder().embed(texts)
        for chunk, vec in zip(all_chunks, fallback_vecs):
            chunk.embedding = vec

    # Store in database
    store.save_document_chunks(interview_id, all_chunks)

    return {
        "status": "ok",
        "interview_id": interview_id,
        "resume_chunks": len(resume_chunks),
        "jd_chunks": len(jd_chunks),
        "total_indexed": len(all_chunks),
        "provider": svc.provider_name,
    }


def retrieve_evidence(
    store: Any,
    interview_id: str,
    query: str,
    doc_type: Optional[str] = None,
    top_k: int = 3,
    min_similarity: float = 0.05,
    embedding_service: Optional[EmbeddingService] = None,
) -> List[Dict[str, Any]]:
    """Retrieve the most relevant document chunks for the candidate.

    PRIVACY & ISOLATION:
    Queries the database strictly for chunks matching the given interview_id.
    Never returns another candidate's document chunks.
    """
    if not interview_id or not query or not query.strip():
        return []

    stored_chunks = store.get_document_chunks(interview_id, doc_type=doc_type)
    if not stored_chunks:
        return []

    svc = embedding_service or EmbeddingService.get_instance()
    query_vec = svc.embed_query(query)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for chunk in stored_chunks:
        chunk_vec = chunk.get("embedding")
        if not chunk_vec:
            continue
        sim = cosine_similarity(query_vec, chunk_vec)
        if sim >= min_similarity:
            scored.append((sim, chunk))

    # Sort descending by similarity score
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for sim, c in scored[:top_k]:
        results.append({
            "chunk_id": c.get("id"),
            "doc_type": c.get("doc_type"),
            "section": c.get("section"),
            "chunk_index": c.get("chunk_index"),
            "text": c.get("text"),
            "score": round(sim, 3),
        })

    return results


def format_evidence_block(evidence_list: List[Dict[str, Any]]) -> str:
    """Format retrieved evidence chunks for LLM prompt context."""
    if not evidence_list:
        return "None retrieved."

    lines = []
    for i, ev in enumerate(evidence_list, 1):
        doc = "Resume" if ev.get("doc_type") == "resume" else "Job Description"
        sec = ev.get("section", "general").capitalize()
        score = ev.get("score", 0)
        lines.append(f"[{i}] [{doc} · Section: {sec} (Match: {int(score * 100)}%)]\n\"{ev.get('text')}\"")
    return "\n\n".join(lines)
