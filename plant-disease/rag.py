"""
RAG (Retrieval-Augmented Generation) Knowledge Engine for Plant Pathology
Indexes expert documents (documents/tomato.txt, potato.txt, rice.txt) and retrieves
pathology data, treatment plans, chemical/organic remedies, and answers agronomic queries.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / os.getenv("DOCUMENTS_PATH", "documents")


class PlantRAGKnowledgeBase:
    """
    RAG Engine for indexing agricultural pathology manuals and retrieving
    exact disease treatments, symptoms, and agronomic management plans.
    """
    def __init__(self, docs_dir: Path = DOCS_DIR):
        self.docs_dir = Path(docs_dir)
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.build_index()

    def build_index(self):
        """
        Parses documents in docs_dir into semantically rich topic chunks.
        """
        self.chunks = []
        if not self.docs_dir.exists():
            print(f"[RAG Warning] Documents folder not found: {self.docs_dir}")
            return

        doc_files = list(self.docs_dir.glob("*.txt")) + list(self.docs_dir.glob("*.md"))
        for file_path in doc_files:
            crop_name = file_path.stem.capitalize()
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            file_chunks = self._chunk_document(content, crop_name, file_path.name)
            self.chunks.extend(file_chunks)

        if self.chunks:
            texts = [f"{c['crop']} {c['disease']}\n{c['text']}" for c in self.chunks]
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            print(f"[RAG] Indexed {len(self.chunks)} knowledge chunks from {len(doc_files)} agronomy documents.")
        else:
            print("[RAG] No document content found to index.")

    def _chunk_document(self, content: str, crop_name: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Splits structured agronomy documents into topic blocks (e.g. per disease).
        """
        chunks = []
        # Split by numbered section header e.g. "1. OVERVIEW...", "2. TOMATO EARLY BLIGHT..."
        sections = [s.strip() for s in re.split(r"(?m)^(?=[0-9]+\.\s+[A-Z])", content) if s.strip()]

        for sec_text in sections:
            if len(sec_text) < 40:
                continue

            first_line = sec_text.splitlines()[0].strip()

            # Match title e.g. "2. TOMATO EARLY BLIGHT (Alternaria...)"
            match = re.search(r"^[0-9]+\.\s+(?:(?:TOMATO|POTATO|RICE)\s+)?([A-Z\s]+?)(?:\s*\(|\s*$)", first_line, re.IGNORECASE)
            if match:
                disease = match.group(1).strip().title()
            elif "OVERVIEW" in first_line.upper():
                disease = "Overview & Biology"
            elif "HEALTHY" in first_line.upper():
                disease = "Healthy Foliage Care"
            else:
                disease = first_line[:40]

            chunks.append({
                "crop": crop_name,
                "disease": disease,
                "source": source_file,
                "text": sec_text
            })

        return chunks

    def search(self, query: str, crop: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search across indexed agricultural manuals.
        """
        if not self.vectorizer or self.tfidf_matrix is None or not self.chunks:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        filtered = []
        for idx, score in enumerate(scores):
            if crop:
                chunk_crop = self.chunks[idx]["crop"].lower()
                if crop.lower() not in chunk_crop:
                    continue
            filtered.append((idx, score))

        filtered.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in filtered[:top_k]:
            chunk = dict(self.chunks[idx])
            chunk["score"] = round(float(score), 4)
            results.append(chunk)

        return results

    def get_disease_profile(self, crop: str, disease: str) -> Dict[str, Any]:
        """
        Extracts structured disease information (Symptoms, Organic, Chemical, Prevention)
        specifically for a detected plant disease.
        """
        query = f"{crop} {disease}"
        matches = self.search(query, crop=crop, top_k=2)
        if not matches:
            matches = self.search(query, top_k=2)

        profile = {
            "crop": crop,
            "disease": disease,
            "symptoms": [],
            "organic_treatments": [],
            "chemical_treatments": [],
            "prevention": [],
            "raw_text": ""
        }

        if matches:
            best_chunk = matches[0]["text"]
            profile["raw_text"] = best_chunk

            profile["symptoms"] = self._extract_section(best_chunk, ["visual symptoms", "symptoms"])
            profile["organic_treatments"] = self._extract_section(best_chunk, ["organic treatments", "organic & preventative solutions", "organic"])
            profile["chemical_treatments"] = self._extract_section(best_chunk, ["chemical treatments", "treatment & management", "chemical controls"])
            profile["prevention"] = self._extract_section(best_chunk, ["preventive agronomic practices", "prevention", "cultural practices"])

        # Optional GraphRAG enrichment from Neo4j
        try:
            from neo4j_graph import get_neo4j_graph
            graph = get_neo4j_graph()
            if graph.is_connected():
                graph_data = graph.query_disease_relations(crop, disease)
                profile["neo4j_graph"] = graph_data
        except Exception:
            profile["neo4j_graph"] = None

        return profile

    def _extract_section(self, text: str, header_candidates: List[str]) -> List[str]:
        """Extracts bullet items under a matching section title."""
        lines = text.splitlines()
        capturing = False
        items = []

        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue

            # Check if this line is a target header
            lower_line = line_clean.lower().rstrip(":")
            if any(h in lower_line for h in header_candidates):
                capturing = True
                continue

            # If capturing, check if we hit another section header
            if capturing:
                if line_clean.endswith(":") and not (line_clean.startswith("-") or line_clean.startswith("•")):
                    break
                if line_clean.startswith("-") or line_clean.startswith("•") or line_clean.startswith("*"):
                    items.append(line_clean.lstrip("-•* ").strip())
                elif any(h in line_clean.lower() for h in ["severity level", "pathogen type", "biology & spread"]):
                    break

        return items

    def answer_query(self, user_question: str, crop: Optional[str] = None, disease: Optional[str] = None) -> Dict[str, Any]:
        """
        Synthesizes an agronomic answer grounded directly in the indexed documentation.
        """
        search_query = f"{crop or ''} {disease or ''} {user_question}".strip()
        relevant_chunks = self.search(search_query, crop=crop, top_k=3)

        if not relevant_chunks:
            return {
                "answer": f"No specific documentation found for '{user_question}' in {crop or 'indexed crops'}.",
                "sources": []
            }

        context_text = "\n\n".join([f"[{c['crop']} - {c['disease']}]:\n{c['text']}" for c in relevant_chunks])

        # Optional Gemini API integration if key provided
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = (
                    f"You are an expert AI Agronomist on an NVIDIA DGX Server. Answer the farmer's question using strictly "
                    f"the reference documentation below.\n\n"
                    f"Context:\n{context_text}\n\n"
                    f"Farmer Question: {user_question}\n\n"
                    f"Provide a clear, practical, bulleted agronomic recommendation."
                )
                response = model.generate_content(prompt)
                return {
                    "answer": response.text,
                    "sources": [c["source"] for c in relevant_chunks],
                    "context": context_text
                }
            except Exception as e:
                print(f"[RAG LLM Error] {e}. Falling back to document extraction.")

        # Local extraction
        summary_points = []
        for c in relevant_chunks:
            for line in c["text"].splitlines():
                line_str = line.strip()
                if line_str.startswith("-") and any(w in line_str.lower() for w in user_question.lower().split()):
                    summary_points.append(line_str.lstrip("-•* "))

        if not summary_points:
            snippet = relevant_chunks[0]["text"][:600] + "..."
            answer_text = f"**Relevant Agronomic Guidance ({relevant_chunks[0]['crop']} - {relevant_chunks[0]['disease']}):**\n\n{snippet}"
        else:
            bullets = "\n".join([f"• {pt}" for pt in summary_points[:5]])
            answer_text = f"**Agronomic Guidance:**\n{bullets}"

        return {
            "answer": answer_text,
            "sources": list(set(c["source"] for c in relevant_chunks)),
            "context": context_text
        }


# Singleton RAG instance
_rag_instance = None

def get_rag() -> PlantRAGKnowledgeBase:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = PlantRAGKnowledgeBase()
    return _rag_instance


if __name__ == "__main__":
    rag = get_rag()
    print("\n--- Testing RAG Profile Extraction for Tomato Early Blight ---")
    profile = rag.get_disease_profile("Tomato", "Early Blight")
    print("Crop:", profile["crop"])
    print("Disease:", profile["disease"])
    print("Symptoms count:", len(profile["symptoms"]))
    for s in profile["symptoms"][:2]:
        print(" •", s)
    print("Organic treatments count:", len(profile["organic_treatments"]))
    for o in profile["organic_treatments"][:2]:
        print(" •", o)
    print("Chemical treatments count:", len(profile["chemical_treatments"]))
    for c in profile["chemical_treatments"][:2]:
        print(" •", c)

    print("\n--- Testing RAG Question Answering ---")
    ans = rag.answer_query("What fungicides should I apply for late blight?", crop="Tomato")
    print(ans["answer"])
