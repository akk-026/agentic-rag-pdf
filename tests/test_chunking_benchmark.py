from pathlib import Path
from statistics import mean

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HierarchicalChunker

from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from sentence_transformers import SentenceTransformer, util

from src.config import EMBEDDING_MODEL

PDF_PATH = Path("data/uploads/healthcare_india_hipaa.pdf")

QUESTIONS = [
    "What are the three main HIPAA rules?",
    "What does the Security Rule require?",
    "What standards are listed for EHR in India?",
]

# same embedding model for fair comparison
embedder = SentenceTransformer(EMBEDDING_MODEL)


def convert_pdf_cpu(pdf_path: Path):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CPU,
    )
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(pdf_path)
    return result.document


def chunk_stats(name, chunks):
    texts = [c for c in chunks if c and c.strip()]
    lengths = [len(t.split()) for t in texts]
    print(f"\n=== {name} ===")
    print(f"Chunks: {len(texts)}")
    print(f"Avg words/chunk: {mean(lengths):.1f}" if lengths else "Avg words/chunk: n/a")
    print(f"Min words/chunk: {min(lengths) if lengths else 'n/a'}")
    print(f"Max words/chunk: {max(lengths) if lengths else 'n/a'}")
    return texts


def top_hit(question, chunks, top_k=3):
    q_vec = embedder.encode(question, normalize_embeddings=True)
    c_vecs = embedder.encode(chunks, normalize_embeddings=True)
    sims = util.cos_sim(q_vec, c_vecs)[0]
    top = sims.topk(top_k)

    print(f"\nQuestion: {question}")
    for rank, idx in enumerate(top.indices.tolist(), start=1):
        print(f"\n[{rank}] score={float(sims[idx]):.4f}")
        print(chunks[idx][:600])


def baseline_docling_export(doc):
    md = doc.export_to_markdown()
    blocks = [b.strip() for b in md.split("\n\n") if b.strip()]
    return blocks


def hierarchical_docling(doc):
    chunker = HierarchicalChunker()
    chunks = list(chunker.chunk(doc))
    return [c.text for c in chunks if c.text and c.text.strip()]


def semantic_splitter_llamaindex(doc):
    text = doc.export_to_markdown()
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL
    )

    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model,
    )

    nodes = splitter.get_nodes_from_documents(
        [LlamaDocument(text=text, metadata={"source": str(PDF_PATH)})]
    )
    return [n.text for n in nodes if n.text and n.text.strip()]


if __name__ == "__main__":
    doc = convert_pdf_cpu(PDF_PATH)

    baseline = chunk_stats("1) Docling markdown export baseline", baseline_docling_export(doc))
    hierarchical = chunk_stats("2) Docling HierarchicalChunker", hierarchical_docling(doc))
    semantic = chunk_stats("3) LlamaIndex SemanticSplitterNodeParser", semantic_splitter_llamaindex(doc))

    for q in QUESTIONS:
        print("\n" + "=" * 100)
        print("BASELINE")
        top_hit(q, baseline)

        print("\n" + "=" * 100)
        print("HIERARCHICAL")
        top_hit(q, hierarchical)

        print("\n" + "=" * 100)
        print("SEMANTIC")
        top_hit(q, semantic)