from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HierarchicalChunker


def _extract_page_numbers(chunk) -> List[int]:
    """
    Pull page numbers from Docling chunk provenance.
    A chunk can sometimes span multiple pages, so we store a list.
    """
    page_numbers = set()

    try:
        meta = getattr(chunk, "meta", None)
        doc_items = getattr(meta, "doc_items", []) if meta else []

        for item in doc_items:
            prov_list = getattr(item, "prov", []) or []
            for prov in prov_list:
                page_no = getattr(prov, "page_no", None)
                if page_no is not None:
                    page_numbers.add(int(page_no))
    except Exception:
        pass

    return sorted(page_numbers)


def load_pdf_documents(file_paths: Union[str, List[str]]) -> List[Document]:
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CPU,  # force CPU, avoid MPS
    )
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    chunker = HierarchicalChunker()
    all_docs: List[Document] = []

    for file_path in file_paths:
        pdf_path = Path(file_path)
        result = converter.convert(pdf_path)
        doc = result.document

        for i, chunk in enumerate(chunker.chunk(doc)):
            text = getattr(chunk, "text", "") or ""
            if not text.strip():
                continue

            page_numbers = _extract_page_numbers(chunk)

            metadata = {
                "source": pdf_path.name,
                "chunk_index": i,
                "page_numbers": page_numbers,
                "page": page_numbers[0] if page_numbers else None,
            }

            all_docs.append(
                Document(
                    page_content=text.strip(),
                    metadata=metadata,
                )
            )

    return all_docs

def preview_documents(docs: List[Document], n: int = 5) -> None:
    print(f"\nLoaded {len(docs)} document chunks\n")

    for i, doc in enumerate(docs[:n], start=1):
        print("=" * 80)
        print(f"Chunk {i}")
        print("Metadata:", doc.metadata)
        print("Content preview:", doc.page_content[:500])
        print()