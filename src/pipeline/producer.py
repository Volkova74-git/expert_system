from typing import List

from src.pipeline.models import DocumentSchema, ChunkSchema
from src.pipeline.text_processing import filter_document, build_search_chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Producer:
    def __init__(self):
        pass

    @staticmethod
    def _get_chunks(text: str) -> List[str]:
        base_chunks = build_search_chunks(text)

        MAX_LEN = 2000

        # сплиттер для перерезки слишком длинных чанков.
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_LEN,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        )

        final_chunks: List[str] = []

        for ch in base_chunks:
            ch = ch.strip()
            if not ch:
                continue

            if len(ch) <= MAX_LEN:
                final_chunks.append(ch)
            else:
                # дополнительно нарезаем сверхдлинный чанк.
                sub_chunks = fallback_splitter.split_text(ch)
                for sub in sub_chunks:
                    sub = sub.strip()
                    if sub:
                        final_chunks.append(sub)

        return final_chunks

    def process(self, documents: List[DocumentSchema]) -> List[ChunkSchema]:
        results = []
        for document in documents:
            for file in document.Files:
                if not file.Text:
                    continue

                filtered_document = filter_document(file.Text)

                chunks = self._get_chunks(filtered_document['body'])
                total_chunks = len(chunks)

                for i, chunk in enumerate(chunks):
                    results.append(ChunkSchema(
                        document_id=document.ID,
                        document_name=document.Name,
                        file_name=file.Name,
                        chunk_text=chunk.replace("\n", " ").strip(),
                        total_chunks=total_chunks,
                        chunk_index=i,
                        reg_date=document.RegDate,
                        reg_number=document.RegNumber,
                    ))
        return results