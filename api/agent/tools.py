import json
import os

import pymupdf
from PIL import Image
from pydantic import BaseModel, Field

from util.dependency import FileNameRetriever


class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


class RAGInput(BaseModel):
    query: str = Field(description="질문")


class GetPDFInput(BaseModel):
    filename: str = Field(description="filename")
    page_number: int = Field(description="page number")


def multiply(a: int, b: int) -> str:
    """두 정수를 곱한 결과를 반환합니다."""
    return json.dumps({"result": a * b})


def add(a: int, b: int) -> str:
    """두 정수를 더한 결과를 반환합니다."""
    return json.dumps({"result": a + b})


def rag(retriever):
    def _get_refined_document_list(docs) -> list[dict]:
        """
        문서 리스트를 받아 각 문서의 본문 내용과 출처를 포함한 문자열을 생성
        """

        result = []
        for i, doc in enumerate(docs):
            result.append(
                {
                    "본문": doc.page_content,
                    "출처": doc.metadata["document_name"]
                    + " - "
                    + str(doc.metadata["page_number"])
                    + "페이지",
                }
            )
        return result

    def _rag(r_, query: str) -> list[dict]:
        """RAG를 사용하여 질문에 대한 답변을 반환합니다."""
        document_list = r_.invoke(query)
        refined_document_list = _get_refined_document_list(document_list)

        return refined_document_list

    return lambda query: _rag(retriever, query)


def get_pdf(filename: str, page_number: int) -> dict:
    """파일 이름과 페이지 번호를 받아 해당 페이지의 이미지 URL을 반환합니다."""

    # 가장 비슷한 파일명을 가져온다
    retriever = FileNameRetriever()
    similar_filename_list = retriever(filename)

    # 미리 만들어둔 PDF 페이지 이미지가 없다면 생성
    for similar_filename in similar_filename_list:
        if not os.path.exists(
            f"./documents/page_chunk/{similar_filename}_{page_number}.png"
        ):
            try:
                pdf = pymupdf.open(f"./documents/{similar_filename}")
                page = pdf.load_page(page_number)
                scale = 2450 / page.rect.height
                pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale))
                imgs_ = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                imgs = imgs_.convert("L")

                imgs.save(
                    f"./documents/page_chunk/{similar_filename}_{page_number}.png",
                    "PNG",
                )
            except Exception as e:
                continue

        # 이미지 파일 요청 URL을 반환 -> S3 버킷등으로 변경한다면 좋을것
        return {
            "filename": f"{similar_filename} - {page_number}페이지",
            "image_url": f"/agent/document/{similar_filename}/{page_number}",
        }

    return {"filename": None, "image_url": None}
