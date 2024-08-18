import os
import pickle
from io import BytesIO

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


def get_documents_from_file(
    document_file: BytesIO, filename: str = None
) -> list[Document]:
    """특정 파일에 대한 Document 리스트 객체를 반환한다

    - 로컬 파일에 pickle 데이터가 있다면, 이를 역직렬화하여 반환한다
    - 없다면, PDF를 읽어서 Document 리스트를 만들고, 직렬화하여 반환한다
        - 이 때 만든 데이터는 pickle 파일로 저장한다
    """

    # 직렬화한 바이너리 데이터가 있는 경우, 이를 불러온다
    pickle_filename = filename.replace(".pdf", ".pickle")
    if os.path.exists(f"documents//pickle/{pickle_filename}"):
        with open(f"documents//pickle/{pickle_filename}", "rb") as f:
            return pickle.load(f)

    # 직렬화한 바이너리 데이터가 없는 경우, PDF를 읽어서 document_list에 추가한다
    page_texts = []
    with pdfplumber.open(document_file) as pdf:
        pages = pdf.pages
        for page in tqdm(pages):
            page_texts.append((page.extract_text(), page.to_dict()))

    document_list: list[Document] = []
    last_chunk = None
    for page_number, page in tqdm(enumerate(page_texts)):
        # 텍스트를 chunk로 분리
        text, page_info = page
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=128, length_function=len
        )
        # last_chunk가 존재하면 앞에 붙여줌 (페이지가 넘어가서, chunk context가 끊기는 경우를 방지)
        text = last_chunk + text if last_chunk else text
        chunks = splitter.split_text(text)

        if not text or not chunks:
            continue

        last_chunk = chunks[-1]

        chunk_list = []
        for c_idx, chunk in enumerate(chunks):
            chunk_list.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "document_name": filename,
                        "page_number": page_info.get("page_number", None),
                        "page_width": page_info.get("width", None),
                        "page_height": page_info.get("height", None),
                        "chunk_number": c_idx,
                    },
                )
            )

        document_list.extend(chunk_list)

    # 나중에 사용할 수 있도록 직렬화한다
    with open(f"documents/pickle/{pickle_filename}", "wb") as f:
        pickle.dump(document_list, f)

    return document_list
