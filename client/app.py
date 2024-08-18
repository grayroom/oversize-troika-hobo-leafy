import json

import requests
import streamlit as st

st.set_page_config(
    page_icon="assets/favicon.png",
    page_title="마이데이터 레퍼런스 Q&A 챗봇",
)
st.title("🎓 마이데이터 레퍼런스 Q&A 챗봇")
st.caption("마이데이터 공식 문서를 기반으로 정책과 기술 사양 등을 답변해드립니다.")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": {
                "type": "chat",
                "content": '질문이 떠오르지 않나요? "토큰이 중복 발급되었을 경우 어떻게 되나요?" 라고 물어보는건 어떤가요?',
            },
        }
    ]

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="assets/profile/default.png").write(
            msg["content"]
        )
    if msg["role"] == "assistant":
        if msg["content"]["type"] == "chat":
            st.chat_message(msg["role"], avatar="assets/profile/ryan.png").write(
                msg["content"]["content"]
            )
        if msg["content"]["type"] == "image":
            image_expander = st.expander(msg["content"]["filename"], expanded=False)
            image_expander.image(msg["content"]["content"])


if query := st.chat_input(placeholder="오늘은 어떤 내용이 궁금하신가요?"):
    st.session_state.messages.append({"role": "user", "content": query})

    st.chat_message("user", avatar="assets/profile/default.png").write(query)
    with requests.session() as session:
        if hasattr(st.session_state, "session_id"):
            session_id = st.session_state.session_id
        else:
            session_id = None

        response_list = []
        with st.status("답변을 생성하는중 입니다.", expanded=True) as status:
            status.write("답변에 필요한 추가정보를 수집하는 중...")
            response = session.post(
                "http://api:8000/agent/ask",
                json={
                    "query": query,
                    "session_id": session_id,
                },
            )

            for line in response.iter_lines():
                if line:
                    # SSE임을 감안하여 line앞에 data: 를 떼어낸 뒤 json으로 파싱
                    chunk = json.loads(line[5:])
                    st.session_state.session_id = chunk["session_id"]
                    match chunk["type"]:
                        case "answer":
                            status.update(
                                label="답변 생성 완료!",
                                state="complete",
                                expanded=False,
                            )
                            if isinstance(chunk["answer"], str):
                                response_list.append(
                                    {"type": "chat", "content": chunk["answer"]}
                                )
                            if isinstance(chunk["answer"], dict):
                                response_list.append(
                                    {
                                        "type": "chat",
                                        "content": chunk["answer"]["answer"],
                                    }
                                )
                                response_list.append(
                                    {
                                        "type": "chat",
                                        "content": chunk["answer"]["meta"],
                                    }
                                )
                        case "event":
                            if "image_url" in chunk["answer"]:
                                response_list.append(
                                    {
                                        "type": "image",
                                        "image_url": chunk["answer"]["image_url"],
                                        "filename": chunk["answer"]["filename"],
                                    }
                                )
                            else:
                                status.write(
                                    "수집한 정보를 분석하여 답변을 생성하는 중..."
                                )

        for response in response_list:
            if response["type"] == "chat":
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                st.chat_message("assistant", avatar="assets/profile/ryan.png").write(
                    response["content"]
                )
            if response["type"] == "image":
                img_resp = session.get(f"http://api:8000{response['image_url']}")
                response["content"] = img_resp.content

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

                image_expander = st.expander(response["filename"], expanded=True)
                image_expander.image(response["content"])
