import json

import pytest
import requests
from fastapi import FastAPI

from main import app


@pytest.fixture
def fastapi_app():
    return app


def test_ask(
    fastapi_app: FastAPI,
):
    url = fastapi_app.url_path_for("ask")

    with requests.session() as session:
        resp = session.post(
            f"http://127.0.0.1:8000/{url}", json={"query": "안녕", "session_id": None}
        )

        response_list = []
        for line in resp.iter_lines():
            if line:
                resp_json = json.loads(line[5:])
                assert resp_json["session_id"] is not None
                assert resp_json["type"] in ["answer", "event"]
                assert resp_json["answer"] is not None

        assert resp.status_code == 200
