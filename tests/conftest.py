import pytest

from app.agent import Conversation, PropertyAgent
from app.config import Settings
from app.tools import PropertyTools


@pytest.fixture(autouse=True)
def _no_log_file(monkeypatch):
    monkeypatch.setenv("PROPERTYAGENT_LOG", "")


@pytest.fixture
def settings():
    return Settings(prompt_version="v3", retrieval_mode="vector")


@pytest.fixture
def tools(settings):
    return PropertyTools(settings=settings)


@pytest.fixture
def make_agent(tools, settings):
    def _make(llm):
        return PropertyAgent(llm, tools, settings), Conversation()
    return _make
