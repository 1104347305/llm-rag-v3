import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.main.python.steps.agents.rag_agent import AgnoRAGAgent
from src.main.python.steps.retrieval.answer import answer_query


class TestAnswerPrompt(unittest.TestCase):
    def test_answer_uses_llm_wiki_black_style_prompt(self):
        context = {
            "page_list": "[1] 家庭医生服务 (entities/家庭医生服务.md)",
            "index": "[1] 家庭医生服务 (entities/家庭医生服务.md)",
            "pages_context": "### [1] 家庭医生服务\nPath: entities/家庭医生服务.md\n\n家庭医生服务不限次。",
        }
        with patch("src.main.python.agents.rag_agent.retrieve_context", return_value=context), patch("src.main.python.agents.rag_agent.DashScopeClient") as client_cls:
            client_cls.return_value.chat.return_value = "家庭医生服务不限次。[1]\n\n<!-- cited: 1 -->"

            result = answer_query("p", "家庭医生服务次数")

        messages = client_cls.return_value.chat.call_args.args[0]
        assert result["answer"]
        assert "You are a knowledgeable wiki assistant" in messages[0]["content"]
        assert "Answer based ONLY on the numbered wiki pages provided below" in messages[0]["content"]
        assert "## Wiki Pages" in messages[1]["content"]
        assert "保持简洁" in messages[1]["content"]

    def test_agno_agent_passes_session_id_to_run(self):
        context = {
            "page_list": "[1] 家庭医生服务 (entities/家庭医生服务.md)",
            "index": "[1] 家庭医生服务 (entities/家庭医生服务.md)",
            "pages_context": "### [1] 家庭医生服务\nPath: entities/家庭医生服务.md\n\n家庭医生服务不限次。",
        }
        fake_agent = SimpleNamespace(calls=[])

        def run(prompt, session_id=None, user_id=None):
            fake_agent.calls.append({"prompt": prompt, "session_id": session_id, "user_id": user_id})
            return SimpleNamespace(content="家庭医生服务不限次。[1]")

        fake_agent.run = run
        agent = AgnoRAGAgent()
        with patch("src.main.python.agents.rag_agent.retrieve_context", return_value=context), patch.object(agent, "_get_agent", return_value=fake_agent):
            result = agent.answer("p", "这个服务有几次？", session_id="s1", user_id="u1", history=[{"role": "user", "content": "家庭医生服务"}])

        assert result["answer"] == "家庭医生服务不限次。[1]"
        assert result["agent_engine"] == "agno"
        assert fake_agent.calls[0]["session_id"] == "s1"
        assert fake_agent.calls[0]["user_id"] == "u1"
        assert "Conversation History" in fake_agent.calls[0]["prompt"]


if __name__ == "__main__":
    unittest.main()
