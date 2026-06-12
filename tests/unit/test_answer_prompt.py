import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.main.python.steps.agents.rag_agent import AgnoRAGAgent
from src.main.python.steps.retrieval.pipeline import RetrievalPipeline


class TestAnswerPrompt(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_answer_query_delegates_to_agent(self):
        result = {
            "answer": "不限次。[1]",
            "context": {"pages": [{"page_id": "family"}], "metrics": {}},
            "agent_engine": "async_openai",
        }
        with patch(
            "src.main.python.steps.agents.rag_agent.AgnoRAGAgent"
        ) as agent_cls:
            agent_cls.return_value.answer = AsyncMock(return_value=result)

            pipeline = RetrievalPipeline.__new__(RetrievalPipeline)
            actual = await pipeline.answer_query(
                "p",
                "家庭医生服务次数",
                session_id="s1",
                retrieval_query="家庭医生 服务次数",
            )

        self.assertIs(actual, result)
        agent_cls.return_value.answer.assert_awaited_once_with(
            project_id="p",
            query="家庭医生服务次数",
            retrieval_query="家庭医生 服务次数",
            session_id="s1",
            user_id=None,
            history=None,
        )

    async def test_agent_uses_shared_dashscope_client(self):
        context = {
            "page_list": "[1] 家庭医生服务",
            "index": "[1] 家庭医生服务",
            "pages_context": "家庭医生服务不限次。",
        }
        pipeline = AsyncMock()
        pipeline.retrieve.return_value = context
        client = SimpleNamespace(chat=AsyncMock(return_value="不限次。[1]"))
        agent = AgnoRAGAgent(client=client)
        agent._pipeline = pipeline
        result = await agent.answer(
            "p", "这个服务有几次？", session_id="s1", user_id="u1"
        )

        self.assertEqual(result["answer"], "不限次。[1]")
        self.assertEqual(result["agent_engine"], "async_openai")
        client.chat.assert_awaited_once()

    def test_prompt_contains_context_sections(self):
        agent = AgnoRAGAgent(client=SimpleNamespace())
        prompt = agent.build_user_prompt(
            "家庭医生服务次数",
            {"page_list": "[1] A", "index": "[1] A", "pages_context": "不限次"},
        )
        self.assertIn("## Wiki Pages", prompt)
        self.assertIn("Conversation History", prompt)
