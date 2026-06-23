from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage


class EnglishOnlyLLM(ChatAnthropic):
    """强制 RAGAS 的 LLM judge 用英文回复，避免 JSON 解析失败"""

    def invoke(self, input, **kwargs):
        if isinstance(input, list):
            input = [SystemMessage(content="Always respond in English with valid JSON only.")] + input
        return super().invoke(input, **kwargs)
