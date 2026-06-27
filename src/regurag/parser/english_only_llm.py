from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage


class EnglishOnlyLLM(ChatAnthropic):
    """Force the RAGAS judge model to return English JSON."""

    def invoke(self, input, **kwargs):
        if isinstance(input, list):
            input = [SystemMessage(content="Always respond in English with valid JSON only.")] + input
        return super().invoke(input, **kwargs)
