# -*- coding: utf-8 -*-
"""硅基流动真实 LLM 的最小 smoke test：只做一次 Parser 调用。"""

from agentsense.agents import DisturbanceParser
from agentsense.llm import LLMClient

client = LLMClient("siliconflow", "deepseek-ai/DeepSeek-V3.2", temperature=0.1, max_tokens=256)
parser = DisturbanceParser(client)
result = parser.parse("Bad weather reduces sensing quality in the northern area.")
print(result)
print(client.usage)
