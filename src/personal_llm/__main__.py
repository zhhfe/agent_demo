import src.config as config
from typing import List, Dict
from openai import OpenAI


class HelloAgentsLLM:
    def __init__(self, model: str = config.BYTEPLUS_SEED, apiKey: str = config.BYTEPLUS_API_KEY,
                 baseUrl: str = config.BYTEPLUS_DOMAIN, timeout: int = 60):

        self.apiKey = apiKey
        self.model = model
        self.baseUrl = baseUrl
        self.timeout = timeout

        if not all([self.apiKey, self.model, self.baseUrl]):
            raise

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0):
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            print("✅ 大语言模型触发成功:")

            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return


if __name__ == "__main__":
    llm = HelloAgentsLLM()
    exampleMessages = [
        {"role": "system", "content": "You are a helpful assistant that writes Python code."},
        {"role": "user", "content": "介绍下你自己"}
    ]
    llm.think(exampleMessages)
    print("\n\n--- 完整模型响应 ---")

