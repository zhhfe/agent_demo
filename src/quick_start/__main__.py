import re
from typing import Optional

from src.quick_start.llm import OpenAICompatibleClient
from src.config import BYTEPLUS_DOMAIN, BYTEPLUS_API_KEY, BYTEPLUS_SEED, validate_required_config
from src.quick_start.prompt import AGENT_SYSTEM_PROMPT
from src.quick_start.tool import available_tools


def main():
    validate_required_config()
    llm = OpenAICompatibleClient(
        model=BYTEPLUS_SEED,
        api_key=BYTEPLUS_API_KEY,
        base_url=BYTEPLUS_DOMAIN
    )

    user_prompt = "帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
    prompt_history = [f"user_prompt:{user_prompt}"]

    for i in range(4):
        print(f"--- 循环 {i + 1} ---\n")

        full_prompt = "\n".join(prompt_history)

        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)

        print(f"模型原始输出:\n{llm_output}\n")

        # 模型可能会输出多余的Thought-Action，需要截断
        match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output,
                          re.DOTALL)
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("已截断多余的 Thought-Action 对")

        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)

        action_match = re.search(r'Action:\s*(.*)', llm_output)
        if not action_match:
            observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "="*40)
            prompt_history.append(observation_str)
            continue
        action_str = action_match.group(1).strip()

        if action_str.startswith("Finish"):
            final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
            print(f"任务完成，最终答案: {final_answer}")
            break

        tool_name = re.search(r"(\w+)\(", action_str).group(1)
        args_str = re.search(r"\((.*)\)", action_str).group(1)
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

        if tool_name in available_tools:
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误:未定义的工具 '{tool_name}'"

        # 3.4. 记录观察结果
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
