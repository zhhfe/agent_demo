import requests
import os

from tavily import TavilyClient

from src.config import TAVILY_API_KEY


def get_weather(city: str) -> str:
    """
    :param city: 城市名称
    :return: 城市的天气信息
    """
    url = f"https://wttr.in/{city}?format=j1"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # 提取当前天气
        current_condition = data['current_condition'][0]
        temp_c = current_condition['temp_C']
        weather_desc = current_condition['weatherDesc'][0]['value']

        return f"当前{city}的温度是{temp_c}摄氏度，天气是{weather_desc}"
    except requests.RequestException as e:
        return f"查询{city}天气时出错: {e}"
    except (KeyError, IndexError) as e:
        return f"解析{city}天气数据时出错: {e}"


def get_attraction(city: str, weather: str) -> str:
    """
    :param city: 城市名称
    :param weather: 天气描述
    :return: 城市的旅游景点推荐
    """

    api_key = TAVILY_API_KEY
    if not api_key:
        return "未设置TAVILY_API_KEY环境变量"

    tavily = TavilyClient(api_key=api_key)

    query = f"在{city}天气是{weather}的情况下，推荐一些旅游景点"

    try:
        response = tavily.search(query, search_depth="basic", include_answer=True)
        print(f"tavily result:{response}")

        if response.get("answer"):
            return response["answer"]

        formatted_result = []
        for result in response.get("result", []):
            formatted_result.append(f"- {result['title']}:{result['content']}")

        if not formatted_result:
            return "抱歉，没有找到推荐景点"

        return "根据搜索，为你找到以下信息:\n" + "\n".join(formatted_result)

    except Exception as e:
        return f"error:{e}"


available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}