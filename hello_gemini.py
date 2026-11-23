"""最小可运行示例：加载 .env 中的密钥并调用 Gemini API。"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


def main() -> None:
    # 1. 从当前脚本同目录的 .env 文件加载环境变量，避免把密钥硬编码到代码里
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)

    # 2. 读取并检查 API Key 是否存在
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("错误：未找到 GEMINI_API_KEY，请检查 .env 文件。", file=sys.stderr)
        sys.exit(1)

    # 3. 初始化 Gemini 客户端，并设置 30 秒请求超时（单位：毫秒）
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30_000),
    )

    # 4. 选择响应快且免费的轻量模型，发送一次最基础的文本生成请求
    model_name = "gemini-3.5-flash-lite"
    prompt = "你好！请用两句话向一位正在入门 AI 开发的计算机大一新生打个招呼。"

    print(f"正在调用模型 {model_name} ...")
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    # 5. 输出模型回复和本次请求的 Token 消耗
    print("\n=== Gemini 回复 ===")
    print(response.text)

    if response.usage_metadata:
        print("\n=== Token 用量 ===")
        print(
            f"输入 tokens: {response.usage_metadata.prompt_token_count} | "
            f"输出 tokens: {response.usage_metadata.candidates_token_count} | "
            f"总计: {response.usage_metadata.total_token_count}"
        )


if __name__ == "__main__":
    main()
