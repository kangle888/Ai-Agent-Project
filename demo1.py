import os
import logging
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import LangChainException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_llm_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise LangChainException("DEEPSEEK_API_KEY not set")

    llm = init_chat_model(
        model="deepseek-v3.2",
        model_provider="openai",  # 👈 兼容模式
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.7,
        max_tokens=1000,
    )

    return llm


def main():
    try:
        load_dotenv()

        llm = init_llm_client()
        logger.info("LLM客户端初始化成功！")

        # 普通调用
        response = llm.invoke("你是谁？")
        print("回答：", response.content)

        # 流式调用
        print("\n=================== 流式输出 ===================")
        for chunk in llm.stream("介绍下LangChain，300字以内"):
            print(chunk.content, end="")

    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    main()