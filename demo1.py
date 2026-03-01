import os
import logging
from dotenv import load_dotenv
from langchain_core.exceptions import LangChainException
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_llm_client() -> ChatOpenAI:
    # 读取环境变量
    api_key = SecretStr(os.getenv("DEEPSEEK_API_KEY"))
    if not api_key:
        raise LangChainException("DEEPSEEK_API_KEY not set")

    # 初始化LLM客户端
    llm = ChatOpenAI(
        model="deepseek-v3.2",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.7,
        max_tokens=1000,
    )
    return llm


def main():
    try:
        # 加载 .env
        load_dotenv()

        # 初始化客户端
        llm = init_llm_client()
        logger.info("LLM客户端初始化成功！")

        # 普通调用
        question = "你是谁？"
        response = llm.invoke(question)
        logger.info(f"问题: {question}")
        logger.info(f"回答：{response.content}")

        # 流式调用
        print("\n=================== 以下是流式输出 ===================")
        print("*" * 50)

        for chunk in llm.stream("介绍下langchain, 300字以内"):
            print(chunk.content, end="")

    except LangChainException as e:
        logger.error(e)
    except Exception as e:
        logger.error(e)


if __name__ == '__main__':
    main()