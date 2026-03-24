from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI

from utils.schemas import RequestType
from utils.characters import CHARACTER_PROMPTS


primary_model = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="qwen2.5-7b-instruct-1m",
    streaming=True,
    stream_options={"include_usage": True} 
)

fallback_model = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="glm-4.6v-flash",
    streaming=True,
    stream_options={"include_usage": True} 
)

model = primary_model.with_fallbacks([fallback_model])

def create_handler_chains(character: str = "friendly"):

    if character not in CHARACTER_PROMPTS:
        character = "friendly"

    base_system = CHARACTER_PROMPTS[character]

    question_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""{base_system} Ты — полезный и точный помощник.
    Отвечай информативно, структурировано и по делу.
    Если не знаешь точного ответа или информация устарела — честно скажи об этом.
    Не выдумывай факты."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}")
    ])

    task_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""{base_system} Ты выполняешь конкретные задачи пользователя.
    Делай всё максимально качественно, аккуратно и полностью.
    Если задача требует кода — пиши чистый, прокомментированный код.
    Если нужно что-то составить/перевести/посчитать — делай это тщательно."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}")
    ])

    small_talk_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""{base_system} Ты дружелюбный и приятный собеседник.
    Поддерживай лёгкую, тёплую беседу.
    Если пользователь представился — используй его имя в дальнейшем.
    Будь позитивным, используй смайлики уместно, но не переборщи."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}")
    ])

    complaint_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""{base_system} Пользователь недоволен. Твоя задача:
    1. Проявить искреннюю эмпатию
    2. Извиниться, если это уместно
    3. Понять суть проблемы
    4. Предложить конкретное решение или улучшение."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}")
    ])

    unknown_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""{base_system} Запрос пользователя непонятен или слишком неоднозначен.
    Вежливо попроси уточнить, что именно имелось в виду.
    Не придумывай смысл там, где его нет."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}")
    ])

    chains = {
        RequestType.QUESTION:    question_prompt | model,
        RequestType.TASK:        task_prompt     | model,
        RequestType.SMALL_TALK:  small_talk_prompt | model,
        RequestType.COMPLAINT:   complaint_prompt | model,
        RequestType.UNKNOWN:     unknown_prompt   | model,
    }
    return chains

def wrap_with_response(chain, request_type: RequestType):
    def inner(input_dict):
        return {
            "stream": chain.stream(input_dict),
            "request_type": request_type,
            "confidence": input_dict["classification"].confidence
        }

    return RunnableLambda(inner)

def get_request_branch(character: str = "friendly"):
    chains = create_handler_chains(character)
    
    branch = RunnableBranch(
        (lambda x: x["classification"].request_type == RequestType.QUESTION,
         wrap_with_response(chains[RequestType.QUESTION], RequestType.QUESTION)),
        (lambda x: x["classification"].request_type == RequestType.TASK,
         wrap_with_response(chains[RequestType.TASK], RequestType.TASK)),
        (lambda x: x["classification"].request_type == RequestType.SMALL_TALK,
         wrap_with_response(chains[RequestType.SMALL_TALK], RequestType.SMALL_TALK)),
        (lambda x: x["classification"].request_type == RequestType.COMPLAINT,
         wrap_with_response(chains[RequestType.COMPLAINT], RequestType.COMPLAINT)),
        wrap_with_response(chains[RequestType.UNKNOWN], RequestType.UNKNOWN)
    )
    return branch
