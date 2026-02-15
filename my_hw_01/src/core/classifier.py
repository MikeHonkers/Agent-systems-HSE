from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from utils.schemas import Classification, RequestType
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """\
Ты — точный классификатор пользовательских сообщений.

Твоя задача — определить основной тип запроса и вернуть структурированный ответ.

Возможные категории:

• question     — любой вопрос, запрос информации, уточнение, "что такое", "как работает", "почему", "сколько стоит", "где найти" и т.п.
• task         — просьба что-то сделать, инструкция, команда, "напиши", "посчитай", "переведи", "составь", "сделай", "сгенерируй", "проверь"
• small_talk   — приветствие, прощание, болтовня, шутки, комплименты, "как дела?", "ты крутой", "привет", "пока", "спасибо", "хорошего дня"
• complaint    — жалоба, критика, негативная эмоция по отношению к тебе / ответу / системе, "ты тупой", "это ужасно", "очень плохо", "разочарован"
• unknown      — текст, который не удаётся однозначно отнести ни к одной из вышеперечисленных категорий (мусор, случайный набор символов, слишком неоднозначно)

Примеры:

Привет, как дела?                  → small_talk
Какой сегодня курс доллара?        → question
Напиши мне письмо директору        → task
Ты вообще ничего не понимаешь!     → complaint
Спасибо за помощь, ты молодец      → small_talk
Что такое LCEL в LangChain?        → question
Переведи это предложение на немецкий → task
Это худший ответ в моей жизни      → complaint
Пока, до завтра                    → small_talk
asdfghjkl 123 !!! ???              → unknown
Расскажи анекдот                   → task
Почему небо голубое?               → question

Правила:
• Выбирай только одну категорию — ту, которая лучше всего описывает основной замысел сообщения.
• Оцени уверенность реалистично (от 0.0 до 1.0)
• В поле reasoning пиши короткое (1–2 предложения) объяснение своего выбора

{format_instructions}
"""

parser = PydanticOutputParser(pydantic_object=Classification)

model = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", model="qwen2.5-7b-instruct-1m")

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Запрос: {query}")
])

raw_chain = (
    RunnablePassthrough.assign(
        format_instructions=lambda _: parser.get_format_instructions()
    )
    | prompt
    | model
    | parser
)

def classify_with_fallback(query: str) -> Classification:
    try:
        result = raw_chain.invoke({"query": query})
        return result
    except Exception as e:
        return Classification(
            request_type=RequestType.UNKNOWN,
            confidence=0.5,
            reasoning=f"Ошибка парсинга ответа модели: {str(e)[:120]}…"
        )

if __name__ == "__main__":
    tests = [
        "Привет, как дела?",
        "Напиши мне код для сортировки списка",
        "Ты вообще бесполезный бот",
        "Спасибо большое, очень помог!",
        "Что такое LCEL?",
        "Пока-пока",
        "123 !!! ???",
    ]

    for q in tests:
        cls = classify_with_fallback(q)
        print(f"«{q}» → {cls.request_type:12}  {cls.confidence:.2f}  {cls.reasoning}")