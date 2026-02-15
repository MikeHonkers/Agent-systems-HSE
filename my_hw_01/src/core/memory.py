from typing import List, Dict, Optional, Literal
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate

from utils.characters import CHARACTER_PROMPTS
from core.qhandlers import model


class MemoryManager:
    def __init__(
        self,
        strategy: Literal["buffer", "summary"] = "buffer",
        max_buffer_size: int = 10,
        summary_prompt: Optional[str] = None,
    ):
        self.strategy = strategy
        self.max_buffer_size = max_buffer_size
        self.messages: List[BaseMessage] = []
        self.summary: Optional[str] = None

        self.summary_prompt = summary_prompt or """\
            Ты — помощник, который делает краткое и точное содержание диалога.
            Сохрани ключевые факты, имена, предпочтения, важные договорённости.
            Не добавляй свои интерпретации. Пиши нейтрально и кратко.

            Диалог:
            {history}

            Краткое содержание (3–8 предложений):"""

    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content))

    def add_assistant_message(self, content: str):
        self.messages.append(AIMessage(content=content))

    def get_history(self) -> List[BaseMessage]:
        if self.strategy == "buffer":
            return self.messages[-self.max_buffer_size :]

        elif self.strategy == "summary":
            self._update_summary()
            summary_msg = SystemMessage(content=f"Краткое содержание предыдущего диалога:\n{self.summary}")

            return [summary_msg]

        else:
            return self.messages[-self.max_buffer_size :]

    def _update_summary(self):
        recent_history = self.messages.copy()
        if not recent_history:
            return None

        prompt = ChatPromptTemplate.from_template(self.summary_prompt)
        chain = prompt | model

        try:
            summary_text = chain.invoke({"history": "\n".join(m.content for m in recent_history)}).content
            self.summary = summary_text.strip()
        except Exception as e:
            print(f"[Memory] Ошибка при суммаризации: {e}")

    def clear(self):
        self.messages.clear()
        self.summary = None

    def get_debug_info(self) -> Dict:
        return {
            "strategy": self.strategy,
            "message_count": len(self.messages),
            "summary_exists": self.summary is not None,
            "summary_preview": (self.summary or "")[:120] + "..." if self.summary else None,
        }