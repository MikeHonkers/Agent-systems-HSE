import argparse
import sys
from typing import Optional

from core.classifier import classify_with_fallback
from core.qhandlers import request_branch, change_character
from core.memory import MemoryManager
from utils.characters import CHARACTER_PROMPTS


class SmartAssistant:
    def __init__(
        self,
        initial_character: str = "friendly",
        initial_memory_strategy: str = "buffer",
        model_name: Optional[str] = None,
    ):
        self.memory = MemoryManager(
            strategy=initial_memory_strategy,
            max_buffer_size=10,
        )
        self.current_character = initial_character
        change_character(initial_character)

        self.model_name = model_name or "qwen2.5-7b-instruct"

    def process(self, user_input: str) -> str:
        if not user_input.strip():
            return ""

        classification = classify_with_fallback(user_input)
        self.memory.add_user_message(user_input)

        try:
            response_obj = request_branch.invoke({
                "classification": classification,
                "query": user_input,
                "history": self.memory.get_history(),
            })

            answer = response_obj.content

            self.memory.add_assistant_message(answer)

            return (
                f"[{response_obj.request_type.value}] {answer}\n"
                f"confidence: {response_obj.confidence:.2f} | "
                f"tokens: {response_obj.tokens_used}"
            )
        except Exception as e:
            return f"[ERROR] {str(e)}"

    def show_status(self):
        debug = self.memory.get_debug_info()
        print("\nCurrent settings:")
        print(f"  Character   : {self.current_character}")
        print(f"  Strategy    : {self.memory.strategy}")
        print(f"  Messages    : {debug['message_count']}")
        if debug['summary_exists']:
            print(f"  Summary preview: {debug['summary_preview']}")
        print(f"  Model       : {self.model_name}")
        print()

    def run_cli(self):
        print("Smart Assistant CLI")
        print("Type /help for command list")
        print("─" * 60)

        while True:
            try:
                user_input = input("> ").strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    self._handle_command(user_input)
                else:
                    response = self.process(user_input)
                    if response:
                        print(response)
                        print("─" * 60)

            except KeyboardInterrupt:
                sys.exit(0)

            except Exception as e:
                print(f"[CLI Error] {str(e)}")

    def _handle_command(self, cmd: str):
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if command == "/clear":
            self.memory.clear()
            print("Dialog history cleared.\n")

        elif command == "/character":
            name = arg.lower()
            if name in CHARACTER_PROMPTS:
                change_character(name)
                self.current_character = name
                print(f"Changed character: {name}\n")
            else:
                print(f"Available characters: {', '.join(CHARACTER_PROMPTS.keys())}\n")

        elif command == "/memory":
            strategy = arg.lower()
            if strategy in ("buffer", "summary"):
                self.memory = MemoryManager(strategy=strategy, max_buffer_size=5)
                print(f"Changed startegy: {strategy}\n")
            else:
                print("Available strategies: buffer, summary\n")

        elif command == "/status":
            self.show_status()

        elif command == "/help":
            print("List of commands:")
            print("  /clear          — очистить историю диалога")
            print("  /character <name> — сменить характер (friendly, professional, sarcastic, pirate)")
            print("  /memory <type>    — сменить стратегию памяти (buffer, summary)")
            print("  /status           — показать текущие настройки")
            print("  /help             — эта справка")
            print("  /quit             — выход")
            print()

        elif command == "/quit":
            sys.exit(0)

        else:
            print(f"Unknown command: {command}. Type /help to see commands list.\n")


def main():
    parser = argparse.ArgumentParser(description="Smart Assistant CLI")
    parser.add_argument("--character", default="friendly",
                        choices=list(CHARACTER_PROMPTS.keys()),
                        help="Assistant's character")
    parser.add_argument("--memory", default="buffer",
                        choices=["buffer", "summary"],
                        help="Memory strategy")
    parser.add_argument("--model", default=None,
                        help="Assistant's model")

    args = parser.parse_args()

    assistant = SmartAssistant(
        initial_character=args.character,
        initial_memory_strategy=args.memory,
        model_name=args.model,
    )

    assistant.run_cli()


if __name__ == "__main__":
    main()