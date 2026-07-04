from datetime import datetime, timezone

from timbot.discord_bot import build_conversation_log, clean_message_content, strip_bot_mention


def test_strip_bot_mention_handles_both_discord_mention_forms():
    assert strip_bot_mention("<@123> hello", 123) == "hello"
    assert strip_bot_mention("<@!123> hello", 123) == "hello"


def test_clean_message_content_strips_all_user_mentions():
    assert clean_message_content("<@123> hello <@!456>") == "hello"


def test_build_conversation_log_labels_and_orders_messages():
    messages = [
        FakeMessage(3, "why?", False),
        FakeMessage(2, "blue tbh", True),
        FakeMessage(1, "what's your favorite color?", False),
    ]
    channel = FakeChannel(messages)
    log = run_async(build_conversation_log(channel, limit=6))
    assert log == "User: what's your favorite color?\nTim: blue tbh\nUser: why?"


def test_build_conversation_log_adds_current_message_once_if_missing_from_history():
    current = FakeMessage(3, "<@999> why?", False)
    channel = FakeChannel([FakeMessage(2, "blue tbh", True), FakeMessage(1, "favorite color?", False)])
    log = run_async(build_conversation_log(channel, limit=6, current_message=current))
    assert log.endswith("User: why?")
    assert log.count("User: why?") == 1


class FakeAuthor:
    def __init__(self, bot: bool):
        self.bot = bot


class FakeMessage:
    def __init__(self, message_id: int, content: str, bot: bool):
        self.id = message_id
        self.content = content
        self.author = FakeAuthor(bot)
        self.created_at = datetime.fromtimestamp(message_id, tz=timezone.utc)


class FakeChannel:
    def __init__(self, messages):
        self.messages = messages

    def history(self, limit: int):
        return FakeHistory(self.messages[:limit])


class FakeHistory:
    def __init__(self, messages):
        self.messages = messages

    def __aiter__(self):
        self.index = 0
        return self

    async def __anext__(self):
        if self.index >= len(self.messages):
            raise StopAsyncIteration
        message = self.messages[self.index]
        self.index += 1
        return message


def run_async(coro):
    import asyncio

    return asyncio.run(coro)
