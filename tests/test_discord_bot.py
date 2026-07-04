from timbot.discord_bot import strip_bot_mention


def test_strip_bot_mention_handles_both_discord_mention_forms():
    assert strip_bot_mention("<@123> hello", 123) == "hello"
    assert strip_bot_mention("<@!123> hello", 123) == "hello"
