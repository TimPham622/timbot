from timbot.cleaning import CleanConfig, clean_text


def test_clean_text_removes_urls_and_mentions():
    assert clean_text("hey <@123456789012345678> look https://example.com") == "hey look"


def test_clean_text_drops_commands():
    assert clean_text("!rank", CleanConfig()) is None
