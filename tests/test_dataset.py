from timbot.dataset import extract_assistant_text, row_to_mlx_text


def test_row_to_mlx_text_creates_text_example():
    assert row_to_mlx_text({"content": "nah im tired lmao"}) == {"text": "nah im tired lmao"}


def test_extract_assistant_text_from_chat_format():
    row = {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "nah"}]}
    assert extract_assistant_text(row) == "nah"
