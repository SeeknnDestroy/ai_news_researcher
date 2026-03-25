from src.domain.contracts import DraftOutline, SummaryPayload
from src.llm import _text_format


def test_text_format_closes_summary_schema_for_responses_api():
    text_format = _text_format("article_summary", SummaryPayload)
    schema = text_format["schema"]

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_text_format_closes_nested_object_schemas_for_responses_api():
    text_format = _text_format("draft_outline", DraftOutline)
    schema = text_format["schema"]
    defs = schema["$defs"]

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert defs["DraftOutlineTheme"]["additionalProperties"] is False
    assert set(defs["DraftOutlineTheme"]["required"]) == set(defs["DraftOutlineTheme"]["properties"])
    assert defs["DraftOutlineArticle"]["additionalProperties"] is False
    assert set(defs["DraftOutlineArticle"]["required"]) == set(defs["DraftOutlineArticle"]["properties"])
