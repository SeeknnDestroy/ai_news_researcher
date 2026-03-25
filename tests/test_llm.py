from src.domain.contracts import DraftOutline, SummaryPayload
from src.llm import OpenAIConfig, _build_request, _response_incomplete_reason, _text_format


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


def test_build_request_omits_output_cap_but_keeps_draft_outline_estimate():
    config = OpenAIConfig(api_key="test-key", max_output_tokens=2000)
    _, _, payload, estimated_tokens = _build_request(
        config,
        "system",
        "user",
        "draft_outline",
        DraftOutline,
    )

    assert "max_output_tokens" not in payload
    assert estimated_tokens >= 12000


def test_response_incomplete_reason_reads_responses_api_status():
    reason = _response_incomplete_reason(
        {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        }
    )

    assert reason == "max_output_tokens"
