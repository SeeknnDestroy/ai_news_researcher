from src.config import Settings, task_model_routes


def test_settings_default_to_routed_openai_models():
    settings = Settings()

    assert settings.openai_base_url == "https://api.openai.com/v1"
    assert settings.openai_story_card_model == "gpt-5.4-nano"
    assert settings.openai_merge_classifier_model == "gpt-5.4"
    assert settings.openai_judge_model == "gpt-5.4-mini"


def test_task_model_routes_follow_locked_defaults():
    settings = Settings()
    routes = task_model_routes(settings)

    assert routes["story_card_extraction"] == "gpt-5.4-nano"
    assert routes["merge_classifier"] == "gpt-5.4"
    assert routes["theme_assignment"] == "gpt-5.4"
    assert routes["judge_evaluation"] == "gpt-5.4-mini"
    assert routes["repair_planner"] == "gpt-5.4-mini"
    assert routes["story_article"] == "gpt-5.4-nano"
    assert routes["cod_gelisme"] == "gpt-5.4-nano"
