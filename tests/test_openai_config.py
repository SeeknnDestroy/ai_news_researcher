from src.config import Settings


def test_settings_default_to_openai_gpt_5_4_nano():
    settings = Settings()

    assert settings.openai_base_url == "https://api.openai.com/v1"
    assert settings.openai_model == "gpt-5.4-nano"
