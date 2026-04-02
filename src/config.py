from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_default_model: str = "gpt-5.4-nano"
    openai_story_card_model: str = "gpt-5.4-nano"
    openai_newsletter_split_model: str = "gpt-5.4-nano"
    openai_merge_classifier_model: str = "gpt-5.4"
    openai_theme_assigner_model: str = "gpt-5.4"
    openai_judge_model: str = "gpt-5.4-mini"
    openai_repair_planner_model: str = "gpt-5.4-mini"
    openai_intro_writer_model: str = "gpt-5.4-mini"
    openai_story_writer_model: str = "gpt-5.4-nano"
    openai_cod_model: str = "gpt-5.4-nano"
    openai_temperature: float = 0.2
    openai_timeout_s: int = 60
    openai_reasoning_effort: str = "none"
    openai_verbosity: str = "low"
    openai_max_output_tokens: int = 2000
    openai_rpm_limit: int = 500
    openai_tpm_limit: int = 200_000
    openai_tpd_limit: int = 2_000_000

    eval_revision_threshold: float = 75.0
    eval_groundedness_threshold: float = 70.0
    eval_coverage_threshold: float = 70.0
    eval_default_trials: int = 2

    max_report_chars: int = 12000
    max_factcheck_items: int = 8

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def task_model_routes(settings_obj: Settings) -> dict[str, str]:
    return {
        "story_card_extraction": settings_obj.openai_story_card_model,
        "newsletter_split": settings_obj.openai_newsletter_split_model,
        "merge_classifier": settings_obj.openai_merge_classifier_model,
        "theme_assignment": settings_obj.openai_theme_assigner_model,
        "judge_evaluation": settings_obj.openai_judge_model,
        "repair_planner": settings_obj.openai_repair_planner_model,
        "intro_writer": settings_obj.openai_intro_writer_model,
        "story_article": settings_obj.openai_story_writer_model,
        "cod_gelisme": settings_obj.openai_cod_model,
    }
