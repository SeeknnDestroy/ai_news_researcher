from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SummaryPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    source_name: str = ""
    summary_tr: str
    why_it_matters_tr: str
    tags: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    @field_validator("title", "source_name", "summary_tr", "why_it_matters_tr", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("summary_tr", "why_it_matters_tr")
    @classmethod
    def _require_text(cls, value: str) -> str:
        if not value:
            raise ValueError("required text field is empty")
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            text = str(item).strip().lower()
            if not text:
                continue
            cleaned.append(text[:30])
        return cleaned[:5]

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: object) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.0
        return max(0.0, min(1.0, confidence))


class NewsletterSplitItemPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    start_marker: str = ""
    end_marker: str = ""
    url: str = ""

    @field_validator("title", "start_marker", "end_marker", "url", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> str:
        return str(value or "").strip()


class NewsletterSplitPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[NewsletterSplitItemPayload] = Field(default_factory=list)


class DraftOutlineArticle(BaseModel):
    model_config = ConfigDict(extra="ignore")

    heading: str
    news_urls_included: list[str] = Field(default_factory=list)
    content_plan: str = ""

    @field_validator("heading", "content_plan", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("heading")
    @classmethod
    def _require_heading(cls, value: str) -> str:
        if not value:
            raise ValueError("heading is required")
        return value

    @field_validator("news_urls_included", mode="before")
    @classmethod
    def _normalize_urls(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class DraftOutlineTheme(BaseModel):
    model_config = ConfigDict(extra="ignore")

    theme_name: str
    theme_commentary: str = ""
    articles: list[DraftOutlineArticle] = Field(default_factory=list)

    @field_validator("theme_name", "theme_commentary", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("theme_name")
    @classmethod
    def _require_theme_name(cls, value: str) -> str:
        if not value:
            raise ValueError("theme_name is required")
        return value


class DraftOutline(BaseModel):
    model_config = ConfigDict(extra="ignore")

    report_title: str
    introduction_commentary: str = ""
    themes: list[DraftOutlineTheme] = Field(default_factory=list)

    @field_validator("report_title", "introduction_commentary", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("report_title")
    @classmethod
    def _require_report_title(cls, value: str) -> str:
        if not value:
            raise ValueError("report_title is required")
        return value


class JudgeEvaluation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    critique: str = ""
    specific_fixes_required: list[str] = Field(default_factory=list)
    passes_criteria: bool = False

    @field_validator("critique", mode="before")
    @classmethod
    def _normalize_critique(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("specific_fixes_required", mode="before")
    @classmethod
    def _normalize_fixes(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]
