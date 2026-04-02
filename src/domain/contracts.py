from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class StoryCardPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    story_title_tr: str
    story_type: str
    key_facts: list[str] = Field(default_factory=list)
    must_keep_entities: list[str] = Field(default_factory=list)
    must_keep_facts: list[str] = Field(default_factory=list)
    why_it_matters_tr: str
    technical_relevance: float = 0.0
    strategic_relevance: float = 0.0
    confidence: float = 0.0

    @field_validator("story_title_tr", "story_type", "why_it_matters_tr", mode="before")
    @classmethod
    def _normalize_story_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("story_title_tr", "story_type", "why_it_matters_tr")
    @classmethod
    def _require_story_text(cls, value: str) -> str:
        if not value:
            raise ValueError("required text field is empty")
        return value

    @field_validator("key_facts", "must_keep_entities", "must_keep_facts", mode="before")
    @classmethod
    def _normalize_string_list(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        normalized_values: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            normalized_key = text.casefold()
            if normalized_key in seen:
                continue
            seen.add(normalized_key)
            normalized_values.append(text)
        return normalized_values[:8]

    @field_validator("technical_relevance", "strategic_relevance", "confidence", mode="before")
    @classmethod
    def _normalize_score(cls, value: object) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = 0.0
        return max(0.0, min(1.0, score))


class MergeDecisionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    decision: Literal["same_story", "same_event_supporting", "related_but_separate", "unrelated"]
    rationale: str = ""

    @field_validator("rationale", mode="before")
    @classmethod
    def _normalize_rationale(cls, value: object) -> str:
        return str(value or "").strip()


class MergePlanItemPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    primary_url: str
    supporting_url: str
    decision: Literal["same_story", "same_event_supporting"]
    rationale: str = ""

    @field_validator("primary_url", "supporting_url", "rationale", mode="before")
    @classmethod
    def _normalize_merge_item_text(cls, value: object) -> str:
        return str(value or "").strip()

    @model_validator(mode="after")
    def _validate_pair_urls(self):
        if not self.primary_url:
            raise ValueError("primary_url is required")
        if not self.supporting_url:
            raise ValueError("supporting_url is required")
        if self.primary_url == self.supporting_url:
            raise ValueError("supporting_url must differ from primary_url")
        return self


class MergePlanPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    merges: list[MergePlanItemPayload] = Field(default_factory=list)


class ThemeAssignmentThemePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    theme_name: str
    theme_commentary: str = ""
    story_unit_ids: list[str] = Field(default_factory=list)

    @field_validator("theme_name", "theme_commentary", mode="before")
    @classmethod
    def _normalize_theme_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("theme_name")
    @classmethod
    def _require_theme_name(cls, value: str) -> str:
        if not value:
            raise ValueError("theme_name is required")
        return value

    @field_validator("story_unit_ids", mode="before")
    @classmethod
    def _normalize_story_unit_ids(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class ThemeAssignmentPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    report_title: str
    introduction_signal: str = ""
    themes: list[ThemeAssignmentThemePayload] = Field(default_factory=list)

    @field_validator("report_title", "introduction_signal", mode="before")
    @classmethod
    def _normalize_plan_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("report_title")
    @classmethod
    def _require_report_title(cls, value: str) -> str:
        if not value:
            raise ValueError("report_title is required")
        return value


class RepairOperationPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    operation: Literal[
        "assign_missing_story_unit",
        "move_story_unit",
        "rename_theme",
        "reorder_story_units",
        "retitle_story_unit",
        "set_primary_url",
        "split_story_unit",
    ]
    story_unit_id: str = ""
    theme_name: str = ""
    target_theme_name: str = ""
    ordered_story_unit_ids: list[str] = Field(default_factory=list)
    new_value: str = ""
    reason: str = ""

    @field_validator(
        "story_unit_id", "theme_name", "target_theme_name", "new_value", "reason", mode="before"
    )
    @classmethod
    def _normalize_operation_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("ordered_story_unit_ids", mode="before")
    @classmethod
    def _normalize_ordered_story_unit_ids(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class RepairPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    critique: str = ""
    operations: list[RepairOperationPayload] = Field(default_factory=list)

    @field_validator("critique", mode="before")
    @classmethod
    def _normalize_repair_critique(cls, value: object) -> str:
        return str(value or "").strip()


class DraftOutlineArticle(BaseModel):
    model_config = ConfigDict(extra="ignore")

    heading: str
    primary_url: str
    news_urls_included: list[str] = Field(default_factory=list)
    content_plan: str = ""

    @field_validator("heading", "primary_url", "content_plan", mode="before")
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

    @field_validator("primary_url")
    @classmethod
    def _require_primary_url(cls, value: str) -> str:
        if not value:
            raise ValueError("primary_url is required")
        return value

    @field_validator("news_urls_included")
    @classmethod
    def _require_limited_urls(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("news_urls_included must include at least one URL")
        if len(value) > 2:
            raise ValueError("news_urls_included may contain at most two URLs")
        return value

    @model_validator(mode="after")
    def _validate_primary_url_membership(self):
        if self.primary_url not in self.news_urls_included:
            raise ValueError("primary_url must be one of news_urls_included")
        return self


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


class FinalReportArticlePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gelisme: str
    neden_onemli: str

    @field_validator("gelisme", "neden_onemli", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("gelisme", "neden_onemli")
    @classmethod
    def _require_text(cls, value: str) -> str:
        if not value:
            raise ValueError("required text field is empty")
        return value


class DenseGelismePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    missing_entities: list[str] = Field(default_factory=list)
    gelisme: str

    @field_validator("missing_entities", mode="before")
    @classmethod
    def _normalize_missing_entities(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()][:2]

    @field_validator("gelisme", mode="before")
    @classmethod
    def _normalize_gelisme(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("gelisme")
    @classmethod
    def _require_gelisme(cls, value: str) -> str:
        if not value:
            raise ValueError("required text field is empty")
        return value


class IntroPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    introduction_commentary: str

    @field_validator("introduction_commentary", mode="before")
    @classmethod
    def _normalize_intro(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("introduction_commentary")
    @classmethod
    def _require_intro(cls, value: str) -> str:
        if not value:
            raise ValueError("required text field is empty")
        return value
