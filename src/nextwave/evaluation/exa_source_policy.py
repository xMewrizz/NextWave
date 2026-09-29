"""Frozen provider-independent source typing for Exa web results."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

EXA_SOURCE_POLICY_VERSION = "exa-source-policy-v1"

_ACADEMIC_HOSTS = (
    "arxiv.org",
    "alphaxiv.org",
    "dl.acm.org",
    "ieeexplore.ieee.org",
    "link.springer.com",
    "mdpi.com",
    "onlinelibrary.wiley.com",
    "opg.optica.org",
    "pith.science",
    "proceedings.mlr.press",
    "pubmed.ncbi.nlm.nih.gov",
    "pubs.rsc.org",
    "researchgate.net",
    "sciencedirect.com",
    "semanticscholar.org",
    "tandfonline.com",
)
_ACADEMIC_HOST_FRAGMENTS = ("iopscience.iop.org",)
_PRESS_RELEASE_HOSTS = (
    "businesswire.com",
    "eurekalert.org",
    "globenewswire.com",
    "prnewswire.com",
)
_COMPANY_HOSTS = (
    "amazon.com",
    "apple.com",
    "arm.com",
    "aws.amazon.com",
    "cloud.google.com",
    "cloudflare.com",
    "developers.googleblog.com",
    "developer.nvidia.com",
    "googleblog.com",
    "huggingface.co",
    "ibm.com",
    "machinelearning.apple.com",
    "meta.ai",
    "microsoft.com",
    "newsroom.arm.com",
    "nvidia.com",
    "openai.com",
    "research.ibm.com",
    "salesforce.com",
)


@dataclass(frozen=True, slots=True)
class ExaSourceDecision:
    eligible_for_industry: bool
    source_type: str
    reason: str


def _matches(host: str, patterns: tuple[str, ...]) -> bool:
    return any(host == item or host.endswith(f".{item}") for item in patterns)


def classify_exa_source(url: str) -> ExaSourceDecision:
    """Classify by stable publisher/path rules, never by candidate or label."""

    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    path = parsed.path.casefold()
    if not host:
        return ExaSourceDecision(False, "other", "invalid_host")
    if host == "exa.ai" and path.startswith("/library/publication"):
        return ExaSourceDecision(False, "scientific_publication", "exa_publication_proxy")
    if _matches(host, _ACADEMIC_HOSTS) or any(
        fragment in host for fragment in _ACADEMIC_HOST_FRAGMENTS
    ):
        return ExaSourceDecision(False, "scientific_publication", "academic_domain")
    if (host == "nature.com" or host.endswith(".nature.com")) and path.startswith(
        "/articles/"
    ):
        return ExaSourceDecision(False, "scientific_publication", "journal_article_path")
    if _matches(host, _PRESS_RELEASE_HOSTS):
        return ExaSourceDecision(True, "press_release", "press_release_domain")
    if _matches(host, _COMPANY_HOSTS):
        return ExaSourceDecision(True, "company_technical", "company_domain")
    if _matches(host, ("linkedin.com", "medium.com", "substack.com")):
        return ExaSourceDecision(True, "social_or_blog", "social_or_blog_domain")
    return ExaSourceDecision(True, "industry_media", "web_news_or_trade_source")
