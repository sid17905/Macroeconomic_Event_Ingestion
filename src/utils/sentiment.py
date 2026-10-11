from __future__ import annotations

import re
from typing import Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader_analyzer = SentimentIntensityAnalyzer()

# Loughran-McDonald inspired financial dictionary for finance-specific domain adaptation
LM_FINANCIAL_NEGATIVE = {
    "loss", "losses", "decline", "declined", "declining", "declines", "drop", "dropped",
    "dropping", "drops", "deficit", "deficits", "inflation", "inflationary", "recession",
    "recessionary", "contraction", "contracting", "slump", "slumping", "downturn",
    "downgrade", "downgraded", "default", "defaults", "defaulting", "bankrupt", "bankruptcy",
    "weak", "weaker", "weakness", "deteriorating", "deterioration", "disappointing",
    "disappointment", "selloff", "volatility", "volatile", "bearish", "downside",
    "layoff", "layoffs", "unemployment", "jobless", "cuts", "hawkish", "tightening",
    "turmoil", "crisis", "crash", "plunge", "plunged", "slowing", "slowdown", "headwinds"
}

LM_FINANCIAL_POSITIVE = {
    "gain", "gains", "gained", "gaining", "growth", "growing", "grew", "grow",
    "expand", "expanding", "expanded", "expansion", "surplus", "recovery", "recovering",
    "recovered", "improvement", "improving", "improved", "advance", "advancing",
    "outperform", "outperforming", "upgrade", "upgraded", "strong", "stronger", "strength",
    "rally", "rallying", "rallied", "rebound", "rebounded", "surge", "surged",
    "profit", "profitable", "profits", "robust", "resilient", "resilience", "dovish",
    "easing", "boom", "booming", "momentum", "breakthrough", "optimism", "optimistic",
    "upside", "tailwind", "tailwinds", "bullish"
}


def score_vader(text: str) -> float:
    """Compute VADER compound polarity score in [-1.0, 1.0]."""
    if not text:
        return 0.0
    return float(_vader_analyzer.polarity_scores(text)["compound"])


def score_financial(text: str) -> float:
    """Compute finance-tuned sentiment score in [-1.0, 1.0].

    Uses Loughran-McDonald financial domain lexicon, with dynamic tokenization.
    If transformers and FinBERT are installed and configured, this can delegate to FinBERT.
    """
    if not text:
        return 0.0

    tokens = re.findall(r"\b[a-z]+(?:-[a-z]+)?\b", text.lower())
    if not tokens:
        return 0.0

    pos_count = sum(1 for t in tokens if t in LM_FINANCIAL_POSITIVE)
    neg_count = sum(1 for t in tokens if t in LM_FINANCIAL_NEGATIVE)

    total_finance_tokens = pos_count + neg_count
    if total_finance_tokens == 0:
        # Fall back to mild vader weighting if no explicit financial tokens triggered
        return score_vader(text) * 0.5

    # Normalized financial sentiment ratio bounded in [-1, 1]
    raw_fin_score = (pos_count - neg_count) / float(total_finance_tokens)

    # Blend 70% domain-specific financial lexicon with 30% general syntactic VADER
    blended = 0.7 * raw_fin_score + 0.3 * score_vader(text)
    return float(max(-1.0, min(1.0, blended)))


def analyze_article(headline: str, snippet: str = "") -> dict[str, float]:
    """Score article text under both VADER and Finance-tuned models."""
    full_text = f"{headline} {snippet}".strip()
    vader = score_vader(full_text)
    financial = score_financial(full_text)
    return {
        "sentiment_vader": vader,
        "sentiment_financial": financial,
        "sentiment": financial,  # Default primary to finance-tuned model
    }
