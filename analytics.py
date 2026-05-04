from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from model_loader import ModelLoader

## use this API Key for testing "Iv23likPAU7X8kn2b3Px".


class ContentAnalytics:
    """Performs sentiment, confidence, extractive summary, and satisfaction flagging."""

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
        "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with",
        "i", "you", "your", "we", "they", "them", "this", "those", "these", "our", "or", "but",
        "if", "then", "than", "so", "because", "about", "into", "out", "up", "down", "can",
        "could", "should", "would", "have", "had", "do", "does", "did", "not", "no", "yes",
        "my", "me", "mine", "their", "his", "her", "hers", "him", "what", "when", "where",
        "why", "how", "which", "who", "whom", "been", "being", "am", "very", "just", "also",
    }

    POSITIVE_KEYWORDS = {
        "happy", "great", "good", "excellent", "thanks", "thank you", "resolved", "satisfied",
        "awesome", "perfect", "love", "helpful", "appreciate", "works", "working", "fixed",
    }

    NEGATIVE_KEYWORDS = {
        "bad", "terrible", "angry", "upset", "unhappy", "issue", "problem", "complaint",
        "frustrated", "not working", "failed", "delay", "poor", "worst", "cancel", "refund",
    }

    def __init__(self, model_loader: ModelLoader) -> None:
        self.sentiment_analyzer = model_loader.load_sentiment_analyzer()

    def analyze_text(self, text: str) -> dict[str, Any]:
        cleaned_text = (text or "").strip()
        sentiment_scores = self.sentiment_analyzer.polarity_scores(cleaned_text)
        sentiment_label = self._map_sentiment(sentiment_scores["compound"])
        sentiment_confidence = self._calculate_confidence(sentiment_scores)
        summary = self._extractive_summary(cleaned_text)
        satisfaction_flag = self._infer_customer_satisfaction(cleaned_text, sentiment_scores["compound"])

        return {
            "sentiment": sentiment_label,
            "sentiment_confidence": sentiment_confidence,
            "summary": summary,
            "customer_satisfaction": satisfaction_flag,
            "sentiment_scores": sentiment_scores,
        }

    def _map_sentiment(self, compound: float) -> str:
        if compound >= 0.05:
            return "positive"
        if compound <= -0.05:
            return "negative"
        return "neutral"

    def _calculate_confidence(self, scores: dict[str, float]) -> float:
        compound = abs(scores["compound"])
        margin = abs(scores["pos"] - scores["neg"])
        confidence = max(compound, margin)
        return round(min(1.0, confidence), 4)

    def _extractive_summary(self, text: str, max_sentences: int = 3) -> str:
        sentences = self._split_sentences(text)
        if not sentences:
            return ""
        if len(sentences) <= max_sentences:
            return " ".join(sentences)

        words = self._tokenize_words(text)
        words = [word for word in words if word not in self.STOPWORDS]
        if not words:
            return " ".join(sentences[:max_sentences])

        frequencies = Counter(words)
        sentence_scores: list[tuple[int, float]] = []

        for index, sentence in enumerate(sentences):
            sentence_words = [w for w in self._tokenize_words(sentence) if w not in self.STOPWORDS]
            if not sentence_words:
                continue
            score = sum(frequencies[word] for word in sentence_words) / len(sentence_words)
            sentence_scores.append((index, score))

        if not sentence_scores:
            return " ".join(sentences[:max_sentences])

        selected_indexes = sorted(
            idx for idx, _ in sorted(sentence_scores, key=lambda item: item[1], reverse=True)[:max_sentences]
        )
        return " ".join(sentences[idx] for idx in selected_indexes)

    def _infer_customer_satisfaction(self, text: str, compound_score: float) -> bool:
        normalized = text.lower()
        positive_hits = sum(1 for word in self.POSITIVE_KEYWORDS if word in normalized)
        negative_hits = sum(1 for word in self.NEGATIVE_KEYWORDS if word in normalized)

        if negative_hits > positive_hits:
            return False
        if compound_score >= 0.2 or positive_hits > negative_hits:
            return True
        return False

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        candidates = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [sentence.strip() for sentence in candidates if sentence.strip()]
        return sentences

    @staticmethod
    def _tokenize_words(text: str) -> list[str]:
        return re.findall(r"\b[a-zA-Z']+\b", text.lower())
