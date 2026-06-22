import pytest
from layer1.core.test_case import TestCase, TransformedCase
from layer1.strategies.symbolic import (
    PunctuationPerturbation,
    UnicodeVariantSubstitution,
    EmojiInjection,
    SeparatorRestructuring,
    CaseManipulation,
)


class TestPunctuationPerturbation:
    def test_validates_non_empty(self):
        s = PunctuationPerturbation()
        case = TransformedCase(instruction="Hello, world! How are you?")
        assert s.validate(case) is True

    def test_rejects_empty(self):
        s = PunctuationPerturbation()
        case = TransformedCase(instruction="")
        assert s.validate(case) is False

    def test_modifies_punctuation(self):
        s = PunctuationPerturbation(intensity=0.5, seed=42)
        case = TransformedCase(instruction="Hello, world! This is a test. With punctuation?")
        modified, trace = s.apply(case)
        assert modified.instruction != case.instruction
        assert trace.modification_type == "punctuation_perturbation"

    def test_no_punctuation_returns_unchanged(self):
        s = PunctuationPerturbation(seed=42)
        case = TransformedCase(instruction="Hello world this has no punctuation")
        modified, trace = s.apply(case)
        assert modified.instruction == "Hello world this has no punctuation"
        assert trace.token_change_ratio == 0.0


class TestUnicodeVariantSubstitution:
    def test_replaces_chars(self):
        s = UnicodeVariantSubstitution(intensity=0.5, seed=42)
        case = TransformedCase(instruction="Hello computer science")
        modified, trace = s.apply(case)
        assert trace.modification_type == "unicode_variant_substitution"

    def test_no_replaceable_cases(self):
        s = UnicodeVariantSubstitution(seed=42)
        case = TransformedCase(instruction="456789!@#$%^&*()")
        modified, trace = s.apply(case)
        assert trace.token_change_ratio == 0.0


class TestEmojiInjection:
    def test_injects_emojis(self):
        s = EmojiInjection(intensity=0.3, seed=42)
        case = TransformedCase(instruction="This is a long enough sentence to inject emojis into it")
        modified, trace = s.apply(case)
        assert trace.modification_type == "emoji_injection"

    def test_short_text_no_injection(self):
        s = EmojiInjection(seed=42)
        case = TransformedCase(instruction="Hi")
        modified, trace = s.apply(case)
        assert trace.token_change_ratio == 0.0


class TestSeparatorRestructuring:
    def test_restructures_separators(self):
        s = SeparatorRestructuring(intensity=0.5, seed=42)
        case = TransformedCase(instruction="First paragraph. Second paragraph. Third one.\nFourth line here.")
        modified, trace = s.apply(case)
        assert trace.modification_type == "separator_restructuring"

    def test_validates_non_empty(self):
        s = SeparatorRestructuring()
        assert s.validate(TransformedCase(instruction="test")) is True
        assert s.validate(TransformedCase(instruction="")) is False


class TestCaseManipulation:
    def test_manipulates_case(self):
        s = CaseManipulation(intensity=0.5, seed=42)
        case = TransformedCase(instruction="This is a test sentence for case manipulation")
        modified, trace = s.apply(case)
        assert trace.modification_type == "case_manipulation"

    def test_validates_non_empty(self):
        s = CaseManipulation()
        assert s.validate(TransformedCase(instruction="test")) is True
        assert s.validate(TransformedCase(instruction="")) is False
