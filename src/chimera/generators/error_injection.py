"""Error injection utilities for CHIMERA benchmark.

This module provides tools for systematically injecting various types of
errors into correct responses. The errors are designed to test a model's
ability to detect mistakes in reasoning, facts, and computations.

Error Types:
- FACTUAL: Incorrect facts or data
- COMPUTATIONAL: Mathematical errors
- LOGICAL: Reasoning mistakes or contradictions
- OMISSION: Missing critical information
- HALLUCINATION: Made-up details or fabrications
- TEMPORAL: Incorrect dates or time references
- CAUSAL: Wrong cause-effect relationships
- MAGNITUDE: Order of magnitude errors

Design Principles:
1. Errors should be realistic (the kind a model might actually make)
2. Errors should be unambiguously wrong (verifiable)
3. Error severity should be controllable
4. Injection should preserve overall response coherence
"""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorType(str, Enum):
    """Types of errors that can be injected."""

    FACTUAL = "factual"
    COMPUTATIONAL = "computational"
    LOGICAL = "logical"
    OMISSION = "omission"
    HALLUCINATION = "hallucination"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    MAGNITUDE = "magnitude"


@dataclass
class InjectedError:
    """Represents an error that was injected into a response.

    Attributes:
        error_type: Type of error injected
        location: The erroneous text in the response
        original: The original correct text
        description: Human-readable description of the error
        correction: How to fix the error
        severity: Severity level (subtle, moderate, obvious)
    """

    error_type: ErrorType
    location: str
    original: str
    description: str
    correction: str
    severity: str = "moderate"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InjectionConfig:
    """Configuration for error injection.

    Attributes:
        error_type: Type of error to inject
        severity: Error severity (subtle, moderate, obvious)
        num_errors: Number of errors to inject
        preserve_structure: Whether to preserve response structure
    """

    error_type: ErrorType
    severity: str = "moderate"
    num_errors: int = 1
    preserve_structure: bool = True


class BaseErrorInjector(ABC):
    """Abstract base class for error injectors."""

    @abstractmethod
    def can_inject(self, question: str, response: str) -> bool:
        """Check if this injector can inject errors into the response."""
        pass

    @abstractmethod
    def inject(
        self,
        question: str,
        response: str,
        severity: str = "moderate",
    ) -> tuple[str, InjectedError | None]:
        """Inject an error into the response.

        Args:
            question: The original question
            response: The correct response
            severity: Error severity level

        Returns:
            Tuple of (modified_response, error_info)
        """
        pass


class FactualErrorInjector(BaseErrorInjector):
    """Injects factual errors (incorrect facts, names, places, etc.)."""

    # Common factual substitutions
    SUBSTITUTIONS = {
        # Countries and capitals
        "Paris": ["London", "Berlin", "Madrid"],
        "France": ["Germany", "Spain", "Italy"],
        "London": ["Paris", "Dublin", "Edinburgh"],
        "Tokyo": ["Beijing", "Seoul", "Shanghai"],
        "Washington": ["Philadelphia", "Boston", "New York"],
        # People
        "Einstein": ["Newton", "Galileo", "Bohr"],
        "Newton": ["Einstein", "Galileo", "Kepler"],
        "Shakespeare": ["Dickens", "Milton", "Chaucer"],
        "Darwin": ["Mendel", "Lamarck", "Wallace"],
        "Lincoln": ["Jefferson", "Adams", "Franklin"],
        # Scientific terms
        "oxygen": ["nitrogen", "carbon", "hydrogen"],
        "hydrogen": ["helium", "oxygen", "nitrogen"],
        "electron": ["proton", "neutron", "photon"],
        "proton": ["electron", "neutron", "quark"],
        # Elements and compounds
        "H₂O": ["H₂O₂", "CO₂", "NaCl"],
        "CO₂": ["CO", "O₂", "H₂O"],
        # Numbers and measurements
        "8": ["7", "9", "10"],
        "1945": ["1944", "1946", "1943"],
        "1789": ["1776", "1799", "1792"],
    }

    def can_inject(self, question: str, response: str) -> bool:
        """Check if factual error can be injected."""
        text = question + " " + response
        return any(word in text for word in self.SUBSTITUTIONS)

    def inject(
        self,
        question: str,
        response: str,
        severity: str = "moderate",
    ) -> tuple[str, InjectedError | None]:
        """Inject a factual error."""
        # Find injectable words
        injectable = []
        for word, replacements in self.SUBSTITUTIONS.items():
            if word in response:
                injectable.append((word, replacements))

        if not injectable:
            return response, None

        # Select word to replace
        original, replacements = random.choice(injectable)  # nosec B311

        # Select replacement based on severity
        if severity == "obvious":
            # Pick obviously wrong replacement
            replacement = replacements[-1] if len(replacements) > 1 else replacements[0]
        elif severity == "subtle":
            # Pick similar-sounding or plausible replacement
            replacement = replacements[0]
        else:
            replacement = random.choice(replacements)  # nosec B311

        # Inject error
        modified = response.replace(original, replacement, 1)

        error = InjectedError(
            error_type=ErrorType.FACTUAL,
            location=replacement,
            original=original,
            description=f"Incorrectly states '{replacement}' instead of '{original}'",
            correction=f"Should be '{original}', not '{replacement}'",
            severity=severity,
        )

        return modified, error


class ComputationalErrorInjector(BaseErrorInjector):
    """Injects computational/mathematical errors."""

    # Patterns to find numbers in mathematical context
    NUMBER_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\b")
    EQUATION_PATTERN = re.compile(r"=\s*(\d+(?:\.\d+)?)")

    def can_inject(self, question: str, response: str) -> bool:
        """Check if computational error can be injected."""
        return bool(
            self.EQUATION_PATTERN.search(response)
            or (
                self.NUMBER_PATTERN.search(response)
                and any(
                    word in response.lower()
                    for word in ["equals", "is", "result", "answer", "sum", "total", "="]
                )
            )
        )

    def inject(
        self,
        question: str,
        response: str,
        severity: str = "moderate",
    ) -> tuple[str, InjectedError | None]:
        """Inject a computational error."""
        # Find numbers after equals signs
        matches = list(self.EQUATION_PATTERN.finditer(response))

        if not matches:
            # Find standalone numbers
            matches = list(self.NUMBER_PATTERN.finditer(response))

        if not matches:
            return response, None

        # Select a match to modify
        match = random.choice(matches)  # nosec B311
        original_num = match.group(1)

        try:
            num = float(original_num)
        except ValueError:
            return response, None

        # Generate wrong answer based on severity
        if severity == "obvious":
            # Major error (factor of 10 or more)
            wrong_num = num * random.choice([10, 0.1, 100])  # nosec B311
        elif severity == "subtle":
            # Off by one or small percentage
            wrong_num = num + random.choice([1, -1, 0.1, -0.1])  # nosec B311
        else:
            # Moderate error
            wrong_num = num + num * random.uniform(-0.2, 0.3)  # nosec B311

        # Format the wrong number
        wrong_str = str(int(wrong_num)) if original_num.isdigit() else f"{wrong_num:.2f}"

        # Replace in response
        start, end = match.start(1), match.end(1)
        modified = response[:start] + wrong_str + response[end:]

        error = InjectedError(
            error_type=ErrorType.COMPUTATIONAL,
            location=wrong_str,
            original=original_num,
            description=f"Computational error: {wrong_str} should be {original_num}",
            correction=f"The correct value is {original_num}",
            severity=severity,
        )

        return modified, error


class LogicalErrorInjector(BaseErrorInjector):
    """Injects logical errors (contradictions, invalid reasoning)."""

    # Logical connectors to flip
    LOGICAL_FLIPS = {
        "therefore": "however",
        "because": "despite",
        "since": "although",
        "thus": "nevertheless",
        "so": "but",
        "follows that": "doesn't follow that",
        "we can conclude": "we cannot conclude",
        "implies": "does not imply",
        "must be": "cannot be",
        "all": "some",
        "always": "sometimes",
        "never": "occasionally",
        "every": "few",
    }

    # Negation patterns
    NEGATION_PATTERNS = [
        (r"\bis\b", "is not"),
        (r"\bcan\b", "cannot"),
        (r"\bwill\b", "will not"),
        (r"\bdoes\b", "does not"),
        (r"\bhave\b", "do not have"),
        (r"\bare\b", "are not"),
    ]

    def can_inject(self, question: str, response: str) -> bool:
        """Check if logical error can be injected."""
        text = response.lower()
        has_connector = any(conn in text for conn in self.LOGICAL_FLIPS)
        has_negatable = any(re.search(pat, text) for pat, _ in self.NEGATION_PATTERNS)
        return has_connector or has_negatable

    def inject(
        self,
        question: str,
        response: str,
        severity: str = "moderate",
    ) -> tuple[str, InjectedError | None]:
        """Inject a logical error."""
        modified = response
        original_phrase = ""
        replacement_phrase = ""

        # Try flipping logical connectors
        for original, replacement in self.LOGICAL_FLIPS.items():
            if original in response.lower():
                # Case-insensitive replacement
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                match = pattern.search(response)
                if match:
                    original_phrase = match.group()
                    # Preserve case
                    if original_phrase[0].isupper():
                        replacement_phrase = replacement.capitalize()
                    else:
                        replacement_phrase = replacement
                    modified = pattern.sub(replacement_phrase, response, count=1)
                    break

        if modified == response:
            # Try negation
            for neg_pattern, neg_replacement in self.NEGATION_PATTERNS:
                match = re.search(neg_pattern, response, re.IGNORECASE)
                if match:
                    original_phrase = match.group()
                    replacement_phrase = neg_replacement
                    modified = re.sub(neg_pattern, neg_replacement, response, count=1)
                    break

        if modified == response:
            return response, None

        error = InjectedError(
            error_type=ErrorType.LOGICAL,
            location=replacement_phrase,
            original=original_phrase,
            description=f"Logical error: '{replacement_phrase}' contradicts the correct reasoning",
            correction=f"Should use '{original_phrase}' for valid logic",
            severity=severity,
        )

        return modified, error


class TemporalErrorInjector(BaseErrorInjector):
    """Injects temporal errors (wrong dates, years, sequences)."""

    YEAR_PATTERN = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
    DATE_PATTERN = re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b",
        re.IGNORECASE,
    )

    MONTH_SHIFTS = {
        "January": "March",
        "February": "April",
        "March": "May",
        "April": "June",
        "May": "July",
        "June": "August",
        "July": "September",
        "August": "October",
        "September": "November",
        "October": "December",
        "November": "January",
        "December": "February",
    }

    def can_inject(self, question: str, response: str) -> bool:
        """Check if temporal error can be injected."""
        return bool(self.YEAR_PATTERN.search(response) or self.DATE_PATTERN.search(response))

    def inject(
        self,
        question: str,
        response: str,
        severity: str = "moderate",
    ) -> tuple[str, InjectedError | None]:
        """Inject a temporal error."""
        # Try year modification first
        year_matches = list(self.YEAR_PATTERN.finditer(response))

        if year_matches:
            match = random.choice(year_matches)  # nosec B311
            original_year = match.group(1)
            year = int(original_year)

            if severity == "obvious":
                wrong_year = year + random.choice([100, -100, 50, -50])  # nosec B311
            elif severity == "subtle":
                wrong_year = year + random.choice([1, -1])  # nosec B311
            else:
                wrong_year = year + random.choice([5, -5, 10, -10])  # nosec B311

            wrong_str = str(wrong_year)
            start, end = match.start(1), match.end(1)
            modified = response[:start] + wrong_str + response[end:]

            error = InjectedError(
                error_type=ErrorType.TEMPORAL,
                location=wrong_str,
                original=original_year,
                description=f"Incorrect year: {wrong_str} should be {original_year}",
                correction=f"The correct year is {original_year}",
                severity=severity,
            )

            return modified, error

        # Try month modification
        month_match = self.DATE_PATTERN.search(response)
        if month_match:
            original_month = month_match.group(1)
            # Normalize case for lookup
            month_key = original_month.capitalize()
            wrong_month = self.MONTH_SHIFTS.get(month_key, "June")

            # Preserve original case
            if original_month.isupper():
                wrong_month = wrong_month.upper()
            elif original_month.islower():
                wrong_month = wrong_month.lower()

            modified = response.replace(original_month, wrong_month, 1)

            error = InjectedError(
                error_type=ErrorType.TEMPORAL,
                location=wrong_month,
                original=original_month,
                description=f"Incorrect month: {wrong_month} should be {original_month}",
                correction=f"The correct month is {original_month}",
                severity=severity,
            )

            return modified, error

        return response, None


class MagnitudeErrorInjector(BaseErrorInjector):
    """Injects magnitude errors (order of magnitude mistakes)."""

    NUMBER_WITH_UNIT = re.compile(
        r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(meters?|kilometers?|miles?|feet|"
        r"grams?|kilograms?|pounds?|seconds?|minutes?|hours?|years?|"
        r"dollars?|euros?|percent|%|million|billion|thousand)\b",
        re.IGNORECASE,
    )

    MAGNITUDE_MULTIPLIERS = {
        "obvious": [1000, 0.001, 1000000],
        "moderate": [10, 0.1, 100],
        "subtle": [2, 0.5, 3],
    }

    def can_inject(self, question: str, response: str) -> bool:
        """Check if magnitude error can be injected."""
        return bool(self.NUMBER_WITH_UNIT.search(response))

    def inject(
        self,
        question: str,
        response: str,
        severity: str = "moderate",
    ) -> tuple[str, InjectedError | None]:
        """Inject a magnitude error."""
        matches = list(self.NUMBER_WITH_UNIT.finditer(response))

        if not matches:
            return response, None

        match = random.choice(matches)  # nosec B311
        original_num = match.group(1).replace(",", "")
        unit = match.group(2)

        try:
            num = float(original_num)
        except ValueError:
            return response, None

        multiplier = random.choice(self.MAGNITUDE_MULTIPLIERS.get(severity, [10]))  # nosec B311
        wrong_num = num * multiplier

        # Format with commas for large numbers
        if wrong_num >= 1000:
            wrong_str = f"{wrong_num:,.0f}"
        elif wrong_num < 1:
            wrong_str = f"{wrong_num:.4f}"
        else:
            wrong_str = f"{wrong_num:.2f}".rstrip("0").rstrip(".")

        original_full = match.group(0)
        wrong_full = f"{wrong_str} {unit}"

        modified = response.replace(original_full, wrong_full, 1)

        error = InjectedError(
            error_type=ErrorType.MAGNITUDE,
            location=wrong_full,
            original=original_full,
            description=f"Order of magnitude error: {wrong_full} is off by ~{multiplier}x",
            correction=f"Should be {original_full}",
            severity=severity,
        )

        return modified, error


class HallucinationInjector(BaseErrorInjector):
    """Injects hallucinated (made-up) details."""

    # Templates for hallucinated additions
    HALLUCINATION_TEMPLATES = {
        "attribution": [
            " (as first noted by Dr. {name} in {year})",
            ", according to the {name} study of {year}",
            ", a finding published in the {journal} journal",
        ],
        "detail": [
            " (specifically, {detail})",
            ", particularly in the {region} region",
            ", especially when considering {factor}",
        ],
        "statistic": [
            " (approximately {percent}% of cases)",
            ", with a {percent}% success rate",
            ", affecting roughly {number} million people",
        ],
    }

    FAKE_NAMES = ["Hendricks", "Kowalski", "Tanaka", "Mueller", "Peterson"]
    FAKE_JOURNALS = ["Nature Reviews", "Global Science", "International Research", "World Studies"]
    FAKE_REGIONS = ["northern", "southern", "eastern", "western", "central"]
    FAKE_FACTORS = ["environmental conditions", "temporal variations", "systemic factors"]

    def can_inject(self, question: str, response: str) -> bool:
        """Hallucinations can be added to almost any response."""
        return len(response) > 20

    def inject(
        self,
        question: str,
        response: str,
        severity: str = "moderate",
    ) -> tuple[str, InjectedError | None]:
        """Inject a hallucination."""
        # Find sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", response)

        if not sentences:
            return response, None

        # Select a sentence to modify
        target_idx = random.randint(0, len(sentences) - 1)  # nosec B311
        target_sentence = sentences[target_idx]

        # Choose hallucination type based on severity
        if severity == "obvious":
            hall_type = "statistic"  # Made-up statistics are more obvious
        elif severity == "subtle":
            hall_type = "detail"  # Extra details are harder to verify
        else:
            hall_type = random.choice(["attribution", "detail", "statistic"])  # nosec B311

        template = random.choice(self.HALLUCINATION_TEMPLATES[hall_type])  # nosec B311

        # Fill in template
        hallucination = template.format(
            name=random.choice(self.FAKE_NAMES),  # nosec B311
            year=random.randint(1990, 2023),  # nosec B311
            journal=random.choice(self.FAKE_JOURNALS),  # nosec B311
            region=random.choice(self.FAKE_REGIONS),  # nosec B311
            factor=random.choice(self.FAKE_FACTORS),  # nosec B311
            detail="the secondary mechanism",
            percent=random.randint(15, 85),  # nosec B311
            number=random.randint(1, 100),  # nosec B311
        )

        # Insert hallucination before period
        if target_sentence.endswith("."):
            modified_sentence = target_sentence[:-1] + hallucination + "."
        else:
            modified_sentence = target_sentence + hallucination

        sentences[target_idx] = modified_sentence
        modified = " ".join(sentences)

        error = InjectedError(
            error_type=ErrorType.HALLUCINATION,
            location=hallucination.strip(),
            original="",
            description=f"Hallucinated detail: '{hallucination.strip()}' is fabricated",
            correction="Remove the fabricated information",
            severity=severity,
        )

        return modified, error


class ErrorInjector:
    """Main error injector that coordinates different error types.

    This class provides a unified interface for injecting various types
    of errors into responses.

    Example:
        >>> injector = ErrorInjector()
        >>> config = InjectionConfig(
        ...     error_type=ErrorType.FACTUAL,
        ...     severity="moderate",
        ... )
        >>> modified, errors = injector.inject(
        ...     question="What is the capital of France?",
        ...     response="The capital of France is Paris.",
        ...     config=config,
        ... )
    """

    def __init__(self) -> None:
        """Initialize the error injector with all injector types."""
        self._injectors: dict[ErrorType, BaseErrorInjector] = {
            ErrorType.FACTUAL: FactualErrorInjector(),
            ErrorType.COMPUTATIONAL: ComputationalErrorInjector(),
            ErrorType.LOGICAL: LogicalErrorInjector(),
            ErrorType.TEMPORAL: TemporalErrorInjector(),
            ErrorType.MAGNITUDE: MagnitudeErrorInjector(),
            ErrorType.HALLUCINATION: HallucinationInjector(),
        }

    def register_injector(
        self,
        error_type: ErrorType,
        injector: BaseErrorInjector,
    ) -> None:
        """Register a custom error injector.

        Args:
            error_type: Type of error this injector handles
            injector: The injector instance
        """
        self._injectors[error_type] = injector

    def can_inject(
        self,
        error_type: ErrorType,
        question: str,
        response: str,
    ) -> bool:
        """Check if a specific error type can be injected.

        Args:
            error_type: Type of error to check
            question: The question
            response: The response

        Returns:
            True if error can be injected
        """
        injector = self._injectors.get(error_type)
        if not injector:
            return False
        return injector.can_inject(question, response)

    def get_injectable_types(
        self,
        question: str,
        response: str,
    ) -> list[ErrorType]:
        """Get all error types that can be injected.

        Args:
            question: The question
            response: The response

        Returns:
            List of injectable error types
        """
        return [
            error_type
            for error_type, injector in self._injectors.items()
            if injector.can_inject(question, response)
        ]

    def inject(
        self,
        question: str,
        response: str,
        config: InjectionConfig,
    ) -> tuple[str, list[InjectedError]]:
        """Inject error(s) into a response.

        Args:
            question: The original question
            response: The correct response
            config: Injection configuration

        Returns:
            Tuple of (modified_response, list_of_errors)
        """
        injector = self._injectors.get(config.error_type)
        if not injector:
            return response, []

        if not injector.can_inject(question, response):
            # Try to find an alternative injector
            alternatives = self.get_injectable_types(question, response)
            if alternatives:
                injector = self._injectors[alternatives[0]]
            else:
                return response, []

        errors: list[InjectedError] = []
        modified = response

        for _ in range(config.num_errors):
            new_modified, error = injector.inject(
                question,
                modified,
                config.severity,
            )

            if error:
                modified = new_modified
                errors.append(error)

        return modified, errors

    def inject_random(
        self,
        question: str,
        response: str,
        severity: str = "moderate",
        num_errors: int = 1,
    ) -> tuple[str, list[InjectedError]]:
        """Inject random error type(s) into a response.

        Args:
            question: The original question
            response: The correct response
            severity: Error severity level
            num_errors: Number of errors to inject

        Returns:
            Tuple of (modified_response, list_of_errors)
        """
        injectable = self.get_injectable_types(question, response)

        if not injectable:
            return response, []

        error_type = random.choice(injectable)  # nosec B311
        config = InjectionConfig(
            error_type=error_type,
            severity=severity,
            num_errors=num_errors,
        )

        return self.inject(question, response, config)
