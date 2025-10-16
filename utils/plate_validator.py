"""
Indonesian License Plate Validator
Validates plate text against Indonesian plate number formats
"""

import re
import logging
from typing import Optional

# Indonesian plate patterns
INDONESIAN_PLATE_PATTERNS = [
    r'^[A-Z]{1,2}\d{3,4}[A-Z]{1,3}$',  # Standard: B1234ABC, D5678XYZ
    r'^[A-Z]\d{3}[A-Z]{2,3}$',         # Short format: B123AB
    r'^[A-Z]{2}\d{3}[A-Z]{2}$',        # Regional: AB123CD
]

# Known Indonesian regional codes (first 1-2 letters)
VALID_REGIONAL_CODES = [
    'A', 'AA', 'AB', 'AD', 'AE', 'AG',  # Banten
    'B', 'BB', 'BD', 'BE', 'BG', 'BH', 'BK', 'BL', 'BM', 'BN', 'BP',  # Jakarta & surrounding
    'D', 'DA', 'DB', 'DC', 'DD', 'DE', 'DH', 'DK', 'DM', 'DN', 'DP', 'DS', 'DT',  # West Java
    'E', 'EA', 'EB', 'ED', 'EG', 'EH', 'EK',  # Cirebon
    'F', 'FA', 'FB', 'FD', 'FH',  # Bogor
    'G', 'H', 'K', 'KA', 'KB', 'KH', 'KT',  # Central Java
    'L', 'LA', 'LB', 'LD', 'LH', 'LK', 'LM', 'LN', 'LP', 'LT',  # East Java
    'M', 'N', 'P', 'R', 'S', 'T', 'W', 'Z',  # Eastern Indonesia
]

class PlateValidator:
    """Validator for Indonesian license plates"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.patterns = [re.compile(p) for p in INDONESIAN_PLATE_PATTERNS]

    def validate(self, text: str, check_regional_code: bool = False) -> bool:
        """
        Validate if text matches Indonesian plate format

        Args:
            text: Plate text to validate
            check_regional_code: Also validate regional code (optional, stricter)

        Returns:
            bool: True if valid, False otherwise
        """
        if not text or not isinstance(text, str):
            return False

        # Normalize text
        text = text.upper().strip()

        # Length check (Indonesian plates: 5-10 characters)
        if len(text) < 5 or len(text) > 10:
            self.logger.debug(f"Invalid length: {text} ({len(text)} chars)")
            return False

        # Pattern match
        if not any(pattern.match(text) for pattern in self.patterns):
            self.logger.debug(f"Invalid pattern: {text}")
            return False

        # Optional: Check regional code
        if check_regional_code:
            regional_code = self._extract_regional_code(text)
            if regional_code not in VALID_REGIONAL_CODES:
                self.logger.debug(f"Invalid regional code: {regional_code} in {text}")
                return False

        return True

    def _extract_regional_code(self, text: str) -> str:
        """Extract regional code (first 1-2 letters) from plate text"""
        # Try 2 letters first
        if len(text) >= 2 and text[:2].isalpha():
            return text[:2]
        # Then 1 letter
        elif len(text) >= 1 and text[0].isalpha():
            return text[0]
        return ""

    def get_validation_score(self, text: str) -> float:
        """
        Get validation score (0.0 - 1.0)
        Higher score = more likely to be valid Indonesian plate

        Args:
            text: Plate text to score

        Returns:
            float: Validation score (0.0 - 1.0)
        """
        if not text:
            return 0.0

        text = text.upper().strip()
        score = 0.0

        # Length check (5-10 chars)
        if 5 <= len(text) <= 10:
            score += 0.3

        # Pattern match
        if any(pattern.match(text) for pattern in self.patterns):
            score += 0.4

        # Regional code check
        regional_code = self._extract_regional_code(text)
        if regional_code in VALID_REGIONAL_CODES:
            score += 0.3

        return min(score, 1.0)

    def filter_invalid_plates(self, plates: list, min_score: float = 0.7) -> list:
        """
        Filter out invalid plates from a list

        Args:
            plates: List of plate texts or PlateDetection objects
            min_score: Minimum validation score (default: 0.7)

        Returns:
            list: Filtered list of valid plates
        """
        valid_plates = []

        for plate in plates:
            # Extract text (handle both string and object)
            text = plate if isinstance(plate, str) else getattr(plate, 'text', '')

            # Validate
            score = self.get_validation_score(text)
            if score >= min_score:
                valid_plates.append(plate)
            else:
                self.logger.debug(f"Filtered out invalid plate: {text} (score: {score:.2f})")

        return valid_plates

# Global validator instance
_validator = None

def get_validator() -> PlateValidator:
    """Get global validator instance (singleton)"""
    global _validator
    if _validator is None:
        _validator = PlateValidator()
    return _validator

def validate_plate(text: str, check_regional_code: bool = False) -> bool:
    """
    Quick validation function

    Args:
        text: Plate text to validate
        check_regional_code: Also validate regional code

    Returns:
        bool: True if valid
    """
    return get_validator().validate(text, check_regional_code)

def validate_and_score(text: str) -> tuple:
    """
    Validate and return score

    Args:
        text: Plate text

    Returns:
        tuple: (is_valid, score)
    """
    validator = get_validator()
    score = validator.get_validation_score(text)
    is_valid = score >= 0.7
    return (is_valid, score)

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    validator = PlateValidator()

    # Test cases
    test_plates = [
        "B1234ABC",   # Valid
        "D5678XYZ",   # Valid
        "F9012GHI",   # Valid
        "ET",         # Invalid (too short, wrong format)
        "T",          # Invalid (too short)
        "8123",       # Invalid (no letters)
        "B12345C",    # Invalid (5 digits)
        "ABC",        # Invalid (no numbers)
        "B1234A8C",   # Invalid (number in suffix)
    ]

    print("\n=== Plate Validation Tests ===")
    for plate in test_plates:
        is_valid = validator.validate(plate)
        score = validator.get_validation_score(plate)
        print(f"{plate:15s} → Valid: {is_valid:5s} | Score: {score:.2f}")
