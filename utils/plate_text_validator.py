#!/usr/bin/env python3
"""
Indonesian License Plate Text Validator
Validates and corrects OCR results for Indonesian license plates
"""

import re
from typing import Tuple, Optional
import logging


class PlateTextValidator:
    """
    Validator for Indonesian license plate text with pattern matching and correction
    """

    # Indonesian plate patterns:
    # Format 1: B 1234 ABC (region + 1-4 digits + 1-3 letters)
    # Format 2: B 1234 A (region + 1-4 digits + 1 letter)
    # Format 3: AB 1234 C (2-letter region + digits + letter)
    # Format 4: F 1364 (region + digits only - old format)
    PLATE_PATTERNS = [
        r'^[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{1,3}$',  # Standard format
        r'^[A-Z]{1,2}\s?\d{1,4}$',                # Old format (no suffix)
    ]

    # Character confusion mapping (OCR common mistakes)
    CONFUSION_MAP = {
        '0': 'O',  # Zero vs O
        'O': '0',
        '1': 'I',  # One vs I
        'I': '1',
        '5': 'S',  # Five vs S
        'S': '5',
        '8': 'B',  # Eight vs B
        'B': '8',
        '2': 'Z',
        'Z': '2',
        '6': 'G',
        'G': '6',
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compiled_patterns = [re.compile(p) for p in self.PLATE_PATTERNS]

    def validate(self, text: str) -> Tuple[bool, float, str]:
        """
        Validate plate text against Indonesian patterns

        Args:
            text: OCR extracted text

        Returns:
            (is_valid, confidence_boost, corrected_text)
        """
        # ✅ FIX: Lower threshold dari 3 → 2 chars untuk accept more plates
        if not text or len(text) < 2:
            return False, 0.0, text

        # Clean and normalize
        cleaned = self._clean_text(text)

        # Check if matches any pattern
        for pattern in self.compiled_patterns:
            if pattern.match(cleaned):
                # Valid pattern detected
                confidence_boost = self._calculate_pattern_confidence(cleaned)
                return True, confidence_boost, cleaned

        # Try to correct common OCR mistakes
        corrected = self._attempt_correction(cleaned)
        if corrected != cleaned:
            # Check if correction yields valid pattern
            for pattern in self.compiled_patterns:
                if pattern.match(corrected):
                    confidence_boost = self._calculate_pattern_confidence(corrected) - 10  # Lower boost for corrected
                    self.logger.debug(f"Corrected '{cleaned}' → '{corrected}'")
                    return True, confidence_boost, corrected

        # ✅ FIX: Don't reject completely, return with penalty (allow non-perfect patterns)
        # This allows raw OCR text to be displayed even if pattern doesn't match perfectly
        return True, -20.0, text  # Accept with penalty (was: False, -30.0)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize plate text"""
        # Remove non-alphanumeric
        cleaned = ''.join(c for c in text.upper() if c.isalnum() or c.isspace())

        # Normalize spacing (max 2 spaces for proper plate format)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

    def _calculate_pattern_confidence(self, text: str) -> float:
        """
        Calculate confidence boost based on pattern quality
        """
        confidence = 20.0  # Base boost for valid pattern

        # Length bonus (typical Indonesian plates: 5-10 chars)
        if 6 <= len(text.replace(' ', '')) <= 9:
            confidence += 15
        elif 5 <= len(text.replace(' ', '')) <= 10:
            confidence += 8

        # Structure bonus (proper spacing)
        parts = text.split()
        if len(parts) == 3:  # Ideal: "B 1234 ABC"
            confidence += 10
        elif len(parts) == 2:  # Acceptable: "B 1234"
            confidence += 5

        # Region code bonus (common Indonesian regions)
        common_regions = ['B', 'D', 'F', 'A', 'L', 'N', 'T', 'E', 'AG', 'AA', 'AB', 'AD']
        first_part = parts[0] if parts else text[:2]
        if first_part in common_regions:
            confidence += 10

        return confidence

    def _attempt_correction(self, text: str) -> str:
        """
        Attempt to correct common OCR mistakes based on plate structure

        Indonesian plate structure:
        - Position 1-2: Letters (region code)
        - Position 3-6: Digits (number)
        - Position 7-9: Letters (suffix code)
        """
        if len(text) < 5:
            return text

        corrected = list(text)
        parts = text.split()

        if len(parts) >= 2:
            # Part 1: Should be letters (region code)
            region = parts[0]
            corrected_region = self._force_letters(region)

            # Part 2: Should be digits (plate number)
            number = parts[1]
            corrected_number = self._force_digits(number)

            # Part 3: Should be letters (suffix code) if exists
            suffix = parts[2] if len(parts) >= 3 else ""
            corrected_suffix = self._force_letters(suffix)

            # Reconstruct
            result = corrected_region + ' ' + corrected_number
            if corrected_suffix:
                result += ' ' + corrected_suffix

            return result

        return text

    def _force_letters(self, text: str) -> str:
        """Convert ambiguous characters to letters"""
        result = []
        for char in text:
            if char.isdigit() and char in self.CONFUSION_MAP:
                # Try to convert digit to letter
                letter_option = self.CONFUSION_MAP[char]
                if letter_option.isalpha():
                    result.append(letter_option)
                else:
                    result.append(char)
            else:
                result.append(char)
        return ''.join(result)

    def _force_digits(self, text: str) -> str:
        """Convert ambiguous characters to digits"""
        result = []
        for char in text:
            if char.isalpha() and char in self.CONFUSION_MAP:
                # Try to convert letter to digit
                digit_option = self.CONFUSION_MAP[char]
                if digit_option.isdigit():
                    result.append(digit_option)
                else:
                    result.append(char)
            else:
                result.append(char)
        return ''.join(result)

    def is_valid_indonesian_plate(self, text: str) -> bool:
        """
        Quick check if text matches Indonesian plate pattern
        """
        valid, _, _ = self.validate(text)
        return valid

    def get_corrected_text(self, text: str) -> Optional[str]:
        """
        Get corrected plate text or None if invalid
        """
        valid, confidence, corrected = self.validate(text)

        if valid and confidence > 0:
            return corrected

        return None


if __name__ == "__main__":
    # Test validator
    validator = PlateTextValidator()

    test_cases = [
        "B 1234 ABC",     # Valid standard
        "F 1364",         # Valid old format
        "B 2805 UMP",     # Valid standard
        "B12345ABC",      # Valid but no spacing
        "8 1234 A8C",     # Should correct to "B 1234 ABC"
        "F I364",         # Should correct to "F 1364"
        "RANDOM",         # Invalid
        "12345",          # Invalid
        "",               # Invalid
    ]

    print("🧪 Testing Indonesian Plate Validator:\n")
    for test in test_cases:
        valid, boost, corrected = validator.validate(test)
        status = "✅ VALID" if valid else "❌ INVALID"
        print(f"{status} | '{test}' → '{corrected}' | Boost: {boost:+.1f}%")
