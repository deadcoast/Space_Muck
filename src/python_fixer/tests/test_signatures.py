"""Unit tests for signature analysis components."""

import unittest
from pathlib import Path

import pytest

from python_fixer.core.signatures import (
    TypeInfo,
    SignatureMetrics,
    SignatureComponent,
    CodeSignature
)


class TestTypeInfo(unittest.TestCase):
    """Test cases for TypeInfo class."""

    def setUp(self):
        self.type_info = TypeInfo(
            type_hint="str",
            inferred_type="str",
            confidence=0.9,
            source_locations={"test.py:10"},
            constraints=["len(value) > 0"]
        )

    def test_validation_success(self):
        """Test successful validation."""
        self.assertTrue(self.type_info.validate())
        self.assertEqual(len(self.type_info.get_validation_errors()), 0)

    def test_validation_type_mismatch(self):
        """Test validation with type hint mismatch."""
        self.type_info.inferred_type = "int"
        self.assertFalse(self.type_info.validate())
        errors = self.type_info.get_validation_errors()
        self.assertTrue(any("conflicts with inferred type" in err for err in errors))

    def test_validation_no_type_info(self):
        """Test validation with no type information."""
        type_info = TypeInfo(
            type_hint=None,
            inferred_type=None,
            confidence=0.0
        )
        self.assertFalse(type_info.validate())
        errors = type_info.get_validation_errors()
        self.assertTrue(any("Missing type information" in err for err in errors))


class TestSignatureMetrics(unittest.TestCase):
    """Test cases for SignatureMetrics class."""

    def setUp(self):
        self.metrics = SignatureMetrics(
            complexity=0.5,
            cohesion=0.8,
            coupling=0.3,
            maintainability=0.7,
            documentation_score=0.9,
            type_safety=0.85,
            type_hint_coverage=0.9,
            type_inference_confidence=0.8,
            constraint_coverage=0.7,
            validation_score=0.95,
            validation_coverage=0.85,
            compatibility_score=1.0,
            error_rate=0.1
        )

    def test_metrics_bounds(self):
        """Test that all metrics are properly bounded between 0 and 1."""
        for field, value in self.metrics.dict().items():
            self.assertGreaterEqual(value, 0.0, f"{field} below minimum")
            self.assertLessEqual(value, 1.0, f"{field} above maximum")

    def test_metrics_consistency(self):
        """Test metric value consistency."""
        self.assertAlmostEqual(
            1.0 - self.metrics.error_rate,
            self.metrics.validation_coverage,
            msg="Validation coverage should be complement of error rate"
        )


class TestSignatureComponent(unittest.TestCase):
    """Test cases for SignatureComponent class."""

    def setUp(self):
        self.type_info = TypeInfo(
            type_hint="str",
            inferred_type="str",
            confidence=0.9
        )
        self.component = SignatureComponent(
            name="test_param",
            type_info=self.type_info,
            default_value=None,
            is_optional=False,
            constraints=["len(value) > 0"]
        )

    def test_get_signature(self):
        """Test signature generation."""
        signature = self.component.get_signature()
        self.assertEqual(signature.name, "test_param")
        self.assertEqual(signature.return_type, self.type_info)

    def test_get_type_info(self):
        """Test type info retrieval."""
        type_info = self.component.get_type_info()
        self.assertEqual(type_info["test_param"], self.type_info)

    def test_validation(self):
        """Test component validation."""
        self.assertTrue(self.component.validate())
        self.assertEqual(len(self.component.get_validation_errors()), 0)

    def test_compatibility(self):
        """Test component compatibility."""
        other_component = SignatureComponent(
            name="test_param",
            type_info=TypeInfo(type_hint="str", inferred_type="str", confidence=0.8),
            default_value=None,
            is_optional=False
        )
        self.assertTrue(self.component.is_compatible_with(other_component))


class TestCodeSignature(unittest.TestCase):
    """Test cases for CodeSignature class."""

    def setUp(self):
        self.component = SignatureComponent(
            name="test_param",
            type_info=TypeInfo(type_hint="str", inferred_type="str", confidence=0.9),
            default_value=None,
            is_optional=False
        )
        self.signature = CodeSignature(
            name="test_function",
            module_path=Path("test.py"),
            components=[self.component],
            return_type=TypeInfo(type_hint="int", inferred_type="int", confidence=0.9),
            docstring="Test function docstring"
        )

    def test_get_type_info(self):
        """Test type info retrieval."""
        type_info = self.signature.get_type_info()
        self.assertEqual(type_info["test_param"], self.component.type_info)
        self.assertEqual(type_info["return"], self.signature.return_type)

    def test_validation(self):
        """Test signature validation."""
        self.assertTrue(self.signature.validate())
        self.assertEqual(len(self.signature.get_validation_errors()), 0)

    def test_compatibility(self):
        """Test signature compatibility."""
        other_signature = CodeSignature(
            name="test_function",
            module_path=Path("test.py"),
            components=[self.component],
            return_type=TypeInfo(type_hint="int", inferred_type="int", confidence=0.8),
            docstring="Another test function"
        )
        self.assertTrue(self.signature.is_compatible_with(other_signature))

    def test_similarity_score(self):
        """Test signature similarity calculation."""
        other_signature = CodeSignature(
            name="test_function",
            module_path=Path("test.py"),
            components=[self.component],
            return_type=TypeInfo(type_hint="int", inferred_type="int", confidence=0.8),
            docstring="Test function docstring"
        )
        similarity = self.signature.similarity_score(other_signature)
        self.assertGreaterEqual(similarity, 0.9)  # High similarity expected


if __name__ == "__main__":
    pytest.main([__file__])
