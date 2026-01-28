"""Tests for custom exceptions."""

import pytest

from imputed_prs.core.exceptions import (
    ImputedPRSError,
    DataLoadError,
    ValidationError,
    IncompatibleBuildError,
    MissingVariantsError,
    ModelNotFittedError,
)


class TestExceptionHierarchy:
    """Tests for exception inheritance."""

    def test_data_load_error_inherits_from_base(self):
        """Test DataLoadError inherits from ImputedPRSError."""
        assert issubclass(DataLoadError, ImputedPRSError)

    def test_validation_error_inherits_from_base(self):
        """Test ValidationError inherits from ImputedPRSError."""
        assert issubclass(ValidationError, ImputedPRSError)

    def test_incompatible_build_error_inherits_from_base(self):
        """Test IncompatibleBuildError inherits from ImputedPRSError."""
        assert issubclass(IncompatibleBuildError, ImputedPRSError)

    def test_missing_variants_error_inherits_from_base(self):
        """Test MissingVariantsError inherits from ImputedPRSError."""
        assert issubclass(MissingVariantsError, ImputedPRSError)

    def test_model_not_fitted_error_inherits_from_base(self):
        """Test ModelNotFittedError inherits from ImputedPRSError."""
        assert issubclass(ModelNotFittedError, ImputedPRSError)

    def test_base_inherits_from_exception(self):
        """Test ImputedPRSError inherits from Exception."""
        assert issubclass(ImputedPRSError, Exception)


class TestRaiseAndCatch:
    """Tests for raising and catching exceptions."""

    def test_raise_imputed_prs_error(self):
        """Test raising and catching ImputedPRSError."""
        with pytest.raises(ImputedPRSError) as exc_info:
            raise ImputedPRSError("Base error message")
        assert str(exc_info.value) == "Base error message"

    def test_raise_data_load_error(self):
        """Test raising and catching DataLoadError."""
        with pytest.raises(DataLoadError) as exc_info:
            raise DataLoadError("Failed to load VCF file")
        assert str(exc_info.value) == "Failed to load VCF file"

    def test_raise_validation_error(self):
        """Test raising and catching ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Invalid column format")
        assert str(exc_info.value) == "Invalid column format"

    def test_raise_incompatible_build_error(self):
        """Test raising and catching IncompatibleBuildError."""
        with pytest.raises(IncompatibleBuildError) as exc_info:
            raise IncompatibleBuildError("Expected GRCh38, got GRCh37")
        assert str(exc_info.value) == "Expected GRCh38, got GRCh37"

    def test_raise_missing_variants_error(self):
        """Test raising and catching MissingVariantsError."""
        with pytest.raises(MissingVariantsError) as exc_info:
            raise MissingVariantsError("10 required variants not found")
        assert str(exc_info.value) == "10 required variants not found"

    def test_raise_model_not_fitted_error(self):
        """Test raising and catching ModelNotFittedError."""
        with pytest.raises(ModelNotFittedError) as exc_info:
            raise ModelNotFittedError("Call fit() before predict()")
        assert str(exc_info.value) == "Call fit() before predict()"


class TestExceptionCatching:
    """Tests for catching exceptions by base class."""

    def test_catch_data_load_error_as_base(self):
        """Test catching DataLoadError as ImputedPRSError."""
        with pytest.raises(ImputedPRSError):
            raise DataLoadError("File not found")

    def test_catch_validation_error_as_base(self):
        """Test catching ValidationError as ImputedPRSError."""
        with pytest.raises(ImputedPRSError):
            raise ValidationError("Invalid data")

    def test_catch_incompatible_build_error_as_base(self):
        """Test catching IncompatibleBuildError as ImputedPRSError."""
        with pytest.raises(ImputedPRSError):
            raise IncompatibleBuildError("Build mismatch")

    def test_catch_missing_variants_error_as_base(self):
        """Test catching MissingVariantsError as ImputedPRSError."""
        with pytest.raises(ImputedPRSError):
            raise MissingVariantsError("Missing variants")

    def test_catch_model_not_fitted_error_as_base(self):
        """Test catching ModelNotFittedError as ImputedPRSError."""
        with pytest.raises(ImputedPRSError):
            raise ModelNotFittedError("Not fitted")


class TestExceptionChaining:
    """Tests for exception chaining."""

    def test_chain_with_from_clause(self):
        """Test exception chaining with 'from' clause."""
        try:
            try:
                raise FileNotFoundError("file.vcf not found")
            except FileNotFoundError as e:
                raise DataLoadError("Failed to load VCF") from e
        except DataLoadError as e:
            assert str(e) == "Failed to load VCF"
            assert isinstance(e.__cause__, FileNotFoundError)
            assert str(e.__cause__) == "file.vcf not found"

    def test_chain_validation_from_value_error(self):
        """Test chaining ValidationError from ValueError."""
        try:
            try:
                raise ValueError("Invalid integer")
            except ValueError as e:
                raise ValidationError("Column 'position' must be integer") from e
        except ValidationError as e:
            assert isinstance(e.__cause__, ValueError)
