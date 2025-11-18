"""Unit tests for core.llm.tools module."""

import pytest

from core.llm.tools import calculate_bmi, get_drug_interaction


class TestCalculateBMI:
    """Test suite for the calculate_bmi function."""

    def test_calculate_bmi_valid_input(self) -> None:
        """Test BMI calculation with valid inputs."""
        # Test case: 90kg, 1.8m -> BMI = 90 / (1.8^2) = 27.78
        result = calculate_bmi(weight_kg=90.0, height_m=1.8)
        assert result == pytest.approx(27.78, rel=0.01)

    def test_calculate_bmi_returns_rounded_value(self) -> None:
        """Test that BMI is rounded to 2 decimal places."""
        # Test case: 70kg, 1.75m -> BMI = 70 / (1.75^2) = 22.857...
        result = calculate_bmi(weight_kg=70.0, height_m=1.75)
        assert result == pytest.approx(22.86, rel=0.01)

    def test_calculate_bmi_edge_case_very_small_height(self) -> None:
        """Test BMI calculation with a very small but valid height."""
        # Test case: 50kg, 0.5m -> BMI = 50 / (0.5^2) = 200
        result = calculate_bmi(weight_kg=50.0, height_m=0.5)
        assert result == pytest.approx(200.0, rel=0.01)

    def test_calculate_bmi_zero_height_raises_error(self) -> None:
        """Test that zero height raises ValueError."""
        with pytest.raises(ValueError, match="Height must be positive"):
            calculate_bmi(weight_kg=70.0, height_m=0.0)

    def test_calculate_bmi_negative_height_raises_error(self) -> None:
        """Test that negative height raises ValueError."""
        with pytest.raises(ValueError, match="Height must be positive"):
            calculate_bmi(weight_kg=70.0, height_m=-1.5)

    def test_calculate_bmi_negative_weight_raises_error(self) -> None:
        """Test that negative weight raises ValueError."""
        with pytest.raises(ValueError, match="Weight cannot be negative"):
            calculate_bmi(weight_kg=-70.0, height_m=1.75)


class TestGetDrugInteraction:
    """Test suite for the get_drug_interaction function."""

    def test_get_drug_interaction_metformin_ibuprofen(self) -> None:
        """Test known interaction: metformin and ibuprofen."""
        result = get_drug_interaction("metformin", "ibuprofen")
        assert result == "Generally safe, but monitor kidney function."

    def test_get_drug_interaction_warfarin_aspirin(self) -> None:
        """Test known interaction: warfarin and aspirin."""
        result = get_drug_interaction("warfarin", "aspirin")
        assert result == "High risk of bleeding. Avoid concurrent use."

    def test_get_drug_interaction_order_independent(self) -> None:
        """Test that drug order doesn't matter (A-B == B-A)."""
        result_ab = get_drug_interaction("metformin", "ibuprofen")
        result_ba = get_drug_interaction("ibuprofen", "metformin")
        assert result_ab == result_ba

    def test_get_drug_interaction_case_insensitive(self) -> None:
        """Test that drug names are case-insensitive."""
        result_lower = get_drug_interaction("metformin", "ibuprofen")
        result_upper = get_drug_interaction("METFORMIN", "IBUPROFEN")
        result_mixed = get_drug_interaction("Metformin", "Ibuprofen")

        assert result_lower == result_upper
        assert result_lower == result_mixed

    def test_get_drug_interaction_no_interaction_found(self) -> None:
        """Test that unknown drug pairs return the default message."""
        result = get_drug_interaction("acetaminophen", "lisinopril")
        assert result == "No significant interaction found."

    def test_get_drug_interaction_same_drug(self) -> None:
        """Test querying the same drug twice."""
        result = get_drug_interaction("metformin", "metformin")
        assert result == "No significant interaction found."
