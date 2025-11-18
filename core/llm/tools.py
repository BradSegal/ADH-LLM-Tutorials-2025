"""
Simple tool functions for LLM function calling demonstrations.

This module contains lightweight, educational functions that can be used
as "tools" in LLM function calling workflows. These are designed for
pedagogical purposes to demonstrate the function calling pattern.
"""


def _ensure_float(value: float | int | str, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{field_name} must be numeric.") from exc


def calculate_bmi(weight_kg: float | int | str, height_m: float | int | str) -> float:
    """
    Calculates the Body Mass Index (BMI).

    Args:
        weight_kg: Weight in kilograms.
        height_m: Height in meters.

    Returns:
        The calculated BMI value, rounded to 2 decimal places.

    Raises:
        ValueError: If weight_kg is negative or height_m is non-positive.
    """
    weight = _ensure_float(weight_kg, "Weight")
    height = _ensure_float(height_m, "Height")

    if weight < 0:
        raise ValueError("Weight cannot be negative")
    if height <= 0:
        raise ValueError("Height must be positive")

    bmi = weight / (height**2)
    return round(bmi, 2)


def get_drug_interaction(drug_a: str, drug_b: str) -> str:
    """
    A toy function to look up interactions between two drugs.

    In a real system, this would query a comprehensive drug interaction database.
    For educational purposes, this function contains a small, hardcoded set of
    known interactions to demonstrate the function calling pattern.

    Args:
        drug_a: Name of the first drug.
        drug_b: Name of the second drug.

    Returns:
        A string describing the interaction between the two drugs, or a message
        indicating no significant interaction was found.
    """
    # Convert to lowercase for case-insensitive matching
    drug_a = drug_a.lower()
    drug_b = drug_b.lower()

    # Toy interaction database using frozenset for order-independent matching
    interactions = {
        frozenset(
            ["metformin", "ibuprofen"]
        ): "Generally safe, but monitor kidney function.",
        frozenset(
            ["warfarin", "aspirin"]
        ): "High risk of bleeding. Avoid concurrent use.",
    }

    key = frozenset([drug_a, drug_b])
    return interactions.get(key, "No significant interaction found.")
