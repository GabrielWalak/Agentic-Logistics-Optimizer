"""Typed carrier quote tools used by the logistics agents.

The module deliberately keeps pricing deterministic.  The language model may
compare verified quotes, but it is not allowed to invent operational data such
as price, availability, or estimated transit time.
"""

from dataclasses import dataclass
from math import ceil
from typing import List, Literal, Mapping, Optional

from pydantic import BaseModel, Field


CarrierName = Literal[
    "Standard Shipping",
    "Regional",
    "SEDEX",
    "Premium Express",
    "International",
]


class CarrierQuoteInput(BaseModel):
    """Validated input contract for the carrier quote tool."""

    carrier: CarrierName
    distance_km: float = Field(gt=0, le=20_000)
    weight_g: float = Field(gt=0, le=100_000)


class CarrierQuote(BaseModel):
    """Structured quote returned to the carrier optimization agent."""

    carrier: CarrierName
    estimated_cost: float = Field(ge=0)
    estimated_transit_days: float = Field(gt=0)
    available: bool
    availability_reason: Optional[str] = None
    currency: str = "BRL"
    source: str = "portfolio_rate_card_v1"


@dataclass(frozen=True)
class _CarrierProfile:
    """Internal rate-card entry; not exposed as part of the tool contract."""

    base_cost: float
    cost_per_km: float
    cost_per_kg: float
    base_days: float
    km_per_day: float
    min_distance_km: float = 0
    max_distance_km: Optional[float] = None
    max_weight_g: Optional[float] = None


_PROFILES: Mapping[CarrierName, _CarrierProfile] = {
    "Standard Shipping": _CarrierProfile(
        base_cost=12,
        cost_per_km=0.012,
        cost_per_kg=1.5,
        base_days=2,
        km_per_day=350,
        max_distance_km=750,
        max_weight_g=2_000,
    ),
    "Regional": _CarrierProfile(
        base_cost=18,
        cost_per_km=0.016,
        cost_per_kg=2.0,
        base_days=1.5,
        km_per_day=550,
        max_distance_km=1_500,
        max_weight_g=5_000,
    ),
    "SEDEX": _CarrierProfile(
        base_cost=25,
        cost_per_km=0.018,
        cost_per_kg=2.5,
        base_days=1,
        km_per_day=700,
        min_distance_km=250,
        max_distance_km=3_000,
        max_weight_g=5_000,
    ),
    "Premium Express": _CarrierProfile(
        base_cost=32,
        cost_per_km=0.022,
        cost_per_kg=3.0,
        base_days=1,
        km_per_day=950,
        max_distance_km=3_500,
        max_weight_g=15_000,
    ),
    "International": _CarrierProfile(
        base_cost=55,
        cost_per_km=0.028,
        cost_per_kg=4.0,
        base_days=3,
        km_per_day=600,
        min_distance_km=2_500,
        max_weight_g=30_000,
    ),
}


def get_carrier_quote(tool_input: CarrierQuoteInput) -> CarrierQuote:
    """Return a deterministic quote for one validated carrier request.

    The local rate card is intentionally transparent and replaceable.  A real
    deployment could keep this contract while swapping the implementation for
    an authenticated carrier API or an MCP tool.
    """

    profile = _PROFILES[tool_input.carrier]
    weight_kg = tool_input.weight_g / 1_000

    estimated_cost = (
        profile.base_cost
        + tool_input.distance_km * profile.cost_per_km
        + weight_kg * profile.cost_per_kg
    )
    estimated_days = profile.base_days + ceil(
        tool_input.distance_km / profile.km_per_day
    )

    reasons = []
    if tool_input.distance_km < profile.min_distance_km:
        reasons.append(
            f"minimum supported distance is {profile.min_distance_km:.0f} km"
        )
    if (
        profile.max_distance_km is not None
        and tool_input.distance_km > profile.max_distance_km
    ):
        reasons.append(
            f"maximum supported distance is {profile.max_distance_km:.0f} km"
        )
    if (
        profile.max_weight_g is not None
        and tool_input.weight_g > profile.max_weight_g
    ):
        reasons.append(
            f"maximum supported weight is {profile.max_weight_g:.0f} g"
        )

    return CarrierQuote(
        carrier=tool_input.carrier,
        estimated_cost=round(estimated_cost, 2),
        estimated_transit_days=float(estimated_days),
        available=not reasons,
        availability_reason="; ".join(reasons) if reasons else None,
    )


def get_all_carrier_quotes(
    distance_km: float,
    weight_g: float,
) -> List[CarrierQuote]:
    """Return comparable quotes for every carrier in the local rate card."""

    return [
        get_carrier_quote(
            CarrierQuoteInput(
                carrier=carrier,
                distance_km=distance_km,
                weight_g=weight_g,
            )
        )
        for carrier in _PROFILES
    ]


def select_carrier_quote(
    quotes: List[CarrierQuote],
    requested_carrier: str,
    should_upgrade: bool,
) -> CarrierQuote:
    """Resolve an LLM recommendation to a valid, available quote.

    The model's recommendation is preferred when it matches an available
    carrier.  Otherwise the application applies a deterministic fallback:
    fastest for an upgrade, cheapest for a standard delivery.
    """

    aliases = {
        "standard": "Standard Shipping",
        "standard shipping": "Standard Shipping",
        "regional": "Regional",
        "sedex": "SEDEX",
        "premium": "Premium Express",
        "premium express": "Premium Express",
        "international": "International",
    }
    normalized_name = aliases.get(
        str(requested_carrier).strip().lower(),
        str(requested_carrier).strip(),
    )

    available_quotes = [quote for quote in quotes if quote.available]
    if not available_quotes:
        raise ValueError("No carrier is available for the supplied scenario")

    requested_quote = next(
        (
            quote
            for quote in available_quotes
            if quote.carrier == normalized_name
        ),
        None,
    )
    if requested_quote is not None:
        return requested_quote

    if should_upgrade:
        return min(
            available_quotes,
            key=lambda quote: (
                quote.estimated_transit_days,
                quote.estimated_cost,
            ),
        )

    return min(available_quotes, key=lambda quote: quote.estimated_cost)
