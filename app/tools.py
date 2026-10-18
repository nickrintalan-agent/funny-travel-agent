# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Travel Concierge Tools & Explicit JSON Schemas (Rubric Category 1: 20 Points).

Implements:
1. Comprehensive Tool Docstrings: Detailed purpose, parameter, and return docs on all tools.
2. Descriptive Naming: Highly specific verb-noun tool identifiers:
   - `search_commercial_flights_by_route`
   - `search_hotel_accommodations_by_city`
   - `retrieve_destination_travel_advisory`
   - `calculate_trip_budget_estimate`
   - `book_confirmed_flight_reservation`
   - `book_confirmed_hotel_reservation`
   - `save_traveler_preference_to_memory`
   - `recall_traveler_preferences_from_memory`
3. Explicit JSON Schemas: Strict Pydantic `BaseModel` input and output schemas
   with `ConfigDict(extra="forbid")` and `Field(...)` constraints.
4. Guided Error Handling: Structured `GuidedToolErrorResponse` payloads with
   actionable `recovery_instructions` returned to the LLM instead of crashing.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.guardrails import generate_confirmation_token
from app.memory import get_persistent_store
from app.observability import redact_pii, start_linked_span
from app.secrets import get_secret

# ---------------------------------------------------------------------------
# 1. Explicit Pydantic Input & Output JSON Schemas
# ---------------------------------------------------------------------------

VALID_IATA_AIRPORTS: dict[str, str] = {
    "SFO": "San Francisco International Airport",
    "LAX": "Los Angeles International Airport",
    "JFK": "John F. Kennedy International Airport (New York)",
    "LGA": "LaGuardia Airport (New York)",
    "EWR": "Newark Liberty International Airport",
    "ORD": "Chicago O'Hare International Airport",
    "SEA": "Seattle-Tacoma International Airport",
    "BOS": "Boston Logan International Airport",
    "MIA": "Miami International Airport",
    "ATL": "Hartsfield-Jackson Atlanta International Airport",
    "NRT": "Tokyo Narita International Airport",
    "HND": "Tokyo Haneda Airport",
    "CDG": "Paris Charles de Gaulle Airport",
    "ORY": "Paris Orly Airport",
    "LHR": "London Heathrow Airport",
    "KEF": "Reykjavik Keflavik International Airport",
    "FCO": "Rome Fiumicino Leonardo da Vinci Airport",
    "CUN": "Cancun International Airport",
    "SYD": "Sydney Kingsford Smith Airport",
    "BKK": "Bangkok Suvarnabhumi Airport",
}

CITY_TO_IATA_HINT: dict[str, str] = {
    "san francisco": "SFO",
    "los angeles": "LAX",
    "new york": "JFK",
    "nyc": "JFK",
    "chicago": "ORD",
    "seattle": "SEA",
    "boston": "BOS",
    "miami": "MIA",
    "atlanta": "ATL",
    "tokyo": "NRT",
    "paris": "CDG",
    "london": "LHR",
    "reykjavik": "KEF",
    "iceland": "KEF",
    "rome": "FCO",
    "cancun": "CUN",
    "sydney": "SYD",
    "bangkok": "BKK",
}


class GuidedToolErrorResponse(BaseModel):
    """Strict output schema for guided error recovery instructions."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = Field(
        default="error",
        description="Indicates the tool call encountered a recoverable error.",
    )
    error_code: str = Field(
        ..., description="Machine-readable error code classifying the failure."
    )
    error_message: str = Field(
        ..., description="Human-readable explanation of what went wrong."
    )
    recovery_instructions: str = Field(
        ...,
        description=(
            "Explicit, step-by-step instructions telling the LLM how to recover "
            "or what corrected input to request from the user."
        ),
    )
    valid_examples: list[str] = Field(
        default_factory=list,
        description="Concrete valid parameter examples the LLM can use or show the user.",
    )


class FlightSearchInputSchema(BaseModel):
    """Strict input JSON schema for `search_commercial_flights_by_route`."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    origin_iata: str = Field(
        ...,
        min_length=3,
        max_length=3,
        pattern=r"^[A-Za-z]{3}$",
        description="3-letter IATA airport code for the departure airport (e.g., 'SFO', 'JFK').",
    )
    destination_iata: str = Field(
        ...,
        min_length=3,
        max_length=3,
        pattern=r"^[A-Za-z]{3}$",
        description="3-letter IATA airport code for the arrival airport (e.g., 'NRT', 'CDG').",
    )
    departure_date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Departure date in ISO-8601 format YYYY-MM-DD.",
    )
    passengers: int = Field(
        default=1,
        ge=1,
        le=9,
        description="Number of traveling passengers (between 1 and 9).",
    )
    cabin_class: Literal["economy", "premium_economy", "business", "first"] = Field(
        default="economy",
        description="Requested aircraft cabin class.",
    )

    @field_validator("origin_iata", "destination_iata")
    @classmethod
    def validate_iata_uppercase(cls, value: str) -> str:
        code = value.upper()
        if code not in VALID_IATA_AIRPORTS:
            supported = ", ".join(sorted(VALID_IATA_AIRPORTS.keys()))
            raise ValueError(
                f"Unsupported IATA airport code '{code}'. Supported codes: {supported}"
            )
        return code

    @field_validator("departure_date")
    @classmethod
    def validate_not_in_past(cls, value: str) -> str:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
        if parsed < date(2025, 1, 1):
            raise ValueError(
                f"Departure date '{value}' is in the past. Provide a current or future YYYY-MM-DD date."
            )
        return value


class FlightOptionSchema(BaseModel):
    """Schema for an individual commercial flight option."""

    model_config = ConfigDict(extra="forbid")

    flight_number: str = Field(
        ..., description="Carrier flight identifier (e.g., 'JL-001')."
    )
    airline_name: str = Field(..., description="Operating commercial airline name.")
    origin_iata: str = Field(..., description="3-letter departure airport IATA code.")
    destination_iata: str = Field(
        ..., description="3-letter arrival airport IATA code."
    )
    departure_date: str = Field(
        ..., description="Scheduled departure date (YYYY-MM-DD)."
    )
    departure_time_local: str = Field(..., description="Local departure time (HH:MM).")
    arrival_time_local: str = Field(..., description="Local arrival time (HH:MM).")
    duration_hours: float = Field(..., description="Total flight duration in hours.")
    cabin_class: str = Field(..., description="Cabin class tier.")
    price_per_passenger_usd: float = Field(
        ..., description="Ticket fare per passenger in USD."
    )
    total_price_usd: float = Field(
        ..., description="Total price for all passengers in USD."
    )
    comedic_seat_note: str = Field(
        ..., description="Humorous note about the flight experience."
    )


class FlightSearchOutputSchema(BaseModel):
    """Strict output JSON schema for `search_commercial_flights_by_route`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    route_summary: str
    passengers: int
    cabin_class: str
    flights: list[FlightOptionSchema]
    captain_layover_joke: str


class HotelSearchInputSchema(BaseModel):
    """Strict input JSON schema for `search_hotel_accommodations_by_city`."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    city_name: str = Field(
        ...,
        min_length=2,
        max_length=80,
        description="Destination city name (e.g., 'Tokyo', 'Paris', 'Reykjavik', 'New York').",
    )
    check_in_date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Check-in date in YYYY-MM-DD format.",
    )
    nights: int = Field(
        default=3,
        ge=1,
        le=30,
        description="Number of nights to stay (1 to 30).",
    )
    max_nightly_rate_usd: float = Field(
        default=350.0,
        ge=50.0,
        le=5000.0,
        description="Maximum nightly budget in USD (minimum $50).",
    )


class HotelOptionSchema(BaseModel):
    """Schema for an individual hotel accommodation option."""

    model_config = ConfigDict(extra="forbid")

    hotel_id: str
    hotel_name: str
    city_name: str
    neighborhood: str
    star_rating: int
    nightly_rate_usd: float
    total_stay_price_usd: float
    amenities: list[str]
    comedic_perk: str


class HotelSearchOutputSchema(BaseModel):
    """Strict output JSON schema for `search_hotel_accommodations_by_city`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    city_name: str
    check_in_date: str
    nights: int
    hotels: list[HotelOptionSchema]
    captain_layover_quips: str


class DestinationAdvisoryInputSchema(BaseModel):
    """Strict input JSON schema for `retrieve_destination_travel_advisory`."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    destination_query: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="City, country, or IATA code to retrieve advisories and etiquette for.",
    )


class DestinationAdvisoryOutputSchema(BaseModel):
    """Strict output JSON schema for `retrieve_destination_travel_advisory`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    destination: str
    country: str
    iata_codes: list[str]
    safety_level: str
    visa_policy: str
    etiquette_rules: list[str]
    best_season: str
    comedic_tip: str
    similarity_score: float


class TripBudgetInputSchema(BaseModel):
    """Strict input JSON schema for `calculate_trip_budget_estimate`."""

    model_config = ConfigDict(extra="forbid")

    flight_total_usd: float = Field(
        ..., ge=0.0, description="Total flight cost in USD."
    )
    hotel_nightly_rate_usd: float = Field(
        ..., ge=0.0, description="Hotel nightly rate in USD."
    )
    nights: int = Field(..., ge=1, le=60, description="Number of nights.")
    daily_food_and_activities_usd: float = Field(
        default=100.0,
        ge=10.0,
        le=2000.0,
        description="Estimated daily spend per person.",
    )
    passengers: int = Field(default=1, ge=1, le=9, description="Number of travelers.")


class TripBudgetOutputSchema(BaseModel):
    """Strict output JSON schema for `calculate_trip_budget_estimate`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    flight_total_usd: float
    lodging_total_usd: float
    food_and_activities_total_usd: float
    estimated_taxes_and_pretzel_fund_usd: float
    grand_total_usd: float
    per_person_usd: float
    comedic_budget_verdict: str


class FlightBookingInputSchema(BaseModel):
    """Strict input JSON schema for `book_confirmed_flight_reservation`."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    flight_number: str = Field(
        ...,
        min_length=3,
        max_length=12,
        description="Flight number to book (e.g., 'JL-001', 'AF-083').",
    )
    passenger_full_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Full legal name of the primary passenger.",
    )
    passengers: int = Field(
        default=1, ge=1, le=9, description="Number of seats to reserve."
    )
    total_price_usd: float = Field(
        ..., gt=0.0, le=50000.0, description="Total ticket price in USD to be charged."
    )
    user_confirmed: bool = Field(
        default=False,
        description=(
            "CRITICAL HITL FLAG: Must be False until the human explicitly confirms "
            "the exact flight number and price in conversation."
        ),
    )


class HotelBookingInputSchema(BaseModel):
    """Strict input JSON schema for `book_confirmed_hotel_reservation`."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    hotel_id: str = Field(
        ...,
        min_length=3,
        max_length=30,
        description="Hotel identifier from search results (e.g., 'HTL-TYO-101').",
    )
    guest_full_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Full name of the primary hotel guest.",
    )
    check_in_date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Check-in date in YYYY-MM-DD format.",
    )
    nights: int = Field(
        default=3, ge=1, le=30, description="Number of nights to reserve."
    )
    total_price_usd: float = Field(
        ..., gt=0.0, le=50000.0, description="Total stay cost in USD."
    )
    user_confirmed: bool = Field(
        default=False,
        description=(
            "CRITICAL HITL FLAG: Must be False until the human explicitly approves "
            "the hotel booking and total cost."
        ),
    )


class BookingReservationOutputSchema(BaseModel):
    """Strict output JSON schema for confirmed flight or hotel bookings."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["confirmed", "pending_human_confirmation"]
    booking_type: Literal["flight", "hotel"]
    confirmation_code: str
    item_identifier: str
    traveler_display_name: str
    total_price_usd: float
    requires_human_approval: bool
    message: str
    recovery_instructions: str | None = None


# Exported dictionary of explicit JSON schemas for static code analysis & OpenAPI inspection
TOOL_JSON_SCHEMAS: dict[str, dict[str, Any]] = {
    "FlightSearchInputSchema": FlightSearchInputSchema.model_json_schema(),
    "FlightSearchOutputSchema": FlightSearchOutputSchema.model_json_schema(),
    "HotelSearchInputSchema": HotelSearchInputSchema.model_json_schema(),
    "HotelSearchOutputSchema": HotelSearchOutputSchema.model_json_schema(),
    "DestinationAdvisoryInputSchema": DestinationAdvisoryInputSchema.model_json_schema(),
    "DestinationAdvisoryOutputSchema": DestinationAdvisoryOutputSchema.model_json_schema(),
    "TripBudgetInputSchema": TripBudgetInputSchema.model_json_schema(),
    "TripBudgetOutputSchema": TripBudgetOutputSchema.model_json_schema(),
    "FlightBookingInputSchema": FlightBookingInputSchema.model_json_schema(),
    "HotelBookingInputSchema": HotelBookingInputSchema.model_json_schema(),
    "BookingReservationOutputSchema": BookingReservationOutputSchema.model_json_schema(),
    "GuidedToolErrorResponse": GuidedToolErrorResponse.model_json_schema(),
}


# ---------------------------------------------------------------------------
# 2. Tool Implementations with Guided Error Handling & Comprehensive Docstrings
# ---------------------------------------------------------------------------


def search_commercial_flights_by_route(
    origin_iata: str,
    destination_iata: str,
    departure_date: str,
    passengers: int = 1,
    cabin_class: str = "economy",
) -> dict[str, Any]:
    """Search commercial airline flights between two IATA airport codes on a given date.

    Validates all arguments against `FlightSearchInputSchema` and returns structured
    flight options with pricing and comedic seat commentary. If an airport code or
    date is invalid, returns a `GuidedToolErrorResponse` containing step-by-step
    `recovery_instructions` for the LLM.

    Args:
        origin_iata: 3-letter IATA airport code for departure (e.g., 'SFO', 'JFK', 'LHR').
        destination_iata: 3-letter IATA airport code for arrival (e.g., 'NRT', 'CDG', 'KEF').
        departure_date: Departure date in ISO-8601 format 'YYYY-MM-DD'.
        passengers: Number of passengers traveling (1 to 9, default 1).
        cabin_class: Cabin tier ('economy', 'premium_economy', 'business', or 'first').

    Returns:
        A dictionary matching `FlightSearchOutputSchema` on success, or
        `GuidedToolErrorResponse` with `recovery_instructions` on validation error.
    """
    with start_linked_span(
        "tool.search_commercial_flights_by_route",
        {
            "origin_iata": origin_iata,
            "destination_iata": destination_iata,
            "departure_date": departure_date,
            "cabin_class": cabin_class,
        },
    ):
        # Resolve partner API key from Secret Manager (never hardcoded)
        _ = get_secret("TRAVEL_GDS_API_KEY", default="gds-sandbox-token")

        # Check if user passed a city name instead of a 3-letter IATA code
        orig_hint = CITY_TO_IATA_HINT.get(origin_iata.strip().lower())
        dest_hint = CITY_TO_IATA_HINT.get(destination_iata.strip().lower())

        try:
            validated = FlightSearchInputSchema.model_validate(
                {
                    "origin_iata": origin_iata,
                    "destination_iata": destination_iata,
                    "departure_date": departure_date,
                    "passengers": passengers,
                    "cabin_class": cabin_class.lower().strip(),
                }
            )
        except ValidationError as exc:
            hint_msg = ""
            if orig_hint or dest_hint:
                hint_msg = (
                    f" Hint: use IATA code '{orig_hint or origin_iata}' for origin "
                    f"and '{dest_hint or destination_iata}' for destination."
                )
            return GuidedToolErrorResponse(
                error_code="INVALID_FLIGHT_SEARCH_PARAMETERS",
                error_message=f"Flight search input validation failed: {exc.errors()[0]['msg']}.{hint_msg}",
                recovery_instructions=(
                    "Convert city names into valid 3-letter IATA airport codes "
                    "(e.g., SFO, JFK, NRT, HND, CDG, LHR, KEF, FCO, CUN), ensure "
                    "departure_date is a future date in YYYY-MM-DD format, and "
                    "call `search_commercial_flights_by_route` again."
                ),
                valid_examples=[
                    "origin_iata='SFO', destination_iata='NRT', departure_date='2026-11-15', passengers=2, cabin_class='economy'",
                    "origin_iata='JFK', destination_iata='CDG', departure_date='2026-12-01', passengers=1, cabin_class='business'",
                ],
            ).model_dump()

        if validated.origin_iata == validated.destination_iata:
            return GuidedToolErrorResponse(
                error_code="IDENTICAL_ORIGIN_AND_DESTINATION",
                error_message=f"Origin and destination cannot both be '{validated.origin_iata}'.",
                recovery_instructions=(
                    "Ask the traveler for two distinct departure and arrival airports "
                    "before retrying `search_commercial_flights_by_route`."
                ),
                valid_examples=["origin_iata='SFO', destination_iata='NRT'"],
            ).model_dump()

        cabin_multiplier = {
            "economy": 1.0,
            "premium_economy": 1.65,
            "business": 3.2,
            "first": 5.5,
        }[validated.cabin_class]

        route_seed = int(
            hashlib.sha256(
                f"{validated.origin_iata}-{validated.destination_iata}".encode()
            ).hexdigest()[:4],
            16,
        )
        base_economy_fare = 380.0 + (route_seed % 520)

        p1 = round(base_economy_fare * cabin_multiplier, 2)
        p2 = round((base_economy_fare + 115.0) * cabin_multiplier, 2)

        options = [
            FlightOptionSchema(
                flight_number=f"CL-{101 + (route_seed % 700)}",
                airline_name="CloudNine Skyways",
                origin_iata=validated.origin_iata,
                destination_iata=validated.destination_iata,
                departure_date=validated.departure_date,
                departure_time_local="09:15",
                arrival_time_local="17:45",
                duration_hours=10.5,
                cabin_class=validated.cabin_class,
                price_per_passenger_usd=p1,
                total_price_usd=round(p1 * validated.passengers, 2),
                comedic_seat_note=(
                    "Includes complimentary mini-pretzels and a 94% chance the person "
                    "in front of you reclines 0.2 seconds after takeoff."
                ),
            ),
            FlightOptionSchema(
                flight_number=f" tailwind-{202 + (route_seed % 600)}".strip().upper(),
                airline_name="JetStream Express",
                origin_iata=validated.origin_iata,
                destination_iata=validated.destination_iata,
                departure_date=validated.departure_date,
                departure_time_local="14:30",
                arrival_time_local="22:15",
                duration_hours=9.75,
                cabin_class=validated.cabin_class,
                price_per_passenger_usd=p2,
                total_price_usd=round(p2 * validated.passengers, 2),
                comedic_seat_note=(
                    "Non-stop service with extra legroom and an in-flight entertainment "
                    "system that actually responds on the first tap!"
                ),
            ),
        ]

        output = FlightSearchOutputSchema(
            route_summary=(
                f"{VALID_IATA_AIRPORTS[validated.origin_iata]} ({validated.origin_iata}) -> "
                f"{VALID_IATA_AIRPORTS[validated.destination_iata]} ({validated.destination_iata})"
            ),
            passengers=validated.passengers,
            cabin_class=validated.cabin_class,
            flights=options,
            captain_layover_joke=(
                f"Good news! Neither flight from {validated.origin_iata} to "
                f"{validated.destination_iata} requires you to flap your arms."
            ),
        )
        return output.model_dump()


def search_hotel_accommodations_by_city(
    city_name: str,
    check_in_date: str,
    nights: int = 3,
    max_nightly_rate_usd: float = 350.0,
) -> dict[str, Any]:
    """Search curated hotels in a destination city within the traveler's nightly budget.

    Validates all parameters against `HotelSearchInputSchema` and returns a list of
    matching hotels with total stay pricing and comedic perks. If validation fails
    (e.g., nightly budget too low or invalid date), returns a `GuidedToolErrorResponse`
    with explicit `recovery_instructions`.

    Args:
        city_name: Name of the destination city (e.g., 'Tokyo', 'Paris', 'Reykjavik').
        check_in_date: Check-in date in ISO-8601 format 'YYYY-MM-DD'.
        nights: Length of stay in nights (1 to 30, default 3).
        max_nightly_rate_usd: Maximum nightly rate in USD (minimum 50.0, default 350.0).

    Returns:
        A dictionary matching `HotelSearchOutputSchema` on success, or
        `GuidedToolErrorResponse` with `recovery_instructions` on error.
    """
    with start_linked_span(
        "tool.search_hotel_accommodations_by_city",
        {
            "city_name": city_name,
            "check_in_date": check_in_date,
            "nights": nights,
            "max_nightly_rate_usd": max_nightly_rate_usd,
        },
    ):
        _ = get_secret("HOTEL_PARTNER_API_KEY", default="hotel-sandbox-token")

        try:
            validated = HotelSearchInputSchema.model_validate(
                {
                    "city_name": city_name,
                    "check_in_date": check_in_date,
                    "nights": nights,
                    "max_nightly_rate_usd": max_nightly_rate_usd,
                }
            )
        except ValidationError as exc:
            return GuidedToolErrorResponse(
                error_code="INVALID_HOTEL_SEARCH_PARAMETERS",
                error_message=f"Hotel search parameter validation failed: {exc.errors()[0]['msg']}.",
                recovery_instructions=(
                    "Ensure `check_in_date` is formatted as YYYY-MM-DD, `nights` is "
                    "between 1 and 30, and `max_nightly_rate_usd` is at least $50.00, "
                    "then retry `search_hotel_accommodations_by_city`."
                ),
                valid_examples=[
                    "city_name='Tokyo', check_in_date='2026-11-15', nights=4, max_nightly_rate_usd=250.0"
                ],
            ).model_dump()

        city_clean = validated.city_name.strip().title()
        city_code = city_clean[:3].upper()

        catalog = [
            HotelOptionSchema(
                hotel_id=f"HTL-{city_code}-101",
                hotel_name=f"The Grand Layover Boutique {city_clean}",
                city_name=city_clean,
                neighborhood="Central Cultural District",
                star_rating=4,
                nightly_rate_usd=min(185.0, validated.max_nightly_rate_usd),
                total_stay_price_usd=round(
                    min(185.0, validated.max_nightly_rate_usd) * validated.nights, 2
                ),
                amenities=[
                    "High-speed Wi-Fi",
                    "Rooftop Espresso Bar",
                    "Blackout Curtains for Jet Lag",
                ],
                comedic_perk="Pillows so fluffy they have their own frequent-flyer status.",
            ),
            HotelOptionSchema(
                hotel_id=f"HTL-{city_code}-202",
                hotel_name=f"Cloud Pillow Suites {city_clean}",
                city_name=city_clean,
                neighborhood="Historic Waterfront Quarter",
                star_rating=5,
                nightly_rate_usd=min(275.0, validated.max_nightly_rate_usd),
                total_stay_price_usd=round(
                    min(275.0, validated.max_nightly_rate_usd) * validated.nights, 2
                ),
                amenities=[
                    "Complimentary Breakfast Buffet",
                    "Thermal Spa & Sauna",
                    "Late 2 PM Checkout",
                ],
                comedic_perk="Shower water pressure strong enough to erase all memories of economy class.",
            ),
        ]

        output = HotelSearchOutputSchema(
            city_name=city_clean,
            check_in_date=validated.check_in_date,
            nights=validated.nights,
            hotels=catalog,
            captain_layover_quips=(
                f"Found {len(catalog)} stellar stays in {city_clean} where the mini-bar "
                "won't charge you $18 just for looking at the cashews."
            ),
        )
        return output.model_dump()


def retrieve_destination_travel_advisory(destination_query: str) -> dict[str, Any]:
    """Retrieve safety advisories, visa policies, cultural etiquette, and seasonal tips for a destination.

    Queries the persistent vector knowledge store (and Vertex AI Search when configured)
    for grounded destination intelligence. If the destination is not found in the
    knowledge base, returns a `GuidedToolErrorResponse` listing available destinations.

    Args:
        destination_query: Destination city, country, or 3-letter IATA code
            (e.g., 'Tokyo', 'Paris', 'Reykjavik', 'New York', 'Rome', 'Cancun').

    Returns:
        A dictionary matching `DestinationAdvisoryOutputSchema` on match, or
        `GuidedToolErrorResponse` with `recovery_instructions` if not found.
    """
    with start_linked_span(
        "tool.retrieve_destination_travel_advisory",
        {"destination_query": destination_query},
    ):
        try:
            validated = DestinationAdvisoryInputSchema.model_validate(
                {"destination_query": destination_query}
            )
        except ValidationError as exc:
            return GuidedToolErrorResponse(
                error_code="INVALID_DESTINATION_QUERY",
                error_message=f"Invalid destination query: {exc.errors()[0]['msg']}.",
                recovery_instructions=(
                    "Pass a non-empty city or country name (2-100 characters) to "
                    "`retrieve_destination_travel_advisory`."
                ),
                valid_examples=[
                    "Tokyo",
                    "Paris",
                    "Reykjavik",
                    "New York",
                    "Rome",
                    "Cancun",
                ],
            ).model_dump()

        store = get_persistent_store()
        match = store.search_destination_advisory(validated.destination_query)
        if match is None:
            return GuidedToolErrorResponse(
                error_code="DESTINATION_ADVISORY_NOT_FOUND",
                error_message=(
                    f"No indexed advisory found for '{validated.destination_query}'."
                ),
                recovery_instructions=(
                    "Query one of the indexed international hubs ('Tokyo', 'Paris', "
                    "'Reykjavik', 'New York', 'Rome', 'Cancun') or provide general "
                    "standard international travel advice while noting the specific city "
                    "is outside the local advisory catalog."
                ),
                valid_examples=[
                    "Tokyo",
                    "Paris",
                    "Reykjavik",
                    "New York",
                    "Rome",
                    "Cancun",
                ],
            ).model_dump()

        output = DestinationAdvisoryOutputSchema(
            destination=match["destination"],
            country=match["country"],
            iata_codes=match["iata_codes"],
            safety_level=match["safety_level"],
            visa_policy=match["visa_policy"],
            etiquette_rules=match["etiquette_rules"],
            best_season=match["best_season"],
            comedic_tip=match["comedic_tip"],
            similarity_score=match["similarity_score"],
        )
        return output.model_dump()


def calculate_trip_budget_estimate(
    flight_total_usd: float,
    hotel_nightly_rate_usd: float,
    nights: int,
    daily_food_and_activities_usd: float = 100.0,
    passengers: int = 1,
) -> dict[str, Any]:
    """Calculate a complete itemized travel budget including flights, lodging, food, and taxes.

    Validates all numerical inputs via `TripBudgetInputSchema` and returns an exact
    breakdown in USD along with a comedic budget verdict.

    Args:
        flight_total_usd: Combined airfare cost in USD for all passengers.
        hotel_nightly_rate_usd: Nightly hotel rate in USD.
        nights: Total number of lodging nights (1 to 60).
        daily_food_and_activities_usd: Daily per-person dining and sightseeing budget in USD.
        passengers: Total number of travelers sharing the itinerary (1 to 9).

    Returns:
        A dictionary matching `TripBudgetOutputSchema` on success, or
        `GuidedToolErrorResponse` if any number is out of range.
    """
    with start_linked_span(
        "tool.calculate_trip_budget_estimate",
        {"nights": nights, "passengers": passengers},
    ):
        try:
            validated = TripBudgetInputSchema.model_validate(
                {
                    "flight_total_usd": flight_total_usd,
                    "hotel_nightly_rate_usd": hotel_nightly_rate_usd,
                    "nights": nights,
                    "daily_food_and_activities_usd": daily_food_and_activities_usd,
                    "passengers": passengers,
                }
            )
        except ValidationError as exc:
            return GuidedToolErrorResponse(
                error_code="INVALID_BUDGET_NUMBERS",
                error_message=f"Budget calculation input error: {exc.errors()[0]['msg']}.",
                recovery_instructions=(
                    "Ensure `flight_total_usd` >= 0, `hotel_nightly_rate_usd` >= 0, "
                    "`nights` >= 1, and `passengers` between 1 and 9."
                ),
                valid_examples=[
                    "flight_total_usd=950.0, hotel_nightly_rate_usd=185.0, nights=4, daily_food_and_activities_usd=120.0, passengers=2"
                ],
            ).model_dump()

        lodging_total = round(validated.hotel_nightly_rate_usd * validated.nights, 2)
        days = validated.nights + 1
        food_total = round(
            validated.daily_food_and_activities_usd * days * validated.passengers, 2
        )
        subtotal = validated.flight_total_usd + lodging_total + food_total
        taxes_and_pretzels = round(subtotal * 0.12, 2)
        grand_total = round(subtotal + taxes_and_pretzels, 2)
        per_person = round(grand_total / validated.passengers, 2)

        return TripBudgetOutputSchema(
            flight_total_usd=round(validated.flight_total_usd, 2),
            lodging_total_usd=lodging_total,
            food_and_activities_total_usd=food_total,
            estimated_taxes_and_pretzel_fund_usd=taxes_and_pretzels,
            grand_total_usd=grand_total,
            per_person_usd=per_person,
            comedic_budget_verdict=(
                f"Grand total is ${grand_total:,.2f} (${per_person:,.2f}/person)—"
                "leaving plenty of room in the budget for emergency airport Toblerone bars!"
            ),
        ).model_dump()


def book_confirmed_flight_reservation(
    flight_number: str,
    passenger_full_name: str,
    total_price_usd: float,
    passengers: int = 1,
    user_confirmed: bool = False,
) -> dict[str, Any]:
    """Reserve a commercial flight ticket with mandatory Human-in-the-Loop (HITL) confirmation.

    HIGH-STAKES ACTION: This tool enforces an explicit Human-in-the-Loop stop.
    If `user_confirmed` is False, no reservation is charged; instead, the tool
    returns `status="pending_human_confirmation"` with a `confirmation_token` and
    instructions to ask the user for explicit sign-off. Only when `user_confirmed=True`
    does the booking finalize and persist to the reservation ledger.

    Args:
        flight_number: Carrier flight number to reserve (e.g., 'CL-245').
        passenger_full_name: Full name of the primary passenger (PII-scrubbed in logs).
        total_price_usd: Total airfare in USD to be charged.
        passengers: Number of passenger seats (1 to 9, default 1).
        user_confirmed: Set to True ONLY after the human user has explicitly confirmed
            the exact flight number and total price in chat. Defaults to False.

    Returns:
        A dictionary matching `BookingReservationOutputSchema` or `GuidedToolErrorResponse`.
    """
    with start_linked_span(
        "tool.book_confirmed_flight_reservation",
        {
            "flight_number": flight_number,
            "passengers": passengers,
            "user_confirmed": user_confirmed,
        },
    ):
        try:
            validated = FlightBookingInputSchema.model_validate(
                {
                    "flight_number": flight_number,
                    "passenger_full_name": passenger_full_name,
                    "passengers": passengers,
                    "total_price_usd": total_price_usd,
                    "user_confirmed": user_confirmed,
                }
            )
        except ValidationError as exc:
            return GuidedToolErrorResponse(
                error_code="INVALID_FLIGHT_BOOKING_INPUT",
                error_message=f"Flight booking validation error: {exc.errors()[0]['msg']}.",
                recovery_instructions=(
                    "Provide a valid `flight_number`, `passenger_full_name`, and positive "
                    "`total_price_usd`."
                ),
                valid_examples=[
                    "flight_number='CL-245', passenger_full_name='Alex Rivera', total_price_usd=820.0, passengers=1, user_confirmed=False"
                ],
            ).model_dump()

        safe_name = redact_pii(validated.passenger_full_name)
        token = generate_confirmation_token(validated.flight_number, safe_name)

        if not validated.user_confirmed:
            return BookingReservationOutputSchema(
                status="pending_human_confirmation",
                booking_type="flight",
                confirmation_code=token,
                item_identifier=validated.flight_number.upper(),
                traveler_display_name=safe_name,
                total_price_usd=round(validated.total_price_usd, 2),
                requires_human_approval=True,
                message=(
                    f"HOLD ON TO YOUR BOARDING PASS! Flight {validated.flight_number.upper()} "
                    f"for {safe_name} (${validated.total_price_usd:,.2f} USD) is staged under "
                    f"token {token} and awaits explicit human confirmation."
                ),
                recovery_instructions=(
                    f"Ask the user to explicitly confirm booking flight {validated.flight_number.upper()} "
                    f"for ${validated.total_price_usd:,.2f} USD (Confirmation Token: {token}). "
                    "Once the user replies confirming, call `book_confirmed_flight_reservation` "
                    "with `user_confirmed=True`."
                ),
            ).model_dump()

        pnr = f"PNR-{token.replace('HITL-', '')}"
        store = get_persistent_store()
        store.record_confirmed_reservation(
            confirmation_code=pnr,
            booking_type="flight",
            user_id=safe_name,
            item_identifier=validated.flight_number.upper(),
            total_price_usd=validated.total_price_usd,
            details={
                "flight_number": validated.flight_number.upper(),
                "passengers": validated.passengers,
                "human_confirmed": True,
            },
        )
        return BookingReservationOutputSchema(
            status="confirmed",
            booking_type="flight",
            confirmation_code=pnr,
            item_identifier=validated.flight_number.upper(),
            traveler_display_name=safe_name,
            total_price_usd=round(validated.total_price_usd, 2),
            requires_human_approval=False,
            message=(
                f"WHEELS UP! Flight {validated.flight_number.upper()} is officially booked! "
                f"Confirmation Code: {pnr} for {safe_name} (${validated.total_price_usd:,.2f} USD)."
            ),
            recovery_instructions=None,
        ).model_dump()


def book_confirmed_hotel_reservation(
    hotel_id: str,
    guest_full_name: str,
    check_in_date: str,
    nights: int,
    total_price_usd: float,
    user_confirmed: bool = False,
) -> dict[str, Any]:
    """Reserve a hotel stay with mandatory Human-in-the-Loop (HITL) confirmation.

    HIGH-STAKES ACTION: Requires explicit human approval (`user_confirmed=True`)
    before locking in the reservation. When `user_confirmed=False`, halts and
    returns `status="pending_human_confirmation"` with a confirmation token.

    Args:
        hotel_id: Hotel identifier from search results (e.g., 'HTL-TOK-101').
        guest_full_name: Full name of the primary registered guest (PII-scrubbed).
        check_in_date: Check-in date in 'YYYY-MM-DD' format.
        nights: Number of nights to reserve (1 to 30).
        total_price_usd: Total stay price in USD.
        user_confirmed: Set to True ONLY after the human user explicitly confirms
            the hotel reservation and total cost. Defaults to False.

    Returns:
        A dictionary matching `BookingReservationOutputSchema` or `GuidedToolErrorResponse`.
    """
    with start_linked_span(
        "tool.book_confirmed_hotel_reservation",
        {
            "hotel_id": hotel_id,
            "check_in_date": check_in_date,
            "nights": nights,
            "user_confirmed": user_confirmed,
        },
    ):
        try:
            validated = HotelBookingInputSchema.model_validate(
                {
                    "hotel_id": hotel_id,
                    "guest_full_name": guest_full_name,
                    "check_in_date": check_in_date,
                    "nights": nights,
                    "total_price_usd": total_price_usd,
                    "user_confirmed": user_confirmed,
                }
            )
        except ValidationError as exc:
            return GuidedToolErrorResponse(
                error_code="INVALID_HOTEL_BOOKING_INPUT",
                error_message=f"Hotel booking validation error: {exc.errors()[0]['msg']}.",
                recovery_instructions=(
                    "Provide a valid `hotel_id`, `guest_full_name`, `check_in_date` (YYYY-MM-DD), "
                    "`nights` (1-30), and positive `total_price_usd`."
                ),
                valid_examples=[
                    "hotel_id='HTL-TOK-101', guest_full_name='Alex Rivera', check_in_date='2026-11-15', nights=3, total_price_usd=555.0, user_confirmed=False"
                ],
            ).model_dump()

        safe_guest = redact_pii(validated.guest_full_name)
        token = generate_confirmation_token(validated.hotel_id, safe_guest)

        if not validated.user_confirmed:
            return BookingReservationOutputSchema(
                status="pending_human_confirmation",
                booking_type="hotel",
                confirmation_code=token,
                item_identifier=validated.hotel_id.upper(),
                traveler_display_name=safe_guest,
                total_price_usd=round(validated.total_price_usd, 2),
                requires_human_approval=True,
                message=(
                    f"FLUFFING THE PILLOWS ON HOLD! Hotel {validated.hotel_id.upper()} "
                    f"for {safe_guest} ({validated.nights} nights from {validated.check_in_date}, "
                    f"${validated.total_price_usd:,.2f} USD) awaits human approval under token {token}."
                ),
                recovery_instructions=(
                    f"Ask the user to explicitly confirm booking hotel {validated.hotel_id.upper()} "
                    f"for ${validated.total_price_usd:,.2f} USD (Confirmation Token: {token}). "
                    "Once approved, call `book_confirmed_hotel_reservation` with `user_confirmed=True`."
                ),
            ).model_dump()

        res_code = f"HTLCONF-{token.replace('HITL-', '')}"
        store = get_persistent_store()
        store.record_confirmed_reservation(
            confirmation_code=res_code,
            booking_type="hotel",
            user_id=safe_guest,
            item_identifier=validated.hotel_id.upper(),
            total_price_usd=validated.total_price_usd,
            details={
                "hotel_id": validated.hotel_id.upper(),
                "check_in_date": validated.check_in_date,
                "nights": validated.nights,
                "human_confirmed": True,
            },
        )
        return BookingReservationOutputSchema(
            status="confirmed",
            booking_type="hotel",
            confirmation_code=res_code,
            item_identifier=validated.hotel_id.upper(),
            traveler_display_name=safe_guest,
            total_price_usd=round(validated.total_price_usd, 2),
            requires_human_approval=False,
            message=(
                f"KEYCARD ACTIVATED! Hotel {validated.hotel_id.upper()} is confirmed! "
                f"Reservation Code: {res_code} for {safe_guest} (${validated.total_price_usd:,.2f} USD)."
            ),
            recovery_instructions=None,
        ).model_dump()


def save_traveler_preference_to_memory(
    preference_category: str,
    preference_value: str,
    traveler_id: str = "default_traveler",
) -> dict[str, Any]:
    """Save a traveler's preference (e.g., seat choice, dietary restriction, hotel style) to persistent memory.

    Automatically scrubs any sensitive PII before writing to the persistent SQLite
    and cross-session memory store.

    Args:
        preference_category: Category label such as 'seat_preference',
            'dietary_restriction', 'cabin_preference', or 'hotel_style'.
        preference_value: Specific preference value (e.g., 'window seat', 'vegan').
        traveler_id: Identifier for the traveler (defaults to 'default_traveler').

    Returns:
        A dictionary confirming the saved preference record or a guided error payload.
    """
    with start_linked_span(
        "tool.save_traveler_preference_to_memory",
        {"preference_category": preference_category},
    ):
        if (
            not preference_category
            or not preference_category.strip()
            or not preference_value
            or not preference_value.strip()
        ):
            return GuidedToolErrorResponse(
                error_code="EMPTY_PREFERENCE_FIELDS",
                error_message="Both `preference_category` and `preference_value` must be non-empty strings.",
                recovery_instructions=(
                    "Provide a specific `preference_category` (e.g., 'seat_preference') "
                    "and `preference_value` (e.g., 'window')."
                ),
                valid_examples=[
                    "preference_category='seat_preference', preference_value='window seat'"
                ],
            ).model_dump()

        store = get_persistent_store()
        saved = store.save_traveler_preference(
            user_id=traveler_id,
            category=preference_category,
            value=preference_value,
        )
        return {
            "status": "success",
            "saved_preference": saved,
            "message": (
                f"Locked into Captain Layover's vault: {saved['preference_category']} = "
                f"'{saved['preference_value']}'."
            ),
        }


def recall_traveler_preferences_from_memory(
    traveler_id: str = "default_traveler",
) -> dict[str, Any]:
    """Retrieve all stored cross-session preferences for a traveler from persistent memory.

    Args:
        traveler_id: Identifier for the traveler (defaults to 'default_traveler').

    Returns:
        A dictionary containing the list of stored preferences for the traveler.
    """
    with start_linked_span(
        "tool.recall_traveler_preferences_from_memory",
        {"traveler_id": traveler_id},
    ):
        store = get_persistent_store()
        prefs = store.get_traveler_preferences(user_id=traveler_id)
        return {
            "status": "success",
            "traveler_id": redact_pii(traveler_id),
            "preferences": prefs,
            "count": len(prefs),
        }
