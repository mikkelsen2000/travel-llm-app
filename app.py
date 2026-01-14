import streamlit as st
from datetime import date
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class TripSpec(BaseModel):
    traveler_type: Optional[str] = Field(None, description="e.g., student, solo, family, couple")
    budget_per_night_usd: int
    origin_airport: str
    start_date: str
    end_date: str

    prefer_mountains_over_cities: bool = True
    adventure_level: Literal["low", "medium", "high"] = "medium"

    preferred_cuisines: List[str] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list)
    restaurant_vibes: List[str] = Field(default_factory=list)

    nonstop_preference: Literal["require", "prefer", "dont_care"] = "prefer"
    max_flight_hours: Optional[int] = None


def parse_trip_spec(prefs: str, origin: str, start: str, end: str, budget: int) -> TripSpec:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    system = (
        "You convert a user's travel preferences into a TripSpec object. "
        "Be conservative: if something is not stated, choose a reasonable default. "
        "Do not invent specific destinations yet."
    )

    user = f"""
User free-text preferences:
{prefs}

Hard constraints:
- origin_airport: {origin}
- start_date: {start}
- end_date: {end}
- budget_per_night_usd: {budget}
"""

    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text_format=TripSpec,
    )

    return response.output_parsed

st.set_page_config(page_title="Travel Finder (MVP)", layout="wide")

st.title("Travel Finder (MVP) V0.1")
st.write("Paste your preferences. We'll turn them into a trip plan.")

with st.form("trip_form"):
    prefs = st.text_area(
        "Your travel prompt",
        value=(
            "I'm a student traveling on a budget. Max $120/night. "
            "I prefer mountains and adventure over big cities. "
            "I like places like Banff and Innsbruck. "
            "Food: ramen, tacos, cozy bakeries; avoid loud clubs."
        ),
        height=140,
    )
    origin = st.text_input("Origin airport (IATA)", value="SEA")
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start date", value=date.today())
    with col2:
        end = st.date_input("End date", value=date.today())
    budget = st.number_input("Max hotel budget per night (USD)", min_value=20, value=120, step=10)

    submitted = st.form_submit_button("Plan my trip")

if submitted:

    try:
        trip_spec = parse_trip_spec(
            prefs=prefs,
            origin=origin,
            start=str(start),
            end=str(end),
            budget=int(budget),
        )
        with st.expander("Debug: TripSpec (structured)"):
            st.json(trip_spec.model_dump())
    except Exception as e:
        st.error("TripSpec parsing failed. Check your API key and logs.")
        with st.expander("Debug: error details"):
            st.write(e)


    st.subheader("Results (placeholder)")
    st.info("Next: we’ll convert your prompt into structured TripSpec JSON, then call real APIs.")

    # This creates a collapsible section (accordion)
    with st.expander("Debug: show raw inputs"):
        # Anything inside this block only appears when you expand it
        st.json(
            {
                "prefs": prefs,
                "origin_airport": origin,
                "dates": {"start": str(start), "end": str(end)},
                "budget_per_night_usd": budget,
            }
        )


    st.subheader("Results (placeholder)")
    st.info("Next: we’ll convert your prompt into structured TripSpec JSON, then call real APIs.")
