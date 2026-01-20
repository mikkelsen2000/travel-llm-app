import requests
import urllib.parse
import streamlit as st
from datetime import date
import random
import traceback  # lets us show full error details (very useful while learning)

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Literal, Optional


# ---------------------------
# 1) DATA MODEL (TripSpec)
# ---------------------------
# This is the structured "shape" we want the LLM to output.
# Pydantic validates types for us (e.g., budget must be an int).
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


# ---------------------------
# 2) LLM PARSING FUNCTION
# ---------------------------
def parse_trip_spec(prefs: str, origin: str, start: str, end: str, budget: int) -> TripSpec:
    """
    Takes the user's free-text preferences + hard constraints (origin/dates/budget),
    calls OpenAI, and returns a TripSpec object.

    If something goes wrong, we "fail loud" with a clear error message.
    """

    # Create an OpenAI client using the API key stored in Streamlit secrets
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # System message tells the model its job
    system = (
        "You convert a user's travel preferences into a TripSpec object. "
        "Be conservative: if something is not stated, choose a reasonable default. "
        "Do not invent specific destinations yet."
    )

    # User message includes user's free text + constraints we already know from the form
    user = f"""
User free-text preferences:
{prefs}

Hard constraints:
- origin_airport: {origin}
- start_date: {start}
- end_date: {end}
- budget_per_night_usd: {budget}
"""

    # Ask the model to produce output matching the TripSpec schema
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text_format=TripSpec,
    )

    # FAIL LOUD: if structured parsing failed, output_parsed can be None
    if response.output_parsed is None:
        raise ValueError("Structured parsing returned None. (SDK/model mismatch or parse failed.)")

    return response.output_parsed


# ---------------------------
# 3) MOCK DATA + MOCK "SEARCH" FUNCTIONS
# ---------------------------
# We use mock data to make the experience feel real before we integrate real APIs.

CURATED_DESTINATIONS = [
    {"name": "Innsbruck, Austria", "tags": ["mountains", "adventure", "walkable", "transit"]},
    {"name": "Salzburg, Austria", "tags": ["mountains", "culture", "walkable"]},
    {"name": "Geneva + Chamonix, France", "tags": ["mountains", "adventure", "scenic"]},
    {"name": "Calgary + Banff, Canada", "tags": ["mountains", "adventure", "lakes"]},
    {"name": "Vancouver + Whistler, Canada", "tags": ["mountains", "adventure", "food"]},
    {"name": "Salt Lake City, USA", "tags": ["mountains", "adventure", "budget"]},
    {"name": "Denver + Boulder, USA", "tags": ["mountains", "adventure", "food"]},
    {"name": "Queenstown, New Zealand", "tags": ["mountains", "adventure", "scenic"]},
]


def pick_destinations(trip_spec: TripSpec) -> List[dict]:
    """
    Pick top 3 destinations from a curated list using a simple scoring approach.
    (Later, replace this with a real destination search.)
    """
    scored = []

    for d in CURATED_DESTINATIONS:
        score = 0

        # Prefer mountains if user prefers mountains
        if trip_spec.prefer_mountains_over_cities and "mountains" in d["tags"]:
            score += 3

        # If they want high adventure, give "adventure" destinations extra points
        if trip_spec.adventure_level == "high" and "adventure" in d["tags"]:
            score += 2

        scored.append((score, d))

    # Sort by score descending and take top 3
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [d for _, d in scored[:3]]

    # Add a readable reason string for the UI
    for d in top:
        d["reason"] = "Matches your preferences: " + ", ".join(d["tags"][:3])

    return top


def get_mock_hotels(trip_spec: TripSpec, destination_name: str) -> List[dict]:
    """
    Return fake hotels with placeholder images.
    We attempt to filter by the user's budget to make it feel plausible.
    """
    base = [
        {"name": "Alpine Budget Hotel", "price_per_night": 110},
        {"name": "Mountain View Inn", "price_per_night": 95},
        {"name": "Hostel Central", "price_per_night": 55},
        {"name": "Cozy Lodge", "price_per_night": 120},
        {"name": "Transit-Friendly Stay", "price_per_night": 85},
    ]

    budget = trip_spec.budget_per_night_usd
    filtered = [h for h in base if h["price_per_night"] <= budget]

    # If budget is extremely low, don't starve the UI during prototyping
    if len(filtered) < 3:
        filtered = base

    out = []
    for i, h in enumerate(filtered[:5], start=1):
        out.append(
            {
                **h,
                "image": f"https://picsum.photos/seed/{destination_name.replace(' ', '')}-hotel{i}/900/550",
                "neighborhood": random.choice(["Central", "Old Town", "Near transit", "Quiet area"]),
            }
        )
    return out

@st.cache_data(ttl=60 * 60)  # cache for 1 hour to avoid hitting Yelp rate limits
def yelp_search_restaurants(destination_name: str, cuisines: tuple, price_filter: str, limit: int = 8):
    """
    Calls Yelp Fusion API to find restaurants in/near destination_name.

    destination_name: e.g. "Innsbruck, Austria"
    cuisines: tuple of strings, e.g. ("ramen", "tacos")
    price_filter: Yelp uses "1,2,3,4" where 1=$ and 4=$$$$
    limit: number of results to return
    """

    api_key = st.secrets.get("YELP_API_KEY")
    if not api_key:
        raise ValueError("Missing YELP_API_KEY in Streamlit secrets.")

    url = "https://api.yelp.com/v3/businesses/search"  # Yelp business search endpoint :contentReference[oaicite:4]{index=4}

    # If user listed cuisines, use them as the search term; otherwise use "restaurants"
    if cuisines:
        term = ", ".join([c for c in cuisines if c])  # join cuisines into a single query
    else:
        term = "restaurants"

    headers = {
        # Yelp auth: Authorization: Bearer API_KEY :contentReference[oaicite:5]{index=5}
        "Authorization": f"Bearer {api_key}",
        "accept": "application/json",
    }

    params = {
        "term": term,
        "location": destination_name,   # simplest approach; later we can use lat/long
        "categories": "restaurants",
        "limit": limit,
        "sort_by": "rating",
        "price": price_filter,          # e.g. "1,2"
    }

    resp = requests.get(url, headers=headers, params=params, timeout=20)

    # If Yelp returns an error, show a useful message
    if resp.status_code != 200:
        raise RuntimeError(f"Yelp API error {resp.status_code}: {resp.text}")

    data = resp.json()
    businesses = data.get("businesses", [])

    # Normalize Yelp fields into a simple list of dicts your UI can render
    out = []
    for b in businesses:
        out.append({
            "name": b.get("name"),
            "image_url": b.get("image_url"),
            "yelp_url": b.get("url"),  # link to Yelp business page (good for attribution) :contentReference[oaicite:6]{index=6}
            "rating": b.get("rating"),
            "review_count": b.get("review_count"),
            "price": b.get("price"),
            "categories": ", ".join([c["title"] for c in b.get("categories", []) if "title" in c]),
            "address": ", ".join(b.get("location", {}).get("display_address", [])),
        })

    return out



def get_mock_restaurants(trip_spec: TripSpec, destination_name: str) -> List[dict]:
    """
    Return fake restaurants with placeholder images.
    Uses user's preferred cuisines if available.
    """
    cuisines = trip_spec.preferred_cuisines or ["Ramen", "Tacos", "Bakery", "Casual"]

    base = [
        {"name": "Ramen Corner", "type": cuisines[0] if len(cuisines) > 0 else "Ramen"},
        {"name": "Taco Spot", "type": cuisines[1] if len(cuisines) > 1 else "Tacos"},
        {"name": "Cozy Bakery", "type": "Bakery"},
        {"name": "Casual Alpine Eats", "type": "Casual"},
        {"name": "Noodle House", "type": "Noodles"},
        {"name": "Cafe with Pastries", "type": "Cafe"},
        {"name": "Street Food Market", "type": "Street Food"},
        {"name": "Quick Bites", "type": "Casual"},
    ]

    out = []
    for i, r in enumerate(base[:8], start=1):
        out.append(
            {
                **r,
                "image": f"https://picsum.photos/seed/{destination_name.replace(' ', '')}-food{i}/900/550",
                "price_level": random.choice(["$", "$$", "$$$"]),
            }
        )
    return out


def get_mock_flights(trip_spec: TripSpec, destination_name: str) -> List[dict]:
    """
    Return fake flight options.
    Implements simple nonstop preference logic.
    """
    origin = trip_spec.origin_airport
    nonstop_pref = trip_spec.nonstop_preference

    options = [
        {"route": f"{origin} → {destination_name}", "stops": 0, "duration": "2h 15m", "price": 210},
        {"route": f"{origin} → {destination_name}", "stops": 1, "duration": "6h 40m", "price": 180},
        {"route": f"{origin} → {destination_name}", "stops": 1, "duration": "8h 10m", "price": 160},
    ]

    if nonstop_pref == "require":
        nonstop = [o for o in options if o["stops"] == 0]
        return nonstop if nonstop else [options[0]]

    # Otherwise: fewer stops first, then cheaper
    options.sort(key=lambda x: (x["stops"], x["price"]))
    return options[:3]


# ---------------------------
# 4) STREAMLIT UI (THE ACTUAL APP)
# ---------------------------
st.set_page_config(page_title="Travel Finder (MVP)", layout="wide")
st.title("Travel Finder (MVP) v0.3")
st.write("Paste your preferences. We'll turn them into a trip plan.")

if st.button("Start over (clear cached results)"):
    st.session_state.clear()
    st.rerun()


with st.form("trip_form"):
    prefs = st.text_area(
        "Your travel prompt",
        value=(
            "I'm a student traveling on a budget. Max $120/night. "
            "I prefer mountains and adventure over big cities. "
            "I like places like Banff and Innsbruck. "
            "Food: Italian, Mexican, cozy bakeries; avoid loud clubs."
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
        # 1) Parse user text into structured TripSpec using the LLM
        trip_spec = parse_trip_spec(
            prefs=prefs,
            origin=origin,
            start=str(start),
            end=str(end),
            budget=int(budget),
        )

        # SAFETY CHECK: if parse_trip_spec returned None, stop here
        if trip_spec is None:
            raise ValueError("parse_trip_spec returned None (expected a TripSpec object).")

        # 2) Show TripSpec for debugging
        with st.expander("Debug: TripSpec (structured)"):
            st.json(trip_spec.model_dump())

        # 3) Show raw inputs for debugging
        with st.expander("Debug: raw inputs"):
            st.json(
                {
                    "prefs": prefs,
                    "origin_airport": origin,
                    "dates": {"start": str(start), "end": str(end)},
                    "budget_per_night_usd": int(budget),
                }
            )

        # 4) MOCK RESULTS (so the app feels real before APIs)
        st.subheader("Results (mock for now)")
        st.info("These results are mock data so we can build the experience before real travel APIs.")

        destinations = pick_destinations(trip_spec)
        if not destinations:
            st.warning("No destinations found (unexpected).")
            st.stop()

        # Build a list of destination names for the radio button UI
        destination_names = [d["name"] for d in destinations]

        # Let the user choose which destination to explore
        top_destination = st.radio(
            "Choose a destination to explore:",
            options=destination_names,
            index=0,  # default: first suggestion
        )

        # ----------------------------
        # (C) Budget -> Yelp price filter
        # Yelp uses: 1=$, 2=$$, 3=$$$, 4=$$$$
        # We'll map hotel budget to a rough restaurant price preference.
        # ----------------------------
        if trip_spec.budget_per_night_usd <= 60:
            yelp_price = "1"
        elif trip_spec.budget_per_night_usd <= 120:
            yelp_price = "1,2"
        elif trip_spec.budget_per_night_usd <= 200:
            yelp_price = "1,2,3"
        else:
            yelp_price = "1,2,3,4"

        # ----------------------------
        # Cache results so they don't reshuffle on every Streamlit rerun
        # ----------------------------
        cache_key = f"mock::{top_destination}::{trip_spec.budget_per_night_usd}::{trip_spec.origin_airport}"

        if cache_key not in st.session_state:
            # Try to fetch real restaurants from Yelp.
            # If Yelp fails (missing key, rate limit, location not found, etc.),
            # we fall back to mock restaurants so the app still works.
            try:
                yelp_restaurants = yelp_search_restaurants(
                    destination_name=top_destination,
                    cuisines=tuple(trip_spec.preferred_cuisines),  # tuple is cache-friendly
                    price_filter=yelp_price,
                    limit=8,
                )
                restaurant_source = "Yelp"
            except Exception as yelp_err:
                yelp_restaurants = get_mock_restaurants(trip_spec, top_destination)
                restaurant_source = "Mock (Yelp failed)"
                # Save the error so you can inspect it in the UI
                st.session_state["last_yelp_error"] = str(yelp_err)

            st.session_state[cache_key] = {
                "hotels": get_mock_hotels(trip_spec, top_destination),
                "restaurants": yelp_restaurants,
                "flights": get_mock_flights(trip_spec, top_destination),
                "restaurant_source": restaurant_source,
            }


        hotels = st.session_state[cache_key]["hotels"]
        restaurants = st.session_state[cache_key]["restaurants"]
        flights = st.session_state[cache_key]["flights"]
        restaurant_source = st.session_state[cache_key].get("restaurant_source", "Mock")


        st.subheader("Suggested destinations")
        for d in destinations:
            st.write(f"**{d['name']}** — {d['reason']}")

        st.divider()
        st.subheader(f"Top pick: {top_destination}")

        # Hotels cards
        st.markdown("### Hotels (mock)")
        cols = st.columns(3)
        for i, h in enumerate(hotels):
            with cols[i % 3]:
                st.image(h["image"], use_container_width=True)
                st.write(f"**{h['name']}**")
                st.write(f"${h['price_per_night']}/night • {h['neighborhood']}")

        # Restaurants cards
        st.markdown(f"### Restaurants ({restaurant_source})")
        st.caption("Powered by Yelp")  # basic attribution; keep links to Yelp pages too :contentReference[oaicite:8]{index=8}

        cols = st.columns(4)
        for i, r in enumerate(restaurants):
            with cols[i % 4]:
                # Yelp returns image_url; mocks return "image"
                img = r.get("image_url") or r.get("image")
                if img:
                    st.image(img, use_container_width=True)

                name = r.get("name", "Unknown")
                yelp_url = r.get("yelp_url")

                # If we have a Yelp URL, make the name clickable (good for attribution)
                if yelp_url:
                    st.markdown(f"**[{name}]({yelp_url})**")
                else:
                    st.write(f"**{name}**")

                # Show some extra Yelp details if present
                details = []
                if r.get("categories"):
                    details.append(r["categories"])
                if r.get("price"):
                    details.append(r["price"])
                if r.get("rating") is not None:
                    details.append(f"{r['rating']}⭐ ({r.get('review_count', 0)})")
                if details:
                    st.write(" • ".join(details))

                if r.get("address"):
                    st.caption(r["address"])


        # Flights list
        st.markdown("### Flights (mock)")
        for f in flights:
            st.write(f"**{f['route']}** — {f['stops']} stop(s), {f['duration']}, ~${f['price']}")

    except Exception:
        # Show a friendly error + full traceback for you (developer) to diagnose
        st.error("TripSpec parsing failed. Check your API key and logs.")
        with st.expander("Debug: error details (full traceback)"):
            st.code(traceback.format_exc())
