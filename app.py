import streamlit as st
from datetime import date

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
