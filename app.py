import streamlit as st
import numpy as np
import pandas as pd
import random
import plotly.express as px

st.set_page_config(layout="wide")

st.title("🤖 100 Buyer Bots Simulation (UK)")

# -----------------------------
# PRODUCT INPUT
# -----------------------------
price = st.slider("Price (£)", 10, 300, 80)
rgb = st.checkbox("RGB Lighting")
mechanical = st.checkbox("Mechanical")

product = {
    "price": price,
    "features": []
}

if rgb:
    product["features"].append("RGB")
if mechanical:
    product["features"].append("mechanical")

# -----------------------------
# REGIONS
# -----------------------------
regions = {
    "London": {"income": 3500, "tech": 0.9},
    "South East": {"income": 3200, "tech": 0.85},
    "North West": {"income": 2500, "tech": 0.75},
    "Scotland": {"income": 2700, "tech": 0.7},
    "Wales": {"income": 2300, "tech": 0.6},
}

region_names = list(regions.keys())

# -----------------------------
# BOT GENERATION
# -----------------------------
def create_bot(bot_id):
    region = random.choice(region_names)
    r = regions[region]

    return {
        "id": bot_id,
        "region": region,
        "budget": np.random.normal(r["income"], 400),
        "likes_rgb": random.random(),
        "needs_keyboard": random.random(),
        "tech_interest": r["tech"]
    }

# -----------------------------
# DECISION FUNCTION
# -----------------------------
def will_buy(bot, product):
    score = 0

    if product["price"] < bot["budget"] * 0.05:
        score += 0.5
    else:
        score -= 0.3

    if "RGB" in product["features"]:
        score += bot["likes_rgb"] * 0.3

    score += bot["needs_keyboard"] * 0.5
    score += bot["tech_interest"] * 0.2

    return score > 0.5

# -----------------------------
# RUN SIMULATION
# -----------------------------
if st.button("Run Simulation"):

    bots = []
    buyers = 0

    for i in range(100):
        bot = create_bot(i)
        bot["buy"] = will_buy(bot, product)

        if bot["buy"]:
            buyers += 1

        bots.append(bot)

    df = pd.DataFrame(bots)

    st.subheader("📊 Summary")
    st.metric("Total Buyers", buyers)
    st.metric("Conversion Rate", f"{buyers}%")

    region_stats = df.groupby("region")["buy"].mean().reset_index()
    region_stats["buy"] *= 100

    st.subheader("🌍 Buyers by Region")
    fig_bar = px.bar(region_stats, x="region", y="buy", title="Conversion % per Region")
    st.plotly_chart(fig_bar, use_container_width=True)

    buy_counts = df["buy"].value_counts().reset_index()
    buy_counts.columns = ["Buy", "Count"]

    fig_pie = px.pie(buy_counts, names="Buy", values="Count", title="Buy vs Not Buy")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("💰 Budget vs Decision")

    fig_scatter = px.scatter(
        df,
        x="budget",
        y="needs_keyboard",
        color="buy",
        hover_data=["region"],
        title="Bot Behavior"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("📄 Bot Data")
    st.dataframe(df)
