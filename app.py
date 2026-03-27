import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("pricing_model.pkl")

st.title("🧠 AI Dynamic Pricing Agent")

# Inputs
demand = st.slider("Demand", 0, 100, 50)
competitor_price = st.number_input("Competitor Price", value=2000)
cost_price = st.number_input("Your Cost Price", value=1500)

# Predict
if st.button("Predict Price"):
    price = model.predict([[demand, competitor_price]])[0]

    st.success(f"Recommended Price: ₹{int(price)}")

    # Profit calculation
    profit = price - cost_price
    st.info(f"Estimated Profit: ₹{int(profit)}")

    # Smart suggestion
    if price < competitor_price:
        st.warning("💡 Strategy: Lower price than competitor → High sales volume")
    else:
        st.success("💡 Strategy: Premium pricing → Higher profit per sale")

    # Graph
    demands = list(range(10, 100, 10))
    prices = [model.predict([[d, competitor_price]])[0] for d in demands]

    plt.plot(demands, prices)
    plt.xlabel("Demand")
    plt.ylabel("Price")
    plt.title("Demand vs Price")

    st.pyplot(plt)
