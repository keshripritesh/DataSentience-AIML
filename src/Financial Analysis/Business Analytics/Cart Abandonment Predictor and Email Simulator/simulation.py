import streamlit as st
import pickle
import numpy as np
import random
import os

st.set_page_config(page_title="🛒 Cart Abandonment Email Simulator", layout="centered")
st.title("🛒 Real-Time Cart Abandonment Email Simulator")
st.markdown("Simulate an e-commerce user session and check whether the model predicts cart abandonment. If yes, trigger a marketing email and see if the user returns.")

#Model selection
model_choice = st.selectbox("Choose Model", ["Random Forest", "Logistic Regression", "XGBoost"])

model_files = {
    "Random Forest": "random_forest_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "XGBoost": "xg_boosting_model.pkl"
}
model_path = os.path.join("notebook", model_files[model_choice])
with open(model_path, "rb") as f:
    model = pickle.load(f)

#Input mode toggle
st.subheader("Session Input")
manual_mode = st.toggle("Manual Mode", value=False)

def simulate_checkout_behavior():
    checkout_initiated = random.choices([0, 1, 2, 3], weights=[0.6, 0.25, 0.1, 0.05])[0]

    if checkout_initiated == 0:
        checkout_confirmed = 0
    elif checkout_initiated == 1:
        checkout_confirmed = random.choices([0, 1], weights=[0.7, 0.3])[0]
    elif checkout_initiated == 2:
        checkout_confirmed = random.choices([0, 1, 2], weights=[0.5, 0.3, 0.2])[0]
    else:
        checkout_confirmed = random.choices([1, 2], weights=[0.4, 0.6])[0]

    customer_login = random.choices([0, 1, 2], weights=[0.7, 0.25, 0.05])[0]

    return checkout_initiated, checkout_confirmed, customer_login


#Random session generator
checkout_initiated, checkout_confirmed, customer_login = simulate_checkout_behavior()

def generate_random_session():
    return {
        "No_Checkout_Initiated": checkout_initiated,
        "No_Checkout_Confirmed": checkout_confirmed,
        "No_Customer_Login": customer_login,
        "Session_Activity_Count": random.randint(1, 35),
        "No_Page_Viewed": random.randint(1, 10),
        "No_Items_Added_InCart": random.randint(0, 10)
    }


submitted = False
session_input = {}

if manual_mode:
    st.info("Manual Input Mode Enabled")

    # Checkout Initiated
    checkout_initiated = st.selectbox(
        "Checkout Initiated", options=[0, 1, 2, 3], index=1
    )

    # Restrict Checkout Confirmed
    checkout_confirmed = st.selectbox(
        "Checkout Confirmed",
        options=list(range(0, checkout_initiated + 1))
    )

    customer_login = st.selectbox(
        "Customer Login (0=Not Logged In, 1=Once, 2=Multiple Times)", options=[0, 1, 2]
    )

    activity_count = st.slider("Session Activity Count", 1, 35, 10)
    pages_viewed = st.slider("Pages Viewed", 1, 10, 3)
    items_added = st.slider("Items Added to Cart", 0, 10, 2)

    session_input = {
        "No_Checkout_Initiated": checkout_initiated,
        "No_Checkout_Confirmed": checkout_confirmed,
        "No_Customer_Login": customer_login,
        "Session_Activity_Count": activity_count,
        "No_Page_Viewed": pages_viewed,
        "No_Items_Added_InCart": items_added
    }

    submitted = st.button("Submit for Prediction")  
    if submitted:
        st.write("Session input submitted:", session_input)

else:
    st.info("Auto-generate Mode Enabled")
    session_input = generate_random_session()

    st.write("### Auto-Generated Session Input")
    for key, value in session_input.items():
        st.write(f"**{key}**: {value}")

    submitted = st.button("Simulate Prediction")

# Make prediction
if submitted:
    st.subheader("Prediction Result")
    input_vector = np.array([list(session_input.values())])

    try:
        prediction = model.predict(input_vector)[0]
    except ValueError as e:
        st.error(f"⚠️ Model Input Error:\n{e}")
    else:
        abandoned = prediction == 1
        if abandoned:
            st.error("🛒 Cart was abandoned!")
            st.markdown("✉️ Sending recovery email...")
            returned = random.choices([True, False], weights=[0.4, 0.6])[0]
            if returned:
                st.success("🎉 User returned after email!")
            else:
                st.warning("🚫 User did not return despite the email.")
        else:
            st.success("✅ No cart abandonment. No email sent.")

    st.caption(f"Powered by **{model_choice}** model.")
