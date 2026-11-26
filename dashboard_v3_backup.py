import streamlit as st
import requests
import pandas as pd

API = "http://localhost:8001"

st.set_page_config(
    page_title="EdgeGuard | PaySim V3",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ EdgeGuard")
st.subheader("Edge Financial Fraud Scanner")
st.caption("PaySim dataset | Random Forest V3 | NVIDIA Jetson")

try:
    health = requests.get(f"{API}/health", timeout=3)
    if health.ok:
        st.success("V3 Detection API Online")
    else:
        st.error("V3 API unavailable")
except requests.RequestException:
    st.error("V3 API offline. Check the V3 container.")

st.divider()

monitor, history_tab, about = st.tabs([
    "Transaction Monitor", "Detection History", "Model Information"
])

with monitor:
    st.subheader("Assess a transaction")

    c1, c2 = st.columns(2)

    with c1:
        amount = st.number_input(
            "Transaction amount",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )
        step = st.number_input(
            "Simulation step",
            min_value=1,
            max_value=743,
            value=400
        )
        transaction_type = st.selectbox(
            "Transaction type",
            ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
        )

    with c2:
        sender = st.selectbox(
            "Sender category", ["CUSTOMER", "MERCHANT"]
        )
        recipient = st.selectbox(
            "Recipient category", ["CUSTOMER", "MERCHANT"]
        )

    if st.button("Analyse Transaction", type="primary"):
        payload = {
            "step": step,
            "amount": amount,
            "transaction_type": transaction_type,
            "sender_category": sender,
            "recipient_category": recipient
        }

        try:
            with st.spinner("Running local Random Forest inference..."):
                response = requests.post(
                    f"{API}/predict", json=payload, timeout=20
                )
                response.raise_for_status()
                result = response.json()

            st.session_state["v3_result"] = result
            st.session_state["v3_payload"] = payload

        except requests.RequestException as e:
            st.error(f"Prediction failed: {e}")

    result = st.session_state.get("v3_result")

    if result:
        st.divider()
        st.subheader("Detection Result")

        a, b, c = st.columns(3)
        a.metric("Classification", result["prediction"])
        b.metric(
            "Fraud Probability",
            f"{result['fraud_probability'] * 100:.2f}%"
        )
        c.metric("Inference Latency", f"{result['latency_ms']:.2f} ms")

        st.progress(
            min(max(result["fraud_probability"], 0.0), 1.0)
        )

        if result["prediction"] == "FRAUD":
            st.error("Elevated risk detected. Human review recommended.")
        else:
            st.success("Classified as legitimate by the model.")

        st.caption(
            f"Decision threshold: "
            f"{result['decision_threshold'] * 100:.2f}%"
        )

        if st.button("Generate Prediction Summary"):
            try:
                explanation = requests.post(
                    f"{API}/explain",
                    json=st.session_state["v3_payload"],
                    timeout=20
                )
                explanation.raise_for_status()
                st.info(explanation.json()["explanation"])
            except requests.RequestException as e:
                st.error(f"Summary unavailable: {e}")

        st.caption(
            "Prototype risk assessment using simulated PaySim data. "
            "A model prediction does not confirm actual financial fraud."
        )

with history_tab:
    st.subheader("Recent detection history")
    if st.button("Refresh history"):
        st.rerun()

    try:
        response = requests.get(f"{API}/history", timeout=5)
        response.raise_for_status()
        records = response.json()
        if records:
            st.dataframe(pd.DataFrame(records), use_container_width=True)
        else:
            st.info("No transactions assessed yet.")
    except requests.RequestException:
        st.warning("History API unavailable.")

with about:
    st.subheader("Model information")
    st.write("**Model:** Random Forest V3")
    st.write("**Dataset:** PaySim simulated mobile-money transactions")
    st.write("**Training transactions:** 253,643")
    st.write("**Training fraud cases retained:** 3,643")
    st.write("**Test ROC-AUC:** 0.8812")
    st.write("**Test PR-AUC:** 0.2321")
    st.write("**Test fraud precision:** 0.86")
    st.write("**Test fraud recall:** 0.19")
    st.write("**Inference:** Local model execution")
    st.warning(
        "The test set has a different fraud prevalence from the training "
        "and validation sets. Results should be interpreted accordingly."
    )
