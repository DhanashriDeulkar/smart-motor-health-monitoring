
import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Smart Motor Health Monitoring System",
    page_icon="⚙️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load("motor_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, model_columns = load_model()

# ---------------- LOAD DATASET ----------------
@st.cache_data
def load_dataset():
    path = os.path.join("data", "ai4i2020.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ---------------- NAVIGATION ----------------
st.markdown("""
<style>
div.st-key-home_btn button,
div.st-key-prediction_btn button,
div.st-key-analytics_btn button,
div.st-key-performance_btn button {
    width: 100%;
    min-height: 52px;
    border-radius: 10px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"

c1, c2, c3, c4 = st.columns(4)

with c1:
    if st.button("🏠 Home", key="home_btn",
                 use_container_width=True):
        st.session_state.selected_page = "Home"

with c2:
    if st.button("⚙️ Motor Prediction", key="prediction_btn",
                 use_container_width=True):
        st.session_state.selected_page = "Motor Prediction"

with c3:
    if st.button("📊 Data Analytics", key="analytics_btn",
                 use_container_width=True):
        st.session_state.selected_page = "Data Analytics"

with c4:
    if st.button("🧠 Model Performance", key="performance_btn",
                 use_container_width=True):
        st.session_state.selected_page = "Model Performance"

selected_page = st.session_state.selected_page
st.divider()

# ---------------- HOME PAGE ----------------
if selected_page == "Home":

    st.title("⚙️ Smart Motor Health Monitoring System")
    st.subheader("AI-Powered Predictive Maintenance")

    st.write(
        "This project uses a Machine Learning model to estimate "
        "motor failure risk from sensor readings."
    )

    st.info(
        "Use the navigation buttons above to explore the project."
    )

    c1, c2, c3 = st.columns(3)

    c1.info("🤖 **AI Model**\n\nRandom Forest Classifier")
    c2.info("📊 **Dataset**\n\nAI4I 2020 Predictive Maintenance")
    c3.info("🛠️ **Technology**\n\nPython and Streamlit")

    df = load_dataset()

    if df is not None:
        total = len(df)
        failures = int(df["Machine failure"].sum())
        normal = total - failures

        a, b, c = st.columns(3)
        a.metric("Total Records", f"{total:,}")
        b.metric("Normal Records", f"{normal:,}")
        c.metric("Failure Records", f"{failures:,}")

    st.caption(
        "Educational project. Predictions do not guarantee actual machine safety."
    )

# ---------------- MOTOR PREDICTION PAGE ----------------
elif selected_page == "Motor Prediction":

    st.title("⚙️ Motor Health Prediction")
    st.write("Enter the motor sensor readings below.")

    with st.form("motor_prediction_form"):

        c1, c2 = st.columns(2)

        with c1:
            motor_type = st.selectbox(
                "Motor Type", ["L", "M", "H"]
            )
            air_temp = st.number_input(
                "Air Temperature (K)",
                min_value=250.0, max_value=350.0,
                value=298.1
            )
            process_temp = st.number_input(
                "Process Temperature (K)",
                min_value=250.0, max_value=400.0,
                value=308.6
            )

        with c2:
            speed = st.number_input(
                "Rotational Speed (rpm)",
                min_value=0, max_value=5000, value=1500
            )
            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0, max_value=500.0, value=40.0
            )
            tool_wear = st.number_input(
                "Tool Wear (min)",
                min_value=0, max_value=500, value=100
            )

        submitted = st.form_submit_button(
            "🔍 Predict Motor Health",
            type="primary",
            use_container_width=True
        )

    if submitted:
        input_df = pd.DataFrame([{
            "Type": motor_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }])

        input_df = pd.get_dummies(
            input_df, columns=["Type"]
        )

        input_df = input_df.reindex(
            columns=model_columns, fill_value=0
        )

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        a, b = st.columns(2)
        a.metric(
            "Estimated Failure Probability",
            f"{probability * 100:.2f}%"
        )
        b.metric(
            "Predicted Motor Status",
            "Potential Failure" if prediction == 1
            else "No Failure Predicted"
        )

        if probability >= 0.70:
            st.error("🚨 Estimated Risk Level: HIGH")
        elif probability >= 0.30:
            st.warning("⚠️ Estimated Risk Level: MEDIUM")
        else:
            st.success("✅ Estimated Risk Level: LOW")

        if prediction == 1:
            st.error(
                "Potential failure detected. Inspect the machine "
                "and verify sensor readings."
            )
        else:
            st.success(
                "The model did not predict a failure for these readings."
            )

        st.caption(
            "Educational estimate only. This is not a guarantee of machine safety."
        )

# ---------------- DATA ANALYTICS PAGE ----------------
elif selected_page == "Data Analytics":

    st.title("📊 Dataset Analytics")

    df = load_dataset()

    if df is not None:
        total = len(df)
        failures = int(df["Machine failure"].sum())
        normal = total - failures
        failure_rate = failures / total * 100 if total else 0

        a, b, c, d = st.columns(4)
        a.metric("Total Records", f"{total:,}")
        b.metric("Normal Records", f"{normal:,}")
        c.metric("Failure Records", f"{failures:,}")
        d.metric("Failure Rate", f"{failure_rate:.2f}%")

        st.subheader("Machine Condition Overview")

        chart_data = pd.DataFrame({
            "Condition": ["Normal", "Failure"],
            "Records": [normal, failures]
        }).set_index("Condition")

        st.bar_chart(chart_data)

        st.subheader("📈 Sensor Reading Distribution")

        sensor = st.selectbox(
            "Choose a sensor",
            [
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]"
            ]
        )

        if sensor in df.columns:
            st.line_chart(df[sensor].reset_index(drop=True))

        st.caption(
            "The horizontal axis represents dataset record order, not actual time."
        )

    else:
        st.error("Dataset not found: data/ai4i2020.csv")

# ---------------- MODEL PERFORMANCE PAGE ----------------
elif selected_page == "Model Performance":

    st.title("🧠 AI Model Performance")

    evaluation_path = "evaluation_results.pkl"

    if os.path.exists(evaluation_path):
        evaluation = joblib.load(evaluation_path)

        a, b, c, d = st.columns(4)

        a.metric("Accuracy", f"{evaluation['accuracy'] * 100:.2f}%")
        b.metric("Precision", f"{evaluation['precision'] * 100:.2f}%")
        c.metric("Recall", f"{evaluation['recall'] * 100:.2f}%")
        d.metric("F1-score", f"{evaluation['f1_score'] * 100:.2f}%")

        st.subheader("📊 Confusion Matrix")

        cm = pd.DataFrame(
            evaluation["confusion_matrix"],
            index=["Actual Normal", "Actual Failure"],
            columns=["Predicted Normal", "Predicted Failure"]
        )

        st.dataframe(cm, use_container_width=True)

        st.caption(
            "Rows show actual classes; columns show predicted classes. "
            "Class 0 = Normal; Class 1 = Failure."
        )

        st.subheader("🎯 Feature Importance")

        importance_df = pd.DataFrame(
            evaluation["feature_importance"]
        )

        importance_df = importance_df.sort_values(
            "Importance", ascending=True
        )

        st.bar_chart(importance_df.set_index("Feature"))

        st.caption(
            "Feature importance indicates relative model contribution, "
            "not causation."
        )

    else:
        st.warning(
            "Evaluation results not found. Run train_model.py first."
        )

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Smart Motor Health Monitoring System | AIML Project"
)
