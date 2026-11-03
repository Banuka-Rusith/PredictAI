import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
from pathlib import Path
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="PredictAI | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
h1, h2, h3 {
    color: #15304b;
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5eaf0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING ----------------

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "random_forest_model.joblib")

    with open(MODEL_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)

    with open(MODEL_DIR / "model_info.json") as f:
        model_info = json.load(f)

    return model, feature_columns, model_info


try:
    model, feature_columns, model_info = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ---------------- SESSION STATE ----------------

if "results" not in st.session_state:
    st.session_state.results = None

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None


# ---------------- DATA PREPROCESSING ----------------

REQUIRED_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type"
]


def prepare_data(df):

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    data = df[REQUIRED_COLUMNS].copy()

    if data.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    for col in REQUIRED_COLUMNS[:-1]:
        data[col] = pd.to_numeric(data[col], errors="raise")

    data["Type"] = data["Type"].astype(str).str.strip().str.upper()

    if not data["Type"].isin(["L", "M", "H"]).all():
        raise ValueError("Type must contain only L, M or H.")

    data["Type"] = pd.Categorical(
        data["Type"],
        categories=["H", "L", "M"]
    )

    data = pd.get_dummies(
        data,
        columns=["Type"],
        drop_first=True,
        dtype=int
    )

    data = data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return data


# ---------------- PREDICTIONS ----------------

def make_predictions(df):

    X = prepare_data(df)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results = df.copy()

    results["Predicted Condition"] = np.where(
        predictions == 1,
        "Failure Risk",
        "Normal"
    )

    results["Failure Probability (%)"] = (
        probabilities * 100
    ).round(2)

    return results


# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ PredictAI")

st.sidebar.caption(
    "AI-Based Predictive Maintenance System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Dataset Upload",
        "AI Predictions",
        "Analytics",
        "Reports",
        "About"
    ]
)

st.sidebar.divider()
st.sidebar.success("AI Model Loaded")


# ---------------- HEADER ----------------

st.title("⚙️ PredictAI")
st.caption("Intelligent Machine Failure Prediction Platform")


# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.subheader("Dashboard Overview")

    if st.session_state.results is None:

        st.info(
            "Please upload a dataset and run AI predictions "
            "to view your dashboard."
        )

    else:

        results = st.session_state.results

        total = len(results)

        failures = (
            results["Predicted Condition"] == "Failure Risk"
        ).sum()

        normal = total - failures

        failure_rate = (
            failures / total * 100 if total else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Machines", f"{total:,}")
        c2.metric("Predicted Failures", f"{failures:,}")
        c3.metric("Normal Machines", f"{normal:,}")
        c4.metric("Failure Rate", f"{failure_rate:.2f}%")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            fig = px.pie(
                names=["Normal", "Failure Risk"],
                values=[normal, failures],
                title="Machine Condition Distribution",
                hole=0.45
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:

            fig = px.histogram(
                results,
                x="Failure Probability (%)",
                nbins=20,
                title="Failure Probability Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Predictions")

        st.dataframe(
            results.tail(10),
            use_container_width=True
        )


# ---------------- DATASET UPLOAD ----------------

elif page == "Dataset Upload":

    st.subheader("Upload Machine Dataset")

    st.write(
        "Upload your AI4I-compatible CSV or Excel dataset."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.uploaded_data = df

            st.success("Dataset uploaded successfully!")

            c1, c2, c3 = st.columns(3)

            c1.metric("Rows", len(df))
            c2.metric("Columns", len(df.columns))
            c3.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

            st.subheader("Dataset Preview")

            st.dataframe(
                df.head(20),
                use_container_width=True
            )

            if st.button("🚀 Run AI Predictions"):

                with st.spinner("AI model is analyzing data..."):

                    predictions = make_predictions(df)

                    st.session_state.results = predictions

                st.success("AI predictions completed!")

        except Exception as e:

            st.error(f"Error: {e}")


# ---------------- AI PREDICTIONS ----------------

elif page == "AI Predictions":

    st.subheader("AI Machine Failure Predictions")

    results = st.session_state.results

    if results is None:

        st.warning("Upload a dataset and run predictions first.")

    else:

        st.success(
            f"Successfully analyzed {len(results)} machine records."
        )

        failure_results = results[
            results["Predicted Condition"] == "Failure Risk"
        ]

        st.metric(
            "Machines with Failure Risk",
            len(failure_results)
        )

        st.dataframe(
            results,
            use_container_width=True
        )

        if not failure_results.empty:

            st.subheader("⚠️ Failure Risk Alerts")

            st.dataframe(
                failure_results,
                use_container_width=True
            )

        else:

            st.success("No machine failure risks predicted.")


# ---------------- ANALYTICS ----------------

elif page == "Analytics":

    st.subheader("Machine Data Analytics")

    results = st.session_state.results

    if results is None:

        st.info("Upload a dataset and run predictions first.")

    else:

        col1, col2 = st.columns(2)

        with col1:

            fig = px.histogram(
                results,
                x="Rotational speed [rpm]",
                title="Rotational Speed Distribution",
                nbins=30
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:

            fig = px.histogram(
                results,
                x="Torque [Nm]",
                title="Torque Distribution",
                nbins=30
            )

            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            results,
            x="Rotational speed [rpm]",
            y="Torque [Nm]",
            color="Predicted Condition",
            title="Rotational Speed vs Torque"
        )

        st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(
            results,
            x="Failure Probability (%)",
            color="Predicted Condition",
            title="Failure Probability Analysis",
            nbins=20
        )

        st.plotly_chart(fig, use_container_width=True)


# ---------------- REPORTS ----------------

elif page == "Reports":

    st.subheader("Generate and Download Reports")

    results = st.session_state.results

    if results is None:

        st.info("Run AI predictions before generating reports.")

    else:

        st.success("Your reports are ready to download.")

        # CSV REPORT

        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📄 Download CSV Report",
            data=csv_data,
            file_name="PredictAI_Report.csv",
            mime="text/csv"
        )

        # EXCEL REPORT

        excel_buffer = BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl"
        ) as writer:

            results.to_excel(
                writer,
                sheet_name="Predictions",
                index=False
            )

            summary = pd.DataFrame({
                "Metric": [
                    "Total Machines",
                    "Predicted Failures",
                    "Normal Machines",
                    "Failure Rate (%)"
                ],
                "Value": [
                    len(results),
                    (results["Predicted Condition"] == "Failure Risk").sum(),
                    (results["Predicted Condition"] == "Normal").sum(),
                    round(
                        (results["Predicted Condition"] == "Failure Risk").mean() * 100,
                        2
                    )
                ]
            })

            summary.to_excel(
                writer,
                sheet_name="Summary",
                index=False
            )

        st.download_button(
            "📊 Download Excel Report",
            data=excel_buffer.getvalue(),
            file_name="PredictAI_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # PDF REPORT

        pdf_buffer = BytesIO()

        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4
        )

        styles = getSampleStyleSheet()

        elements = []

        elements.append(
            Paragraph("PredictAI - Machine Analysis Report", styles["Title"])
        )

        elements.append(Spacer(1, 20))

        total = len(results)

        failures = (
            results["Predicted Condition"] == "Failure Risk"
        ).sum()

        normal = total - failures

        summary_data = [
            ["Metric", "Value"],
            ["Total Machines", str(total)],
            ["Predicted Failures", str(failures)],
            ["Normal Machines", str(normal)],
            [
                "Failure Rate",
                f"{(failures / total * 100 if total else 0):.2f}%"
            ]
        ]

        elements.append(Table(summary_data))
        elements.append(Spacer(1, 20))

        elements.append(
            Paragraph("Recent Prediction Results", styles["Heading2"])
        )

        sample = results.head(15)

        table_data = [
            ["Machine", "Condition", "Failure Probability"]
        ]

        for i, row in sample.iterrows():

            table_data.append([
                str(i + 1),
                str(row["Predicted Condition"]),
                f'{row["Failure Probability (%)"]:.2f}%'
            ])

        elements.append(Table(table_data))

        doc.build(elements)

        st.download_button(
            "📕 Download PDF Report",
            data=pdf_buffer.getvalue(),
            file_name="PredictAI_Report.pdf",
            mime="application/pdf"
        )


# ---------------- ABOUT ----------------

elif page == "About":

    st.subheader("About PredictAI")

    st.write("""
    PredictAI is an AI-based predictive maintenance
    application designed to identify possible machine
    failures using operational sensor data.

    The application uses a Random Forest classification
    model trained using the AI4I 2020 Predictive
    Maintenance Dataset.
    """)

    st.divider()

    st.subheader("Model Information")

    c1, c2 = st.columns(2)

    c1.metric(
        "Model Accuracy",
        f'{model_info["accuracy"]:.2f}%'
    )

    c2.metric(
        "Number of Estimators",
        model_info["n_estimators"]
    )

    st.write("**Model:**", model_info["model_name"])
    st.write("**Dataset:**", model_info["dataset"])
    st.write("**Precision:**", f'{model_info["precision"]:.2f}%')
    st.write("**Recall:**", f'{model_info["recall"]:.2f}%')
    st.write("**F1 Score:**", f'{model_info["f1_score"]:.2f}%')

    st.divider()

    st.caption("PredictAI | AI-Based Predictive Maintenance System")
