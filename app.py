
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

import matplotlib.pyplot as plt
import json
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="PredictAI",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
h1, h2, h3 {
    color: #173B5E;
}
[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e5eaf0;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE
# =====================================================

defaults = {
    "trained_model": None,
    "model_name": None,
    "problem_type": None,
    "target_column": None,
    "feature_columns": None,
    "model_metrics": None,
    "comparison": None,
    "training_data": None,
    "prediction_results": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# DATASET FUNCTIONS
# =====================================================

def read_dataset(file):

    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)

    return pd.read_excel(file)


def prepare_dataset(df, target):

    data = df.copy()

    data = data.replace([np.inf, -np.inf], np.nan)

    data = data.dropna(
        axis=1,
        how="all"
    )

    if target not in data.columns:
        raise ValueError("Target column not found.")

    data = data.dropna(
        subset=[target]
    )

    if len(data) < 10:
        raise ValueError("At least 10 usable records are required.")

    y = data[target]

    X = data.drop(
        columns=[target]
    )

    # Remove constant columns

    constant_columns = [
        col for col in X.columns
        if X[col].nunique(dropna=True) <= 1
    ]

    X = X.drop(
        columns=constant_columns
    )

    if X.shape[1] == 0:
        raise ValueError("No useful feature columns found.")

    # Convert numeric-looking text columns

    for col in X.columns:

        if X[col].dtype == "object":

            converted = pd.to_numeric(
                X[col],
                errors="coerce"
            )

            if converted.notna().mean() > 0.9:
                X[col] = converted

    # Remove unsupported columns

    X = X.select_dtypes(
        include=["number", "object", "category", "bool"]
    )

    if X.shape[1] == 0:
        raise ValueError("No supported feature columns found.")

    return X, y


# =====================================================
# PREPROCESSING
# =====================================================

def build_preprocessor(X):

    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        col for col in X.columns
        if col not in numeric_columns
    ]

    transformers = []

    if numeric_columns:

        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ])

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    if categorical_columns:

        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ])

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    return ColumnTransformer(
        transformers=transformers
    )


# =====================================================
# ALGORITHMS
# =====================================================

CLASSIFICATION_MODELS = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=15,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


REGRESSION_MODELS = {

    "Linear Regression": LinearRegression(),

    "Decision Tree Regressor": DecisionTreeRegressor(
        max_depth=15,
        random_state=42
    ),

    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees Regressor": ExtraTreesRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}


# =====================================================
# PDF REPORT FUNCTION
# =====================================================

def generate_pdf(comparison, model_name, problem_type):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "PredictAI - Model Training Report",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            f"Selected Model: {model_name}",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            f"Problem Type: {problem_type}",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 20))

    table_data = [
        comparison.columns.tolist()
    ]

    for _, row in comparison.iterrows():

        table_data.append([
            str(value)
            for value in row.tolist()
        ])

    table = Table(
        table_data,
        repeatRows=1
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173B5E")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ])
    )

    elements.append(table)

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            "Generated by PredictAI",
            styles["Italic"]
        )
    )

    doc.build(elements)

    return buffer.getvalue()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🤖 PredictAI")

st.sidebar.caption(
    "Universal AI Training Platform"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "AI Model Training",
        "Predictions",
        "Reports",
        "About"
    ]
)

st.sidebar.divider()

if st.session_state.trained_model is not None:

    st.sidebar.success("AI Model Ready")

else:

    st.sidebar.info("Train an AI Model")


# =====================================================
# MAIN HEADER
# =====================================================

st.title("🤖 PredictAI")

st.caption(
    "Train AI Models | Compare Performance | Make Predictions"
)


# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.subheader("Platform Dashboard")

    if st.session_state.trained_model is None:

        st.info(
            "Welcome to PredictAI! Upload a dataset "
            "and train your first AI model."
        )

    else:

        metrics = st.session_state.model_metrics

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Selected Model",
            st.session_state.model_name
        )

        c2.metric(
            "Problem Type",
            st.session_state.problem_type
        )

        primary_metric = (
            "F1 Score"
            if st.session_state.problem_type == "Classification"
            else "R2 Score"
        )

        c3.metric(
            primary_metric,
            f"{metrics[primary_metric]:.4f}"
        )

        st.divider()

        st.subheader("Model Comparison")

        st.dataframe(
            st.session_state.comparison,
            use_container_width=True
        )

        fig = px.bar(
            st.session_state.comparison,
            x="Model",
            y=primary_metric,
            title="Model Performance Comparison"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =====================================================
# AI MODEL TRAINING
# =====================================================

elif page == "AI Model Training":

    st.subheader("AI Model Training Center")

    st.write(
        "Upload a dataset and select the target column "
        "to train machine learning models."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel Dataset",
        type=["csv", "xlsx"],
        key="training_upload"
    )

    if uploaded_file is not None:

        try:

            df = read_dataset(uploaded_file)

            st.success("Dataset uploaded successfully!")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Total Records",
                f"{len(df):,}"
            )

            c2.metric(
                "Total Columns",
                len(df.columns)
            )

            c3.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

            st.subheader("Dataset Preview")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            target = st.selectbox(
                "Select Target Column",
                df.columns.tolist()
            )

            problem_type = st.radio(
                "Select Problem Type",
                [
                    "Classification",
                    "Regression"
                ],
                horizontal=True
            )

            test_size = st.slider(
                "Test Dataset Percentage",
                10,
                40,
                20,
                5
            )

            available_models = (
                CLASSIFICATION_MODELS
                if problem_type == "Classification"
                else REGRESSION_MODELS
            )

            selected_models = st.multiselect(
                "Select AI Algorithms",
                list(available_models.keys()),
                default=list(available_models.keys())
            )

            st.divider()

            if st.button(
                "🚀 Train AI Models",
                type="primary",
                use_container_width=True
            ):

                if not selected_models:

                    st.error(
                        "Please select at least one algorithm."
                    )

                else:

                    try:

                        X, y = prepare_dataset(
                            df,
                            target
                        )

                        if problem_type == "Classification":

                            if y.nunique() < 2:
                                raise ValueError(
                                    "Classification needs at least two classes."
                                )

                            if y.nunique() > 50:
                                raise ValueError(
                                    "Target contains too many classes."
                                )

                            if y.value_counts().min() < 2:
                                raise ValueError(
                                    "Each class needs at least two records."
                                )

                            stratify = y

                        else:

                            y = pd.to_numeric(
                                y,
                                errors="coerce"
                            )

                            valid = y.notna()

                            X = X.loc[valid]
                            y = y.loc[valid]

                            if len(y) < 10:
                                raise ValueError(
                                    "Not enough numeric target records."
                                )

                            stratify = None

                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=test_size / 100,
                            random_state=42,
                            stratify=stratify
                        )

                        results = []
                        trained_models = {}

                        progress = st.progress(0)
                        status = st.empty()

                        for index, name in enumerate(selected_models):

                            status.write(
                                f"Training {name}..."
                            )

                            pipeline = Pipeline([
                                (
                                    "preprocessing",
                                    build_preprocessor(X_train)
                                ),
                                (
                                    "model",
                                    available_models[name]
                                )
                            ])

                            pipeline.fit(
                                X_train,
                                y_train
                            )

                            predictions = pipeline.predict(
                                X_test
                            )

                            if problem_type == "Classification":

                                accuracy = accuracy_score(
                                    y_test,
                                    predictions
                                )

                                precision = precision_score(
                                    y_test,
                                    predictions,
                                    average="weighted",
                                    zero_division=0
                                )

                                recall = recall_score(
                                    y_test,
                                    predictions,
                                    average="weighted",
                                    zero_division=0
                                )

                                f1 = f1_score(
                                    y_test,
                                    predictions,
                                    average="weighted",
                                    zero_division=0
                                )

                                result = {
                                    "Model": name,
                                    "Accuracy": round(accuracy, 4),
                                    "Precision": round(precision, 4),
                                    "Recall": round(recall, 4),
                                    "F1 Score": round(f1, 4)
                                }

                            else:

                                mae = mean_absolute_error(
                                    y_test,
                                    predictions
                                )

                                rmse = np.sqrt(
                                    mean_squared_error(
                                        y_test,
                                        predictions
                                    )
                                )

                                r2 = r2_score(
                                    y_test,
                                    predictions
                                )

                                result = {
                                    "Model": name,
                                    "MAE": round(mae, 4),
                                    "RMSE": round(rmse, 4),
                                    "R2 Score": round(r2, 4)
                                }

                            results.append(result)

                            trained_models[name] = pipeline

                            progress.progress(
                                (index + 1) / len(selected_models)
                            )

                        comparison = pd.DataFrame(results)

                        if problem_type == "Classification":

                            comparison = comparison.sort_values(
                                "F1 Score",
                                ascending=False
                            )

                        else:

                            comparison = comparison.sort_values(
                                "R2 Score",
                                ascending=False
                            )

                        best_name = comparison.iloc[0]["Model"]

                        best_model = trained_models[best_name]

                        st.session_state.trained_model = best_model

                        st.session_state.model_name = best_name

                        st.session_state.problem_type = problem_type

                        st.session_state.target_column = target

                        st.session_state.feature_columns = X.columns.tolist()

                        st.session_state.model_metrics = (
                            comparison.iloc[0].to_dict()
                        )

                        st.session_state.comparison = comparison

                        st.session_state.training_data = df.copy()

                        st.session_state.prediction_results = None

                        status.success(
                            "Model training completed successfully!"
                        )

                        st.balloons()

                        st.subheader("Model Comparison Results")

                        st.dataframe(
                            comparison,
                            use_container_width=True
                        )

                        st.success(
                            f"Best Model: {best_name}"
                        )

                        if problem_type == "Classification":

                            fig = px.bar(
                                comparison,
                                x="Model",
                                y=[
                                    "Accuracy",
                                    "Precision",
                                    "Recall",
                                    "F1 Score"
                                ],
                                barmode="group",
                                title="Classification Performance"
                            )

                        else:

                            fig = px.bar(
                                comparison,
                                x="Model",
                                y=[
                                    "MAE",
                                    "RMSE",
                                    "R2 Score"
                                ],
                                barmode="group",
                                title="Regression Performance"
                            )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        model_buffer = BytesIO()

                        joblib.dump(
                            best_model,
                            model_buffer
                        )

                        st.download_button(
                            "⬇️ Download Best Trained Model",
                            data=model_buffer.getvalue(),
                            file_name="PredictAI_Best_Model.joblib",
                            mime="application/octet-stream"
                        )

                    except Exception as e:

                        st.error(
                            f"Training Error: {e}"
                        )

        except Exception as e:

            st.error(
                f"Dataset Loading Error: {e}"
            )


# =====================================================
# PREDICTIONS
# =====================================================

elif page == "Predictions":

    st.subheader("AI Predictions")

    model = st.session_state.trained_model

    if model is None:

        st.warning(
            "Please train an AI model first."
        )

    else:

        st.success(
            f"Active Model: {st.session_state.model_name}"
        )

        st.write(
            f"Problem Type: {st.session_state.problem_type}"
        )

        new_file = st.file_uploader(
            "Upload New Data for Prediction",
            type=["csv", "xlsx"],
            key="prediction_upload"
        )

        if new_file is not None:

            try:

                new_df = read_dataset(new_file)

                feature_columns = st.session_state.feature_columns

                missing = [
                    col for col in feature_columns
                    if col not in new_df.columns
                ]

                if missing:

                    st.error(
                        "Missing input columns: "
                        + ", ".join(missing)
                    )

                else:

                    X_new = new_df[feature_columns].copy()

                    if st.button(
                        "🔮 Generate Predictions",
                        type="primary"
                    ):

                        predictions = model.predict(
                            X_new
                        )

                        output = new_df.copy()

                        output["AI Prediction"] = predictions

                        if (
                            st.session_state.problem_type == "Classification"
                            and hasattr(model, "predict_proba")
                        ):

                            probabilities = model.predict_proba(
                                X_new
                            )

                            output["Prediction Confidence (%)"] = (
                                probabilities.max(axis=1) * 100
                            ).round(2)

                        st.session_state.prediction_results = output

                        st.success(
                            "Predictions generated successfully!"
                        )

            except Exception as e:

                st.error(
                    f"Prediction Error: {e}"
                )

        if st.session_state.prediction_results is not None:

            st.subheader("Prediction Results")

            results = st.session_state.prediction_results

            st.dataframe(
                results,
                use_container_width=True
            )

            st.download_button(
                "Download Predictions CSV",
                results.to_csv(index=False).encode("utf-8"),
                file_name="PredictAI_Predictions.csv",
                mime="text/csv"
            )


# =====================================================
# REPORTS
# =====================================================

elif page == "Reports":

    st.subheader("AI Model Reports")

    comparison = st.session_state.comparison

    if comparison is None:

        st.info(
            "Train an AI model before generating reports."
        )

    else:

        st.subheader("Model Performance")

        st.dataframe(
            comparison,
            use_container_width=True
        )

        # CSV REPORT

        csv_data = comparison.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📄 Download CSV Report",
            csv_data,
            file_name="PredictAI_Comparison.csv",
            mime="text/csv"
        )

        # EXCEL REPORT

        excel_buffer = BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl"
        ) as writer:

            comparison.to_excel(
                writer,
                sheet_name="Model Comparison",
                index=False
            )

            if st.session_state.prediction_results is not None:

                st.session_state.prediction_results.to_excel(
                    writer,
                    sheet_name="Predictions",
                    index=False
                )

        st.download_button(
            "📊 Download Excel Report",
            excel_buffer.getvalue(),
            file_name="PredictAI_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # PDF REPORT

        pdf_data = generate_pdf(
            comparison,
            st.session_state.model_name,
            st.session_state.problem_type
        )

        st.download_button(
            "📕 Download PDF Report",
            pdf_data,
            file_name="PredictAI_Report.pdf",
            mime="application/pdf"
        )


# =====================================================
# ABOUT
# =====================================================

elif page == "About":

    st.subheader("About PredictAI")

    st.write("""
    PredictAI is a universal machine learning platform.

    Users can upload datasets, select target columns,
    train multiple classification and regression models,
    compare algorithm performance and generate predictions.
    """)

    st.divider()

    st.subheader("Classification Algorithms")

    st.write("""
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Extra Trees
    """)

    st.subheader("Regression Algorithms")

    st.write("""
    - Linear Regression
    - Decision Tree Regressor
    - Random Forest Regressor
    - Extra Trees Regressor
    """)

    st.info(
        "PredictAI is an educational machine learning platform. "
        "Validate model performance before real-world use."
    )

    st.caption(
        "PredictAI | Universal AI Training Platform"
    )
