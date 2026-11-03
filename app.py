
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
from sklearn.inspection import permutation_importance

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

defaults = {
    "trained_model": None,
    "model_name": None,
    "problem_type": None,
    "target_column": None,
    "feature_columns": None,
    "model_metrics": None,
    "comparison": None,
    "training_data": None,
    "prediction_results": None,
    "y_test": None,
    "y_predictions": None,
    "y_probabilities": None,
    "X_test": None
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
        "Dataset Explorer",
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
        
        # =============================================
        # ADVANCED MODEL ANALYTICS
        # =============================================

        st.divider()
        st.header("Advanced Model Analytics")

        model = st.session_state.trained_model
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_predictions
        X_test = st.session_state.X_test
        y_prob = st.session_state.y_probabilities

        if y_test is not None and y_pred is not None:

            # -----------------------------------------
            # CONFUSION MATRIX
            # -----------------------------------------

            if st.session_state.problem_type == "Classification":

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_test,
                    y_pred
                )

                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))

                display = ConfusionMatrixDisplay(
                    confusion_matrix=cm
                )

                display.plot(
                    ax=ax_cm,
                    cmap="Blues",
                    colorbar=False
                )

                st.pyplot(fig_cm)
                plt.close(fig_cm)

                # -------------------------------------
                # ROC CURVE
                # -------------------------------------

                st.subheader("ROC Curve")

                if (
                    y_prob is not None
                    and y_prob.shape[1] == 2
                ):

                    classes = model.named_steps["model"].classes_
                    positive_class = classes[1]

                    y_binary = (
                        np.asarray(y_test) == positive_class
                    ).astype(int)

                    fpr, tpr, thresholds = roc_curve(
                        y_binary,
                        y_prob[:, 1]
                    )

                    auc_value = roc_auc_score(
                        y_binary,
                        y_prob[:, 1]
                    )

                    fig_roc = px.area(
                        x=fpr,
                        y=tpr,
                        title=f"ROC Curve (AUC = {auc_value:.4f})",
                        labels={
                            "x": "False Positive Rate",
                            "y": "True Positive Rate"
                        }
                    )

                    fig_roc.add_shape(
                        type="line",
                        x0=0,
                        y0=0,
                        x1=1,
                        y1=1,
                        line=dict(
                            dash="dash",
                            color="gray"
                        )
                    )

                    st.plotly_chart(
                        fig_roc,
                        use_container_width=True
                    )

                    st.metric(
                        "ROC AUC Score",
                        f"{auc_value:.4f}"
                    )

                else:

                    st.info(
                        "ROC Curve is available for binary "
                        "classification models with probability output."
                    )

            # -----------------------------------------
            # FEATURE IMPORTANCE
            # -----------------------------------------

            st.divider()
            st.subheader("Feature Importance")

            if st.button("Calculate Feature Importance"):

                with st.spinner("Analyzing feature importance..."):

                    if st.session_state.problem_type == "Classification":
                        scoring_method = "f1_weighted"
                    else:
                        scoring_method = "r2"

                    importance_result = permutation_importance(
                        model,
                        X_test,
                        y_test,
                        n_repeats=5,
                        random_state=42,
                        scoring=scoring_method,
                        n_jobs=-1
                    )

                    importance_df = pd.DataFrame({
                        "Feature": X_test.columns,
                        "Importance": importance_result.importances_mean
                    })

                    importance_df = importance_df.sort_values(
                        "Importance",
                        ascending=False
                    )

                    fig_importance = px.bar(
                        importance_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Feature Importance Analysis"
                    )

                    fig_importance.update_layout(
                        yaxis={"categoryorder": "total ascending"}
                    )

                    st.plotly_chart(
                        fig_importance,
                        use_container_width=True
                    )

                    st.dataframe(
                        importance_df,
                        use_container_width=True
                    )

        else:

            st.info(
                "Train a model to view advanced analytics."
            )



# =====================================================
# DATASET EXPLORER
# =====================================================

elif page == "Dataset Explorer":

    st.subheader("🔍 Dataset Explorer | දත්ත කට්ටලය පරීක්ෂා කිරීම")

    st.write(
        "Upload a CSV or Excel dataset to automatically examine its "
        "structure, quality and statistical properties."
    )
    st.caption(
        "CSV හෝ Excel dataset එක upload කළාම එහි rows, columns, "
        "missing values, duplicates සහ statistics ස්වයංක්‍රීයව පෙන්වයි."
    )

    explorer_file = st.file_uploader(
        "Upload Dataset / Dataset එක upload කරන්න",
        type=["csv", "xlsx"],
        key="explorer_upload"
    )

    if explorer_file is not None:

        try:
            explorer_df = read_dataset(explorer_file)

            if explorer_df.empty:
                st.warning("The uploaded dataset is empty. / Upload කළ dataset එක හිස්.")
            else:
                row_count, column_count = explorer_df.shape
                missing_total = int(explorer_df.isna().sum().sum())
                duplicate_total = int(explorer_df.duplicated().sum())

                st.success("Dataset loaded successfully! / Dataset එක සාර්ථකව load වුණා!")

                st.markdown("### 1. Dataset Overview | Dataset එකේ මූලික විස්තර")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Observations / Rows", f"{row_count:,}")
                c2.metric("Variables / Columns", f"{column_count:,}")
                c3.metric("Missing Values", f"{missing_total:,}")
                c4.metric("Duplicate Rows", f"{duplicate_total:,}")

                st.markdown("### 2. Dataset Preview | මුල් දත්ත")
                st.dataframe(explorer_df.head(10), use_container_width=True)

                st.markdown("### 3. Column Information | Column විස්තර")

                column_info = pd.DataFrame({
                    "Column": explorer_df.columns.astype(str),
                    "Data Type": [str(dtype) for dtype in explorer_df.dtypes],
                    "Non-Null Values": explorer_df.notna().sum().values,
                    "Missing Values": explorer_df.isna().sum().values,
                    "Missing (%)": (
                        explorer_df.isna().mean().values * 100
                    ).round(2),
                    "Unique Values": explorer_df.nunique(dropna=True).values
                })

                st.dataframe(
                    column_info.sort_values("Missing (%)", ascending=False),
                    use_container_width=True
                )

                st.markdown("### 4. Data Quality | දත්තවල ගුණාත්මකභාවය")

                q1, q2, q3 = st.columns(3)
                q1.metric(
                    "Complete Cells",
                    f"{(explorer_df.size - missing_total):,}"
                )
                q2.metric(
                    "Missing Data Rate",
                    f"{(missing_total / explorer_df.size * 100):.2f}%"
                    if explorer_df.size else "0%"
                )
                q3.metric(
                    "Duplicate Row Rate",
                    f"{(duplicate_total / row_count * 100):.2f}%"
                    if row_count else "0%"
                )

                if missing_total > 0:
                    st.warning(
                        "Missing values were detected. These may need "
                        "imputation or removal before model training. "
                        "/ Missing values තිබෙන නිසා training කිරීමට පෙර "
                        "ඒවා handle කළ යුතු විය හැක."
                    )
                else:
                    st.success(
                        "No missing values detected. / Missing values හමු නොවීය."
                    )

                if duplicate_total > 0:
                    st.info(
                        f"{duplicate_total} duplicate rows detected. "
                        "Review them before deciding whether to remove them. "
                        "/ Duplicate rows ඉවත් කළ යුතුද කියා පරීක්ෂා කරන්න."
                    )

                st.markdown("### 5. Statistical Summary | සංඛ්‍යානමය සාරාංශය")

                numeric_df = explorer_df.select_dtypes(include=np.number)
                categorical_df = explorer_df.select_dtypes(exclude=np.number)

                if not numeric_df.empty:
                    st.write(
                        "**Numerical variables – Mean, Standard Deviation, "
                        "Minimum, Maximum, etc. / සංඛ්‍යාත්මක variables වල විස්තර**"
                    )
                    st.dataframe(
                        numeric_df.describe().T,
                        use_container_width=True
                    )
                else:
                    st.info("No numerical columns found. / සංඛ්‍යාත්මක columns නොමැත.")

                if not categorical_df.empty:
                    st.write(
                        "**Categorical variables – Count, Unique values and "
                        "Most frequent value / වර්ගීකරණ දත්තවල සාරාංශය**"
                    )
                    categorical_summary = pd.DataFrame({
                        "Column": categorical_df.columns,
                        "Non-Null Count": categorical_df.notna().sum().values,
                        "Unique Values": categorical_df.nunique(dropna=True).values,
                        "Most Frequent Value": [
                            (
                                categorical_df[col].mode(dropna=True).iloc[0]
                                if not categorical_df[col].mode(dropna=True).empty
                                else "N/A"
                            )
                            for col in categorical_df.columns
                        ]
                    })
                    st.dataframe(
                        categorical_summary,
                        use_container_width=True
                    )

                st.markdown("### 6. Target Distribution | Target එකේ දත්ත ව්‍යාප්තිය")

                target_choice = st.selectbox(
                    "Select a target/output column (optional) / Target column එක තෝරන්න",
                    ["-- Select target column --"] + explorer_df.columns.tolist(),
                    key="explorer_target"
                )

                if target_choice != "-- Select target column --":
                    target_series = explorer_df[target_choice].dropna()

                    if target_series.empty:
                        st.info("The selected target column contains no usable values.")
                    elif (
                        pd.api.types.is_numeric_dtype(target_series)
                        and target_series.nunique() > 20
                    ):
                        target_fig = px.histogram(
                            x=target_series,
                            nbins=30,
                            title=f"Distribution of {target_choice}",
                            labels={"x": target_choice, "y": "Count"}
                        )
                        st.plotly_chart(target_fig, use_container_width=True)
                    else:
                        distribution = (
                            target_series.astype(str)
                            .value_counts()
                            .head(30)
                            .rename_axis("Target Value")
                            .reset_index(name="Count")
                        )
                        target_fig = px.bar(
                            distribution,
                            x="Target Value",
                            y="Count",
                            title=f"Distribution of {target_choice}"
                        )
                        st.plotly_chart(target_fig, use_container_width=True)

                    st.caption(
                        "This chart shows how values are distributed in the "
                        "selected target column. / තෝරාගත් target column එකේ "
                        "values පැතිරී ඇති ආකාරය මෙම chart එකෙන් පෙන්වයි."
                    )

                st.download_button(
                    "⬇️ Download Column Analysis (CSV)",
                    data=column_info.to_csv(index=False).encode("utf-8"),
                    file_name="PredictAI_Dataset_Analysis.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Dataset Analysis Error: {e}")


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

                        # Save evaluation data for advanced analytics
                        best_predictions = best_model.predict(X_test)

                        st.session_state.y_test = y_test.copy()
                        st.session_state.y_predictions = best_predictions
                        st.session_state.X_test = X_test.copy()

                        if (
                            problem_type == "Classification"
                            and hasattr(best_model, "predict_proba")
                        ):
                            st.session_state.y_probabilities = (
                                best_model.predict_proba(X_test)
                            )
                        else:
                            st.session_state.y_probabilities = None

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
