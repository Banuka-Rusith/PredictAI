
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from io import BytesIO
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
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
import time
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

def build_preprocessor(X, scaling_method="StandardScaler"):

    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        col for col in X.columns
        if col not in numeric_columns
    ]

    transformers = []

    if numeric_columns:

        numeric_steps = [
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]

        if scaling_method == "StandardScaler":
            numeric_steps.append(("scaler", StandardScaler()))
        elif scaling_method == "RobustScaler":
            numeric_steps.append(("scaler", RobustScaler()))

        numeric_pipeline = Pipeline(numeric_steps)

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
        "Model Comparison",
        "Automatic Dual Analysis",
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
            # REGRESSION EVALUATION
            # -----------------------------------------

            else:
                st.subheader(
                    "Regression Evaluation | Regression ප්‍රතිඵල විශ්ලේෂණය"
                )

                actual_values = np.asarray(y_test, dtype=float)
                predicted_values = np.asarray(y_pred, dtype=float)
                residuals = actual_values - predicted_values

                mae_value = mean_absolute_error(
                    actual_values, predicted_values
                )
                mse_value = mean_squared_error(
                    actual_values, predicted_values
                )
                rmse_value = float(np.sqrt(mse_value))
                r2_value = r2_score(
                    actual_values, predicted_values
                )

                metric1, metric2, metric3, metric4 = st.columns(4)
                metric1.metric("MAE", f"{mae_value:.4f}")
                metric2.metric("MSE", f"{mse_value:.4f}")
                metric3.metric("RMSE", f"{rmse_value:.4f}")
                metric4.metric("R² Score", f"{r2_value:.4f}")

                st.caption(
                    "MAE, MSE and RMSE measure prediction errors; lower values "
                    "are generally better. R² measures explained variation; "
                    "higher is generally better. "
                    "/ Error metrics අඩු වීම සාමාන්‍යයෙන් හොඳයි. R² වැඩි වීම සාමාන්‍යයෙන් හොඳයි."
                )

                st.markdown("#### Actual vs Predicted | සැබෑ සහ පුරෝකථන අගයන්")
                regression_plot_df = pd.DataFrame({
                    "Actual": actual_values,
                    "Predicted": predicted_values
                })

                actual_pred_fig = px.scatter(
                    regression_plot_df,
                    x="Actual",
                    y="Predicted",
                    title="Actual vs Predicted Values",
                    opacity=0.65
                )

                min_value = float(
                    min(actual_values.min(), predicted_values.min())
                )
                max_value = float(
                    max(actual_values.max(), predicted_values.max())
                )
                actual_pred_fig.add_shape(
                    type="line",
                    x0=min_value,
                    y0=min_value,
                    x1=max_value,
                    y1=max_value,
                    line=dict(dash="dash", color="red")
                )
                st.plotly_chart(
                    actual_pred_fig,
                    use_container_width=True
                )
                st.caption(
                    "Points closer to the dashed diagonal line indicate more "
                    "accurate predictions. / තිත් dashed line එකට ළඟ නම් predictions වඩා නිවැරදියි."
                )

                st.markdown("#### Residual Plot | Prediction Error")
                residual_df = pd.DataFrame({
                    "Predicted": predicted_values,
                    "Residual": residuals
                })
                residual_fig = px.scatter(
                    residual_df,
                    x="Predicted",
                    y="Residual",
                    title="Residuals vs Predicted Values",
                    opacity=0.65
                )
                residual_fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="red"
                )
                st.plotly_chart(
                    residual_fig,
                    use_container_width=True
                )
                st.caption(
                    "Residual = Actual − Predicted. A random spread around zero "
                    "is often desirable. / Residual කියන්නේ Actual − Predicted."
                )

                st.markdown("#### Residual Distribution | Error ව්‍යාප්තිය")
                residual_hist = px.histogram(
                    residual_df,
                    x="Residual",
                    nbins=30,
                    title="Distribution of Prediction Errors"
                )
                st.plotly_chart(
                    residual_hist,
                    use_container_width=True
                )

                # Show a small preview to avoid excessive page/memory usage.
                st.markdown("#### Prediction Sample | Prediction උදාහරණ")
                st.dataframe(
                    regression_plot_df.head(100),
                    use_container_width=True
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

elif page == "Model Comparison":

    st.subheader("Automated Model Comparison Dashboard")
    st.caption(
        "Compare trained models using performance metrics and training time. "
        "/ Models වල performance සහ training time එකම තැනක compare කරන්න."
    )

    comparison = st.session_state.comparison

    if comparison is None or comparison.empty:
        st.info(
            "Please train models first in AI Model Training. "
            "/ මුලින් AI Model Training page එකෙන් models train කරන්න."
        )
    else:
        problem_type = st.session_state.problem_type
        selected_model_name = st.session_state.model_name

        st.markdown("### Model Performance Table")
        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        if problem_type == "Classification":
            main_metric = "F1 Score"
            metric_description = (
                "Higher is better. F1-score balances precision and recall."
            )
            chart_metrics = [
                metric for metric in
                ["Accuracy", "Precision", "Recall", "F1 Score"]
                if metric in comparison.columns
            ]
        else:
            main_metric = "R2 Score"
            metric_description = (
                "Higher is generally better. MAE and RMSE should be lower."
            )
            chart_metrics = [
                metric for metric in ["R2 Score"]
                if metric in comparison.columns
            ]

        best_row = comparison.sort_values(
            main_metric,
            ascending=False
        ).iloc[0]

        st.success(
            f"Top model in the comparison: {best_row['Model']} "
            f"({main_metric}: {best_row[main_metric]:.4f})"
        )

        st.info(
            f"Currently selected prediction model: {selected_model_name}. "
            "If automatic hyperparameter tuning improved the selected model, "
            "the comparison table still represents the original model runs. "
            "The tuning result is shown separately on the AI Model Training page."
        )

        st.markdown("### Performance Comparison")
        if problem_type == "Classification":
            metric_to_plot = st.selectbox(
                "Select performance metric",
                chart_metrics,
                key="comparison_classification_metric"
            )
        else:
            metric_to_plot = main_metric

        fig_comparison = px.bar(
            comparison.sort_values(metric_to_plot, ascending=False),
            x="Model",
            y=metric_to_plot,
            title=f"Model Comparison – {metric_to_plot}",
            text_auto=".4f"
        )
        st.plotly_chart(
            fig_comparison,
            use_container_width=True
        )

        if problem_type == "Regression":
            st.markdown("### Regression Error Comparison")
            error_metrics = [
                metric for metric in ["MAE", "RMSE"]
                if metric in comparison.columns
            ]
            if error_metrics:
                error_data = comparison.melt(
                    id_vars=["Model"],
                    value_vars=error_metrics,
                    var_name="Error Metric",
                    value_name="Error"
                )
                fig_errors = px.bar(
                    error_data,
                    x="Model",
                    y="Error",
                    color="Error Metric",
                    barmode="group",
                    title="Regression Errors (lower is better)"
                )
                st.plotly_chart(
                    fig_errors,
                    use_container_width=True
                )

        if "Training Time (s)" in comparison.columns:
            st.markdown("### Training Time Comparison")
            fig_time = px.bar(
                comparison.sort_values("Training Time (s)"),
                x="Model",
                y="Training Time (s)",
                title="Training Time (seconds) – lower is faster",
                text_auto=".3f"
            )
            st.plotly_chart(
                fig_time,
                use_container_width=True
            )

        st.caption(
            metric_description
            + " Training time measures model fitting only; it excludes "
            "data upload, preprocessing outside the pipeline, and prediction."
        )

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

                # -------------------------------------------------
                # AUTOMATIC DATA VISUALIZATION
                # -------------------------------------------------
                st.divider()
                st.markdown("## 7. Data Visualization | දත්ත දෘශ්‍යකරණය")
                st.caption(
                    "Explore distributions, relationships, outliers and trends. "
                    "/ දත්තවල ව්‍යාප්තිය, සම්බන්ධතා, outliers සහ trends පරීක්ෂා කරන්න."
                )

                numeric_columns = explorer_df.select_dtypes(
                    include=np.number
                ).columns.tolist()

                if numeric_columns:
                    st.markdown("### A. Histogram | දත්ත ව්‍යාප්තිය")
                    histogram_column = st.selectbox(
                        "Select a numerical variable / සංඛ්‍යාත්මක variable එකක් තෝරන්න",
                        numeric_columns,
                        key="explorer_hist_column"
                    )
                    hist_fig = px.histogram(
                        explorer_df,
                        x=histogram_column,
                        nbins=30,
                        title=f"Distribution of {histogram_column}",
                        labels={histogram_column: histogram_column, "count": "Frequency"}
                    )
                    st.plotly_chart(hist_fig, use_container_width=True)
                    st.caption(
                        "Histogram shows how frequently values occur. "
                        "/ Histogram එකෙන් values කොපමණ වාරයක් ඇතිද පෙන්වයි."
                    )

                    st.markdown("### B. Box Plot | Outlier පරීක්ෂාව")
                    box_column = st.selectbox(
                        "Select a variable for box plot / Box plot සඳහා variable එකක් තෝරන්න",
                        numeric_columns,
                        key="explorer_box_column"
                    )
                    box_fig = px.box(
                        explorer_df,
                        y=box_column,
                        points="outliers",
                        title=f"Box Plot of {box_column}"
                    )
                    st.plotly_chart(box_fig, use_container_width=True)
                    st.caption(
                        "Points outside the whiskers may be potential outliers; "
                        "investigate them before removing anything. "
                        "/ Whiskers වලින් පිටත points potential outliers විය හැක."
                    )

                    st.markdown("### C. Scatter Plot | Variables අතර සම්බන්ධතාව")
                    if len(numeric_columns) >= 2:
                        scatter_col1, scatter_col2 = st.columns(2)
                        with scatter_col1:
                            scatter_x = st.selectbox(
                                "X-axis variable",
                                numeric_columns,
                                index=0,
                                key="explorer_scatter_x"
                            )
                        with scatter_col2:
                            default_y = 1 if len(numeric_columns) > 1 else 0
                            scatter_y = st.selectbox(
                                "Y-axis variable",
                                numeric_columns,
                                index=default_y,
                                key="explorer_scatter_y"
                            )

                        scatter_fig = px.scatter(
                            explorer_df,
                            x=scatter_x,
                            y=scatter_y,
                            title=f"{scatter_x} vs {scatter_y}",
                            trendline=None
                        )
                        st.plotly_chart(scatter_fig, use_container_width=True)
                        st.caption(
                            "A scatter plot helps identify possible relationships "
                            "between two numerical variables. "
                            "/ Variables දෙකක් අතර සම්බන්ධතාවයක් තිබේද බලන්න."
                        )
                    else:
                        st.info(
                            "At least two numerical columns are needed for a scatter plot. "
                            "/ Scatter plot එකක් සඳහා numerical columns දෙකක් අවශ්‍යයි."
                        )

                    st.markdown("### D. Line Plot | Measurement Trend")
                    line_column = st.selectbox(
                        "Select a variable for line plot / Line plot සඳහා variable එකක් තෝරන්න",
                        numeric_columns,
                        key="explorer_line_column"
                    )
                    line_fig = px.line(
                        explorer_df.reset_index(),
                        x="index",
                        y=line_column,
                        title=f"Trend of {line_column}",
                        labels={"index": "Record Number", line_column: line_column}
                    )
                    st.plotly_chart(line_fig, use_container_width=True)
                    st.caption(
                        "This plot displays the values in the current row order. "
                        "If the dataset has timestamps, sort by time before interpreting trends. "
                        "/ මෙය දැනට ඇති row order එක අනුව values පෙන්වයි. "
                        "Timestamp තිබේ නම් time අනුව sort කර බලන්න."
                    )

                    st.markdown("### E. Correlation Heatmap | Correlation පරීක්ෂාව")
                    if len(numeric_columns) >= 2:
                        correlation = explorer_df[numeric_columns].corr()
                        heatmap_fig = px.imshow(
                            correlation,
                            text_auto=".2f",
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            zmin=-1,
                            zmax=1,
                            title="Numerical Feature Correlation Heatmap"
                        )
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                        st.caption(
                            "Correlation ranges from -1 to +1. Correlation does not prove causation. "
                            "/ Correlation එකෙන් සම්බන්ධතාව පෙන්වන නමුත් හේතු-ඵල සම්බන්ධය තහවුරු නොවේ."
                        )
                    else:
                        st.info(
                            "At least two numerical columns are needed for a correlation heatmap. "
                            "/ Heatmap එකක් සඳහා numerical columns දෙකක් අවශ්‍යයි."
                        )
                else:
                    st.info(
                        "No numerical columns are available for these plots. "
                        "/ මෙම plots සඳහා numerical columns හමු නොවීය."
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
# AUTOMATIC DUAL ANALYSIS
# =====================================================

elif page == "Automatic Dual Analysis":

    st.subheader("🔁 Automatic Dual Analysis | Classification + Regression")
    st.write(
        "Upload one dataset and choose one numerical target. PredictAI will run "
        "both Regression and Classification automatically using the same target."
    )
    st.caption(
        "එකම numerical target එක භාවිතා කර Analysis දෙකම ස්වයංක්‍රීයව සිදු කරයි."
    )

    dual_file = st.file_uploader(
        "Upload CSV or Excel Dataset",
        type=["csv", "xlsx"],
        key="dual_analysis_upload"
    )

    if dual_file is not None:
        try:
            dual_df = read_dataset(dual_file)
            st.success(f"Dataset loaded: {len(dual_df):,} rows, {len(dual_df.columns)} columns")

            dual_target = st.selectbox(
                "Select numerical target column / Numerical target එක තෝරන්න",
                dual_df.select_dtypes(include=["number"]).columns.tolist(),
                key="dual_analysis_target"
            )

            if dual_target:
                target_values = pd.to_numeric(
                    dual_df[dual_target], errors="coerce"
                )
                usable_mask = target_values.notna()
                dual_data = dual_df.loc[usable_mask].copy()
                y_reg = target_values.loc[usable_mask].astype(float)
                X_dual = dual_data.drop(columns=[dual_target])
                X_dual = X_dual.select_dtypes(
                    include=["number", "object", "category", "bool"]
                )
                constant_cols = [
                    col for col in X_dual.columns
                    if X_dual[col].nunique(dropna=True) <= 1
                ]
                X_dual = X_dual.drop(columns=constant_cols)

                if len(dual_data) < 20:
                    st.error("At least 20 valid rows are recommended for dual analysis.")
                elif X_dual.shape[1] == 0:
                    st.error("No usable feature columns found after removing the target.")
                elif y_reg.nunique() < 2:
                    st.error("The selected target must contain at least two different numeric values.")
                else:
                    # Classification labels are derived from the median.
                    # This rule is shown explicitly so the generated classes are transparent.
                    threshold = float(y_reg.median())
                    y_class = (y_reg >= threshold).astype(int).map({
                        0: "Below Median",
                        1: "Median or Above"
                    })

                    st.markdown("### Automatic classification rule")
                    st.info(
                        f"Values below {threshold:.3f} → **Below Median**; "
                        f"values equal to or above {threshold:.3f} → **Median or Above**. "
                        "This is a generated grouping rule, not an original label in your dataset."
                    )

                    st.markdown("### Target distribution")
                    distribution = pd.DataFrame({
                        "Target": y_class.value_counts().index,
                        "Records": y_class.value_counts().values
                    })
                    st.plotly_chart(
                        px.bar(
                            distribution, x="Target", y="Records",
                            title="Automatically Generated Classification Classes"
                        ),
                        use_container_width=True
                    )

                    run_dual = st.button(
                        "🚀 Run Classification + Regression",
                        type="primary",
                        use_container_width=True,
                        key="run_dual_analysis"
                    )

                    if run_dual:
                        try:
                            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                            from sklearn.metrics import (
                                accuracy_score, precision_score, recall_score,
                                f1_score, confusion_matrix, mean_absolute_error,
                                mean_squared_error, r2_score
                            )

                            stratify_values = (
                                y_class if y_class.value_counts().min() >= 2 else None
                            )
                            X_train_d, X_test_d, yc_train, yc_test, yr_train, yr_test = train_test_split(
                                X_dual, y_class, y_reg,
                                test_size=0.2,
                                random_state=42,
                                stratify=stratify_values
                            )

                            classifier = Pipeline([
                                ("preprocessor", build_preprocessor(X_train_d)),
                                ("model", RandomForestClassifier(
                                    n_estimators=80, random_state=42,
                                    n_jobs=1, class_weight="balanced"
                                ))
                            ])
                            regressor = Pipeline([
                                ("preprocessor", build_preprocessor(X_train_d)),
                                ("model", RandomForestRegressor(
                                    n_estimators=80, random_state=42, n_jobs=1
                                ))
                            ])

                            with st.spinner("Training both models... / Models දෙකම train කරමින් පවතී..."):
                                classifier.fit(X_train_d, yc_train)
                                class_pred = classifier.predict(X_test_d)
                                regressor.fit(X_train_d, yr_train)
                                reg_pred = regressor.predict(X_test_d)

                            st.success("Both analyses completed successfully!")

                            st.markdown("## 1. Classification Results")
                            ca, cp, cr, cf = st.columns(4)
                            ca.metric("Accuracy", f"{accuracy_score(yc_test, class_pred):.3f}")
                            cp.metric("Precision", f"{precision_score(yc_test, class_pred, pos_label='Median or Above', zero_division=0):.3f}")
                            cr.metric("Recall", f"{recall_score(yc_test, class_pred, pos_label='Median or Above', zero_division=0):.3f}")
                            cf.metric("F1-score", f"{f1_score(yc_test, class_pred, pos_label='Median or Above', zero_division=0):.3f}")

                            cm = confusion_matrix(
                                yc_test, class_pred,
                                labels=["Below Median", "Median or Above"]
                            )
                            st.plotly_chart(
                                px.imshow(
                                    cm, text_auto=True,
                                    x=["Predicted Below", "Predicted Above"],
                                    y=["Actual Below", "Actual Above"],
                                    labels={"x": "Predicted", "y": "Actual", "color": "Records"},
                                    title="Classification Confusion Matrix"
                                ),
                                use_container_width=True
                            )

                            st.markdown("## 2. Regression Results")
                            rmse = float(np.sqrt(mean_squared_error(yr_test, reg_pred)))
                            rma, rmr, rrr = st.columns(3)
                            rma.metric("MAE", f"{mean_absolute_error(yr_test, reg_pred):.3f}")
                            rmr.metric("RMSE", f"{rmse:.3f}")
                            rrr.metric("R² Score", f"{r2_score(yr_test, reg_pred):.3f}")

                            regression_plot = pd.DataFrame({
                                "Actual": yr_test.to_numpy(),
                                "Predicted": reg_pred
                            })
                            st.plotly_chart(
                                px.scatter(
                                    regression_plot, x="Actual", y="Predicted",
                                    title="Regression: Actual vs Predicted"
                                ).add_shape(
                                    type="line",
                                    x0=float(regression_plot["Actual"].min()),
                                    y0=float(regression_plot["Actual"].min()),
                                    x1=float(regression_plot["Actual"].max()),
                                    y1=float(regression_plot["Actual"].max())
                                ),
                                use_container_width=True
                            )

                            residuals = yr_test.to_numpy() - reg_pred
                            residual_plot = pd.DataFrame({
                                "Predicted": reg_pred,
                                "Residual": residuals
                            })
                            st.plotly_chart(
                                px.scatter(
                                    residual_plot, x="Predicted", y="Residual",
                                    title="Regression: Residual vs Predicted"
                                ).add_hline(y=0),
                                use_container_width=True
                            )

                            st.markdown("### Prediction Preview")
                            preview = pd.DataFrame({
                                "Actual Numeric Target": yr_test.to_numpy(),
                                "Regression Prediction": reg_pred,
                                "Actual Generated Class": yc_test.to_numpy(),
                                "Classification Prediction": class_pred
                            }).head(100)
                            st.dataframe(preview, use_container_width=True)

                            st.download_button(
                                "Download Dual Analysis Results CSV",
                                preview.to_csv(index=False).encode("utf-8"),
                                file_name="PredictAI_Dual_Analysis_Results.csv",
                                mime="text/csv"
                            )

                            st.warning(
                                "Interpretation note: Classification labels are generated from the "
                                "selected target's median. This allows both task types to run, but "
                                "the classification result answers a different question from exact "
                                "numeric regression. Review the threshold before using the results."
                            )

                        except Exception as dual_error:
                            st.error(f"Dual Analysis Error: {dual_error}")

        except Exception as upload_error:
            st.error(f"Dataset Error: {upload_error}")


# =====================================================
# AI MODEL TRAINING
# =====================================================

elif page == "AI Model Training":

    st.subheader("AI Model Training Center")

    st.write(
        "Upload a dataset and select the target column "
        "to train machine learning models."
    )
    st.caption(
        "Dataset එක upload කරලා පහත modes දෙකෙන් එකක් තෝරන්න. "
        "/ Upload your dataset and choose one of the two learning modes."
    )

    workflow_mode = st.radio(
        "Choose Your Experience / ඔබට අවශ්‍ය ක්‍රමය තෝරන්න",
        ["Guided Mode", "Automatic Mode"],
        horizontal=True,
        key="training_workflow_mode",
        help=(
            "Guided Mode explains the steps. Automatic Mode uses all available "
            "algorithms with the selected target and settings."
        )
    )

    if workflow_mode == "Guided Mode":
        with st.container(border=True):
            st.markdown("### 📘 Guided Mode | පියවරෙන් පියවර")
            st.markdown(
                "**1. Upload Dataset** – CSV / Excel file එක upload කරන්න.  \n"
                "**2. Select Target** – Predict කරන්න ඕන output column එක තෝරන්න.  \n"
                "**3. Select Problem Type** – Classification හෝ Regression තෝරන්න.  \n"
                "**4. Choose Algorithms** – Train කරන්න ඕන algorithms තෝරන්න.  \n"
                "**5. Train & Review** – Results සහ model comparison බලන්න."
            )
            st.info(
                "Guided Mode: You can choose which algorithms to train. "
                "/ ඔබට අවශ්‍ය algorithms ඔබම තෝරාගත හැක."
            )
    else:
        with st.container(border=True):
            st.markdown("### ⚡ Automatic Mode | ස්වයංක්‍රීය ක්‍රමය")
            st.markdown(
                "Select the target and problem type. PredictAI will prepare the "
                "data, apply preprocessing and train all available algorithms. "
                "/ Target එක සහ problem type එක තෝරන්න. ඉතිරි preprocessing සහ "
                "algorithms සියල්ල PredictAI මගින් සිදු කරයි."
            )
            st.success(
                "All available algorithms will be selected automatically. "
                "/ සියලුම algorithms ස්වයංක්‍රීයව තෝරාගනු ලැබේ."
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

            st.markdown("### Preprocessing Settings | දත්ත සකස් කිරීමේ සැකසුම්")
            st.caption(
                "Choose how duplicate rows and numerical feature scaling should be handled. "
                "/ Duplicate rows සහ numerical scaling handle කරන ආකාරය තෝරන්න."
            )

            remove_duplicates = st.checkbox(
                "Remove exact duplicate rows / සම්පූර්ණයෙන්ම සමාන duplicate rows ඉවත් කරන්න",
                value=False,
                help=(
                    "Disabled by default because repeated rows may be valid observations. "
                    "Enable only when duplicates should be removed."
                )
            )

            duplicate_count = int(df.duplicated().sum())
            if duplicate_count:
                st.info(
                    f"{duplicate_count:,} exact duplicate rows detected. "
                    "Review before removing them. / Duplicate rows හඳුනාගෙන ඇත."
                )

            if remove_duplicates and duplicate_count:
                df = df.drop_duplicates().copy()
                st.success(
                    f"Removed {duplicate_count:,} duplicate rows. "
                    "/ Duplicate rows ඉවත් කළා."
                )

            scaling_method = st.selectbox(
                "Numerical Feature Scaling / Numerical features scale කරන ක්‍රමය",
                ["StandardScaler", "RobustScaler", "No Scaling"],
                index=0,
                help=(
                    "StandardScaler centers and scales features. RobustScaler uses "
                    "the median and interquartile range and is less affected by outliers. "
                    "No Scaling leaves numeric feature magnitudes unchanged."
                )
            )

            if scaling_method == "RobustScaler":
                st.info(
                    "RobustScaler uses median and interquartile range. "
                    "It can reduce the influence of extreme values, but it does not remove outliers. "
                    "/ Outliers ඉවත් නොකර ඒවායේ බලපෑම අඩු කිරීමට උපකාරී වේ."
                )
            elif scaling_method == "No Scaling":
                st.info(
                    "Numerical features will not be scaled. "
                    "/ Numerical features scale කරන්නේ නැහැ."
                )
            else:
                st.caption(
                    "StandardScaler is the default scaling method. "
                    "/ StandardScaler default ලෙස භාවිතා වේ."
                )

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

            # Smart target validation and recommendations
            id_name_tokens = ("id", "udi", "uuid", "serial", "code")
            id_like_columns = []

            for column in df.columns:
                column_name = str(column).strip().lower().replace(" ", "_")
                unique_ratio = (
                    df[column].nunique(dropna=True) / max(len(df), 1)
                )
                name_looks_like_id = (
                    column_name in {"id", "udi", "product_id", "machine_id"}
                    or column_name.endswith("_id")
                    or any(token in column_name for token in ("uuid", "serial_number"))
                )
                if name_looks_like_id or (
                    unique_ratio >= 0.98 and df[column].nunique(dropna=True) > 20
                ):
                    id_like_columns.append(column)

            target_unique_count = df[target].nunique(dropna=True)
            target_is_numeric = pd.api.types.is_numeric_dtype(df[target])
            target_name = str(target).strip().lower().replace(" ", "_")
            target_looks_like_id = (
                target in id_like_columns
                or target_name in {"id", "udi", "product_id", "machine_id"}
                or target_name.endswith("_id")
            )

            target_validation_error = None

            st.markdown("### Smart Target Assistant | Target එක තෝරාගැනීමට සහාය")

            if id_like_columns:
                st.info(
                    "Possible ID columns detected: "
                    + ", ".join(map(str, id_like_columns))
                    + ". These usually identify records and are not suitable prediction targets. "
                    + "/ මේ columns බොහෝවිට records හඳුනාගැනීමට භාවිතා කරන නිසා target ලෙස තෝරාගැනීමෙන් වළකින්න."
                )

            if target_looks_like_id:
                target_validation_error = (
                    f"'{target}' looks like an identifier column. "
                    "Choose an outcome column such as Machine failure or Fever."
                )
                st.error(
                    "⚠️ " + target_validation_error
                    + " / ID column එකක් වෙනුවට predict කළ යුතු result column එක තෝරන්න."
                )
            elif problem_type == "Classification":
                if target_unique_count < 2:
                    target_validation_error = (
                        "Classification requires at least two different target classes."
                    )
                    st.error(target_validation_error)
                elif target_unique_count > 50:
                    target_validation_error = (
                        f"'{target}' has {target_unique_count} unique values. "
                        "It looks like a continuous or high-cardinality target. "
                        "Try Regression if this is a numeric measurement, or select a class label."
                    )
                    st.error(
                        target_validation_error
                        + " / මෙහි unique values ගොඩක් තිබෙනවා. Numeric measurement එකක් නම් Regression තෝරන්න."
                    )
                else:
                    st.success(
                        f"'{target}' has {target_unique_count} distinct classes and can be used for classification. "
                        "/ මෙම target එක Classification සඳහා භාවිතා කළ හැක."
                    )
            elif problem_type == "Regression":
                if not target_is_numeric:
                    target_validation_error = (
                        f"'{target}' is not numeric. Regression requires a numerical target."
                    )
                    st.error(
                        target_validation_error
                        + " / Regression සඳහා සංඛ්‍යාත්මක target column එකක් තෝරන්න."
                    )
                else:
                    st.success(
                        f"'{target}' is numerical and can be used for regression. "
                        "/ මෙම target එක Regression සඳහා සුදුසුයි."
                    )

            available_models = (
                CLASSIFICATION_MODELS
                if problem_type == "Classification"
                else REGRESSION_MODELS
            )

            if workflow_mode == "Guided Mode":
                selected_models = st.multiselect(
                    "Select AI Algorithms / Algorithms තෝරන්න",
                    list(available_models.keys()),
                    default=list(available_models.keys())
                )
            else:
                # Automatically select every available algorithm.
                # No multiselect widget is needed in Automatic Mode.
                selected_models = list(available_models.keys())

                st.info(
                    "Automatically selected algorithms: "
                    + ", ".join(selected_models)
                    + " / සියලුම algorithms ස්වයංක්‍රීයව තෝරාගෙන ඇත."
                )

            improve_model = st.checkbox(
                "Improve the best model automatically / Best model එක automatically improve කරන්න",
                value=False,
                help=(
                    "After comparing the selected algorithms, PredictAI will tune "
                    "the best supported model using training data only and compare "
                    "its performance on the held-out test set."
                )
            )

            if improve_model:
                st.info(
                    "PredictAI will try a small hyperparameter search for the best "
                    "supported model. Linear Regression has no tuning search in this version. "
                    "/ හොඳම model එකට සුදුසු parameters සොයනු ලැබේ."
                )

            st.divider()

            train_button_label = (
                "🚀 Train Selected Models / තෝරාගත් Models Train කරන්න"
                if workflow_mode == "Guided Mode"
                else "⚡ Run Automatic Training / ස්වයංක්‍රීයව Train කරන්න"
            )

            if st.button(
                train_button_label,
                type="primary",
                use_container_width=True
            ):

                if not selected_models:

                    st.error(
                        "Please select at least one algorithm."
                    )

                elif target_validation_error:

                    st.error(
                        "Please correct the target column or problem type before training. "
                        "/ Training කිරීමට පෙර target column එක හෝ problem type එක නිවැරදි කරන්න."
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
                                    build_preprocessor(X_train, scaling_method)
                                ),
                                (
                                    "model",
                                    available_models[name]
                                )
                            ])

                            training_start_time = time.perf_counter()

                            pipeline.fit(
                                X_train,
                                y_train
                            )

                            training_seconds = time.perf_counter() - training_start_time

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
                                    "F1 Score": round(f1, 4),
                                    "Training Time (s)": round(training_seconds, 3)
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
                                    "R2 Score": round(r2, 4),
                                    "Training Time (s)": round(training_seconds, 3)
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
                        baseline_model = best_model
                        baseline_predictions = baseline_model.predict(X_test)

                        if problem_type == "Classification":
                            baseline_score = f1_score(
                                y_test,
                                baseline_predictions,
                                average="weighted",
                                zero_division=0
                            )
                            tuning_scoring = "f1_weighted"
                        else:
                            baseline_score = r2_score(
                                y_test,
                                baseline_predictions
                            )
                            tuning_scoring = "r2"

                        tuning_summary = None

                        if improve_model:
                            tuning_spaces = {
                                "Logistic Regression": {
                                    "model__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
                                },
                                "Decision Tree": {
                                    "model__max_depth": [None, 5, 10, 15, 20],
                                    "model__min_samples_split": [2, 5, 10],
                                    "model__min_samples_leaf": [1, 2, 4]
                                },
                                "Random Forest": {
                                    "model__n_estimators": [50, 100, 150],
                                    "model__max_depth": [None, 10, 15, 20],
                                    "model__min_samples_split": [2, 5],
                                    "model__min_samples_leaf": [1, 2]
                                },
                                "Extra Trees": {
                                    "model__n_estimators": [50, 100, 150],
                                    "model__max_depth": [None, 10, 15, 20],
                                    "model__min_samples_split": [2, 5],
                                    "model__min_samples_leaf": [1, 2]
                                },
                                "Decision Tree Regressor": {
                                    "model__max_depth": [None, 5, 10, 15, 20],
                                    "model__min_samples_split": [2, 5, 10],
                                    "model__min_samples_leaf": [1, 2, 4]
                                },
                                "Random Forest Regressor": {
                                    "model__n_estimators": [50, 100, 150],
                                    "model__max_depth": [None, 10, 15, 20],
                                    "model__min_samples_split": [2, 5],
                                    "model__min_samples_leaf": [1, 2]
                                },
                                "Extra Trees Regressor": {
                                    "model__n_estimators": [50, 100, 150],
                                    "model__max_depth": [None, 10, 15, 20],
                                    "model__min_samples_split": [2, 5],
                                    "model__min_samples_leaf": [1, 2]
                                }
                            }

                            if best_name in tuning_spaces:
                                with st.spinner(
                                    "Improving the best model... / Model එක improve කරමින්..."
                                ):
                                    search = RandomizedSearchCV(
                                        estimator=baseline_model,
                                        param_distributions=tuning_spaces[best_name],
                                        n_iter=5,
                                        scoring=tuning_scoring,
                                        cv=3,
                                        random_state=42,
                                        n_jobs=1,
                                        refit=True
                                    )
                                    search.fit(X_train, y_train)
                                    tuned_model = search.best_estimator_
                                    tuned_predictions = tuned_model.predict(X_test)

                                    if problem_type == "Classification":
                                        tuned_score = f1_score(
                                            y_test,
                                            tuned_predictions,
                                            average="weighted",
                                            zero_division=0
                                        )
                                        score_name = "F1 Score"
                                        improved = tuned_score > baseline_score
                                    else:
                                        tuned_score = r2_score(
                                            y_test,
                                            tuned_predictions
                                        )
                                        score_name = "R2 Score"
                                        improved = tuned_score > baseline_score

                                    tuning_summary = pd.DataFrame([
                                        {
                                            "Version": "Original",
                                            score_name: round(float(baseline_score), 4)
                                        },
                                        {
                                            "Version": "Tuned",
                                            score_name: round(float(tuned_score), 4)
                                        }
                                    ])

                                    if improved:
                                        best_model = tuned_model
                                        best_predictions = tuned_predictions
                                        st.success(
                                            "Tuned model improved the test score. "
                                            "/ Tuned model එකේ ප්‍රතිඵල වැඩිදියුණු වුණා."
                                        )
                                    else:
                                        best_predictions = baseline_predictions
                                        st.info(
                                            "Tuning did not improve the held-out test score. "
                                            "The original model was retained. "
                                            "/ Original model එක තබාගත්තා."
                                        )

                                    st.write("Before vs After Model Improvement")
                                    st.dataframe(
                                        tuning_summary,
                                        use_container_width=True
                                    )
                                    st.write("Best parameters found:")
                                    st.json(search.best_params_)
                            else:
                                best_predictions = baseline_predictions
                                st.info(
                                    "Automatic tuning is not configured for this model. "
                                    "The original model will be used."
                                )
                        else:
                            best_predictions = baseline_predictions

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


            # -----------------------------------------
            # INDIVIDUAL PREDICTION EXPLANATION
            # -----------------------------------------

            st.divider()
            st.subheader("Why did the model make this prediction?")
            st.caption(
                "One-feature-at-a-time sensitivity analysis. "
                "This is an educational approximation, not SHAP or a causal explanation. "
                "/ එක් feature එකක් වෙනස් කළාම prediction එක වෙනස් වන ආකාරය පෙන්වයි."
            )

            explanation_features = st.session_state.feature_columns
            available_rows = min(len(results), 1000)

            if available_rows > 0 and explanation_features:

                selected_row = st.selectbox(
                    "Select a prediction row to explain / පැහැදිලි කිරීමට row එක තෝරන්න",
                    options=list(range(available_rows)),
                    format_func=lambda row_number: f"Row {row_number + 1}"
                )

                if st.button(
                    "Explain This Prediction / මේ prediction එක පැහැදිලි කරන්න"
                ):

                    with st.spinner("Analyzing feature effects..."):

                        try:
                            original_row = results.iloc[
                                [selected_row]
                            ][explanation_features].copy()

                            baseline_prediction = model.predict(
                                original_row
                            )[0]

                            explanation_rows = []

                            # Limit the number of tested features to keep
                            # the app responsive on Streamlit Cloud.
                            features_to_check = explanation_features[:20]

                            if st.session_state.problem_type == "Classification":
                                if hasattr(model, "predict_proba"):
                                    baseline_probabilities = model.predict_proba(
                                        original_row
                                    )[0]
                                    model_classes = list(model.classes_)
                                    predicted_class_index = model_classes.index(
                                        baseline_prediction
                                    )
                                    baseline_value = float(
                                        baseline_probabilities[predicted_class_index]
                                    )
                                else:
                                    baseline_value = None
                            else:
                                baseline_value = float(baseline_prediction)

                            for feature_name in features_to_check:

                                changed_row = original_row.copy()
                                reference_column = results[feature_name]

                                if pd.api.types.is_numeric_dtype(
                                    reference_column
                                ):
                                    reference_value = reference_column.median()
                                else:
                                    modes = reference_column.mode(dropna=True)
                                    if modes.empty:
                                        continue
                                    reference_value = modes.iloc[0]

                                changed_row.loc[
                                    changed_row.index[0],
                                    feature_name
                                ] = reference_value

                                changed_prediction = model.predict(
                                    changed_row
                                )[0]

                                if st.session_state.problem_type == "Classification":
                                    if not hasattr(model, "predict_proba"):
                                        continue

                                    changed_probabilities = model.predict_proba(
                                        changed_row
                                    )[0]
                                    changed_value = float(
                                        changed_probabilities[predicted_class_index]
                                    )
                                    impact = changed_value - baseline_value
                                    explanation_rows.append({
                                        "Feature": feature_name,
                                        "Current Value": str(
                                            original_row.iloc[0][feature_name]
                                        ),
                                        "Reference Value": str(reference_value),
                                        "Change in Predicted Class Probability": impact
                                    })
                                else:
                                    changed_value = float(changed_prediction)
                                    impact = changed_value - baseline_value
                                    explanation_rows.append({
                                        "Feature": feature_name,
                                        "Current Value": str(
                                            original_row.iloc[0][feature_name]
                                        ),
                                        "Reference Value": str(reference_value),
                                        "Change in Prediction": impact
                                    })

                            if explanation_rows:
                                explanation_df = pd.DataFrame(
                                    explanation_rows
                                )
                                impact_column = (
                                    "Change in Predicted Class Probability"
                                    if st.session_state.problem_type == "Classification"
                                    else "Change in Prediction"
                                )
                                explanation_df["Absolute Impact"] = (
                                    explanation_df[impact_column].abs()
                                )
                                explanation_df = explanation_df.sort_values(
                                    "Absolute Impact",
                                    ascending=False
                                ).drop(columns=["Absolute Impact"])

                                st.write(
                                    f"Original model prediction: **{baseline_prediction}**"
                                )

                                if st.session_state.problem_type == "Classification":
                                    st.write(
                                        "The changes below show how replacing one feature "
                                        "with its typical value affects the probability of "
                                        "the original predicted class."
                                    )
                                    explanation_df[impact_column] = (
                                        explanation_df[impact_column] * 100
                                    )
                                    explanation_df = explanation_df.rename(
                                        columns={
                                            impact_column:
                                            "Probability Change (percentage points)"
                                        }
                                    )

                                st.dataframe(
                                    explanation_df,
                                    use_container_width=True
                                )

                                st.info(
                                    "Important: features are changed one at a time. "
                                    "Correlated features and interactions are not fully captured. "
                                    "This analysis explains model sensitivity, not cause and effect. "
                                    "/ මෙය model sensitivity එකක් පමණයි; හේතුවක් බව තහවුරු කරන්නේ නැහැ."
                                )
                            else:
                                st.warning(
                                    "No explanation could be calculated for this model."
                                )

                        except Exception as explanation_error:
                            st.error(
                                f"Explanation Error: {explanation_error}"
                            )

            else:
                st.info(
                    "Generate predictions first to use this explanation feature."
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
