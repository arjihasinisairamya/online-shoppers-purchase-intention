
# Import required libraries

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import hashlib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# Page configuration

st.set_page_config(
    page_title="Online Shopper Purchase Intention",
    page_icon="🛍️",
    layout="wide"
)


# Custom styling

st.markdown("""
<style>
    .stApp {
        background-color: #F7F7F2;
    }

    h1, h2, h3 {
        color: #344638;
    }

    div[data-testid="stMetric"] {
        background-color: #E9EDE5;
        padding: 12px;
        border-radius: 8px;
    }

    div.stButton > button {
        background-color: #E9EDE5;
        color: #344638;
        border: 1px solid #AAB9A6;
        border-radius: 10px;
        min-height: 90px;
        font-size: 17px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #DCE6D9;
        color: #344638;
        border-color: #52765B;
    }

</style>
""", unsafe_allow_html=True)


# Application heading

st.title("🛍️ Predict Online Shopper Purchase Intention")

st.write("Will this shopper make a purchase?")

st.write(
    "Explore online shopping behaviour using machine learning. "
    "Upload shopper session data, select a classification model "
    "and predict purchase outcomes or evaluate model performance."
)


# Load trained models

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "Decision Tree": joblib.load("models/decision_tree.pkl"),
        "KNN": joblib.load("models/knn.pkl"),
        "Gaussian Naive Bayes": joblib.load("models/gaussian_nb.pkl"),
        "SVM": joblib.load("models/svm.pkl")
    }

    return models


try:

    models = load_models()

except Exception as e:

    st.error("Unable to load the trained models. Please check the models folder.")
    st.error(e)
    st.stop()


# Upload shopper data

st.divider()

st.subheader("📁 Upload Shopper Data")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)


# Sample data download option

if uploaded_file is None:

    if os.path.exists("test_data.csv"):

        st.caption("Don't have a dataset? Try our sample data.")

        with open("test_data.csv", "rb") as sample_file:

            st.download_button(
                label="⬇️ Download Sample Dataset",
                data=sample_file.read(),
                file_name="sample_test_data.csv",
                mime="text/csv",
                use_container_width=False
            )

    st.caption("Upload a CSV file above to get started.")


# Continue only after uploading a file

if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        st.markdown(
            f"""
            <div style="
                background-color: #DCEFE0;
                color: #245B35;
                padding: 14px 18px;
                border-radius: 8px;
                border: 1px solid #A5C9AA;
                font-weight: 600;
                font-size: 16px;
            ">
                ✓ Dataset uploaded successfully!
                {len(data):,} shopper records are ready.
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:

        st.error("Unable to read the uploaded CSV file.")
        st.error(e)
        st.stop()


    # Check Revenue column

    if "Revenue" not in data.columns:

        st.error(
            "The uploaded dataset must contain the 'Revenue' column "
            "to predict and evaluate purchase outcomes."
        )
        st.stop()


    if data["Revenue"].isnull().any():

        st.error("The Revenue column contains missing values. Please check your dataset.")
        st.stop()


    # Convert Revenue values into Boolean values

    if data["Revenue"].dtype != bool:

        revenue_values = data["Revenue"].astype(str).str.strip().str.lower()

        revenue_mapping = {
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
            "1": True,
            "0": False
        }

        if not revenue_values.isin(revenue_mapping.keys()).all():

            st.error(
                "Revenue must contain only True/False, Yes/No or 1/0 values."
            )
            st.stop()

        data["Revenue"] = revenue_values.map(revenue_mapping)


    if data["Revenue"].nunique() != 2:

        st.error(
            "The Revenue column must contain both purchase and non-purchase records."
        )
        st.stop()


    # Select a classification model

    st.divider()

    st.subheader("🧠 Choose a Classification Model")

    selected_model_name = st.selectbox(
        "Choose a model from the dropdown:",
        list(models.keys())
    )

    selected_model = models[selected_model_name]


    # Display selected model

    st.markdown(
        f"""
        <div style="
            background-color: #DCEFE0;
            color: #245B35;
            padding: 10px 15px;
            border-radius: 8px;
            border: 1px solid #A5C9AA;
            font-weight: 600;
            display: inline-block;
            font-size: 14px;
        ">
            ✓ Selected Model: {selected_model_name}
        </div>
        """,
        unsafe_allow_html=True
    )

    if selected_model_name != "SVM":

        st.caption("⭐ Recommended model: SVM")

    else:

        st.caption("⭐ Recommended model")


    # Choose an option

    st.divider()

    st.subheader("⚙️ Choose an Option")


    if "application_mode" not in st.session_state:

        st.session_state.application_mode = None


    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🛍️ Predict",
            use_container_width=True,
            type="primary" if st.session_state.application_mode == "Predict" else "secondary"
        ):

            st.session_state.application_mode = "Predict"


    with col2:

        if st.button(
            "📊 Evaluate Model",
            use_container_width=True,
            type="primary" if st.session_state.application_mode == "Evaluate Model" else "secondary"
        ):

            st.session_state.application_mode = "Evaluate Model"


    application_mode = st.session_state.application_mode


    # Validate input features

    expected_features = list(selected_model.feature_names_in_)

    actual_features = [
        column for column in data.columns
        if column != "Revenue"
    ]

    missing_features = [
        feature for feature in expected_features
        if feature not in actual_features
    ]

    extra_features = [
        feature for feature in actual_features
        if feature not in expected_features
    ]


    if missing_features:

        st.error(
            f"Missing required columns: {missing_features}"
        )
        st.stop()


    if extra_features:

        st.warning(
            f"Extra columns will be ignored: {extra_features}"
        )


    # Arrange columns in the same order used during training

    X_test = data[expected_features]

    y_test = data["Revenue"]


    # Identify the uploaded dataset

    file_hash = hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()

    current_key = (
        file_hash,
        selected_model_name,
        application_mode
    )


    # Run the selected model automatically after choosing an option

    if application_mode is not None:

        if st.session_state.get("result_key") != current_key:

            try:

                predictions = selected_model.predict(X_test)


                # Get prediction scores for model evaluation

                prediction_scores = None

                if application_mode == "Evaluate Model":

                    if hasattr(selected_model, "predict_proba"):

                        prediction_scores = selected_model.predict_proba(X_test)[:, 1]

                    elif hasattr(selected_model, "decision_function"):

                        prediction_scores = selected_model.decision_function(X_test)

                    else:

                        prediction_scores = predictions


                # Store results

                st.session_state["predictions"] = predictions

                st.session_state["actual_values"] = y_test.to_numpy()

                st.session_state["prediction_scores"] = prediction_scores

                st.session_state["evaluation_metrics"] = None


                # Calculate evaluation metrics

                if application_mode == "Evaluate Model":

                    evaluation_metrics = {

                        "Accuracy": accuracy_score(y_test, predictions),

                        "AUC": roc_auc_score(y_test, prediction_scores),

                        "Precision": precision_score(
                            y_test, predictions, zero_division=0
                        ),

                        "Recall": recall_score(
                            y_test, predictions, zero_division=0
                        ),

                        "F1 Score": f1_score(
                            y_test, predictions, zero_division=0
                        ),

                        "MCC": matthews_corrcoef(y_test, predictions)

                    }

                    st.session_state["evaluation_metrics"] = evaluation_metrics


                st.session_state["result_key"] = current_key


            except Exception as e:

                st.session_state.pop("result_key", None)

                st.error("Something went wrong while running the model.")
                st.error(e)
                st.stop()


        # Display results

        if st.session_state.get("result_key") == current_key:

            predictions = st.session_state["predictions"]

            actual_values = st.session_state["actual_values"]


            # Prediction results

            if application_mode == "Predict":

                st.divider()

                st.subheader("🛍️ Shopper Purchase Outcomes")


                total_sessions = len(predictions)

                purchased = sum(predictions)

                not_purchased = total_sessions - purchased


                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Sessions",
                    f"{total_sessions:,}"
                )

                col2.metric(
                    "Purchased",
                    f"{purchased:,}"
                )

                col3.metric(
                    "Not Purchased",
                    f"{not_purchased:,}"
                )


                # Display predictions

                prediction_df = pd.DataFrame({

                    "Session": range(1, total_sessions + 1),

                    "Purchase Outcome": [
                        "Purchased" if value else "Not Purchased"
                        for value in predictions
                    ]

                })


                def colour_outcome(value):

                    if value == "Purchased":

                        return "background-color: #DCEFE0; color: #245B35"

                    return "background-color: #F8DEDE; color: #8B2929"


                st.dataframe(
                    prediction_df.style.map(
                        colour_outcome,
                        subset=["Purchase Outcome"]
                    ),
                    use_container_width=True,
                    hide_index=True
                )


                # Download predictions

                csv = prediction_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Prediction Results",
                    data=csv,
                    file_name="shopper_purchase_predictions.csv",
                    mime="text/csv"
                )


                # Actual vs predicted results

                st.divider()

                st.subheader("Actual vs Predicted")


                comparison_result = [

                    "Correct" if actual == predicted else "Incorrect"

                    for actual, predicted in zip(actual_values, predictions)

                ]


                actual_prediction_df = pd.DataFrame({

                    "Session": range(1, total_sessions + 1),

                    "Actual Outcome": [
                        "Purchased" if value else "Not Purchased"
                        for value in actual_values
                    ],

                    "Predicted Outcome": [
                        "Purchased" if value else "Not Purchased"
                        for value in predictions
                    ],

                    "Result": comparison_result

                })


                correct_count = comparison_result.count("Correct")

                incorrect_count = comparison_result.count("Incorrect")


                col1, col2 = st.columns(2)

                col1.metric(
                    "Correct Predictions",
                    f"{correct_count:,}"
                )

                col2.metric(
                    "Incorrect Predictions",
                    f"{incorrect_count:,}"
                )


                def colour_result(value):

                    if value == "Correct":

                        return "background-color: #DCEFE0; color: #245B35"

                    return "background-color: #F8DEDE; color: #8B2929"


                st.dataframe(
                    actual_prediction_df.style.map(
                        colour_result,
                        subset=["Result"]
                    ),
                    use_container_width=True,
                    hide_index=True
                )


                csv = actual_prediction_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Actual vs Predicted Results",
                    data=csv,
                    file_name="actual_vs_predicted.csv",
                    mime="text/csv"
                )


            # Model evaluation results

            elif application_mode == "Evaluate Model":

                st.divider()

                st.subheader("Model Performance")


                evaluation_metrics = st.session_state["evaluation_metrics"]


                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Accuracy",
                    f"{evaluation_metrics['Accuracy']:.2%}"
                )

                col2.metric(
                    "AUC",
                    f"{evaluation_metrics['AUC']:.2%}"
                )

                col3.metric(
                    "Precision",
                    f"{evaluation_metrics['Precision']:.2%}"
                )


                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Recall",
                    f"{evaluation_metrics['Recall']:.2%}"
                )

                col2.metric(
                    "F1 Score",
                    f"{evaluation_metrics['F1 Score']:.2%}"
                )

                col3.metric(
                    "MCC",
                    f"{evaluation_metrics['MCC']:.4f}"
                )


                # Confusion matrix

                st.divider()

                st.subheader("Confusion Matrix")


                cm = confusion_matrix(
                    actual_values,
                    predictions,
                    labels=[False, True]
                )


                fig, ax = plt.subplots(figsize=(5, 3.5))

                ax.imshow(
                    cm,
                    cmap="Blues",
                    vmin=0,
                    vmax=cm.max() * 1.5
                )

                ax.set_title(
                    "Confusion Matrix",
                    fontsize=13,
                    pad=12
                )

                ax.set_xlabel(
                    "Predicted Outcome",
                    fontsize=10
                )

                ax.set_ylabel(
                    "Actual Outcome",
                    fontsize=10
                )

                ax.set_xticks([0, 1])

                ax.set_yticks([0, 1])

                ax.set_xticklabels(
                    ["Not Purchased", "Purchased"],
                    fontsize=9
                )

                ax.set_yticklabels(
                    ["Not Purchased", "Purchased"],
                    fontsize=9
                )


                for i in range(2):

                    for j in range(2):

                        ax.text(
                            j,
                            i,
                            cm[i, j],
                            ha="center",
                            va="center",
                            fontsize=12,
                            color="#263D4A"
                        )


                plt.tight_layout()

                st.pyplot(fig, use_container_width=False)

                plt.close(fig)


                # Classification report

                st.divider()

                st.subheader("Classification Report")


                report = classification_report(
                    actual_values,
                    predictions,
                    labels=[False, True],
                    target_names=["Not Purchased", "Purchased"],
                    output_dict=True,
                    zero_division=0
                )


                report_df = pd.DataFrame(report).transpose()

                st.dataframe(
                    report_df.round(4),
                    use_container_width=True
                )


                # Compare with other models

                st.divider()

                st.subheader("Compare Classification Models")


                compare_models = st.radio(
                    "Would you like to compare this model with the other classifiers?",
                    ["Yes", "No"],
                    horizontal=True,
                    index=1
                )


                if compare_models == "Yes":

                    comparison_results = []


                    for model_name, model in models.items():

                        model_predictions = model.predict(X_test)


                        if hasattr(model, "predict_proba"):

                            model_scores = model.predict_proba(X_test)[:, 1]

                        elif hasattr(model, "decision_function"):

                            model_scores = model.decision_function(X_test)

                        else:

                            model_scores = model_predictions


                        model_metrics = {

                            "Model": model_name,

                            "Accuracy": accuracy_score(
                                y_test, model_predictions
                            ),

                            "AUC": roc_auc_score(
                                y_test, model_scores
                            ),

                            "Precision": precision_score(
                                y_test, model_predictions, zero_division=0
                            ),

                            "Recall": recall_score(
                                y_test, model_predictions, zero_division=0
                            ),

                            "F1 Score": f1_score(
                                y_test, model_predictions, zero_division=0
                            ),

                            "MCC": matthews_corrcoef(
                                y_test, model_predictions
                            )

                        }

                        comparison_results.append(model_metrics)


                    comparison_df = pd.DataFrame(comparison_results)


                    st.write(
                        "Performance comparison of all five classification models:"
                    )

                    st.dataframe(
                        comparison_df.set_index("Model").round(4),
                        use_container_width=True
                    )


                    # Grouped bar chart

                    metrics_to_plot = [
                        "Accuracy",
                        "AUC",
                        "Precision",
                        "Recall",
                        "F1 Score",
                        "MCC"
                    ]


                    chart_data = comparison_df.set_index("Model")[metrics_to_plot]


                    fig, ax = plt.subplots(figsize=(12, 5))

                    chart_data.plot(
                        kind="bar",
                        ax=ax,
                        width=0.8
                    )

                    ax.set_title(
                        "Performance Comparison of Classification Models"
                    )

                    ax.set_xlabel("Classification Models")

                    ax.set_ylabel("Score")

                    ax.set_ylim(0, 1)

                    ax.legend(
                        title="Evaluation Metrics",
                        loc="upper right"
                    )

                    plt.xticks(
                        rotation=15,
                        ha="right"
                    )

                    plt.tight_layout()

                    st.pyplot(fig, use_container_width=True)

                    plt.close(fig)


else:

    st.caption("Upload a CSV file above to get started.")
