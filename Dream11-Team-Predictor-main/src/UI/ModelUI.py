import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import sys
import os


sys.path.append(os.path.abspath("../model/"))
from train_model import train_model

sys.path.append(os.path.abspath("../UI/Product_UI/backend/"))


def load_data(match_type):
    """Load data based on match type"""
    if match_type == "T20":
        return pd.read_csv("../data/processed/formatwise/t20.csv")
    elif match_type == "ODI":
        return pd.read_csv("../data/processed/formatwise/odi.csv")
    else:
        return pd.read_csv("../data/processed/formatwise/test.csv")


def filter_data(df, start_date, end_date):
    df["date"] = pd.to_datetime(df["date"])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    return df.loc[mask]


# def train_model(train_df, test_df, train_end_date):

#     # Simulate training delay
#     import time

#     time.sleep(2)

#     # Dummy predictions and metrics
#     train_predictions = np.random.normal(0, 10, len(train_df))
#     test_predictions = np.random.normal(0, 10, len(test_df))

#     train_mae = mean_absolute_error(train_df["label"], train_predictions)
#     test_mae = mean_absolute_error(test_df["label"], test_predictions)

#     return {
#         "train_mae": train_mae,
#         "test_mae": test_mae,
#         "train_predictions": train_predictions,
#         "test_predictions": test_predictions,
#     }


def main():
    st.title("Cricket Match Analysis")

    # Sidebar for input parameters
    st.sidebar.header("Parameters")

    match_type = st.sidebar.selectbox("Select Match Type", ["T20", "ODI", "Test"])

    # Date range selectors
    col1, col2 = st.sidebar.columns(2)

    with col1:
        st.subheader("Training Period")
        train_start = st.date_input(
            "Start Date",
            datetime(1990, 1, 1),
            min_value=datetime(1990, 1, 1),
            max_value=datetime(2024, 12, 31),
        )
        train_end = st.date_input(
            "End Date",
            datetime(2024, 6, 30),
            min_value=datetime(1990, 1, 1),
            max_value=datetime(2024, 12, 31),
        )

    with col2:
        st.subheader("Testing Period")
        test_start = st.date_input(
            "Start Date",
            datetime(2024, 7, 1),
            min_value=datetime(1990, 1, 1),
            max_value=datetime(2024, 12, 31),
        )
        test_end = st.date_input(
            "End Date",
            datetime(2024, 11, 10),
            min_value=datetime(1990, 1, 1),
            max_value=datetime(2024, 12, 31),
        )

    if st.sidebar.button("Run Analysis"):
        # Load data
        with st.spinner("Loading Data..."):
            df = load_data(match_type)

        # Filter data for training and testing
        train_df = filter_data(df, train_start, train_end)
        test_df = filter_data(df, test_start, test_end)
        with st.spinner("Saving training data..."):
            train_df.to_csv(
                f"../data/processed/training_data_{match_type}_{train_end}.csv",
                index=False,
            )
        # Display data info
        st.subheader("Dataset Information")
        st.write(f"Training samples: {len(train_df)}")
        st.write(f"Testing samples: {len(test_df)}")

        # Train model
        with st.spinner("Training Model..."):
            results = train_model(train_df, test_df, match_type.lower(), train_end)

        # Display metrics
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training MAE", f"{results['train_mae']:.2f}")
        with col2:
            st.metric("Testing MAE", f"{results['test_mae']:.2f}")
        with col3:
            st.metric("Testing MAPE", f"{results['mape_loss']:.2f}")

        # Create and display visualizations
        st.subheader("Predictions vs Actual")

        # Training data visualization
        fig_train = px.scatter(
            x=train_df["label"],
            y=results["train_predictions"],
            labels={"x": "Actual Fantasy Score", "y": "Predicted Fantasy Score"},
            title="Training Data: Predicted vs Actual",
        )
        fig_train.add_shape(
            type="line",
            line=dict(dash="dash"),
            x0=train_df["label"].min(),
            y0=train_df["label"].min(),
            x1=train_df["label"].max(),
            y1=train_df["label"].max(),
        )
        st.plotly_chart(fig_train)

        # Testing data visualization
        fig_test = px.scatter(
            x=test_df["label"],
            y=results["test_predictions"],
            labels={"x": "Actual Fantasy Score", "y": "Predicted Fantasy Score"},
            title="Testing Data: Predicted vs Actual",
        )
        fig_test.add_shape(
            type="line",
            line=dict(dash="dash"),
            x0=test_df["label"].min(),
            y0=test_df["label"].min(),
            x1=test_df["label"].max(),
            y1=test_df["label"].max(),
        )
        st.plotly_chart(fig_test)


if __name__ == "__main__":
    main()
