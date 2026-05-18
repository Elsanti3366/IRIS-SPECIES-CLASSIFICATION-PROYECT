import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)
from sklearn.preprocessing import StandardScaler

# ── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Species Classification",
    page_icon="🌸",
    layout="wide"
)

# ── LOAD & TRAIN ───────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    df["species_name"] = df["species"].map({0: "Setosa", 1: "Versicolor", 2: "Virginica"})

    X = df[iris.feature_names]
    y = df["species"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall":    recall_score(y_test, y_pred, average="weighted"),
        "F1-Score":  f1_score(y_test, y_pred, average="weighted"),
    }

    return model, scaler, df, iris, metrics, y_test, y_pred

model, scaler, df, iris, metrics, y_test, y_pred = load_and_train()
feature_names = iris.feature_names
species_map = {0: "🌸 Setosa", 1: "🌺 Versicolor", 2: "🌼 Virginica"}
colors = {"Setosa": "#1E2761", "Versicolor": "#4A90D9", "Virginica": "#2ECC71"}

# ── SIDEBAR ────────────────────────────────────────────────────────────────
st.sidebar.title("🌸 Iris Classifier")
st.sidebar.markdown("**Universidad de la Costa**")
st.sidebar.markdown("Data Mining — José Escorcia-Gutiérrez, Ph.D.")
st.sidebar.markdown("**Estudiante:** Santiago Orozco")
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Go to", ["📊 Overview & Metrics", "🔮 Predict Species", "📈 Data Exploration"])

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & METRICS
# ══════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & Metrics":
    st.title("🌸 Iris Species Classification Dashboard")
    st.markdown("End-to-end classification project using **Random Forest** on the Iris dataset.")
    st.markdown("---")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{metrics['Accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['Precision']:.4f}")
    col3.metric("Recall",    f"{metrics['Recall']:.4f}")
    col4.metric("F1-Score",  f"{metrics['F1-Score']:.4f}")

    st.markdown("---")

    # Confusion Matrix + Feature Importance
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Setosa", "Versicolor", "Virginica"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Feature Importance")
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
        fig, ax = plt.subplots(figsize=(5, 4))
        importances.plot(kind="barh", ax=ax, color="#1E2761")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Workflow
    st.subheader("📋 Workflow")
    st.markdown("""
    1. **Data Understanding** — Load Iris dataset (150 samples, 4 features, 3 classes)
    2. **Preprocessing** — StandardScaler normalization
    3. **Train/Test Split** — 80/20 stratified split
    4. **Model Training** — Random Forest (100 trees)
    5. **Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrix
    6. **Deployment** — Interactive Streamlit dashboard
    """)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Species":
    st.title("🔮 Predict Iris Species")
    st.markdown("Enter the flower measurements below to get a prediction.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
        sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.4, 0.1)
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.7, 0.1)
        petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 1.5, 0.1)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    st.markdown("---")
    st.subheader("Prediction Result")
    st.success(f"**Predicted Species: {species_map[prediction]}**")

    # Probability bars
    prob_df = pd.DataFrame({
        "Species": ["Setosa", "Versicolor", "Virginica"],
        "Probability": probabilities
    })
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.barh(prob_df["Species"], prob_df["Probability"],
            color=["#1E2761", "#4A90D9", "#2ECC71"], edgecolor="white")
    ax.set_xlabel("Probability")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("3D Scatter Plot — Position of New Sample")

    plot_df = df.copy()
    new_point = pd.DataFrame({
        "sepal length (cm)": [sepal_length],
        "sepal width (cm)":  [sepal_width],
        "petal length (cm)": [petal_length],
        "petal width (cm)":  [petal_width],
        "species_name": ["New Sample"],
        "species": [prediction]
    })

    combined = pd.concat([plot_df, new_point], ignore_index=True)
    combined["size"] = combined["species_name"].apply(lambda x: 10 if x == "New Sample" else 4)
    combined["symbol"] = combined["species_name"].apply(lambda x: "star" if x == "New Sample" else "circle")

    color_map = {"Setosa": "#1E2761", "Versicolor": "#4A90D9",
                 "Virginica": "#2ECC71", "New Sample": "#E74C3C"}

    fig3d = px.scatter_3d(
        combined,
        x="petal length (cm)", y="petal width (cm)", z="sepal length (cm)",
        color="species_name", color_discrete_map=color_map,
        size="size", symbol="symbol",
        title="3D Scatter — New Sample vs Dataset",
        labels={"species_name": "Species"}
    )
    fig3d.update_layout(height=550)
    st.plotly_chart(fig3d, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA EXPLORATION
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 Data Exploration":
    st.title("📈 Data Exploration")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        df["species_name"].value_counts().plot(kind="bar", ax=ax,
            color=["#1E2761", "#4A90D9", "#2ECC71"], edgecolor="white", width=0.6)
        ax.set_xlabel("Species")
        ax.set_ylabel("Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Feature Distributions")
        feature = st.selectbox("Select feature", feature_names)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for species, color in colors.items():
            subset = df[df["species_name"] == species]
            ax.hist(subset[feature], bins=15, alpha=0.7, label=species, color=color)
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Scatter Matrix")
    fig = px.scatter_matrix(
        df,
        dimensions=feature_names,
        color="species_name",
        color_discrete_map=colors,
        title="Scatter Matrix — All Features",
        labels={f: f.replace(" (cm)", "") for f in feature_names}
    )
    fig.update_traces(diagonal_visible=False, marker=dict(size=3))
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(df[feature_names].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.caption("Universidad de la Costa · Data Mining · 2026")
