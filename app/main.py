import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import math

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon=r"C:\games\Cancer-app\assets\breast-cancer.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
css = """
    <style>
    .diagnosis {
        color: #fff;
        padding: 0.2em 0.5em;
        border-radius: 0.5em;
    }

    .diagnosis.benign {
        background-color: #01DB4B
    }

    .diagnosis.borderline_benign {
        background-color: #A9E34B
    }

    .diagnosis.borderline_malicious {
        background-color: #F2B900
    }

    .diagnosis.malicious {
        background-color: #ff4b4b
    }
    </style>
"""

# Add custom CSS
st.markdown(css, unsafe_allow_html=True)

def main():
    input_data = add_sidebar()
    add_predictions(input_data)
    add_visualizations()

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)
    prediction_proba = model.predict_proba(input_array_scaled)

    malignant_prob = prediction_proba[0][1]
    
    if malignant_prob < 0.3:
        prediction_label = "Benign"
        diagnosis_class = "benign"
    elif 0.3 <= malignant_prob < 0.5:
        prediction_label = "Borderline Benign but Risky"
        diagnosis_class = "borderline_benign"
    elif 0.5 <= malignant_prob < 0.7:
        prediction_label = "Borderline Malicious"
        diagnosis_class = "borderline_malicious"
    else:
        prediction_label = "Malignant"
        diagnosis_class = "malicious"

    st.markdown(f"<div class='diagnosis {diagnosis_class}'>Prediction: {prediction_label}</div>", unsafe_allow_html=True)
    st.write(f"Prediction Probability (Benign): {prediction_proba[0][0]:.2f}")
    st.write(f"Prediction Probability (Malignant): {malignant_prob:.2f}")

def add_visualizations():
    data = get_clean_data()

    st.subheader("Visualizations")

    # Feature Importance Bar Chart
    st.write("### Feature Importance")
    feature_importance_chart(data)

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    correlation_heatmap(data)

    # Bubble Chart
    st.write("### Bubble Chart")
    add_bubble_chart(data)

    

def feature_importance_chart(data):
    model = pickle.load(open("model/model.pkl", "rb"))
    features = data.drop(['diagnosis'], axis=1).columns
    importance = model.coef_[0]  # Coefficients from the model
    feature_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values(by="Importance", ascending=False)

    # Create a plot
    fig = px.bar(feature_df, x="Importance", y="Feature", orientation="h", title="Feature Importance")

    # Show the chart with a unique key
    st.plotly_chart(fig, use_container_width=True, key="feature_importance_chart")
    
    # Add explanation
    st.markdown("""
    ### Interpretation of Feature Importance:
    The chart above displays the importance of each feature in predicting the likelihood of the tumor being malignant (1) or benign (0). Features with higher importance (located on the left) play a greater role in determining the prediction. 

    - **Positive Importance**: Features with positive importance have a direct impact on predicting malignant tumors (higher values increase the likelihood of malignancy).
    - **Negative Importance**: Features with negative importance are less significant or might have a more complex relationship (e.g., larger values might indicate benign tumors).

    Based on this chart, features like `radius_mean`, `perimeter_mean`, and `smoothness_mean` have a higher contribution to determining whether a tumor is malignant. On the other hand, features with lower importance might be less informative for the model.

    In practice, focusing on high-importance features can lead to better understanding and improvement of the model.
    """)
def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)
    prediction_proba = model.predict_proba(input_array_scaled)

    malignant_prob = prediction_proba[0][1]
    benign_prob = prediction_proba[0][0]
    
    if malignant_prob < 0.3:
        prediction_label = "Benign"
        diagnosis_class = "benign"
    elif 0.3 <= malignant_prob < 0.5:
        prediction_label = "Borderline Benign but Risky"
        diagnosis_class = "borderline_benign"
    elif 0.5 <= malignant_prob < 0.7:
        prediction_label = "Borderline Malicious"
        diagnosis_class = "borderline_malicious"
    else:
        prediction_label = "Malignant"
        diagnosis_class = "malicious"

    st.markdown(f"<div class='diagnosis {diagnosis_class}'>Prediction: {prediction_label}</div>", unsafe_allow_html=True)
    st.write(f"Prediction Probability (Benign): {benign_prob:.2f}")
    st.write(f"Prediction Probability (Malignant): {malignant_prob:.2f}")
    
    # Displaying the bar chart for the prediction probabilities
    st.write("### Prediction Probabilities (Bar Chart)")

    # Bar chart for benign vs malignant probabilities
    fig = go.Figure()

    # Adding bars for prediction probabilities
    fig.add_trace(go.Bar(
        x=["Benign", "Malignant"],
        y=[benign_prob, malignant_prob],
        marker_color=['green', 'red'],
        name="Predicted Probability"
    ))

    # Set the title and labels
    fig.update_layout(
        title="Predicted Probability for Benign vs Malignant",
        xaxis_title="Diagnosis",
        yaxis_title="Probability",
        barmode='group',
        showlegend=False
    )

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

    # Add description
    st.markdown("""
    ### Interpretation of the Bar Chart:
    The bar chart above shows the predicted probabilities for the tumor being **Benign** and **Malignant** based on the model's predictions. 
    - The **green bar** represents the probability of the tumor being **Benign**, and the **red bar** represents the probability of the tumor being **Malignant**.
    - The height of each bar indicates how confident the model is in classifying the tumor into one of these categories.

    ### Example:
    - If the **Benign** probability is higher, it means the tumor is more likely to be non-cancerous.
    - If the **Malignant** probability is higher, it indicates a higher chance of the tumor being cancerous.

    ### Conclusion:
    This bar chart provides a quick visual comparison of the predicted probabilities for both benign and malignant diagnoses. By looking at the height of the bars, users can immediately understand the model's confidence in each prediction.
    """)


def correlation_heatmap(data):
    # Compute correlation matrix
    correlation = data.corr()
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot heatmap with adjusted style and annotations
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                center=0, cbar_kws={'label': 'Correlation Coefficient'},
                annot_kws={"size": 5, "weight": 'bold'}, linewidths=1.0, linecolor='gray')
    
    # Improve readability by rotating the x and y labels
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, ha="right", fontsize=10)
    
    # Title of the heatmap
    ax.set_title("Correlation Heatmap of Breast Cancer Features", fontsize=16, weight='bold')
    
    # Display the heatmap
    st.pyplot(fig)
    
    # Add an explanation for the users
    st.markdown("""
    ### Interpretation of the Correlation Heatmap:
    The heatmap above shows the pairwise correlation between various features in the dataset. Correlation values range from -1 to 1:
    
    - **+1**: Perfect positive correlation, meaning that as one feature increases, the other also increases proportionally.
    - **0**: No correlation, meaning that there is no linear relationship between the features.
    - **-1**: Perfect negative correlation, meaning that as one feature increases, the other decreases proportionally.
    
    The color intensity represents the strength of the correlation:
    - **Red (positive)**: Strong positive correlation.
    - **Blue (negative)**: Strong negative correlation.
    - **White (neutral)**: No or weak correlation.
    
    ### How Correlation Relates to Diagnosis (Benign and Malignant):
    - **Benign (0)**: Features associated with benign tumors tend to have less variation and lower values in some characteristics like texture, radius, and smoothness.
    - **Malignant (1)**: Malignant tumors often have more variation and higher values for certain features such as radius, perimeter, and concavity.
    
    Strong correlations between features might indicate that those features share similar characteristics, which could help the model identify benign or malignant tumors. For instance, a high correlation between `radius_mean` and `perimeter_mean` may suggest that tumors with larger radii tend to have larger perimeters, a pattern more likely found in malignant tumors.

    ### Key Takeaways:
    - Highly correlated features may provide similar information, and understanding these relationships can help refine the model.
    - It's important to look for features that distinguish benign from malignant tumors, which might show up as correlations with diagnostic labels.
    """)

def add_bubble_chart(data):
    # Simulating an example of a bubble chart analogy (GDP vs Life Expectancy)
    # Here we will use some of the features from the breast cancer dataset for this analogy
    df = data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'diagnosis']]
    df_2007 = df  # We can consider this as the simplified dataset

    # Define hover text and bubble size
    hover_text = []
    bubble_size = []

    for index, row in df_2007.iterrows():
        hover_text.append(('Radius Mean: {radius}<br>' +
                          'Texture Mean: {texture}<br>' +
                          'Perimeter Mean: {perimeter}<br>' +
                          'Area Mean: {area}<br>' +
                          'Smoothness Mean: {smoothness}<br>' +
                          'Diagnosis: {diagnosis}').format(
                              radius=row['radius_mean'],
                              texture=row['texture_mean'],
                              perimeter=row['perimeter_mean'],
                              area=row['area_mean'],
                              smoothness=row['smoothness_mean'],
                              diagnosis='Malignant' if row['diagnosis'] == 1 else 'Benign'))
        
        # Bubble size based on the perimeter (or another feature of choice)
        bubble_size.append(math.sqrt(row['perimeter_mean']))

    df_2007['text'] = hover_text
    df_2007['size'] = bubble_size

    sizeref = 2.*max(df_2007['size'])/(100**2)

    # Create the Bubble Chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_2007['radius_mean'], y=df_2007['texture_mean'],
        mode='markers',
        text=df_2007['text'],
        marker=dict(
            size=df_2007['size'], sizemode='area', sizeref=sizeref,
            line_width=2, color=df_2007['diagnosis'], 
            colorscale='Viridis', colorbar=dict(title='Diagnosis'),
        ),
        name='Cancer Data'
    ))

    fig.update_layout(
        title="Bubble Chart (Radius Mean vs Texture Mean)",
        xaxis_title="Radius Mean",
        yaxis_title="Texture Mean",
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        showlegend=True
    )

    # Show the chart
    st.plotly_chart(fig, use_container_width=True)

    # Add explanation
    st.markdown("""
    ### Interpretation of the Bubble Chart:
    This bubble chart represents a simple analogy where we show two features: **Radius Mean** and **Texture Mean**.
    
    - The size of the bubble indicates the **Perimeter Mean**, which gives us an idea of how large the tumor is.
    - The color of the bubble indicates the **Diagnosis**: 
        - **Malignant (1)**: Red color.
        - **Benign (0)**: Green color.
        
    The **Radius Mean** is plotted on the X-axis, and the **Texture Mean** is plotted on the Y-axis. Larger tumors with higher perimeters will have larger bubbles.
    """)






def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()

    slider_labels = [
        ("Radius(mean)", "radius_mean"),
        ("Texture(mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict

def get_clean_data():
    data = pd.read_csv(r"C:\games\Cancer-app\data\data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1, errors='ignore')
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

if __name__ == '__main__':
    main()
