import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sys
import os

# 检查Python版本和依赖包版本
st.set_page_config(
    page_title="HSRF K Predictor",
    page_icon="📈",
    layout="wide"
)


def check_versions():
    info = []
    info.append(f"Python version: {sys.version}")
    info.append(f"Streamlit version: {st.__version__}")
    info.append(f"Pandas version: {pd.__version__}")
    info.append(f"NumPy version: {np.__version__}")
    
    try:
        import sklearn
        info.append(f"Scikit-learn version: {sklearn.__version__}")
    except:
        info.append("Scikit-learn: Not installed")
    
    return info


@st.cache_resource
def load_resources():
    try:
        model = joblib.load('hydrogel_k_predictor.pkl')
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None, None
    
    try:
        with open('feature_config.json', 'r') as f:
            config = json.load(f)
        st.success("✅ Configuration file loaded successfully")
    except Exception as e:
        st.error(f"❌ Configuration file loading failed: {str(e)}")
        return model, None
    
    return model, config


with st.sidebar:
    st.subheader("System Information")
    versions = check_versions()
    for v in versions:
        st.text(v)


st.title("📈 HSRF First-order Kinetic Rate Constant (K) Predictor")

model, config = load_resources()

if model is None or config is None:
    st.stop()


left_col, right_col = st.columns([0.7, 0.3])

with left_col:
    st.subheader("Input Parameters")
    
    # Binary features section
    st.markdown("**Binary Features (0 or 1)**")
    bin_cols = st.columns(5)
    input_data = {}
    
    binary_features = config.get('binary_features', ['AA', 'AMPS', 'CMC', 'PAA', 'PVA'])
    for i, feature in enumerate(binary_features):
        with bin_cols[i]:
            input_data[feature] = st.selectbox(
                f"{feature}",
                [0, 1],
                key=f"bin_{feature}"
            )
    
    # Numeric features section
    st.markdown("**Numeric Features**")
    
    num_cols1 = st.columns(4)
    num_features1 = ['CBR', 'RT (℃)', 'Rti (h)', 'DT (℃)']
    
    for i, feature in enumerate(num_features1):
        with num_cols1[i]:
            range_info = config.get('numeric_ranges', {}).get(feature, {'min': 0, 'max': 100})
            input_data[feature] = st.number_input(
                f"{feature}\nRange: {range_info.get('min', 0):.2f} ~ {range_info.get('max', 100):.2f}",
                min_value=float(range_info.get('min', 0)),
                max_value=float(range_info.get('max', 100)),
                value=float((range_info.get('min', 0) + range_info.get('max', 100)) / 2),
                step=0.1,
                key=f"num_{feature}"
            )
    
    num_cols2 = st.columns(4)
    num_features2 = ['UC (%)', 'Rtemp (℃)', 'pH-SR']
    
    for i, feature in enumerate(num_features2):
        with num_cols2[i]:
            range_info = config.get('numeric_ranges', {}).get(feature, {'min': 0, 'max': 100})
            input_data[feature] = st.number_input(
                f"{feature}\nRange: {range_info.get('min', 0):.2f} ~ {range_info.get('max', 100):.2f}",
                min_value=float(range_info.get('min', 0)),
                max_value=float(range_info.get('max', 100)),
                value=float((range_info.get('min', 0) + range_info.get('max', 100)) / 2),
                step=0.1,
                key=f"num_{feature}"
            )
    
    # Categorical features section
    st.markdown("**Categorical Features**")
    cat_cols = st.columns(2)
    
    with cat_cols[0]:
        cra_options = config.get('categorical_options', {}).get('CRA', ['Type A', 'Type B'])
        input_data['CRA'] = st.selectbox("CRA", cra_options, key="cat_CRA")
    
    with cat_cols[1]:
        ulm_options = config.get('categorical_options', {}).get('ULM', ['Type X', 'Type Y'])
        input_data['ULM'] = st.selectbox("ULM", ulm_options, key="cat_ULM")

with right_col:
    st.subheader("Prediction")
    
    if st.button("PREDICT", type="primary", use_container_width=True):
        try:
            # Get feature order from config
            feature_order = config.get('feature_order', 
                                     binary_features + 
                                     num_features1 + num_features2 + 
                                     ['CRA', 'ULM'])
            
            # Validate that all required features are present
            for feature in feature_order:
                if feature not in input_data:
                    st.error(f"Missing feature: {feature}")
                    st.stop()
            
            # Create DataFrame with correct column order
            input_df = pd.DataFrame([input_data])
            input_df = input_df[feature_order]
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Display result
            st.markdown("---")
            st.markdown("**Result:**")
            st.markdown(f"### K = {prediction:.6f} h⁻¹")
            
            # Store results in session state
            st.session_state['last_prediction'] = prediction
            st.session_state['last_input'] = input_data
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.exception(e)
    
    # Display last prediction if available
    if 'last_prediction' in st.session_state:
        st.markdown("---")
        st.markdown("**Last Prediction:**")
        st.markdown(f"K = {st.session_state['last_prediction']:.6f} h⁻¹")

# Add an expander for input summary
with st.expander("Input Summary", expanded=False):
    if input_data:
        summary_df = pd.DataFrame([input_data]).T.reset_index()
        summary_df.columns = ['Parameter', 'Value']
        st.table(summary_df)

st.markdown("---")
st.caption("HSRF K Prediction Model | First-order kinetic rate constant")