"""
Streamlit Web App for RetailRocket Product Recommendation System
Deployment of the trained ML model with interactive features
"""

# ============================================================================
# 1. STREAMLIT APP - MAIN APPLICATION
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import sys
import os

# Set page configuration
st.set_page_config(
    page_title="RetailRocket Product Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .success-message {
        color: #10B981;
        font-weight: bold;
    }
    .warning-message {
        color: #F59E0B;
        font-weight: bold;
    }
    .model-performance {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. LOAD MODELS AND DATA
# ============================================================================

@st.cache_resource
def load_models():
    """Load all trained models and artifacts"""
    models = {}
    
    try:
        # Load neural network model
        import tensorflow as tf
        models['nn_model'] = tf.keras.models.load_model('models/best_nn_model.h5')
        st.success("✅ Neural Network model loaded successfully")
    except Exception as e:
        st.warning(f"⚠ Neural Network model not found: {e}")
        models['nn_model'] = None
    
    try:
        # Load Random Forest model
        models['rf_model'] = joblib.load('models/Random_Forest.pkl')
        st.success("✅ Random Forest model loaded successfully")
    except:
        st.warning("⚠ Random Forest model not found")
        models['rf_model'] = None
    
    try:
        # Load Gradient Boosting model
        models['gb_model'] = joblib.load('models/Gradient_Boosting.pkl')
        st.success("✅ Gradient Boosting model loaded successfully")
    except:
        st.warning("⚠ Gradient Boosting model not found")
        models['gb_model'] = None
    
    try:
        # Load scaler
        models['scaler'] = joblib.load('models/scaler.pkl')
        st.success("✅ Scaler loaded successfully")
    except:
        st.warning("⚠ Scaler not found")
        models['scaler'] = None
    
    try:
        # Load feature columns
        with open('models/feature_columns.json', 'r') as f:
            models['feature_columns'] = json.load(f)
        st.success("✅ Feature columns loaded successfully")
    except:
        st.warning("⚠ Feature columns not found")
        models['feature_columns'] = []
    
    try:
        # Load user features
        models['user_features'] = pd.read_parquet('artifacts/user_features.parquet')
        st.success("✅ User features loaded successfully")
    except:
        st.warning("⚠ User features not found")
        models['user_features'] = None
    
    try:
        # Load product features
        models['product_features'] = pd.read_parquet('artifacts/product_features.parquet')
        st.success("✅ Product features loaded successfully")
    except:
        st.warning("⚠ Product features not found")
        models['product_features'] = None
    
    try:
        # Load performance metrics
        models['metrics'] = pd.read_csv('artifacts/performance_metrics.csv')
        st.success("✅ Performance metrics loaded successfully")
    except:
        st.warning("⚠ Performance metrics not found")
        models['metrics'] = None
    
    return models

# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================

def prepare_features(user_id, product_id, models):
    """Prepare features for prediction"""
    features = {}
    
    # Get user features
    if models['user_features'] is not None:
        user_data = models['user_features'][models['user_features']['visitorid'] == user_id]
        if not user_data.empty:
            for col in user_data.columns:
                if col != 'visitorid':
                    features[f'user_{col}'] = user_data[col].values[0]
    
    # Get product features
    if models['product_features'] is not None:
        product_data = models['product_features'][models['product_features']['itemid'] == product_id]
        if not product_data.empty:
            for col in product_data.columns:
                if col != 'itemid':
                    features[f'product_{col}'] = product_data[col].values[0]
    
    # Add default values for missing features
    default_features = {
        'total_interactions': 0,
        'first_interaction': 0,
        'last_interaction': 0,
        'interaction_duration': 0,
        'view_count': 0,
        'addtocart_count': 0,
        'purchase_count': 0,
        'has_viewed': 0,
        'has_added_to_cart': 0,
        'has_purchased': 0,
        'has_view_to_cart': 0,
        'has_cart_to_purchase': 0,
        'user_activity_normalized': features.get('user_total_events', 0) / 1000,
        'product_popularity_normalized': features.get('product_total_interactions', 0) / 1000,
        'conversion_rate_normalized': features.get('product_conversion_rate', 0) / 0.02,
        'interaction_intensity': 0,
        'user_product_affinity': 0,
        'product_user_affinity': 0,
        'user_product_match': 0
    }
    
    features.update(default_features)
    
    # Ensure all features are in correct order
    feature_vector = []
    for col in models['feature_columns']:
        if col in features:
            feature_vector.append(features[col])
        else:
            feature_vector.append(0)
    
    return np.array(feature_vector).reshape(1, -1)

def predict_relevance(models, user_id, product_id, model_type='nn'):
    """Predict relevance score using specified model"""
    if models['scaler'] is None or len(models['feature_columns']) == 0:
        return None
    
    # Prepare features
    features = prepare_features(user_id, product_id, models)
    features_scaled = models['scaler'].transform(features)
    
    # Make prediction
    if model_type == 'nn' and models['nn_model'] is not None:
        probability = models['nn_model'].predict(features_scaled, verbose=0)[0, 0]
    elif model_type == 'rf' and models['rf_model'] is not None:
        probability = models['rf_model'].predict_proba(features_scaled)[0, 1]
    elif model_type == 'gb' and models['gb_model'] is not None:
        probability = models['gb_model'].predict_proba(features_scaled)[0, 1]
    else:
        return None
    
    return float(probability)

def get_top_recommendations(models, user_id, candidate_products, top_n=10, model_type='nn'):
    """Get top recommendations for a user"""
    recommendations = []
    
    with st.spinner(f"Generating {top_n} recommendations..."):
        progress_bar = st.progress(0)
        
        for i, product_id in enumerate(candidate_products):
            score = predict_relevance(models, user_id, product_id, model_type)
            if score is not None:
                recommendations.append({
                    'product_id': product_id,
                    'relevance_score': score,
                    'probability': score
                })
            
            # Update progress
            progress = (i + 1) / len(candidate_products)
            progress_bar.progress(progress)
    
    # Sort by relevance score
    recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return recommendations[:top_n]

# ============================================================================
# 4. STREAMLIT APP LAYOUT
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.markdown('<h1 class="main-header">🛒 RetailRocket Product Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #6B7280;">
            Advanced ML-powered product recommendations for e-commerce
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading ML models and data..."):
        models = load_models()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a module",
        ["🏠 Dashboard", "🎯 Single Prediction", "📊 Batch Recommendations", 
         "📈 Model Analytics", "⚙️ System Settings"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About this App:**
    - Powered by 7 ML models
    - Real-time predictions
    - Business impact analysis
    - Production-ready deployment
    """)
    
    # ========================================================================
    # DASHBOARD
    # ========================================================================
    if app_mode == "🏠 Dashboard":
        st.markdown('<h2 class="sub-header">📊 System Dashboard</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Total Users", value=f"{len(models['user_features']):,}" if models['user_features'] is not None else "N/A")
        
        with col2:
            st.metric(label="Total Products", value=f"{len(models['product_features']):,}" if models['product_features'] is not None else "N/A")
        
        with col3:
            st.metric(label="Models Loaded", value=f"{sum(1 for m in models.values() if m is not None)}/8")
        
        # Model performance metrics
        if models['metrics'] is not None:
            st.markdown('<h3 class="sub-header">🎯 Model Performance</h3>', unsafe_allow_html=True)
            
            # Get best model
            best_model = models['metrics'].loc[models['metrics']['f1_score'].idxmax()]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Best Model</h4>
                    <p style="font-size: 1.8rem; font-weight: bold;">{best_model['model']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>F1 Score</h4>
                    <p style="font-size: 1.8rem; font-weight: bold; color: #10B981;">{best_model['f1_score']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>ROC AUC</h4>
                    <p style="font-size: 1.8rem; font-weight: bold; color: #3B82F6;">{best_model['roc_auc']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Expected Profit</h4>
                    <p style="font-size: 1.8rem; font-weight: bold; color: #8B5CF6;">${best_model['expected_profit']:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Quick start section
        st.markdown('<h3 class="sub-header">🚀 Quick Start</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Go to Single Prediction", use_container_width=True):
                st.session_state['nav_to'] = "single"
                st.rerun()
        
        with col2:
            if st.button("📊 Go to Batch Recommendations", use_container_width=True):
                st.session_state['nav_to'] = "batch"
                st.rerun()
    
    # ========================================================================
    # SINGLE PREDICTION
    # ========================================================================
    elif app_mode == "🎯 Single Prediction" or (hasattr(st.session_state, 'nav_to') and st.session_state.nav_to == "single"):
        st.markdown('<h2 class="sub-header">🎯 Single Product Recommendation</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # User selection
            if models['user_features'] is not None:
                user_ids = models['user_features']['visitorid'].unique()
                selected_user = st.selectbox(
                    "Select User ID",
                    options=user_ids[:1000],  # Limit for performance
                    index=0
                )
            else:
                selected_user = st.number_input("Enter User ID", min_value=1, value=12345)
        
        with col2:
            # Product selection
            if models['product_features'] is not None:
                product_ids = models['product_features']['itemid'].unique()
                selected_product = st.selectbox(
                    "Select Product ID",
                    options=product_ids[:1000],  # Limit for performance
                    index=0
                )
            else:
                selected_product = st.number_input("Enter Product ID", min_value=1, value=67890)
        
        # Model selection
        model_type = st.radio(
            "Select Prediction Model",
            ["Neural Network", "Random Forest", "Gradient Boosting"],
            horizontal=True
        )
        
        model_map = {
            "Neural Network": "nn",
            "Random Forest": "rf",
            "Gradient Boosting": "gb"
        }
        
        # Prediction threshold
        threshold = st.slider(
            "Recommendation Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Probability threshold above which product is recommended"
        )
        
        # Make prediction
        if st.button("Predict Relevance", type="primary", use_container_width=True):
            with st.spinner("Making prediction..."):
                probability = predict_relevance(
                    models, 
                    selected_user, 
                    selected_product, 
                    model_map[model_type]
                )
            
            if probability is not None:
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Relevance Probability", f"{probability:.3f}")
                
                with col2:
                    recommendation = "✅ RECOMMEND" if probability >= threshold else "❌ NOT RECOMMENDED"
                    st.metric("Recommendation", recommendation)
                
                with col3:
                    confidence = "High" if abs(probability - threshold) > 0.3 else "Medium" if abs(probability - threshold) > 0.1 else "Low"
                    st.metric("Confidence", confidence)
                
                # Visual indicators
                st.progress(probability)
                
                # Detailed analysis
                st.markdown("---")
                st.markdown('<h4 class="sub-header">📋 Detailed Analysis</h4>', unsafe_allow_html=True)
                
                if probability >= threshold:
                    st.success(f"**Strong Match!** This product has a {probability*100:.1f}% chance of being relevant to this user.")
                    
                    # Business impact estimate
                    expected_profit = probability * 10 - (1 - probability) * 2
                    st.info(f"**Estimated Business Impact:** ${expected_profit:.2f} expected profit per recommendation")
                else:
                    st.warning(f"**Weak Match.** This product has only a {probability*100:.1f}% chance of being relevant.")
                
                # Show feature importance (simplified)
                if model_type == "Random Forest" and models['rf_model'] is not None:
                    st.markdown('<h4 class="sub-header">🔍 Top Influencing Factors</h4>', unsafe_allow_html=True)
                    
                    # Get feature importance
                    feature_importance = models['rf_model'].feature_importances_
                    top_indices = np.argsort(feature_importance)[-5:]
                    top_features = np.array(models['feature_columns'])[top_indices]
                    top_importance = feature_importance[top_indices]
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 4))
                    y_pos = np.arange(len(top_features))
                    ax.barh(y_pos, top_importance, color='#3B82F6')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(top_features)
                    ax.set_xlabel('Importance')
                    ax.set_title('Top 5 Influencing Features')
                    st.pyplot(fig)
        
        # Clear navigation state
        if hasattr(st.session_state, 'nav_to'):
            del st.session_state.nav_to
    
    # ========================================================================
    # BATCH RECOMMENDATIONS
    # ========================================================================
    elif app_mode == "📊 Batch Recommendations" or (hasattr(st.session_state, 'nav_to') and st.session_state.nav_to == "batch"):
        st.markdown('<h2 class="sub-header">📊 Batch Product Recommendations</h2>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔍 For Specific User", "📁 Upload CSV"])
        
        with tab1:
            # User selection for batch recommendations
            if models['user_features'] is not None:
                user_ids = models['user_features']['visitorid'].unique()
                selected_user_batch = st.selectbox(
                    "Select User for Recommendations",
                    options=user_ids[:500],
                    index=0,
                    key="batch_user"
                )
            else:
                selected_user_batch = st.number_input("Enter User ID for Recommendations", 
                                                    min_value=1, value=12345, key="batch_user_input")
            
            # Number of recommendations
            top_n = st.slider("Number of Recommendations", min_value=5, max_value=50, value=10, step=5)
            
            # Candidate products
            if models['product_features'] is not None:
                product_ids = models['product_features']['itemid'].unique()
                
                col1, col2 = st.columns(2)
                with col1:
                    min_product = st.number_input("From Product ID", min_value=1, 
                                                value=int(product_ids[0]), max_value=int(product_ids[-1]))
                with col2:
                    max_product = st.number_input("To Product ID", min_value=1, 
                                                value=int(min(min_product + 100, product_ids[-1])), 
                                                max_value=int(product_ids[-1]))
                
                candidate_products = [pid for pid in product_ids if min_product <= pid <= max_product][:200]  # Limit
            else:
                st.info("Product features not available. Using sample product IDs.")
                candidate_products = list(range(1000, 1200))
            
            # Model selection for batch
            batch_model_type = st.radio(
                "Select Model for Batch Predictions",
                ["Neural Network", "Random Forest", "Gradient Boosting"],
                horizontal=True,
                key="batch_model"
            )
            
            if st.button("Generate Recommendations", type="primary", use_container_width=True):
                if len(candidate_products) == 0:
                    st.error("No candidate products selected!")
                else:
                    recommendations = get_top_recommendations(
                        models,
                        selected_user_batch,
                        candidate_products,
                        top_n,
                        model_map[batch_model_type]
                    )
                    
                    if recommendations:
                        # Display recommendations
                        st.markdown(f"### Top {len(recommendations)} Recommendations for User {selected_user_batch}")
                        
                        # Create dataframe for display
                        rec_df = pd.DataFrame(recommendations)
                        rec_df['rank'] = range(1, len(rec_df) + 1)
                        rec_df = rec_df[['rank', 'product_id', 'relevance_score', 'probability']]
                        rec_df.columns = ['Rank', 'Product ID', 'Relevance Score', 'Probability']
                        
                        # Format dataframe
                        styled_df = rec_df.style.format({
                            'Relevance Score': '{:.3f}',
                            'Probability': '{:.3f}'
                        }).background_gradient(subset=['Relevance Score', 'Probability'], cmap='YlOrRd')
                        
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Download button
                        csv = rec_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Recommendations as CSV",
                            data=csv,
                            file_name=f"recommendations_user_{selected_user_batch}.csv",
                            mime="text/csv"
                        )
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(range(len(rec_df)), rec_df['Probability'].values[::-1], color='#3B82F6')
                        ax.set_yticks(range(len(rec_df)))
                        ax.set_yticklabels([f"Product {pid}" for pid in rec_df['Product ID'].values[::-1]])
                        ax.set_xlabel('Probability')
                        ax.set_title(f'Top {top_n} Recommended Products')
                        ax.set_xlim([0, 1])
                        st.pyplot(fig)
        
        with tab2:
            st.markdown("### 📁 Upload CSV for Batch Predictions")
            st.info("Upload a CSV file with columns: user_id, product_id")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    batch_data = pd.read_csv(uploaded_file)
                    
                    if not all(col in batch_data.columns for col in ['user_id', 'product_id']):
                        st.error("CSV must contain 'user_id' and 'product_id' columns")
                    else:
                        st.success(f"File uploaded successfully! {len(batch_data)} rows found.")
                        
                        # Show sample
                        st.dataframe(batch_data.head(), use_container_width=True)
                        
                        # Select model
                        batch_model_csv = st.radio(
                            "Select Prediction Model",
                            ["Neural Network", "Random Forest", "Gradient Boosting"],
                            horizontal=True,
                            key="csv_model"
                        )
                        
                        if st.button("Process Batch File", type="primary"):
                            predictions = []
                            
                            with st.spinner("Processing batch predictions..."):
                                progress_bar = st.progress(0)
                                
                                for i, row in batch_data.iterrows():
                                    prob = predict_relevance(
                                        models,
                                        row['user_id'],
                                        row['product_id'],
                                        model_map[batch_model_csv]
                                    )
                                    
                                    predictions.append({
                                        'user_id': row['user_id'],
                                        'product_id': row['product_id'],
                                        'probability': prob if prob is not None else 0,
                                        'recommended': prob >= 0.5 if prob is not None else False
                                    })
                                    
                                    # Update progress
                                    progress = (i + 1) / len(batch_data)
                                    progress_bar.progress(progress)
                            
                            # Create results dataframe
                            results_df = pd.DataFrame(predictions)
                            
                            # Display results
                            st.markdown("### 📊 Batch Prediction Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Predictions", len(results_df))
                            with col2:
                                st.metric("Recommendations", results_df['recommended'].sum())
                            with col3:
                                st.metric("Recommendation Rate", f"{results_df['recommended'].mean()*100:.1f}%")
                            
                            # Download results
                            csv_results = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv_results,
                                file_name="batch_predictions_results.csv",
                                mime="text/csv"
                            )
                
                except Exception as e:
                    st.error(f"Error processing file: {e}")
        
        # Clear navigation state
        if hasattr(st.session_state, 'nav_to'):
            del st.session_state.nav_to
    
    # ========================================================================
    # MODEL ANALYTICS
    # ========================================================================
    elif app_mode == "📈 Model Analytics":
        st.markdown('<h2 class="sub-header">📈 Model Performance Analytics</h2>', unsafe_allow_html=True)
        
        if models['metrics'] is not None:
            # Model comparison table
            st.markdown("### 🏆 Model Comparison")
            
            display_cols = ['model', 'accuracy', 'precision', 'recall', 'f1_score', 
                          'roc_auc', 'avg_precision', 'expected_profit']
            
            styled_metrics = models['metrics'][display_cols].copy()
            styled_metrics = styled_metrics.style.format({
                'accuracy': '{:.3f}',
                'precision': '{:.3f}',
                'recall': '{:.3f}',
                'f1_score': '{:.3f}',
                'roc_auc': '{:.3f}',
                'avg_precision': '{:.3f}',
                'expected_profit': '${:,.0f}'
            }).background_gradient(subset=['f1_score', 'roc_auc', 'expected_profit'], cmap='YlOrRd')
            
            st.dataframe(styled_metrics, use_container_width=True)
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 F1-Score Comparison")
                fig, ax = plt.subplots(figsize=(8, 6))
                models_sorted = models['metrics'].sort_values('f1_score', ascending=False)
                bars = ax.barh(models_sorted['model'], models_sorted['f1_score'], color='#3B82F6')
                ax.set_xlabel('F1-Score')
                ax.set_xlim([0, 1])
                ax.bar_label(bars, fmt='%.3f')
                st.pyplot(fig)
            
            with col2:
                st.markdown("### 💰 Expected Profit Comparison")
                fig, ax = plt.subplots(figsize=(8, 6))
                models_sorted = models['metrics'].sort_values('expected_profit', ascending=False)
                bars = ax.barh(models_sorted['model'], models_sorted['expected_profit'], color='#10B981')
                ax.set_xlabel('Expected Profit ($)')
                ax.bar_label(bars, fmt='${:,.0f}')
                st.pyplot(fig)
            
            # Feature importance visualization
            if models['rf_model'] is not None:
                st.markdown("### 🔍 Feature Importance Analysis")
                
                feature_importance = models['rf_model'].feature_importances_
                top_n_features = st.slider("Number of top features to show", 5, 20, 10)
                
                top_indices = np.argsort(feature_importance)[-top_n_features:]
                top_features = np.array(models['feature_columns'])[top_indices]
                top_importance = feature_importance[top_indices]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                y_pos = np.arange(len(top_features))
                ax.barh(y_pos, top_importance, color='#8B5CF6')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_features)
                ax.set_xlabel('Importance')
                ax.set_title(f'Top {top_n_features} Most Important Features')
                st.pyplot(fig)
        
        else:
            st.warning("Performance metrics not available. Please train models first.")
    
    # ========================================================================
    # SYSTEM SETTINGS
    # ========================================================================
    elif app_mode == "⚙️ System Settings":
        st.markdown('<h2 class="sub-header">⚙️ System Configuration</h2>', unsafe_allow_html=True)
        
        # Model settings
        st.markdown("### 🤖 Model Settings")
        
        default_threshold = st.slider(
            "Default Recommendation Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Default probability threshold for recommendations"
        )
        
        default_model = st.selectbox(
            "Default Prediction Model",
            ["Neural Network", "Random Forest", "Gradient Boosting"],
            index=0
        )
        
        # Business settings
        st.markdown("### 💼 Business Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            profit_per_relevant = st.number_input(
                "Profit per Relevant Recommendation ($)",
                min_value=0.0,
                value=10.0,
                step=1.0
            )
        
        with col2:
            cost_per_irrelevant = st.number_input(
                "Cost per Irrelevant Recommendation ($)",
                min_value=0.0,
                value=2.0,
                step=0.5
            )
        
        with col3:
            cost_per_missed = st.number_input(
                "Opportunity Cost per Missed Recommendation ($)",
                min_value=0.0,
                value=5.0,
                step=0.5
            )
        
        # System information
        st.markdown("### 🖥️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Loaded Models:**
            - Neural Network: {"✅" if models['nn_model'] is not None else "❌"}
            - Random Forest: {"✅" if models['rf_model'] is not None else "❌"}
            - Gradient Boosting: {"✅" if models['gb_model'] is not None else "❌"}
            - Scaler: {"✅" if models['scaler'] is not None else "❌"}
            """)
        
        with col2:
            st.info(f"""
            **Data Statistics:**
            - Users: {len(models['user_features']) if models['user_features'] is not None else "N/A"}
            - Products: {len(models['product_features']) if models['product_features'] is not None else "N/A"}
            - Features: {len(models['feature_columns']) if models['feature_columns'] is not None else "N/A"}
            """)
        
        # Save settings button
        if st.button("Save Settings", type="primary"):
            settings = {
                'default_threshold': default_threshold,
                'default_model': default_model,
                'profit_per_relevant': profit_per_relevant,
                'cost_per_irrelevant': cost_per_irrelevant,
                'cost_per_missed': cost_per_missed,
                'last_updated': datetime.now().isoformat()
            }
            
            # Save settings (in a real app, this would save to a file/database)
            st.session_state['settings'] = settings
            st.success("Settings saved successfully!")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; padding: 1rem;">
        <p>🛒 <strong>RetailRocket Product Recommendation System</strong> v1.0.0</p>
        <p>Powered by Machine Learning & Deep Learning | December 2024</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 5. RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    # Check if required files exist
    required_files = [
        'models/best_nn_model.h5',
        'models/Random_Forest.pkl',
        'models/scaler.pkl',
        'models/feature_columns.json'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        st.warning(f"⚠ Missing required files: {missing_files}")
        st.info("Please run the main training script first to generate model files.")
        
        if st.button("Generate Sample Data for Demo"):
            # In a real scenario, you would generate sample data
            st.info("Sample data generation would run here...")
            # For now, we'll continue with the app using placeholder data
            main()
    else:
        main()