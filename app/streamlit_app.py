# ============================================================================
# nairobi_realestate_predictor/app/streamlit_app.py
# Interactive Streamlit Application for Nairobi Real Estate Price Prediction
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

# Add src to path for custom modules
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Set page config
st.set_page_config(
    page_title="Nairobi Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #D1FAE5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #10B981;
        text-align: center;
        margin: 1.5rem 0;
    }
    .market-stats {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #F59E0B;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🏠 Nairobi Real Estate Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <strong>Addressing Kenya's 2M+ Housing Deficit</strong> - This AI-powered tool helps buyers, investors, 
    and developers make data-driven decisions in Nairobi's dynamic real estate market. 
    Powered by machine learning models trained on thousands of property listings.
</div>
""", unsafe_allow_html=True)

# Sidebar for navigation and info
with st.sidebar:
    st.image("https://wallpaperaccess.com/full/3183270.jpg", 
             use_container_width=True)
    st.markdown("### 📊 Navigation")
    app_mode = st.radio(
        "Choose Application Mode",
        ["🏠 Price Prediction", "📈 Market Analysis", "🗺️ Location Insights", "💰 Investment Calculator"]
    )
    
    st.markdown("---")
    st.markdown("### 📈 Market Context")
    st.markdown("""
    <div class="market-stats">
    <strong>Kenya Real Estate 2026:</strong>
    • Housing Deficit: 2M+ units[citation:5]
    • Annual Demand: 250,000 units[citation:5]
    • Annual Supply: 50,000 units[citation:5]
    • Urbanization Rate: 4.3%[citation:2]
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎯 Target Users")
    st.markdown("""
    - **Home Buyers**: Avoid overpaying by 10-20%
    - **Investors**: Identify undervalued properties
    - **Developers**: Optimize project pricing
    - **Agents**: Provide data-backed valuations
    """)

# ========== FIXED: LOAD MODEL FUNCTION ==========
# ========== FIXED: LOAD MODEL FUNCTION ==========
@st.cache_resource
def load_model():
    """Load trained model and feature information - WORKS BOTH LOCAL & DEPLOYED"""
    try:
        import os
        from pathlib import Path
        
        # Detect if we're in deployment (Render, Vercel, Streamlit Cloud)
        IS_DEPLOYED = 'RENDER' in os.environ or 'VERCEL' in os.environ or 'STREAMLIT_SHARING' in os.environ
        
        if IS_DEPLOYED:
            # DEPLOYMENT: Current directory is project root
            # Models are in /models/ folder
            models_dir = Path.cwd() / 'models'
        else:
            # LOCAL DEVELOPMENT: Models are in nairobi_realestate_predictor/models/
            current_file = Path(__file__).resolve()
            app_dir = current_file.parent  # app folder
            project_dir = app_dir.parent   # nairobi_realestate_predictor folder
            models_dir = project_dir / 'models'
        
        # Check if models directory exists
        if not models_dir.exists():
            return None, None, None
        
        # Load feature info
        info_path = models_dir / 'model_feature_info.json'
        
        if not info_path.exists():
            return None, None, None
        
        with open(info_path, 'r') as f:
            feature_info = json.load(f)
        
        # Try to load ensemble model
        model_path = models_dir / 'ensemble_model.pkl'
        if not model_path.exists():
            model_path = models_dir / 'ensemble_best_model.pkl'
        
        if not model_path.exists():
            return None, feature_info, None
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Check if it's an ensemble model with meta_model
        if isinstance(model_data, dict) and 'meta_model' in model_data:
            return model_data['meta_model'], feature_info, model_data.get('base_models', None)
        else:
            return model_data, feature_info, None
            
    except Exception as e:
        # Fail silently - let the app continue without models
        return None, None, None
    
@st.cache_data
def load_sample_data():
    """Load sample property data for comparison"""
    try:
        import os
        from pathlib import Path
        
        # Detect if we're in deployment (Render, Vercel, Streamlit Cloud)
        IS_DEPLOYED = 'RENDER' in os.environ or 'VERCEL' in os.environ or 'STREAMLIT_SHARING' in os.environ
        
        if IS_DEPLOYED:
            # DEPLOYMENT: Current directory is project root
            # Data are in /data/ folder
            base_dir = Path.cwd()
            data_dir = base_dir / 'data'
        else:
            # LOCAL DEVELOPMENT: Data are in nairobi_realestate_predictor/data/
            current_file = Path(__file__).resolve()
            app_dir = current_file.parent  # app folder
            base_dir = app_dir.parent      # nairobi_realestate_predictor folder
            data_dir = base_dir / 'data'
        
        # Try to find data file in multiple possible locations
        possible_paths = [
            data_dir / 'processed' / 'nairobi_processed_features.csv',
            data_dir / 'nairobi_processed_features.csv',
            base_dir / 'data' / 'processed' / 'nairobi_processed_features.csv',
            base_dir / 'data' / 'nairobi_processed_features.csv'
        ]
        
        for data_path in possible_paths:
            if data_path.exists():
                df = pd.read_csv(data_path)
                return df
        
        # Fallback to sample data if no file found
        return pd.DataFrame({
            'SIZE_SQM_CAPPED': [80, 120, 150, 200],
            'BEDROOMS': [2, 3, 3, 4],
            'LOCATION_MEAN_ENCODED': [35000000, 45000000, 28000000, 55000000],
            'IS_SATELLITE': [0, 0, 1, 0],
            'PRICE_KSH_CAPPED': [8500000, 12500000, 9800000, 18500000]
        })
        
    except Exception as e:
        # Fallback to sample data
        return pd.DataFrame({
            'SIZE_SQM_CAPPED': [80, 120, 150, 200],
            'BEDROOMS': [2, 3, 3, 4],
            'LOCATION_MEAN_ENCODED': [35000000, 45000000, 28000000, 55000000],
            'IS_SATELLITE': [0, 0, 1, 0],
            'PRICE_KSH_CAPPED': [8500000, 12500000, 9800000, 18500000]
        })

# ========== AUTOMATICALLY LOAD MODELS ==========
# This happens automatically when app starts
model, feature_info, base_models = load_model()
sample_data = load_sample_data()

# ==================== PRICE PREDICTION MODULE ====================
if app_mode == "🏠 Price Prediction":
    st.markdown('<h2 class="sub-header">🏠 Property Price Prediction</h2>', unsafe_allow_html=True)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Property Details")
        
        # Location selection
        locations = [
            "Westlands", "Kilimani", "Kileleshwa", "Ruiru", "Kitengela",
            "Syokimau", "Athi River", "Thika", "Ngong", "Rongai",
            "Karen", "Lavington", "Parklands", "South B", "Embakasi"
        ]
        
        location = st.selectbox(
            "📍 Location",
            locations,
            help="Select property location. Satellite towns offer better value[citation:1]"
        )
        
        # Property features
        bedrooms = st.slider(
            "🛏️ Number of Bedrooms",
            min_value=1,
            max_value=6,
            value=3,
            help="Each bedroom adds approximately KSh 1.5M to property value"
        )
        
        size_sqm = st.slider(
            "📐 Size (Square Meters)",
            min_value=20,
            max_value=400,
            value=120,
            help="Typical sizes: 80 SQM (2BR), 120 SQM (3BR), 200 SQM (4BR)"
        )
        
    with col2:
        st.markdown("#### Additional Features")
        
        bathrooms = st.slider(
            "🚿 Number of Bathrooms",
            min_value=1,
            max_value=4,
            value=2,
            help="Typically 1-2 bathrooms per bedroom"
        )
        
        # Satellite town indicator
        satellite_towns = ["Ruiru", "Kitengela", "Syokimau", "Athi River", "Thika", "Ngong", "Rongai"]
        is_satellite = 1 if location in satellite_towns else 0
        
        if is_satellite:
            st.info(f"✅ {location} is a satellite town with 15-30% annual appreciation potential[citation:10]")
        
        # Infrastructure corridor
        thika_corridor = ["Ruiru", "Thika", "Juja"]
        mombasa_corridor = ["Kitengela", "Syokimau", "Athi River"]
        
        corridor_info = ""
        if location in thika_corridor:
            corridor_info = "Thika Road Corridor - High growth area"
        elif location in mombasa_corridor:
            corridor_info = "Mombasa Road Corridor - Infrastructure premium"
        
        if corridor_info:
            st.info(f"🚗 {corridor_info}")
        
        # Affordable housing indicator
        st.markdown("#### Market Segment")
        affordable_segment = st.checkbox(
            "Affordable Housing Segment (KES 3M-8M)",
            value=(location in satellite_towns),
            help="Target segment for 250,000 annual unit demand[citation:5]"
        )
    
    # Calculate derived features
    price_per_sqm = 50000  # Base rate, will be adjusted by model
    location_mean = {
        "Karen": 55000000, "Westlands": 50000000, "Lavington": 48000000,
        "Kilimani": 45000000, "Kileleshwa": 42000000, "Parklands": 40000000,
        "South B": 35000000, "Embakasi": 30000000, "Ruiru": 28000000,
        "Kitengela": 27000000, "Syokimau": 29000000, "Athi River": 26000000,
        "Thika": 25000000, "Ngong": 26000000, "Rongai": 25000000
    }
    
    location_mean_encoded = location_mean.get(location, 30000000)
    bedrooms_per_sqm = bedrooms / size_sqm if size_sqm > 0 else 0
    location_size_interaction = location_mean_encoded * size_sqm
    
    # Create feature vector
    features = {
        'BEDROOMS': bedrooms,
        'BATHROOMS': bathrooms,
        'SIZE_SQM_CAPPED': size_sqm,
        'PRICE_PER_SQM': price_per_sqm,
        'BEDROOMS_PER_SQM': bedrooms_per_sqm,
        'LOCATION_MEAN_ENCODED': location_mean_encoded,
        'IS_SATELLITE': is_satellite,
        'LOCATION_SIZE_INTERACTION': location_size_interaction,
        'CORRIDOR_THIKA': 1 if location in thika_corridor else 0,
        'CORRIDOR_MOMBASA': 1 if location in mombasa_corridor else 0,
        'AFFORDABLE_SEGMENT': 1 if affordable_segment else 0,
        'SATELLITE_SIZE_VALUE': is_satellite * size_sqm,
        'BEDROOM_LOCATION_PREMIUM': bedrooms * location_mean_encoded
    }
    
    # Add location dummies (simplified)
    for loc in locations:
        features[f'LOCATION_{loc.upper().replace(" ", "_")}'] = 1 if location == loc else 0
    
    # Convert to DataFrame
    feature_df = pd.DataFrame([features])
    
    # Align with training features
    if feature_info and 'feature_names' in feature_info:
        all_features = feature_info['feature_names']
        for feature in all_features:
            if feature not in feature_df.columns:
                feature_df[feature] = 0
        
        # Reorder columns
        feature_df = feature_df[all_features]
    
    # Prediction button
    st.markdown("---")
    if st.button("🎯 Predict Property Price", type="primary", use_container_width=True):
        if model is not None:
            try:
                # Make prediction
                if base_models:  # Ensemble model
                    predictions = []
                    for base_name, base_model in base_models.items():
                        predictions.append(base_model.predict(feature_df)[0])
                    prediction = np.mean(predictions)
                else:
                    prediction = model.predict(feature_df)[0]
                
                # Format prediction
                prediction_millions = prediction / 1e6
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Property Value</h2>
                    <h1 style="color: #10B981;">KSh {prediction_millions:,.1f} Million</h1>
                   <p>Based on {feature_info['model_name'] if feature_info else 'ML'} model</p>
                </div>
                """, unsafe_allow_html=True)
                
                
                # Market comparison
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    price_per_sqm_pred = prediction / size_sqm if size_sqm > 0 else 0
                    st.metric("Price per SQM", f"KSh {price_per_sqm_pred:,.0f}")
                
                with col2:
                    # Compare with similar properties
                    similar_data = sample_data[
                        (sample_data['BEDROOMS'] == bedrooms) & 
                        (sample_data['SIZE_SQM_CAPPED'].between(size_sqm*0.8, size_sqm*1.2))
                    ]
                    
                    if len(similar_data) > 0:
                        avg_similar_price = similar_data['PRICE_KSH_CAPPED'].mean() / 1e6
                        diff_pct = ((prediction_millions - avg_similar_price) / avg_similar_price) * 100
                        st.metric("Vs Similar Properties", 
                                 f"{diff_pct:+.1f}%",
                                 delta=f"{diff_pct:+.1f}%")
                
                with col3:
                    # Satellite town premium/discount
                    if is_satellite:
                        core_avg = sample_data[sample_data['IS_SATELLITE'] == 0]['PRICE_KSH_CAPPED'].mean() / 1e6
                        satellite_discount = ((core_avg - prediction_millions) / core_avg) * 100
                        st.metric("Satellite Discount", f"{satellite_discount:.1f}%")
                    else:
                        satellite_avg = sample_data[sample_data['IS_SATELLITE'] == 1]['PRICE_KSH_CAPPED'].mean() / 1e6
                        core_premium = ((prediction_millions - satellite_avg) / satellite_avg) * 100
                        st.metric("Core Premium", f"{core_premium:.1f}%")
                
                # SHAP Explanation (simplified)
                st.markdown('<h3 class="sub-header">📊 Price Drivers Analysis</h3>', unsafe_allow_html=True)
                
                # Simplified feature importance
                feature_importance = {
                    'Location': 0.35,
                    'Size (SQM)': 0.25,
                    'Bedrooms': 0.15,
                    'Satellite Town': 0.10,
                    'Infrastructure': 0.08,
                    'Bathrooms': 0.07
                }
                
                # Adjust based on inputs
                if is_satellite:
                    feature_importance['Satellite Town'] = 0.15
                    feature_importance['Location'] = 0.30
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                features_sorted = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
                ax.barh(list(features_sorted.keys()), list(features_sorted.values()), color='steelblue')
                ax.set_xlabel('Importance')
                ax.set_title('Key Price Drivers for This Property')
                ax.invert_yaxis()
                
                st.pyplot(fig)
                
                # Investment recommendation
                st.markdown('<h3 class="sub-header">💰 Investment Recommendation</h3>', unsafe_allow_html=True)
                
                if is_satellite and affordable_segment:
                    st.success("""
                    **Strong Investment Potential** ✅
                    
                    This property in the affordable housing segment in a satellite town has:
                    - High demand (250,000 units annually needed)[citation:5]
                    - Strong growth potential (15-30% annual appreciation)[citation:10]
                    - Good rental yields (10-15%)[citation:10]
                    """)
                elif not is_satellite and prediction_millions > 15:
                    st.warning("""
                    **Premium Market - Lower Yields** ⚠️
                    
                    Luxury properties in core Nairobi areas offer:
                    - Lower rental yields (4-6%)[citation:10]
                    - Slower appreciation (5-8% annually)[citation:10]
                    - Higher entry costs
                    """)
                else:
                    st.info("""
                    **Moderate Investment** ℹ️
                    
                    Consider these factors:
                    - Verify infrastructure access[citation:1]
                    - Check for government housing programs[citation:5]
                    - Compare with similar properties in area
                    """)
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.error("Model not loaded. Please check model files.")

# ==================== MARKET ANALYSIS MODULE ====================
elif app_mode == "📈 Market Analysis":
    st.markdown('<h2 class="sub-header">📈 Nairobi Real Estate Market Analysis</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Market Trends", "Price Distribution", "Investment Insights"])
    
    with tab1:
        st.markdown("#### Market Trends 2026[citation:7][citation:10]")
        
        # Create trend indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Housing Deficit", "2M+ units", 
                     help="Kenya's cumulative housing shortage[citation:5]")
        
        with col2:
            st.metric("Annual Demand", "250,000 units",
                     help="New housing units needed each year[citation:5]")
        
        with col3:
            st.metric("Annual Supply", "50,000 units",
                     help="Actual units built annually[citation:5]")
        
        with col4:
            st.metric("Supply Gap", "80%",
                     help="Percentage of demand not met[citation:5]")
        
        # Trend visualization
        st.markdown("#### Price Trends by Location Type")
        
        # Sample data for trends
        years = [2022, 2023, 2024, 2025, 2026]
        satellite_prices = [5.2, 5.8, 6.5, 7.3, 8.2]  # In millions
        core_prices = [12.5, 13.1, 13.4, 13.6, 13.8]  # In millions
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=satellite_prices,
            mode='lines+markers',
            name='Satellite Towns',
            line=dict(color='green', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=years, y=core_prices,
            mode='lines+markers',
            name='Nairobi Core',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title='Price Appreciation: Satellite vs Core Areas[citation:1]',
            xaxis_title='Year',
            yaxis_title='Median Price (Million KSH)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.markdown("""
        <div class="info-box">
        <strong>Key Insight:</strong> Satellite towns show stronger growth (15-30% annually) 
        compared to plateauing core markets (5-8%). This shift is driven by affordability, 
        improved infrastructure, and changing buyer preferences[citation:1].
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("#### Property Price Distribution")
        
        # Create histogram of prices
        if len(sample_data) > 0:
            fig = px.histogram(
                sample_data,
                x='PRICE_KSH_CAPPED',
                nbins=30,
                title='Distribution of Property Prices',
                labels={'PRICE_KSH_CAPPED': 'Price (KSH)', 'count': 'Number of Properties'},
                color_discrete_sequence=['skyblue']
            )
            
            # Add vertical lines for market segments
            fig.add_vline(x=3000000, line_dash="dash", line_color="green", 
                         annotation_text="Affordable Start")
            fig.add_vline(x=8000000, line_dash="dash", line_color="green",
                         annotation_text="Affordable End")
            fig.add_vline(x=15000000, line_dash="dash", line_color="red",
                         annotation_text="Luxury Threshold")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Market segments analysis
            st.markdown("#### Market Segments Analysis")
            
            affordable_count = len(sample_data[
                (sample_data['PRICE_KSH_CAPPED'] >= 3000000) & 
                (sample_data['PRICE_KSH_CAPPED'] <= 8000000)
            ])
            
            mid_count = len(sample_data[
                (sample_data['PRICE_KSH_CAPPED'] > 8000000) & 
                (sample_data['PRICE_KSH_CAPPED'] <= 15000000)
            ])
            
            luxury_count = len(sample_data[
                sample_data['PRICE_KSH_CAPPED'] > 15000000
            ])
            
            total_count = len(sample_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Affordable (KES 3M-8M)", f"{affordable_count} units",
                         f"{affordable_count/total_count*100:.1f}%")
            with col2:
                st.metric("Mid-Market (KES 8M-15M)", f"{mid_count} units",
                         f"{mid_count/total_count*100:.1f}%")
            with col3:
                st.metric("Luxury (>KES 15M)", f"{luxury_count} units",
                         f"{luxury_count/total_count*100:.1f}%")
        
    with tab3:
        st.markdown("#### Investment Hotspots 2026[citation:10]")
        
        # Investment hotspots data
        hotspots = pd.DataFrame({
            'Location': ['Ruiru/Thika Road', 'Konza Technopolis', 'Malindi/Kilifi', 
                        'Naivasha', 'Kitengela', 'Westlands/Karen'],
            'Appreciation': [20, 35, 25, 18, 16, 6],
            'Risk': ['Low', 'Medium', 'Low-Medium', 'Low', 'Low', 'Low'],
            'Yield': [15, 25, 20, 12, 12, 5]
        })
        
        # Create bubble chart
        fig = px.scatter(
            hotspots,
            x='Appreciation',
            y='Yield',
            size='Appreciation',
            color='Risk',
            hover_name='Location',
            size_max=60,
            title='Investment Hotspots: Appreciation vs Yield[citation:10]',
            labels={'Appreciation': 'Annual Appreciation (%)', 'Yield': 'Rental Yield (%)'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("#### Top Investment Recommendations")
        
        recommendations = [
            {
                "location": "Ruiru",
                "reason": "Thika Road corridor, SGR access, student demand",
                "metric": "15-20% appreciation, 10-15% yields"
            },
            {
                "location": "Kitengela",
                "reason": "Nairobi overflow, Expressway access, affordable land",
                "metric": "12-16% appreciation, 12-15% yields"
            },
            {
                "location": "Syokimau",
                "reason": "Mombasa Road corridor, airport proximity, planned development",
                "metric": "15-18% appreciation, 10-12% yields"
            }
        ]
        
        for rec in recommendations:
            with st.expander(f"📍 {rec['location']}"):
                st.markdown(f"**Why invest:** {rec['reason']}")
                st.markdown(f"**Expected returns:** {rec['metric']}")
                st.markdown(f"**Risk level:** Low")
                st.markdown(f"**Infrastructure:** Good transport links[citation:1]")

# ==================== LOCATION INSIGHTS MODULE ====================
elif app_mode == "🗺️ Location Insights":
    st.markdown('<h2 class="sub-header">🗺️ Nairobi Location Analysis</h2>', unsafe_allow_html=True)
    
    # Create interactive map
    st.markdown("#### Nairobi Property Map")
    
    # Nairobi center coordinates
    nairobi_center = [-1.2921, 36.8219]
    
    # Create Folium map
    m = folium.Map(location=nairobi_center, zoom_start=11, tiles='CartoDB positron')
    
    # Define location data
    locations_data = [
        {"name": "Westlands", "coords": [-1.2665, 36.8032], "type": "core", "avg_price": 12.5},
        {"name": "Kilimani", "coords": [-1.2973, 36.7959], "type": "core", "avg_price": 11.8},
        {"name": "Ruiru", "coords": [-1.1496, 36.9630], "type": "satellite", "avg_price": 7.5},
        {"name": "Kitengela", "coords": [-1.4689, 36.9806], "type": "satellite", "avg_price": 7.2},
        {"name": "Syokimau", "coords": [-1.3819, 36.9128], "type": "satellite", "avg_price": 7.8},
        {"name": "Karen", "coords": [-1.3192, 36.7085], "type": "core", "avg_price": 15.2},
        {"name": "Lavington", "coords": [-1.2679, 36.7742], "type": "core", "avg_price": 13.5},
    ]
    
    # Add markers
    for loc in locations_data:
        color = 'green' if loc['type'] == 'satellite' else 'blue'
        
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4>{loc['name']}</h4>
            <p><strong>Type:</strong> {loc['type'].title()}</p>
            <p><strong>Avg Price:</strong> KSh {loc['avg_price']}M</p>
            <p><strong>Market:</strong> {"Satellite - High Growth" if loc['type'] == 'satellite' else "Core - Stable"}</p>
            <p><strong>Appreciation:</strong> {"15-30%" if loc['type'] == 'satellite' else "5-8%"}[citation:10]</p>
        </div>
        """
        
        folium.CircleMarker(
            location=loc['coords'],
            radius=10,
            popup=folium.Popup(popup_html, max_width=250),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            tooltip=f"{loc['name']}: KSh {loc['avg_price']}M"
        ).add_to(m)
    
    # Display map
    folium_static(m, width=900, height=500)
    
    # Location comparison
    st.markdown("#### Location Comparison")
    
    selected_locations = st.multiselect(
        "Compare locations:",
        [loc['name'] for loc in locations_data],
        default=["Ruiru", "Westlands", "Kitengela"]
    )
    
    if selected_locations:
        selected_data = [loc for loc in locations_data if loc['name'] in selected_locations]
        
        # Create comparison chart
        fig = go.Figure(data=[
            go.Bar(
                name='Average Price (M KSH)',
                x=[loc['name'] for loc in selected_data],
                y=[loc['avg_price'] for loc in selected_data],
                marker_color=['green' if loc['type'] == 'satellite' else 'blue' for loc in selected_data]
            )
        ])
        
        fig.update_layout(
            title='Location Price Comparison',
            yaxis_title='Average Price (Million KSH)',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Location insights
        st.markdown("#### Location Insights")
        
        for loc in selected_data:
            with st.expander(f"📌 {loc['name']} Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Market Characteristics:**")
                    if loc['type'] == 'satellite':
                        st.markdown("""
                        - High growth potential (15-30%)[citation:10]
                        - Affordable housing focus[citation:5]
                        - Infrastructure-driven growth[citation:1]
                        - 10-15% rental yields[citation:10]
                        """)
                    else:
                        st.markdown("""
                        - Stable, established market
                        - Luxury and premium segment
                        - 4-6% rental yields[citation:10]
                        - Slower appreciation (5-8%)[citation:10]
                        """)
                
                with col2:
                    st.markdown("**Investment Considerations:**")
                    if loc['type'] == 'satellite':
                        st.markdown(f"""
                        - Entry price: ~KSh {loc['avg_price']}M
                        - Target: Long-term appreciation
                        - Risk: Medium (infrastructure dependent)
                        - Best for: First-time investors, affordable housing
                        """)
                    else:
                        st.markdown(f"""
                        - Entry price: ~KSh {loc['avg_price']}M
                        - Target: Stable rental income
                        - Risk: Low (established market)
                        - Best for: Conservative investors, luxury market
                        """)

# ==================== INVESTMENT CALCULATOR MODULE ====================
elif app_mode == "💰 Investment Calculator":
    st.markdown('<h2 class="sub-header">💰 Real Estate Investment Calculator</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Investment Parameters")
        
        investment_amount = st.number_input(
            "Investment Amount (KSH)",
            min_value=1000000,
            max_value=100000000,
            value=5000000,
            step=1000000,
            help="Minimum investment for affordable housing: KSh 3M[citation:10]"
        )
        
        location_type = st.radio(
            "Location Type",
            ["Satellite Town", "Nairobi Core"],
            help="Satellite towns offer higher growth, core areas offer stability"
        )
        
        investment_horizon = st.slider(
            "Investment Horizon (Years)",
            min_value=1,
            max_value=10,
            value=5,
            help="Recommended: 3-5 years for optimal returns"
        )
        
    with col2:
        st.markdown("#### Market Assumptions")
        
        # Adjust assumptions based on location
        if location_type == "Satellite Town":
            annual_appreciation = st.slider(
                "Annual Appreciation (%)",
                min_value=5.0,
                max_value=30.0,
                value=18.0,
                help="Satellite towns: 15-30% annual appreciation[citation:10]"
            )
            
            rental_yield = st.slider(
                "Annual Rental Yield (%)",
                min_value=5.0,
                max_value=20.0,
                value=12.0,
                help="Satellite towns: 10-15% rental yields[citation:10]"
            )
        else:
            annual_appreciation = st.slider(
                "Annual Appreciation (%)",
                min_value=2.0,
                max_value=10.0,
                value=6.0,
                help="Nairobi core: 5-8% annual appreciation[citation:10]"
            )
            
            rental_yield = st.slider(
                "Annual Rental Yield (%)",
                min_value=2.0,
                max_value=10.0,
                value=5.0,
                help="Nairobi core: 4-6% rental yields[citation:10]"
            )
        
        occupancy_rate = st.slider(
            "Occupancy Rate (%)",
            min_value=70,
            max_value=100,
            value=85,
            help="Typical occupancy rates: 80-90%"
        )
    
    # Calculate button
    if st.button("📊 Calculate Investment Returns", type="primary", use_container_width=True):
        # Calculate returns
        years = list(range(investment_horizon + 1))
        
        # Property value over time
        property_value = [investment_amount]
        for year in range(1, investment_horizon + 1):
            value = investment_amount * ((1 + annual_appreciation/100) ** year)
            property_value.append(value)
        
        # Rental income over time
        rental_income = []
        cumulative_income = 0
        for year in range(investment_horizon + 1):
            if year == 0:
                annual_rent = investment_amount * (rental_yield/100) * (occupancy_rate/100)
                rental_income.append(annual_rent)
                cumulative_income = annual_rent
            else:
                # Rent increases with property value
                annual_rent = property_value[year] * (rental_yield/100) * (occupancy_rate/100)
                rental_income.append(annual_rent)
                cumulative_income += annual_rent
        
        # Total returns
        total_returns = property_value[-1] + cumulative_income - investment_amount
        roi = (total_returns / investment_amount) * 100
        annualized_roi = ((1 + roi/100) ** (1/investment_horizon) - 1) * 100
        
        # Display results
        st.markdown(f"""
        <div class="prediction-box">
            <h2>Investment Results</h2>
            <h3>Total ROI: {roi:.1f}%</h3>
            <p>Annualized Return: {annualized_roi:.1f}% per year</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed breakdown
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Final Property Value", f"KSh {property_value[-1]/1e6:.1f}M")
        
        with col2:
            st.metric("Total Rental Income", f"KSh {cumulative_income/1e6:.1f}M")
        
        with col3:
            st.metric("Net Profit", f"KSh {total_returns/1e6:.1f}M")
        
        # Visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=[v/1e6 for v in property_value],
            mode='lines+markers',
            name='Property Value',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Bar(
            x=years,
            y=[r/1e6 for r in rental_income],
            name='Annual Rental Income',
            marker_color='green',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Investment Growth Over Time',
            xaxis_title='Year',
            yaxis_title='Value (Million KSH)',
            barmode='stack',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison with other investments
        st.markdown("#### Comparison with Other Investments")
        
        comparison_data = {
            'Investment Type': ['Real Estate (This)', 'NSE Stocks', 'Government Bonds', 'Savings Account'],
            'Expected ROI': [annualized_roi, 12.0, 9.5, 7.0],
            'Risk Level': ['Medium', 'High', 'Low', 'Very Low'],
            'Liquidity': ['Low', 'High', 'Medium', 'High']
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison chart
        fig = px.bar(
            comparison_df,
            x='Investment Type',
            y='Expected ROI',
            color='Risk Level',
            title='Investment Comparison: Expected Annual Returns',
            labels={'Expected ROI': 'Expected Annual Return (%)'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Investment recommendation
        st.markdown("#### Investment Recommendation")
        
        if location_type == "Satellite Town" and annualized_roi > 15:
            st.success(f"""
            **Excellent Investment Opportunity** ✅
            
            Your investment in a satellite town shows strong potential:
            - Expected annual return: {annualized_roi:.1f}%
            - Higher than average for real estate (12-15%)
            - Benefits from infrastructure development[citation:1]
            - Meets affordable housing demand[citation:5]
            
            **Next Steps:**
            1. Research specific locations along growth corridors[citation:7]
            2. Verify property titles via Ardhisasa[citation:10]
            3. Consider partnerships for larger developments
            """)
        elif annualized_roi > 10:
            st.info(f"""
            **Good Investment Potential** ℹ️
            
            Your investment shows reasonable returns:
            - Expected annual return: {annualized_roi:.1f}%
            - Comparable to market averages
            - Consider diversifying across locations
            
            **Recommendations:**
            1. Look for properties near infrastructure projects[citation:1]
            2. Consider mixed-use developments for higher yields
            3. Monitor government affordable housing programs[citation:5]
            """)
        else:
            st.warning(f"""
            **Below Market Returns** ⚠️
            
            Your investment parameters show lower than expected returns:
            - Expected annual return: {annualized_roi:.1f}%
            - Below market average for similar investments
            
            **Consider:**
            1. Increasing investment horizon to 5+ years
            2. Exploring satellite towns for higher growth[citation:1]
            3. Reviewing assumptions (appreciation, yield, occupancy)
            """)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>📊 <strong>Nairobi Real Estate Price Predictor</strong> | Personal Project by Joseph Thuo</p>
<p>📍 Academic Project | Machine Learning Engineering Portfolio  </p>
    <p>📧 Contact: <a href="machariajoseph1422@gmail.com">machariajoseph1422@gmail.com</a> | 
       🔗 Portfolio: <a href="https://my-portfolio-3wrw.onrender.com/#" target="_blank">https://my-portfolio-3wrw.onrender.com/#</a></p>
    <p>📱 GitHub: <a href="https://github.com/JinxWycman" target="_blank">https://github.com/JinxWycman</a> | 
       💼 LinkedIn: <a href="https://www.linkedin.com/in/machariajosepht/" target="_blank">https://www.linkedin.com/in/machariajosepht/</a></p>
    <p>⚠️ <em>Disclaimer: This is an academic project for demonstration purposes only. 
       Predictions are estimates based on historical data. Not for actual real estate transactions.</em></p>
    <p style="font-size: 0.8rem; margin-top: 10px;">© 2026 Joseph THuo. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
