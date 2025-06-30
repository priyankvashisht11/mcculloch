"""
Streamlit Dashboard for LCF Group Funding Recommendation System

Provides a user-friendly interface for:
- Business profile submission
- Viewing predictions and risk scores
- Searching similar businesses
- System monitoring
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="LCF Funding Recommendation System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high { color: #d62728; }
    .risk-medium { color: #ff7f0e; }
    .risk-low { color: #2ca02c; }
    .funding-yes { color: #2ca02c; }
    .funding-no { color: #d62728; }
    .funding-maybe { color: #ff7f0e; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/healthcheck", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None

def submit_business_profile(business_data):
    """Submit business profile to API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/submit_business_profile",
            json=business_data,
            timeout=180
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

def search_similar_businesses(query, limit=5):
    """Search for similar businesses"""
    try:
        params = {
            "query": query,
            "limit": limit,
            "search_type": "hybrid"
        }
        response = requests.get(f"{API_BASE_URL}/search_similar", params=params, timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

def get_model_info():
    """Get model information"""
    try:
        response = requests.get(f"{API_BASE_URL}/get_model_info", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

# Main dashboard
def main():
    # Header
    st.markdown('<h1 class="main-header">💰 LCF Funding Recommendation System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Dashboard", "📊 Business Analysis", "🔍 Search Similar", "⚙️ System Info", "📈 Monitoring"]
    )
    
    # Check API health
    api_healthy, health_data = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ API is not accessible. Please ensure the backend services are running.")
        st.info("To start the services, run: `docker-compose up -d`")
        return
    
    # Display health status
    if health_data:
        status_color = "🟢" if health_data.get("status") == "healthy" else "🟡"
        st.sidebar.markdown(f"{status_color} **System Status:** {health_data.get('status', 'unknown')}")
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📊 Business Analysis":
        show_business_analysis()
    elif page == "🔍 Search Similar":
        show_search_similar()
    elif page == "⚙️ System Info":
        show_system_info()
    elif page == "📈 Monitoring":
        show_monitoring()

def show_dashboard():
    """Show main dashboard"""
    st.header("📊 System Overview")
    
    # Get model info
    model_success, model_info = get_model_info()
    
    if model_success:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Version", f"v{model_info.get('version', {}).get('major', '1')}.{model_info.get('version', {}).get('minor', '0')}")
        
        with col2:
            st.metric("Base Model", model_info.get('base_model', 'Unknown'))
        
        with col3:
            st.metric("Device", model_info.get('device', 'Unknown'))
        
        with col4:
            total_params = model_info.get('total_parameters', 0)
            # Handle different data types and None values
            if total_params is None:
                total_params = 0
            try:
                total_params = int(float(total_params))
                st.metric("Parameters", f"{total_params:,}")
            except (ValueError, TypeError):
                st.metric("Parameters", "N/A")
    
    # Quick analysis section
    st.subheader("🚀 Quick Business Analysis")
    
    with st.form("quick_analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            business_name = st.text_input("Business Name", placeholder="Enter business name")
            domain = st.selectbox("Domain", ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing", "Other"])
            location = st.text_input("Location", placeholder="e.g., California")
        
        with col2:
            revenue = st.number_input("Annual Revenue ($)", min_value=0, value=500000, step=10000)
            employee_count = st.number_input("Employee Count", min_value=1, value=15, step=1)
            years_active = st.number_input("Years Active", min_value=0, value=3, step=1)
        
        description = st.text_area("Business Description", 
                                 placeholder="Describe your business, products/services, market position, etc.",
                                 height=100)
        
        submitted = st.form_submit_button("Analyze Business")
        
        if submitted and business_name:
            with st.spinner("Analyzing business profile..."):
                business_data = {
                    "business_name": business_name,
                    "domain": domain,
                    "location": location,
                    "revenue": revenue,
                    "employee_count": employee_count,
                    "years_active": years_active,
                    "description": description
                }
                
                success, result = submit_business_profile(business_data)
                
                if success:
                    st.session_state.analysis_results = result
                    st.success("Analysis completed successfully!")
                    st.rerun()
                else:
                    st.error(f"Analysis failed: {result}")

def show_business_analysis():
    """Show business analysis results"""
    st.header("📊 Business Analysis Results")
    
    if st.session_state.analysis_results is None:
        st.info("No analysis results available. Please submit a business profile first.")
        return
    
    results = st.session_state.analysis_results
    
    # Business information
    st.subheader("🏢 Business Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Business Name", results.get("business_name", "N/A"))
    
    with col2:
        risk_level = results.get("risk_assessment", {}).get("level", "unknown")
        risk_confidence = results.get("risk_assessment", {}).get("confidence", 0)
        risk_class = f"risk-{risk_level}"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Risk Level</h4>
            <p class="{risk_class}"><strong>{risk_level.upper()}</strong></p>
            <p>Confidence: {risk_confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        funding_decision = results.get("funding_recommendation", {}).get("decision", "unknown")
        funding_confidence = results.get("funding_recommendation", {}).get("confidence", 0)
        funding_class = f"funding-{funding_decision}"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Funding Decision</h4>
            <p class="{funding_class}"><strong>{funding_decision.upper()}</strong></p>
            <p>Confidence: {funding_confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk assessment details
    st.subheader("⚠️ Risk Assessment Details")
    risk_probs = results.get("risk_assessment", {}).get("probabilities", {})
    
    if risk_probs:
        risk_df = pd.DataFrame([
            {"Risk Level": k.title(), "Probability": v}
            for k, v in risk_probs.items()
        ])
        
        fig = px.bar(risk_df, x="Risk Level", y="Probability", 
                    title="Risk Level Probabilities",
                    color="Risk Level",
                    color_discrete_map={
                        "Low": "#2ca02c",
                        "Medium": "#ff7f0e", 
                        "High": "#d62728"
                    })
        st.plotly_chart(fig, use_container_width=True)
    
    # Funding recommendation details
    st.subheader("💰 Funding Recommendation Details")
    funding_probs = results.get("funding_recommendation", {}).get("probabilities", {})
    
    if funding_probs:
        funding_df = pd.DataFrame([
            {"Decision": k.title(), "Probability": v}
            for k, v in funding_probs.items()
        ])
        
        fig = px.bar(funding_df, x="Decision", y="Probability",
                    title="Funding Decision Probabilities",
                    color="Decision",
                    color_discrete_map={
                        "Yes": "#2ca02c",
                        "Maybe": "#ff7f0e",
                        "No": "#d62728"
                    })
        st.plotly_chart(fig, use_container_width=True)
    
    # Analysis insights
    st.subheader("🔍 Analysis Insights")
    analysis = results.get("analysis", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        if analysis.get("strengths"):
            st.markdown("**Strengths:**")
            for strength in analysis["strengths"]:
                st.markdown(f"✅ {strength}")
        
        if analysis.get("key_factors"):
            st.markdown("**Key Factors:**")
            for factor in analysis["key_factors"]:
                st.markdown(f"🔑 {factor}")
    
    with col2:
        if analysis.get("concerns"):
            st.markdown("**Concerns:**")
            for concern in analysis["concerns"]:
                st.markdown(f"⚠️ {concern}")
        
        market_position = analysis.get("market_position", "unknown")
        st.markdown(f"**Market Position:** {market_position.title()}")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    recommendations = results.get("recommendations", [])
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    else:
        st.info("No specific recommendations available.")
    
    # Similar businesses
    st.subheader("🔍 Similar Businesses")
    similar_businesses = results.get("similar_businesses", [])
    
    if similar_businesses:
        similar_df = pd.DataFrame(similar_businesses)
        if not similar_df.empty:
            # Select relevant columns
            display_cols = ["business_name", "domain", "location", "similarity_score"]
            available_cols = [col for col in display_cols if col in similar_df.columns]
            
            if available_cols:
                st.dataframe(similar_df[available_cols], use_container_width=True)
    else:
        st.info("No similar businesses found.")

def show_search_similar():
    """Show search similar businesses page"""
    st.header("🔍 Search Similar Businesses")
    
    # Search form
    with st.form("search_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            query = st.text_input("Search Query", placeholder="e.g., AI technology startup")
            search_type = st.selectbox("Search Type", ["hybrid", "semantic", "advanced"])
        
        with col2:
            limit = st.number_input("Number of Results", min_value=1, max_value=50, value=10)
            domain_filter = st.selectbox("Filter by Domain", ["All", "Technology", "Healthcare", "Finance", "Retail", "Manufacturing"])
        
        submitted = st.form_submit_button("Search")
        
        if submitted and query:
            with st.spinner("Searching for similar businesses..."):
                success, result = search_similar_businesses(query, limit)
                
                if success:
                    st.success(f"Found {result.get('total', 0)} similar businesses")
                    
                    # Display results
                    results = result.get("results", [])
                    if results:
                        # Convert to DataFrame for better display
                        df = pd.DataFrame(results)
                        
                        # Select relevant columns
                        display_cols = ["business_name", "domain", "location", "score"]
                        available_cols = [col for col in display_cols if col in df.columns]
                        
                        if available_cols:
                            st.dataframe(df[available_cols], use_container_width=True)
                            
                            # Show detailed view for selected business
                            if st.checkbox("Show detailed view"):
                                selected_idx = st.selectbox("Select business for details:", range(len(results)))
                                selected_business = results[selected_idx]
                                
                                st.subheader(f"Details: {selected_business.get('business_name', 'Unknown')}")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.json(selected_business)
                                with col2:
                                    if "features" in selected_business:
                                        features = selected_business["features"]
                                        if features:
                                            st.markdown("**Key Features:**")
                                            for key, value in features.items():
                                                if isinstance(value, (int, float)):
                                                    try:
                                                        formatted_value = f"{value:,}"
                                                        st.markdown(f"- {key}: {formatted_value}")
                                                    except (ValueError, TypeError):
                                                        st.markdown(f"- {key}: {value}")
                                                else:
                                                    st.markdown(f"- {key}: {value}")
                    else:
                        st.info("No similar businesses found.")
                else:
                    st.error(f"Search failed: {result}")

def show_system_info():
    """Show system information"""
    st.header("⚙️ System Information")
    
    # Get model info
    success, model_info = get_model_info()
    
    if success:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Configuration")
            st.json({
                "Base Model": model_info.get("base_model"),
                "Device": model_info.get("device"),
                "Max Length": model_info.get("max_length"),
                "Total Parameters": str(model_info.get('total_parameters', 0) or 0),
                "Trainable Parameters": str(model_info.get('trainable_parameters', 0) or 0)
            })
        
        with col2:
            st.subheader("Model Version")
            version = model_info.get("version", {})
            st.json({
                "Major": version.get("major"),
                "Minor": version.get("minor"),
                "Patch": version.get("patch"),
                "Description": version.get("description")
            })
        
        st.subheader("Classification Labels")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Levels:**")
            for label in model_info.get("risk_labels", []):
                st.markdown(f"- {label.title()}")
        
        with col2:
            st.markdown("**Funding Decisions:**")
            for label in model_info.get("funding_labels", []):
                st.markdown(f"- {label.title()}")
        
        st.subheader("LoRA Configuration")
        lora_config = model_info.get("lora_config", {})
        st.json(lora_config)
    else:
        st.error(f"Failed to get model info: {model_info}")

def show_monitoring():
    """Show system monitoring"""
    st.header("📈 System Monitoring")
    
    # Get health check data
    api_healthy, health_data = check_api_health()
    
    if health_data:
        st.subheader("Service Health")
        
        services = health_data.get("services", {})
        for service, status in services.items():
            status_icon = "🟢" if "healthy" in status else "🔴"
            st.markdown(f"{status_icon} **{service}**: {status}")
        
        # System metrics
        st.subheader("System Metrics")
        
        # Simulate some metrics (in production, these would come from CloudWatch)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("API Response Time", "120ms", "5ms")
        
        with col2:
            st.metric("Requests/min", "45", "2")
        
        with col3:
            st.metric("Error Rate", "0.2%", "-0.1%")
        
        with col4:
            st.metric("Active Connections", "12", "1")
        
        # System uptime
        st.subheader("System Uptime")
        uptime = time.time() - 1640995200  # Simulated uptime
        days = int(uptime // 86400)
        hours = int((uptime % 86400) // 3600)
        minutes = int((uptime % 3600) // 60)
        
        st.metric("Uptime", f"{days}d {hours}h {minutes}m")
        
        # Recent activity
        st.subheader("Recent Activity")
        st.info("Monitoring dashboard would show recent API calls, errors, and system events.")
    else:
        st.error("Unable to retrieve system health information.")

if __name__ == "__main__":
    main() 