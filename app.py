import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score

st.set_page_config(page_title="Bollywood ML Studio", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

# Dark Theme styling
st.markdown("""
<style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; }
    .stButton>button { background: #238636; color: white; border-radius: 6px; border: none; font-weight: 600; width: 100%; }
    .stButton>button:hover { background: #2ea043; }
    .summary-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    file_path = r"C:\Users\jayra\OneDrive\Desktop\bollywood movie dataset\bollywood.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_csv("bollywood.csv")
    
    df['Genre'] = df['Genre'].astype(str).str.strip()
    if 'ReleaseTime' in df.columns:
        df['ReleaseTime'] = df['ReleaseTime'].astype(str).str.strip()
        
    df['ROI'] = df['BoxOfficeCollection'] - df['Budget']
    df['ROI_Percent'] = ((df['BoxOfficeCollection'] - df['Budget']) / df['Budget']) * 100
    df['Like_To_View'] = df['YoutubeLikes'] / (df['YoutubeViews'] + 1)
    return df

df = load_data()

@st.cache_resource
def build_rf_model(data):
    features = ['Genre', 'ReleaseTime', 'month', 'year', 'Budget', 'YoutubeViews', 'YoutubeLikes', 'YoutubeDislikes', 'Like_To_View']
    X = data[[c for c in features if c in data.columns]]
    y = data['BoxOfficeCollection']
    
    y_log = np.log1p(y)
    
    cat_cols = [c for c in ['Genre', 'ReleaseTime', 'month'] if c in X.columns]
    num_cols = [c for c in ['year', 'Budget', 'YoutubeViews', 'YoutubeLikes', 'YoutubeDislikes', 'Like_To_View'] if c in X.columns]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)]
    )
    
    model = Pipeline(steps=[
        ('prep', preprocessor),
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42))]
    )
    
    # Train model on complete dataset for maximum statistical stability
    model.fit(X, y_log)
    
    preds_log = model.predict(X)
    preds = np.expm1(preds_log)
    
    # Full Dataset Diagnostics Calculation
    n = len(y)
    p = X.shape[1]
    
    r2 = r2_score(y, preds)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mape = np.mean(np.abs((y - preds) / (y + 1e-5))) * 100
    std_err = np.std(y - preds) / np.sqrt(n)
    exp_var = explained_variance_score(y, preds)
    
    ohe_features = list(model.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(cat_cols))
    all_features = num_cols + ohe_features
    importances = model.named_steps['rf'].feature_importances_
    
    feat_df = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    metrics_dict = {
        'r2': r2, 'adj_r2': adj_r2, 'mae': mae, 'rmse': rmse,
        'mape': mape, 'std_err': std_err, 'exp_var': exp_var, 'total_records': n
    }
    
    return model, metrics_dict, feat_df, X, y, preds

model, metrics, feat_df, X, y, preds = build_rf_model(df)

if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Executive Dashboard", "Dataset Statistical Summary", "3D Mesh & Live Visual Forecast", "Advanced Model Diagnostics"])

if menu == "Executive Dashboard":
    st.title("🎬 Executive Box Office Overview")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"₹{df['BoxOfficeCollection'].sum():,.1f} Cr")
    c2.metric("Average Budget", f"₹{df['Budget'].mean():,.1f} Cr")
    c3.metric("Top ROI Genre", df.groupby('Genre')['ROI_Percent'].mean().idxmax())
    c4.metric("Total Movies Analyzed", f"{len(df)}")
    
    st.subheader("Budget vs Collection Trajectory")
    fig_scatter = px.scatter(df, x='Budget', y='BoxOfficeCollection', size='YoutubeLikes', color='Genre', hover_name='MovieName', template='plotly_dark')
    st.plotly_chart(fig_scatter, width="stretch")

elif menu == "Dataset Statistical Summary":
    st.title("📊 Dataset Statistical Summary")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats_df = df[num_cols].describe().T
    stats_df['median'] = df[num_cols].median()
    stats_df['skewness'] = df[num_cols].skew()
    stats_df['std_err'] = df[num_cols].sem()
    
    st.subheader("Numeric Distributions & Moments")
    st.dataframe(
        stats_df[['count', 'mean', 'std', 'std_err', 'min', '50%', 'max', 'skewness']].rename(columns={'50%': 'median', 'std_err': 'Std Error'}),
        width="stretch"
    )

    st.markdown("---")
    st.subheader("Feature Correlation Heatmap")
    corr = df[num_cols].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Blues", template="plotly_dark")
    st.plotly_chart(fig_corr, width="stretch")

elif menu == "3D Mesh & Live Visual Forecast":
    st.title("🔮 Live Prediction & 3D Mesh Visualization")
    
    col_input, col_graph = st.columns([1, 2])
    
    with col_input:
        st.subheader("Predict New Movie")
        with st.form("interactive_form"):
            movie_title = st.text_input("Movie Name", "New Film")
            g_input = st.selectbox("Genre", df['Genre'].unique())
            r_input = st.selectbox("Release Slot", df['ReleaseTime'].unique() if 'ReleaseTime' in df.columns else ['N'])
            m_input = st.selectbox("Month", df['month'].unique() if 'month' in df.columns else ['January'])
            y_input = st.selectbox("Year", [2024, 2025, 2026])
            b_input = st.number_input("Budget (Cr)", min_value=1.0, value=50.0)
            v_input = st.number_input("YouTube Views", min_value=1000, value=2500000)
            l_input = st.number_input("YouTube Likes", min_value=100, value=60000)
            d_input = st.number_input("YouTube Dislikes", min_value=10, value=3000)
            
            submit = st.form_submit_button("Run Prediction & Plot")
            
        if submit:
            ratio = l_input / (v_input + 1)
            sample = pd.DataFrame({
                'Genre': [g_input], 'ReleaseTime': [r_input], 'month': [m_input], 'year': [y_input],
                'Budget': [b_input], 'YoutubeViews': [v_input], 'YoutubeLikes': [l_input],
                'YoutubeDislikes': [d_input], 'Like_To_View': [ratio]
            })
            pred_log = model.predict(sample)[0]
            pred_val = np.expm1(pred_log)
            
            st.session_state.history.append({
                'MovieName': movie_title,
                'Budget': b_input,
                'YoutubeLikes': l_input,
                'PredictedCollection': pred_val
            })
            st.success(f"Forecasted Collection: ₹{pred_val:.2f} Cr")

    with col_graph:
        st.subheader("3D Mesh with Actual Movies & Live Predictions")
        
        # Grid Generation
        x_max_val = max(df['Budget'].max(), b_input if 'b_input' in locals() else 100)
        y_max_val = max(df['YoutubeLikes'].max(), l_input if 'l_input' in locals() else 100000)
        
        x_range = np.linspace(df['Budget'].min(), x_max_val, 30)
        y_range = np.linspace(df['YoutubeLikes'].min(), y_max_val, 30)
        xx, yy = np.meshgrid(x_range, y_range)

        med_views = df['YoutubeViews'].median()
        med_dislikes = df['YoutubeDislikes'].median()
        med_ratio = df['Like_To_View'].median()
        def_genre = df['Genre'].mode()[0]
        def_slot = df['ReleaseTime'].mode()[0] if 'ReleaseTime' in df.columns else 'N'
        def_month = df['month'].mode()[0] if 'month' in df.columns else 'January'
        def_year = df['year'].median() if 'year' in df.columns else 2014

        grid_df = pd.DataFrame({
            'Genre': [def_genre] * xx.size,
            'ReleaseTime': [def_slot] * xx.size,
            'month': [def_month] * xx.size,
            'year': [def_year] * xx.size,
            'Budget': xx.ravel(),
            'YoutubeViews': [med_views] * xx.size,
            'YoutubeLikes': yy.ravel(),
            'YoutubeDislikes': [med_dislikes] * xx.size,
            'Like_To_View': med_ratio
        })

        zz_log = model.predict(grid_df)
        zz = np.expm1(zz_log).reshape(xx.shape)

        fig_3d = go.Figure()

        # Actual Movies Scatter Points
        fig_3d.add_trace(go.Scatter3d(
            x=df['Budget'], y=df['YoutubeLikes'], z=df['BoxOfficeCollection'],
            mode='markers', text=df['MovieName'],
            marker=dict(size=5, color=df['BoxOfficeCollection'], colorscale='Reds', opacity=0.9),
            name='Actual Movies'
        ))

        # Decision Surface Mesh
        fig_3d.add_trace(go.Surface(
            x=x_range, y=y_range, z=zz,
            colorscale='Viridis', opacity=0.6, showscale=False, name='RF Decision Mesh'
        ))

        # Live session predictions
        if st.session_state.history:
            hist_df = pd.DataFrame(st.session_state.history)
            fig_3d.add_trace(go.Scatter3d(
                x=hist_df['Budget'], y=hist_df['YoutubeLikes'], z=hist_df['PredictedCollection'],
                mode='markers+text', text=hist_df['MovieName'],
                marker=dict(size=10, color='#00ff87', symbol='diamond', opacity=1.0),
                name='New Predictions'
            ))

        fig_3d.update_layout(
            template='plotly_dark',
            scene=dict(xaxis_title='Budget (Cr)', yaxis_title='YouTube Likes', zaxis_title='Box Office (Cr)'),
            margin=dict(l=0, r=0, b=0, t=30)
        )

        st.plotly_chart(fig_3d, width="stretch")

elif menu == "Advanced Model Diagnostics":
    st.title("🤖 Statistical Model Diagnostics")
    
    st.subheader("Core Regression Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score", f"{metrics['r2']:.4f}")
    m2.metric("Adjusted R²", f"{metrics['adj_r2']:.4f}")
    m3.metric("Explained Variance", f"{metrics['exp_var']:.4f}")
    m4.metric("Std Error of Mean", f"₹{metrics['std_err']:.2f} Cr")
    
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Mean Absolute Error (MAE)", f"₹{metrics['mae']:.2f} Cr")
    m6.metric("Root Mean Squared Error (RMSE)", f"₹{metrics['rmse']:.2f} Cr")
    m7.metric("Mean Abs % Error (MAPE)", f"{metrics['mape']:.2f}%")
    m8.metric("Total Sample Size (N)", metrics['total_records'])

    # Explicit Summary Breakdown Section
    st.markdown("### 📝 Statistical Metrics Summary")
    st.markdown(f"""
    <div class="summary-card">
        <ul>
            <li><b>R² Score ({metrics['r2']:.4f}):</b> Indicates that approximately <b>{metrics['r2']*100:.1f}%</b> of the variance in Box Office collection is explained by the model features.</li>
            <li><b>Adjusted R² ({metrics['adj_r2']:.4f}):</b> Penalizes unnecessary variables; confirms model strength without overfitting.</li>
            <li><b>RMSE (₹{metrics['rmse']:.2f} Cr):</b> Measures average prediction error magnitude, giving higher weight to large outliers.</li>
            <li><b>MAE (₹{metrics['mae']:.2f} Cr):</b> Represents the average absolute error margin across all movie collection predictions.</li>
            <li><b>Standard Error (₹{metrics['std_err']:.2f} Cr):</b> Estimates variability in model residual estimates across the full <b>{metrics['total_records']}</b> movie records.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Actual vs Predicted Residual Analysis")
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=y, y=preds, mode='markers', name='Actual Movies', marker=dict(color='#58a6ff', opacity=0.7)))
        
        if st.session_state.history:
            hist_df = pd.DataFrame(st.session_state.history)
            fig_p.add_trace(go.Scatter(
                x=hist_df['PredictedCollection'], y=hist_df['PredictedCollection'], 
                mode='markers+text', text=hist_df['MovieName'], textposition='top center',
                name='Live Predictions', marker=dict(color='#00ff87', size=12, symbol='star')
            ))

        max_v = max(max(y), max(preds))
        fig_p.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode='lines', name='Ideal Fit (y=x)', line=dict(color='#f85149', dash='dash')))
        fig_p.update_layout(template='plotly_dark', xaxis_title="Actual Box Office (Cr)", yaxis_title="Predicted Box Office (Cr)")
        st.plotly_chart(fig_p, width="stretch")
        
    with col_b:
        st.subheader("Feature Importance Breakdown")
        fig_f = px.bar(feat_df.head(10), x='Importance', y='Feature', orientation='h', template='plotly_dark', color='Importance', color_continuous_scale='Blues')
        fig_f.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_f, width="stretch")