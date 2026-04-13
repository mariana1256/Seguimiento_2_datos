"""
Mental Health & Productivity 2026 - Streamlit Application
Author: Mariana Gutierrez Restrepo
Course: Ingeniería de Datos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Mental Health & Productivity 2026",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    local_file = "cleaned_dataset.csv"
    
    if os.path.exists(local_file):
        try:
            df = pd.read_csv(local_file)
            if "productivity_level" not in df.columns:
                df["productivity_level"] = pd.cut(
                    df["Productivity_Score"],
                    bins=[-1, 40, 70, 100],
                    labels=["Baja", "Media", "Alta"]
                )
            return df
        except Exception as e:
            st.warning(f"Error reading local file: {e}")
    
    try:
        import kagglehub
        with st.spinner('Descargando dataset de Kaggle...'):
            path = kagglehub.dataset_download("shadab80k/mental-health-productivity-2026")
            df = pd.read_csv(path + "/mental_health_productivity_2026.csv")
            
            df["productivity_level"] = pd.cut(
                df["Productivity_Score"],
                bins=[-1, 40, 70, 100],
                labels=["Baja", "Media", "Alta"]
            )
            
            df.to_csv(local_file, index=False)
            return df
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
    
    np.random.seed(42)
    n_samples = 1500
    
    countries = ["USA", "UK", "Germany", "Brazil", "Australia", "France", "Singapore", "India", "Japan", "Canada"]
    industries = ["Manufacturing", "Finance", "Tech", "Retail", "Education", "Healthcare"]
    work_modes = ["Remote", "On-site", "Hybrid"]
    genders = ["Male", "Female", "Non-binary"]
    burnout_risks = ["Low", "Medium", "High"]
    mental_health_support = ["Yes", "No"]
    
    data = {
        "Employee_ID": [f"EMP_{str(i).zfill(4)}" for i in range(1, n_samples + 1)],
        "Age": np.random.randint(22, 60, n_samples),
        "Gender": np.random.choice(genders, n_samples),
        "Country": np.random.choice(countries, n_samples),
        "Industry": np.random.choice(industries, n_samples),
        "Work_Mode": np.random.choice(work_modes, n_samples),
        "Work_Hours_Per_Week": np.random.randint(30, 65, n_samples),
        "Stress_Level": np.random.randint(1, 11, n_samples),
        "Sleep_Hours": np.random.randint(4, 11, n_samples),
        "Productivity_Score": np.random.randint(40, 101, n_samples),
        "Physical_Activity_Hours": np.random.uniform(0, 10, n_samples).round(1),
        "Mental_Health_Support_Access": np.random.choice(mental_health_support, n_samples),
        "Burnout_Risk": np.random.choice(burnout_risks, n_samples)
    }
    
    df = pd.DataFrame(data)
    df["productivity_level"] = pd.cut(
        df["Productivity_Score"],
        bins=[-1, 40, 70, 100],
        labels=["Baja", "Media", "Alta"]
    )
    
    st.info("ℹ️ Usando datos de ejemplo para demostración.")
    return df

def apply_filters(df):
    st.sidebar.header("🔍 Filtros")
    
    countries = ["Todos"] + sorted(df["Country"].unique().tolist())
    selected_country = st.sidebar.selectbox("País", countries)
    
    industries = ["Todos"] + sorted(df["Industry"].unique().tolist())
    selected_industry = st.sidebar.selectbox("Industria", industries)
    
    work_modes = ["Todos"] + sorted(df["Work_Mode"].unique().tolist())
    selected_work_mode = st.sidebar.selectbox("Modo de Trabajo", work_modes)
    
    genders = ["Todos"] + sorted(df["Gender"].unique().tolist())
    selected_gender = st.sidebar.selectbox("Género", genders)
    
    burnout_risks = ["Todos"] + sorted(df["Burnout_Risk"].unique().tolist())
    selected_burnout = st.sidebar.selectbox("Riesgo de Burnout", burnout_risks)
    
    min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.sidebar.slider("Rango de Edad", min_age, max_age, (min_age, max_age))
    
    min_stress, max_stress = int(df["Stress_Level"].min()), int(df["Stress_Level"].max())
    stress_range = st.sidebar.slider("Nivel de Estrés", min_stress, max_stress, (min_stress, max_stress))
    
    filtered_df = df.copy()
    
    if selected_country != "Todos":
        filtered_df = filtered_df[filtered_df["Country"] == selected_country]
    if selected_industry != "Todos":
        filtered_df = filtered_df[filtered_df["Industry"] == selected_industry]
    if selected_work_mode != "Todos":
        filtered_df = filtered_df[filtered_df["Work_Mode"] == selected_work_mode]
    if selected_gender != "Todos":
        filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
    if selected_burnout != "Todos":
        filtered_df = filtered_df[filtered_df["Burnout_Risk"] == selected_burnout]
    
    filtered_df = filtered_df[(filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1])]
    filtered_df = filtered_df[(filtered_df["Stress_Level"] >= stress_range[0]) & (filtered_df["Stress_Level"] <= stress_range[1])]
    
    return filtered_df

def home_page():
    st.title("🧠 Mental Health & Productivity 2026")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Bienvenido al Dashboard de Análisis
        
        Esta aplicación analiza la relación entre la **salud mental** de los empleados y su **productividad laboral**.
        
        ### Características del Dataset:
        - **1,000+ empleados** de múltiples países
        - Variables: edad, género, industria, modo de trabajo, nivel de estrés, horas de sueño, productividad, actividad física, riesgo de burnout
        - Análisis geográfico, estadístico y predictivo
        
        ### Objetivos del Análisis:
        1. Explorar patrones entre bienestar mental y rendimiento laboral
        2. Identificar factores de riesgo de burnout
        3. Predecir niveles de productividad
        4. Visualizar distribución geográfica de empleados
        """)
    
    with col2:
        st.info("👈 Use el menú lateral para navegar entre las diferentes secciones del análisis.")
    
    st.markdown("---")
    st.markdown("**Autor:** Mariana Gutierrez Restrepo | **Curso:** Ingeniería de Datos")

def data_overview_page(df):
    st.title("📊 Resumen de Datos")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Empleados", len(df))
    with col2:
        st.metric("Países", df["Country"].nunique())
    with col3:
        st.metric("Industrias", df["Industry"].nunique())
    with col4:
        st.metric("Productividad Promedio", f"{df['Productivity_Score'].mean():.1f}")
    
    st.markdown("### Estructura del Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Primeras Filas")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.markdown("#### Información de Columnas")
        info_df = pd.DataFrame({
            "Columna": df.columns,
            "Tipo": df.dtypes.values,
            "No Nulos": df.notna().sum().values,
            "Únicos": df.nunique().values
        })
        st.dataframe(info_df, use_container_width=True)
    
    st.markdown("### Estadísticas Descriptivas")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)

def descriptive_graphics_page(df):
    st.title("📈 Gráficos Descriptivos")
    st.markdown("---")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Histogramas", "Distribuciones", "Correlación", "Scatter Plots", "Box Plots", "Pair Plots"
    ])
    
    with tab1:
        st.subheader("Histogramas de Variables Numéricas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.histogram(df, x="Age", nbins=20, title="Distribución de Edad", 
                              color_discrete_sequence=["#6366f1"])
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x="Work_Hours_Per_Week", nbins=15, title="Horas de Trabajo Semanales",
                              color_discrete_sequence=["#10b981"])
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.histogram(df, x="Sleep_Hours", nbins=15, title="Horas de Sueño",
                              color_discrete_sequence=["#f59e0b"])
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x="Stress_Level", nbins=10, title="Nivel de Estrés",
                              color_discrete_sequence=["#ef4444"])
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x="Productivity_Score", nbins=20, title="Puntuación de Productividad",
                              color_discrete_sequence=["#8b5cf6"])
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Distribuciones de Variables Categóricas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(df, names="productivity_level", title="Nivel de Productividad",
                        color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(df, names="Burnout_Risk", title="Riesgo de Burnout",
                        color_discrete_sequence=px.colors.qualitative.Reds_r)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x="Gender", title="Distribución por Género", 
                              color_discrete_sequence=["#06b6d4"])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x="Work_Mode", title="Modo de Trabajo",
                              color_discrete_sequence=["#84cc16"])
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x="Industry", title="Distribución por Industria",
                              color_discrete_sequence=["#f97316"])
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x="Country", title="Distribución por País",
                              color_discrete_sequence=["#ec4899"])
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Mapa de Correlación")
        
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Matriz de Correlación"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Correlaciones con Productividad")
        productivity_corr = correlation_matrix["Productivity_Score"].sort_values(ascending=False)
        corr_df = pd.DataFrame({
            "Variable": productivity_corr.index,
            "Correlación": productivity_corr.values
        })
        fig = px.bar(corr_df, x="Variable", y="Correlación", title="Correlaciones con Productividad",
                    color="Correlación", color_continuous_scale="RdBu_r")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Diagramas de Dispersión")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x="Sleep_Hours", y="Productivity_Score",
                           color="productivity_level", title="Sueño vs Productividad",
                           trendline="ols", color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x="Stress_Level", y="Productivity_Score",
                           color="Burnout_Risk", title="Estrés vs Productividad",
                           trendline="ols", color_discrete_sequence=px.colors.qualitative.Reds_r)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x="Work_Hours_Per_Week", y="Productivity_Score",
                           color="Work_Mode", title="Horas de Trabajo vs Productividad",
                           trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x="Age", y="Productivity_Score",
                           color="Gender", title="Edad vs Productividad",
                           trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x="Physical_Activity_Hours", y="Productivity_Score",
                           title="Actividad Física vs Productividad", trendline="ols",
                           color_discrete_sequence=["#10b981"])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x="Stress_Level", y="Sleep_Hours",
                           color="productivity_level", title="Estrés vs Sueño",
                           trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("Diagramas de Caja (Box Plots)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df, x="productivity_level", y="Productivity_Score",
                        title="Productividad por Nivel", color="productivity_level")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x="Burnout_Risk", y="Productivity_Score",
                        title="Productividad por Riesgo de Burnout", color="Burnout_Risk")
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df, x="Work_Mode", y="Productivity_Score",
                        title="Productividad por Modo de Trabajo", color="Work_Mode")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x="Industry", y="Productivity_Score",
                        title="Productividad por Industria", color="Industry")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df, x="Gender", y="Stress_Level", title="Estrés por Género", color="Gender")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x="Burnout_Risk", y="Sleep_Hours", title="Sueño por Riesgo de Burnout", 
                        color="Burnout_Risk")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.subheader("Pair Plots")
        
        selected_cols = st.multiselect(
            "Seleccione variables para Pair Plot",
            options=numeric_cols,
            default=["Age", "Stress_Level", "Sleep_Hours", "Productivity_Score", "Work_Hours_Per_Week"]
        )
        
        if selected_cols:
            fig = px.scatter_matrix(df[selected_cols], 
                                   dimensions=selected_cols,
                                   color="productivity_level",
                                   title="Pair Plot de Variables Seleccionadas")
            fig.update_layout(height=800)
            st.plotly_chart(fig, use_container_width=True)

def geographic_map_page(df):
    st.title("🗺️ Mapa Geográfico")
    st.markdown("---")
    
    st.subheader("Distribución de Empleados por País")
    
    country_counts = df["Country"].value_counts().reset_index()
    country_counts.columns = ["Country", "Count"]
    
    fig = px.bar(country_counts, x="Country", y="Count", 
                 title="Número de Empleados por País",
                 color="Count", color_continuous_scale="Viridis")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Productividad por País")
    
    country_productivity = df.groupby("Country").agg({
        "Productivity_Score": "mean",
        "Employee_ID": "count"
    }).reset_index()
    country_productivity.columns = ["Country", "Avg_Productivity", "Employee_Count"]
    
    fig = px.bar(country_productivity, x="Country", y="Avg_Productivity",
                 title="Productividad Promedio por País",
                 color="Avg_Productivity", color_continuous_scale="RdYlGn")
    fig.update_layout(xaxis_tickangle=-45, yaxis_title="Productividad Promedio")
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Estrés Promedio por País")
        country_stress = df.groupby("Country")["Stress_Level"].mean().reset_index()
        fig = px.bar(country_stress, x="Country", y="Stress_Level",
                     title="Estrés Promedio por País",
                     color="Stress_Level", color_continuous_scale="Reds_r")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Riesgo de Burnout por País")
        burnout_by_country = df.groupby(["Country", "Burnout_Risk"]).size().reset_index(name="Count")
        fig = px.bar(burnout_by_country, x="Country", y="Count", color="Burnout_Risk",
                     title="Distribución de Burnout por País",
                     barmode="group")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Modo de Trabajo por País")
    workmode_by_country = df.groupby(["Country", "Work_Mode"]).size().reset_index(name="Count")
    fig = px.bar(workmode_by_country, x="Country", y="Count", color="Work_Mode",
                 title="Modo de Trabajo por País", barmode="group")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def frequency_tables_page(df):
    st.title("📋 Tablas de Frecuencia")
    st.markdown("---")
    
    st.subheader("Distribución por Género")
    gender_freq = df["Gender"].value_counts()
    gender_df = pd.DataFrame({
        "Género": gender_freq.index,
        "Frecuencia": gender_freq.values,
        "Porcentaje": (gender_freq.values / len(df) * 100).round(2)
    })
    st.table(gender_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución por Industria")
        industry_freq = df["Industry"].value_counts()
        industry_df = pd.DataFrame({
            "Industria": industry_freq.index,
            "Frecuencia": industry_freq.values,
            "Porcentaje": (industry_freq.values / len(df) * 100).round(2)
        })
        st.table(industry_df)
    
    with col2:
        st.subheader("Distribución por Modo de Trabajo")
        workmode_freq = df["Work_Mode"].value_counts()
        workmode_df = pd.DataFrame({
            "Modo de Trabajo": workmode_freq.index,
            "Frecuencia": workmode_freq.values,
            "Porcentaje": (workmode_freq.values / len(df) * 100).round(2)
        })
        st.table(workmode_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución por Riesgo de Burnout")
        burnout_freq = df["Burnout_Risk"].value_counts()
        burnout_df = pd.DataFrame({
            "Riesgo de Burnout": burnout_freq.index,
            "Frecuencia": burnout_freq.values,
            "Porcentaje": (burnout_freq.values / len(df) * 100).round(2)
        })
        st.table(burnout_df)
    
    with col2:
        st.subheader("Distribución por Nivel de Productividad")
        prod_level_freq = df["productivity_level"].value_counts()
        prod_level_df = pd.DataFrame({
            "Nivel de Productividad": prod_level_freq.index,
            "Frecuencia": prod_level_freq.values,
            "Porcentaje": (prod_level_freq.values / len(df) * 100).round(2)
        })
        st.table(prod_level_df)
    
    st.subheader("Distribución por País")
    country_freq = df["Country"].value_counts()
    country_df = pd.DataFrame({
        "País": country_freq.index,
        "Frecuencia": country_freq.values,
        "Porcentaje": (country_freq.values / len(df) * 100).round(2)
    })
    st.table(country_df)
    
    st.subheader("Acceso a Apoyo de Salud Mental")
    support_freq = df["Mental_Health_Support_Access"].value_counts()
    support_df = pd.DataFrame({
        "Acceso a Apoyo": support_freq.index,
        "Frecuencia": support_freq.values,
        "Porcentaje": (support_freq.values / len(df) * 100).round(2)
    })
    st.table(support_df)

def predictive_analysis_page(df):
    st.title("🔮 Análisis Predictivo")
    st.markdown("---")
    
    st.markdown("""
    Este módulo permite predecir la **Puntuación de Productividad** basándose en diferentes variables.
    El modelo utiliza algoritmos de Machine Learning para encontrar patrones en los datos.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuración del Modelo")
        
        model_type = st.selectbox("Tipo de Modelo", ["Random Forest", "Regresión Lineal"])
        
        st.markdown("### Variables de Entrada para Predicción")
        
        age_input = st.number_input("Edad", min_value=22, max_value=59, value=35)
        stress_input = st.slider("Nivel de Estrés (1-10)", 1, 10, 5)
        sleep_input = st.number_input("Horas de Sueño", min_value=4.0, max_value=10.0, value=7.0, step=0.5)
        work_hours_input = st.number_input("Horas de Trabajo/Semana", min_value=30, max_value=64, value=40)
        physical_activity_input = st.number_input("Horas de Actividad Física/Semana", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
        
        predict_button = st.button("🔮 Predecir Productividad", type="primary")
    
    with col2:
        features = ["Age", "Stress_Level", "Sleep_Hours", "Work_Hours_Per_Week", "Physical_Activity_Hours"]
        
        X = df[features]
        y = df["Productivity_Score"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        st.subheader("Métricas del Modelo")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("R² Score", f"{r2:.4f}")
        with col_m2:
            st.metric("MAE", f"{mae:.2f}")
        with col_m3:
            st.metric("RMSE", f"{rmse:.2f}")
        
        if predict_button:
            input_data = pd.DataFrame({
                "Age": [age_input],
                "Stress_Level": [stress_input],
                "Sleep_Hours": [sleep_input],
                "Work_Hours_Per_Week": [work_hours_input],
                "Physical_Activity_Hours": [physical_activity_input]
            })
            
            prediction = model.predict(input_data)[0]
            
            st.success(f"### Predicción de Productividad: {prediction:.1f}")
        
        st.subheader("Importancia de Características")
        
        if model_type == "Random Forest":
            importance_df = pd.DataFrame({
                "Feature": features,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True)
            
            fig = px.bar(importance_df, x="Importance", y="Feature",
                        title="Importancia de Características (Random Forest)",
                        orientation="h", color="Importance", color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)
        else:
            coef_df = pd.DataFrame({
                "Feature": features,
                "Coefficient": model.coef_
            }).sort_values("Coefficient", ascending=True)
            
            fig = px.bar(coef_df, x="Coefficient", y="Feature",
                        title="Coeficientes del Modelo (Regresión Lineal)",
                        orientation="h", color="Coefficient", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Predicciones vs Valores Reales")
        
        results_df = pd.DataFrame({
            "Valor Real": y_test.values,
            "Predicción": y_pred
        }).head(20)
        
        fig = px.scatter(results_df, x="Valor Real", y="Predicción",
                        title="Predicciones vs Valores Reales",
                        trendline="ols")
        fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), 
                     x1=y_test.max(), y1=y_test.max(),
                     line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

def main():
    df = load_data()
    
    st.sidebar.title("🧠 Mental Health & Productivity")
    st.sidebar.markdown("---")
    
    pages = {
        "🏠 Inicio": "home",
        "📊 Resumen de Datos": "data_overview",
        "📈 Gráficos Descriptivos": "descriptive_graphics",
        "🗺️ Mapa Geográfico": "geographic_map",
        "📋 Tablas de Frecuencia": "frequency_tables",
        "🔍 Filtros": "filters",
        "🔮 Análisis Predictivo": "predictive_analysis"
    }
    
    selected_page = st.sidebar.radio("Navegación", list(pages.keys()))
    
    filtered_df = apply_filters(df)
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 Registros mostrados: {len(filtered_df)} / {len(df)}")
    
    page = pages[selected_page]
    
    if page == "home":
        home_page()
    elif page == "data_overview":
        data_overview_page(filtered_df)
    elif page == "descriptive_graphics":
        descriptive_graphics_page(filtered_df)
    elif page == "geographic_map":
        geographic_map_page(filtered_df)
    elif page == "frequency_tables":
        frequency_tables_page(filtered_df)
    elif page == "filters":
        st.title("🔍 Filtros Activos")
        st.markdown("---")
        st.write("Los filtros se aplican en tiempo real a través del panel lateral.")
        st.write("Todos los gráficos y tablas reflejan los datos filtrados.")
        st.markdown("### Resumen de Filtros Aplicados")
        st.dataframe(filtered_df.describe(), use_container_width=True)
        st.subheader("Datos Filtrados")
        st.dataframe(filtered_df.head(100), use_container_width=True)
    elif page == "predictive_analysis":
        predictive_analysis_page(filtered_df)

if __name__ == "__main__":
    main()