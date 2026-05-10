import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from regresion_diabetes import mapeo_nombres, regresion

# Configuración de la página
st.set_page_config(
    page_title="Diabetes Risk Analytics",
    layout="wide"
)
css_path = Path(__file__).resolve().parent / 'styles.css'
if css_path.exists():
    with open(css_path, 'r', encoding='utf-8') as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

# Carga de datos

file_path = 'datasets/diabetes_risk_dataset.csv'
df = pd.read_csv(file_path, sep=';')

# TRADUCCIÓN
Gender_Mapped_map = {'Male': 'Masculino', 'Female': 'Femenino'}
df.loc[:, 'Gender_Mapped'] = df['gender'].map(Gender_Mapped_map)

level_map = {
    'Low': 'Bajo',
    'Moderate': 'Moderado',
    'High': 'Alto'
}
df.loc[:, 'Physical_Activity_Level_Mapped'] = df['physical_activity_level'].map(level_map)

risk_map = {
    'Prediabetes': 'Prediabetes',
    'Low Risk': 'Bajo Riesgo',
    'High Risk': 'Alto Riesgo'
}
df.loc[:, 'Diabetes_Risk_Category_Mapped'] = df['diabetes_risk_category'].map(risk_map)

family_map = {
    'No': 'No',
    'Yes': 'Sí'
}
df.loc[:, 'family_history_diabetes'] = df['family_history_diabetes'].map(family_map)

# Categorización de Nivel de Estrés
bins_stress = [0, 3, 7, 10]
labels_stress = ['Bajo (1-3)', 'Moderado (4-7)', 'Alto (8-10)']
df['stress_category'] = pd.cut(df['stress_level'], bins=bins_stress, labels=labels_stress)

# Categorización de Horas de Sueño
bins_sleep = [0, 5.9, 8, 24]
labels_sleep = ['Insuficiente (<6h)', 'Recomendado (6-8h)', 'Prolongado (>8h)']
df['sleep_category'] = pd.cut(df['sleep_hours'], bins=bins_sleep, labels=labels_sleep)

# --- PALETAS ---
riesgo = ['#7FDED5', '#F8E8B7', '#F25E7C']
herencia = ['#F29BB2', '#766FD8']

# --- SIDEBAR: FILTROS ---
st.sidebar.header("Filtros de Análisis")

Gender_Mapped_filter = st.sidebar.multiselect(
    "Género", options=df["Gender_Mapped"].unique(), default=df["Gender_Mapped"].unique()
)

risk_filter = st.sidebar.multiselect(
    "Categoría de Riesgo", options=df["Diabetes_Risk_Category_Mapped"].unique(), default=df["Diabetes_Risk_Category_Mapped"].unique()
)

activity_filter = st.sidebar.multiselect(
    "Nivel de Actividad Física", options=df["Physical_Activity_Level_Mapped"].unique(), default=df["Physical_Activity_Level_Mapped"].unique()
)
    
stress_filter = st.sidebar.multiselect(
    "Nivel de Estrés", options=df["stress_category"].unique(), default=df["stress_category"].unique()
)

sleep_filter = st.sidebar.multiselect(
    "Calidad de Sueño", options=df["sleep_category"].unique(), default=df["sleep_category"].unique()
)

# Aplicar filtros
filtered_df = df[
    (df["Gender_Mapped"].isin(Gender_Mapped_filter)) &
    (df["Physical_Activity_Level_Mapped"].isin(activity_filter)) &
    (df["Diabetes_Risk_Category_Mapped"].isin(risk_filter)) &
    (df['stress_category'].isin(stress_filter)) &
    (df['sleep_category'].isin(sleep_filter))
]

# --- TÍTULO PRINCIPAL ---
st.markdown(
"""
<header class="hero">
    <h1 class="bold-title">Riesgo de Diabetes</h1>
    <p class="tagline">Análisis EDA | Clasificación y Regresión</p>
</header>
""",
unsafe_allow_html=True,
)

# --- SECCIÓN 1: MÉTRICAS (KPIs) ---
cols = st.columns(4)
avg_risk = filtered_df["diabetes_risk_score"].mean()
high_risk_pct = (len(filtered_df[filtered_df["Diabetes_Risk_Category_Mapped"] == "Alto Riesgo"]) / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
avg_bmi = filtered_df["bmi"].mean()
avg_hba1c = filtered_df["HbA1c_level"].mean()

metrics = [
    ("Risk Score Promedio", f"{avg_risk:.2f}"),
    ("% Riesgo Alto", f"{high_risk_pct:.1f}%"),
    ("IMC Promedio", f"{avg_bmi:.2f}"),
    ("HbA1c Promedio", f"{avg_hba1c:.2f}")
]

for col, (title, value) in zip(cols, metrics):
    with col:
        st.markdown(f"""
            <div class="card">
                <div class="card-title">{title}</div>
                <div class="stat-value">{value}</div>
            </div>
        """, unsafe_allow_html=True)

    # avg_calories = filtered_df["daily_calorie_intake"].mean()
    # avg_sugar = filtered_df["sugar_intake_grams_per_day"].mean()
    # sleep_issues_pct = (len(filtered_df[filtered_df["sleep_hours"] < 6]) / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    # high_stress_pct = (len(filtered_df[filtered_df["stress_level"] >= 8]) / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    # metrics2 = [
    #     ("Promedio Calorías Diarias", f"{avg_calories:.0f} kcal"),
    #     ("Promedio Azúcar/Día", f"{avg_sugar:.1f} g"),
    #     ("% Sueño Insuficiente", f"{sleep_issues_pct:.1f}%"),
    #     ("% Estrés Alto", f"{high_stress_pct:.1f}%")
    # ]

    # for col, (title, value) in zip(cols, metrics2):
    #     with col:
    #         st.markdown(f"""
    #             <div class="card">
    #                 <div class="card-title">{title}</div>
    #                 <div class="stat-value">{value}</div>
    #             </div>
    #         """, unsafe_allow_html=True)


# --- SECCIÓN 2: TABS DE ANÁLISIS ---
tab1, tab2,tab_biom, tab_habits, tab_regresion = st.tabs(["Score y Categorías de Riesgo", "Analisis BMI","Ingesta Calórica y de Azúcar", "Hábitos y Estilo de Vida", "Regresión Logística"])

with tab1:
    c1, c2 = st.columns([60,40])

    with c1:
        fig_hist = px.histogram(
            filtered_df, x="diabetes_risk_score",
            nbins=30, color="Diabetes_Risk_Category_Mapped",
            title="Distribución del Score de Riesgo",
            marginal="box", 
            color_discrete_sequence=riesgo,
            labels={"diabetes_risk_score":"Score de Riesgo de Diabetes", "Diabetes_Risk_Category_Mapped": "Categoría de Riesgo de Diabetes"},
            category_orders={'Diabetes_Risk_Category_Mapped':['Bajo Riesgo', 'Prediabetes', 'Alto Riesgo']}
        )
        fig_hist.update_layout(margin = dict(t=60, l=40, r=40, b=40),
                  title={'x':0.5, 'xanchor': 'center', 'font': dict(size=24)})
        st.plotly_chart(fig_hist, use_container_width=True, width='stretch', key='chart-histograma')

    with c2:
        fig_pie = px.pie(
            filtered_df, names="Diabetes_Risk_Category_Mapped",
            title="Proporción por Categoría de Riesgo",
            hole=0.5, 
            color_discrete_sequence=riesgo,
            labels={"diabetes_risk_score":"Score de Riesgo de Diabetes", "Diabetes_Risk_Category_Mapped": "Categoría de Riesgo de Diabetes"},
            category_orders={'Diabetes_Risk_Category_Mapped':['Bajo Riesgo', 'Prediabetes', 'Alto Riesgo']}
        )
        fig_pie.update_layout(margin = dict(t=60, l=100, r=100, b=20),
                  title={'x':0.5, 'xanchor': 'center', 'font': dict(size=24)})
        st.plotly_chart(fig_pie, use_container_width=True, width='stretch', key='chart-proporcion')


with tab2:
    c1, c2 = st.columns(2)
    with c1:
        # Scatter Plot bmi vs glucosa
        fig_glusosa = px.scatter(
            filtered_df,
            x="bmi",
            y="fasting_glucose_level",
            color="diabetes_risk_score",
            hover_data=["age", "HbA1c_level"],
            color_continuous_scale="Temps",
            title="BMI vs Glucosa",
            labels={"bmi": "Índice de Masa Corporal", "fasting_glucose_level": "Glucosa en Ayunas", "diabetes_risk_score": "Riesgo de Diabetes"}
        )
        fig_glusosa.update_layout(margin = dict(t=60, l=40, r=40, b=40),
                  title={'x':0.5, 'xanchor': 'center', 'font': dict(size=24)})
        st.plotly_chart(fig_glusosa, use_container_width=True, width='stretch', key='chart-bm-glucosa')
    
    with c2:
        # Scatter Plot Multivariado
        fig_multi = px.scatter(
            filtered_df,
            x="bmi",
            y="triglycerides_level",
            color="diabetes_risk_score",
            hover_data=["age", "HbA1c_level"],
            color_continuous_scale="Temps",
            title="BMI vs Triglicéridos",
            labels={"bmi": "Índice de Masa Corporal", "triglycerides_level": "Nivel de trigliceridos", "diabetes_risk_score": "Riesgo de Diabetes"}
        )
        fig_multi.update_layout(margin = dict(t=60, l=40, r=40, b=40),
                  title={'x':0.5, 'xanchor': 'center', 'font': dict(size=24)})
        st.plotly_chart(fig_multi, use_container_width=True, width='stretch', key='chart-bmi-trigliceridos')

with tab_biom:
    c1, c2 = st.columns(2)
    with c1:
        fig_calorie_risk = px.box(filtered_df, x="Diabetes_Risk_Category_Mapped", y="daily_calorie_intake",
                                color="Diabetes_Risk_Category_Mapped", title="Ingesta de Calorías por Categoría de Riesgo", color_discrete_sequence=riesgo,
                                labels={"Diabetes_Risk_Category_Mapped": "Categoría de Riesgo de Diabetes", "daily_calorie_intake": "Ingesta Calorías(dia)"},
                                category_orders={'Diabetes_Risk_Category_Mapped':['Bajo Riesgo', 'Prediabetes', 'Alto Riesgo']})
        fig_calorie_risk.update_layout(margin = dict(t=60, l=40, r=40, b=40),
                  title={'x':0.5, 'xanchor': 'center', 'font': dict(size=24)})
        st.plotly_chart(fig_calorie_risk, use_container_width=True, width='stretch', key='chart-bio1')
    with c2:
        fig_sugar_risk = px.box(filtered_df, x="Diabetes_Risk_Category_Mapped", y="sugar_intake_grams_per_day",
                                color="Diabetes_Risk_Category_Mapped", title="Ingesta de Azúcar por Categoría de Riesgo", color_discrete_sequence=riesgo,
                                labels={"Diabetes_Risk_Category_Mapped": "Categoría de Riesgo de Diabetes", "sugar_intake_grams_per_day": "Ingesta Azúcar(diario)"},
                                category_orders={'Diabetes_Risk_Category_Mapped':['Bajo Riesgo', 'Prediabetes', 'Alto Riesgo']})
        fig_sugar_risk.update_layout(margin = dict(t=60, l=40, r=40, b=40),
                  title={'x':0.5, 'xanchor': 'center', 'font': dict(size=24)})
        st.plotly_chart(fig_sugar_risk, use_container_width=True, width='stretch', key='chart-bio2')

    st.divider()
    st.subheader("Relación Dieta vs Score de Riesgo")
    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        fig_cal = px.scatter(filtered_df, x="daily_calorie_intake", y="diabetes_risk_score", 
                            color="Physical_Activity_Level_Mapped", opacity=0.5,
                            title="Calorías vs Riesgo", category_orders={'Physical_Activity_Level_Mapped': ["Alto", "Moderado", "Bajo"]},
                            labels={"Physical_Activity_Level_Mapped": "Nivel de Actividad", "daily_calorie_intake": "Ingesta Calorías(diario)",
                                    "diabetes_risk_score": "Riesgo de Diabetes(Score)"},
                            color_discrete_sequence=riesgo)
        fig_cal.update_layout(margin = dict(t=60, l=40, r=40, b=40),
                  title={'x':0.5, 'xanchor': 'center', 'font': dict(size=24)})
        st.plotly_chart(fig_cal, use_container_width=True, width='stretch', key='chart-dieta')
    with row2_c2:
        fig_sugar_scatter = px.scatter(filtered_df, x="sugar_intake_grams_per_day", y="bmi",
                                        color="diabetes_risk_score",
                                        labels={"sugar_intake_grams_per_day": "Ingesta Azúcar(diario)",
                                    "diabetes_risk_score": "Riesgo de Diabetes(Score)"},
                                        title="Azúcar vs Riesgo", color_continuous_scale="Temps")
        fig_sugar_scatter.update_layout(margin = dict(t=60, l=40, r=40, b=40),
                  title={'x':0.5, 'xanchor': 'center', 'font': dict(size=24)})
        st.plotly_chart(fig_sugar_scatter, use_container_width=True, width='stretch', key='chart-azucar')

with tab_habits:
    
    # Comparativa Estrés vs Riesgo
    stress_impact = filtered_df.groupby(['stress_category', 'Physical_Activity_Level_Mapped'])['diabetes_risk_score'].mean().reset_index()
    fig_stress = px.bar(stress_impact, x='stress_category', y='diabetes_risk_score',
                        title="Riesgo Promedio según Nivel de Estrés", facet_col= "Physical_Activity_Level_Mapped",
                        color='stress_category', color_discrete_sequence=px.colors.sequential.Burg,
                        labels={'diabetes_risk_score': 'Riesgo de Diabetes', 'stress_category':'Nivel de Estrés',
                                "Physical_Activity_Level_Mapped": "Nivel de Actividad"},
                                category_orders={'Physical_Activity_Level_Mapped': ['Bajo', 'Moderado', 'Alto']})
    fig_stress.update_layout(margin = dict(t=60, l=40, r=40, b=40),
                  title={'x':0.5, 'xanchor': 'center', 'font': dict(size=24)})
    st.plotly_chart(fig_stress, use_container_width=True, width='stretch', key='chart-estres')

    
    # Comparativa Sueño vs Riesgo
    sleep_impact = filtered_df.groupby(['sleep_category', 'Physical_Activity_Level_Mapped'])['diabetes_risk_score'].mean().reset_index()
    fig_sleep = px.bar(sleep_impact, x='sleep_category', y='diabetes_risk_score',
                        title="Riesgo Promedio según Horas de Sueño", facet_col= "Physical_Activity_Level_Mapped",
                        color='sleep_category', color_discrete_sequence=px.colors.sequential.Teal_r,
                        labels={'sleep_category': 'Nivel de Sueño', 'diabetes_risk_score': 'Riesgo de Diabetes',
                                "Physical_Activity_Level_Mapped": "Nivel de Actividad"},
                                category_orders={'Physical_Activity_Level_Mapped': ['Bajo', 'Moderado', 'Alto']})
    fig_sleep.update_layout(margin = dict(t=60, l=40, r=40, b=40),
                  title={'x':0.5, 'xanchor': 'center', 'font': dict(size=24)})
    st.plotly_chart(fig_sleep, use_container_width=True, width='stretch', key='chart-suenio')

with tab_regresion:
    # Heatmap de correlación

    # 2. Calculamos la correlación
    corr = filtered_df.select_dtypes(include=['float64', 'int64']).drop(columns=['Patient_ID'], errors='ignore').corr()

    # 3. Renombramos tanto el índice como las columnas
    corr.rename(columns=mapeo_nombres, index=mapeo_nombres, inplace=True)

    # 4. Graficamos
    fig_heat = px.imshow(
        corr, 
        text_auto=".2f", 
        aspect="auto",
        title="Matriz de Correlación (Pearson)",
        color_continuous_scale="Temps", 
        origin="lower"
    )
    fig_heat.update_layout(margin = dict(t=60, l=40, r=40, b=40),
                  title={'x':0.5, 'xanchor': 'center', 'font': dict(size=24)})

    st.plotly_chart(fig_heat, use_container_width=True, key='chart-corr')

    importance_df, r2, mae, mse, rmse = regresion(df)
    fig = px.bar(data_frame=importance_df, x='Importance', y='Feature_Mapped', 
                 color_discrete_sequence=['#F25E7C'], 
                 title='Importancia de las Variables en el Modelo Random Forest (Predicción de Riesgo de Diabetes)',
                 labels={'Importance': 'Importancia (Gini Impurity Decrease)', 'Feature_Mapped': 'Variable'})
    fig.update_layout(margin = dict(t=60, l=40, r=40, b=40),
            title={'x':0.5, 'xanchor': 'center', 'font': dict(size=24)})

    st.plotly_chart(fig, use_container_width=True, width='stretch', key='chart-importancia')


    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div>
                <p class="tagline">R-squared:</p>
                    <div class="stat-value">{f"{r2:.4f}"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )     

    with c2:
        st.markdown(
            f"""
            <div>
                <p class="tagline">Mean Absolute Error (MAE):</p>
                    <div class="stat-value">{f"{mae:.4f}"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )   

    with c3:
        st.markdown(
            f"""
            <div>
                <p class="tagline">Mean Squared Error (MSE):</p>
                    <div class="stat-value">{f"{mse:.4f}"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )   

    with c4:
        st.markdown(
            f"""
            <div>
                <p class="tagline">Root Mean Squared Error (RMSE):</p>
                    <div class="stat-value">{f"{rmse:.4f}"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )      
    