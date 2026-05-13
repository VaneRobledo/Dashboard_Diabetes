import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from regresion_diabetes import mapeo_nombres, regresion, entrenar_y_evaluar_clasificador

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

file_path = 'diabetes_risk_dataset.csv'
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
herencia = ["#A19DD6","#8DD9E0",]

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
tab_demo, tab1, tab2,tab_biom, tab_habits, tab_regresion = st.tabs(["Datos demográficos","Score y Categorías de Riesgo", "Analisis BMI","Ingesta Calórica y de Azúcar", "Hábitos y Estilo de Vida", "Regresión y Clasificación"])

with tab_demo:
    fig_demo = px.histogram(filtered_df,
                            x='age', 
                            color='Gender_Mapped', 
                            nbins=30, barmode="group", 
                            color_discrete_sequence=herencia,
                            title='Ditribución de Género por Edad',
                            labels={'age': 'Edad', 'Gender_Mapped': 'Género'})
    fig_demo.update_layout(margin = dict(t=60, l=40, r=40, b=40),
                  title={'x':0.5, 'xanchor': 'center', 'font': dict(size=24)})
    st.plotly_chart(fig_demo,use_container_width=True, width='stretch', key='chart-histograma')

    filtered_df['Rango_edad'] = pd.cut(x=filtered_df['age'], labels=['18-24', '25-44','45-64','+65'], bins=[0, 24, 44, 64, 100])
    tabla = filtered_df.groupby(['Gender_Mapped', 'Rango_edad'])['Diabetes_Risk_Category_Mapped'].value_counts().reset_index()
    tabla = tabla.rename(columns={'count':'Cantidad'})
    
    fig_gen_riesgo = px.density_heatmap(
        tabla,x='Rango_edad', y='Gender_Mapped', z='Cantidad', 
        facet_col='Diabetes_Risk_Category_Mapped',
        color_continuous_scale="Purp",
        title='Cantidad de Pacientes por Tipo de Riesgo, Género y Edad',
        labels={'Rango_edad': 'Rango Edad', 'Gender_Mapped': 'Género', 'Cantidad': "Pacientes", 'Diabetes_Risk_Category_Mapped':'Categoría'},
        text_auto=True
    )
    
    fig_gen_riesgo.update_layout(margin = dict(t=60, l=40, r=40, b=40),
                  title={'x':0.5, 'xanchor': 'center', 'font': dict(size=24)})
    fig_gen_riesgo.update_coloraxes(showscale=False)
    st.plotly_chart(fig_gen_riesgo, use_container_width=True, width='stretch', key='chart-r')
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
        st.plotly_chart(fig_hist, use_container_width=True, width='stretch', key='chart-hist')

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
    fig_stress = px.box(filtered_df, x='stress_category', y='diabetes_risk_score', boxmode='group',
                        title="Riesgo según Nivel de Estrés", facet_col= "Physical_Activity_Level_Mapped", color='stress_category',
                        labels={'diabetes_risk_score': 'Riesgo de Diabetes', 'stress_category':'Nivel de Estrés',
                                "Physical_Activity_Level_Mapped": "Nivel de Actividad"}, color_discrete_sequence=px.colors.sequential.Burg,
                        category_orders={'Physical_Activity_Level_Mapped': ['Bajo', 'Moderado', 'Alto'], 'stress_category': ['Bajo (1-3)', 'Moderado (4-7)', 'Alto (8-10)']})
    fig_stress.add_hline(y=0, line_width=0, line_dash="dash", line_color=riesgo[1])
    fig_stress.add_hline(y=35, line_width=3, line_dash="dash", line_color=riesgo[1])
    fig_stress.add_hline(y=65, line_width=3, line_dash="dash", line_color=riesgo[2])
    
    fig_stress.add_hrect(y0=35, y1=65, line_width=0, fillcolor=riesgo[1], opacity=0.05)
    fig_stress.add_hrect(y0=0, y1=35, line_width=0, fillcolor=riesgo[0], opacity=0.05)
    fig_stress.add_hrect(y0=65, y1=100, line_width=0, fillcolor=riesgo[2], opacity=0.05)

    fig_stress.update_layout(margin = dict(t=60, l=40, r=40, b=40),
                  title={'x':0.5, 'xanchor': 'center', 'font': dict(size=24)})

    st.plotly_chart(fig_stress, use_container_width=True, width='stretch', key='chart-estres')

    
    # Comparativa Sueño vs Riesgo
    fig_sleep = px.box(filtered_df, x='sleep_category', y='diabetes_risk_score', boxmode='group',
                        title="Riesgo según Horas de Sueño", facet_col= "Physical_Activity_Level_Mapped",
                        color='sleep_category', color_discrete_sequence=px.colors.sequential.Teal_r,
                        labels={'sleep_category': 'Nivel de Sueño', 'diabetes_risk_score': 'Riesgo de Diabetes',
                                "Physical_Activity_Level_Mapped": "Nivel de Actividad"},
                                category_orders={'Physical_Activity_Level_Mapped': ['Bajo', 'Moderado', 'Alto']})
    fig_sleep.add_hline(y=0, line_width=0, line_dash="dash", line_color=riesgo[1])
    fig_sleep.add_hline(y=35, line_width=3, line_dash="dash", line_color=riesgo[1])
    fig_sleep.add_hline(y=65, line_width=3, line_dash="dash", line_color=riesgo[2])
    
    fig_sleep.add_hrect(y0=35, y1=65, line_width=0, fillcolor=riesgo[1], opacity=0.05)
    fig_sleep.add_hrect(y0=0, y1=35, line_width=0, fillcolor=riesgo[0], opacity=0.05)
    fig_sleep.add_hrect(y0=65, y1=100, line_width=0, fillcolor=riesgo[2], opacity=0.05)
    

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

    st.subheader('Regresión Random Forest')

    st.header("Análisis de Predicción de Riesgo")
    
    X_all = df.drop(['Patient_ID', 'diabetes_risk_score', 'diabetes_risk_category', 'Gender_Mapped', 'Physical_Activity_Level_Mapped', 'Diabetes_Risk_Category_Mapped', 'stress_category', 'sleep_category'], axis=1)
    y = df['diabetes_risk_score']

    importance_df_all, r2_all, mae_all, mse_all, rmse_all = regresion(X_all, y)
    fig = px.bar(data_frame=importance_df_all, x='Importance', y='Feature_Mapped', 
                 color_discrete_sequence=['#F25E7C'], 
                 title='Importancia de las Variables en el Modelo Random Forest (Predicción de Riesgo de Diabetes)',
                 labels={'Importance': 'Importancia (Gini Impurity Decrease)', 'Feature_Mapped': 'Variable'})
    fig.update_layout(margin = dict(t=60, l=40, r=40, b=40),
            title={'x':0.5, 'xanchor': 'center', 'font': dict(size=24)})

    st.plotly_chart(fig, use_container_width=True, width='stretch', key='chart-importancia')

    st.subheader('Comparativa de Modelos: Random Forest')
    # 2. Modelo con variables seleccionadas
    X_subset = df[['bmi', 'fasting_glucose_level']]
    y = df['diabetes_risk_score']
    importance_df_sub, r2_sub, mae_sub, mse_sub, rmse_sub = regresion(X_subset, y)

    # --- Creación de la tabla comparativa ---
    # Creamos un diccionario con los resultados
    metricas_comparativas = {
        "Métrica": ["R-squared (R2)", "Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)"],
        "Todas las Variables": [r2_all, mae_all, mse_all, rmse_all],
        "Solo IMC y Glucosa": [r2_sub, mae_sub, mse_sub, rmse_sub]
    }

    df_metrics = pd.DataFrame(metricas_comparativas)

    # Formateamos los números a 4 decimales para la visualización
    df_metrics_style = df_metrics.style.format({
        "Todas las Variables": "{:.4f}",
        "Solo IMC y Glucosa": "{:.4f}"
    })

    # Mostramos la tabla
    st.table(df_metrics_style)

    # Opcional: Resaltar cuál es mejor (si el R2 es mayor, resaltarlo)
    st.info("**Nota:** Un R2 más cercano a 1 y errores (MAE/RMSE) más bajos indican un mejor ajuste.")

    X_clf = df.drop(['Patient_ID', 'diabetes_risk_score', 'diabetes_risk_category', 'Gender_Mapped', 'Physical_Activity_Level_Mapped', 'Diabetes_Risk_Category_Mapped'], axis=1)
    y_clf = df['Diabetes_Risk_Category_Mapped']

    st.subheader('Regresión Logística (Todas las variables)')

    st.header("Análisis de Clasificación de Riesgo")

    # Ejecución automática al cargar el dashboard
    acc, report, matrix, labels = entrenar_y_evaluar_clasificador(X_clf, y_clf)

    # Indicador visual rápido
    st.metric("Exactitud del Modelo (Accuracy)", f"{acc:.2%}")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.write("### Matriz de Confusión")
        fig_cm = px.imshow(
            matrix,
            text_auto=True,
            labels=dict(x="Predicción", y="Valor Real", color="Casos"),
            x=[str(l) for l in labels],
            y=[str(l) for l in labels],
            color_continuous_scale='Burg',
            aspect="auto"
        )
        fig_cm.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_cm, use_container_width=True, width='stretch', key='chart-matriz')

    with col2:
        st.write("### Reporte de Métricas")
        df_report = pd.DataFrame(report).transpose()
        # Limpiamos el dataframe para mostrar solo métricas principales
        st.dataframe(
            df_report.iloc[:-3, :].style.background_gradient(cmap='Purples', subset=['f1-score'], high=1, low=0),
            use_container_width=True, width='stretch'
        )

    st.info("El modelo utiliza un ajuste de pesos balanceado para compensar clases con pocos datos.")

    st.subheader('Regresión logística (Solo IMC y Glucosa)')
    X_clf = df[['bmi', 'fasting_glucose_level']]

    st.header("Análisis de Clasificación de Riesgo")

    # Ejecución automática al cargar el dashboard
    acc, report, matrix, labels = entrenar_y_evaluar_clasificador(X_clf, y_clf)

    # Indicador visual rápido
    st.metric("Exactitud del Modelo (Accuracy)", f"{acc:.2%}")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.write("### Matriz de Confusión")
        fig_cm = px.imshow(
            matrix,
            text_auto=True,
            labels=dict(x="Predicción", y="Valor Real", color="Casos"),
            x=[str(l) for l in labels],
            y=[str(l) for l in labels],
            color_continuous_scale='Burg',
            aspect="auto", zmin=0.4
        )
        fig_cm.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_cm, use_container_width=True, width='stretch', key='chart-confusion'
        )

    with col2:
        st.write("### Reporte de Métricas")
        df_report = pd.DataFrame(report).transpose()
        # Limpiamos el dataframe para mostrar solo métricas principales
        st.dataframe(
            df_report.iloc[:-3, :].style.background_gradient(cmap='Purples', subset=['f1-score'], high=1, low=0).format(precision=4),
        use_container_width=True)
        

    st.info("El modelo utiliza un ajuste de pesos balanceado para compensar clases con pocos datos.")
