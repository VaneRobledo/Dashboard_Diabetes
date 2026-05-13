from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

mapeo_nombres = {
    'age': 'Edad',
    'blood_pressure': 'Presión_arterial',
    'fasting_glucose_level': 'Glucosa_ayunas',
    'insulin_level': 'Nivel_insulina',
    'bmi': 'IMC',
    'diabetes_risk_score': 'Riesgo_diabetes',
    'sleep_hours': 'Horas_sueño',
    'HbA1c_level': 'Nivel_HbA1c',
    'cholesterol_level': 'Nivel_colesterol',
    'triglycerides_level': 'Nivel_trigliceridos',
    'daily_calorie_intake': 'Ingesta calórica diaria',
    'sugar_intake_grams_per_day': 'Ingesta_azúcar_g_día',
    'stress_level': 'Nivel_estrés',
    'waist_circumference_cm': 'Circunferencia_cintura_cm'
}

def regresion(X, y):
    # Definir características (X) y objetivo (y)

    # Identificar columnas categóricas y numéricas
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Dividir los datos en conjuntos de entrenamiento y prueba.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Crea una canalización que primero preprocese los datos y luego entrene el modelo.
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', RandomForestRegressor(random_state=42))])

    # Entrenar al modelo
    model_pipeline.fit(X_train, y_train)

    # Realiza predicciones en el conjunto de prueba
    y_pred = model_pipeline.predict(X_test)

    # Evaluar el modelo
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)


    # Obtener las importancias de las características del Random Forest
    feature_importances = model_pipeline.named_steps['regressor'].feature_importances_

    # Obtener los nombres de las características preprocesadas
    feature_names_processed = model_pipeline.named_steps['preprocessor'].get_feature_names_out()

    # Crear un DataFrame para visualizar las importancias
    importance_df = pd.DataFrame({
        'Feature': feature_names_processed,
        'Importance': feature_importances
    })

    # Ordenar por importancia de forma descendente
    importance_df = importance_df.sort_values(by='Importance', ascending=True)

    # Mapeo de nombres de columnas para una mejor visualización

    def map_feature_name(feature_str):
        if feature_str.startswith('num__'):
            original_col = feature_str.replace('num__', '')
            return mapeo_nombres.get(original_col, original_col)
        elif feature_str.startswith('cat__'):
            # Remove the 'cat__' prefix
            temp_str = feature_str.replace('cat__', '')

            # Find the last underscore to separate the original column name from the encoded value
            last_underscore_index = temp_str.rfind('_')

            if last_underscore_index != -1:
                original_cat_col_name = temp_str[:last_underscore_index]
                cat_value = temp_str[last_underscore_index + 1:]

                # Now map the original categorical column name based on common patterns
                if original_cat_col_name == 'gender':
                    return f"Género: {cat_value}"
                elif original_cat_col_name == 'physical_activity_level':
                    return f"Act. Física: {cat_value}"
                elif original_cat_col_name == 'family_history_diabetes':
                    return f"Hist. Fam. Diabetes: {cat_value}"
                # Fallback for other categorical column names, using mapeo_nombres if available
                return f"{mapeo_nombres.get(original_cat_col_name, original_cat_col_name)}: {cat_value}"
            else:
                # This case would handle features like 'cat__some_feature' without a specific value split by '_'
                # For OneHotEncoder output, this is less common but included for robustness.
                return mapeo_nombres.get(temp_str, temp_str) # Try to map the whole string
        return feature_str # Fallback for any other unexpected format

    importance_df['Feature_Mapped'] = importance_df['Feature'].apply(map_feature_name)

    return importance_df, r2, mae, mse, rmse


def entrenar_y_evaluar_clasificador(X, y):
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cambio de solver a 'lbfgs' para soportar multiclase
    pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                random_state=42, 
                solver='lbfgs',      # Soporta multiclase
                max_iter=1000,       # Aumentado para asegurar convergencia
                class_weight='balanced'
            ))
        ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)
    
    # Extraer etiquetas únicas para los ejes de la matriz
    labels = sorted(y.unique())
    
    return acc, report, matrix, labels
