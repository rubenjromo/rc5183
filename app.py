import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np

# Configuración inicial de la página
st.set_page_config(
    page_title="Análisis de Calidad de Papel y Proceso RC+5183",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de estilo de gráficos (Seaborn/Matplotlib)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (9, 6)

# --- Funciones de Preprocesamiento de Datos ---

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """
    Carga, detecta el encabezado y preprocesa el archivo CSV, renombrando 'HORA' a 'ALMIDÓN'.
    """
    if uploaded_file is None:
        return None
        
    try:
        # Intento 1: Detección de encabezado
        temp_df = pd.read_csv(uploaded_file, encoding='latin1', header=None, sep=';', skip_blank_lines=False)
    except Exception:
        # Intento 2: Usar UTF-8
        uploaded_file.seek(0) # Resetear puntero
        temp_df = pd.read_csv(uploaded_file, encoding='utf-8', header=None, sep=';', skip_blank_lines=False)

    # Buscar la fila de encabezado
    header_row_index = -1
    for i in range(10):
        if temp_df.iloc[i].astype(str).str.contains('REEL', case=False, na=False).any():
            header_row_index = i
            break

    if header_row_index == -1:
        st.error("❌ Error: No se pudo encontrar la fila de encabezado que contiene 'REEL' en las primeras 10 filas.")
        return None
        
    uploaded_file.seek(0) # Resetear puntero para la carga final
    try:
        df = pd.read_csv(uploaded_file, encoding='latin1', header=header_row_index, sep=';')
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-8', header=header_row_index, sep=';')
    
    # Limpieza inicial de columnas
    df.columns = df.columns.str.strip()
    df.columns = [f'Unnamed_{i}' if name.startswith('Unnamed:') or name == '' else name for i, name in enumerate(df.columns)]
    df = df.dropna(axis=1, how='all')

    # *** CAMBIO CLAVE: Renombrar la columna 'HORA' a 'ALMIDÓN' ***
    if 'HORA' in df.columns:
        df.rename(columns={'HORA': 'ALMIDÓN'}, inplace=True)
    
    # Columnas de interés actualizadas
    cols_interes = ['REEL', 'GRAMAJE', 'SCT', 'CMT', 'COBB', 'POROSIDAD', 'DOSIFICACIÓN', 'VELOCIDAD', 'ALMIDÓN']

    # Conversión a numérico y manejo de comas
    for col in cols_interes:
        if col in df.columns:
            if df[col].dtype == 'object':
                 df[col] = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Limpieza de filas y tratamiento de NaNs
    df_limpio = df.dropna(subset=['REEL']).copy()
    
    # Rellenar NaN en propiedades con la media (incluyendo 'ALMIDÓN')
    for col in ['GRAMAJE', 'SCT', 'CMT', 'COBB', 'POROSIDAD', 'DOSIFICACIÓN', 'VELOCIDAD', 'ALMIDÓN']: 
        if col in df_limpio.columns:
            df_limpio[col].fillna(df_limpio[col].mean(), inplace=True)

    df_analisis = df_limpio[df_limpio['REEL'] > 0].copy()

    return df_analisis

# --- Funciones de Visualización ---

def plot_variation_vs_reel(df, features):
    """Genera gráficos de línea para ver la variación de las propiedades vs. REEL."""
    st.subheader("1. Variación de Propiedades Clave vs. REEL")
    existing_features = [f for f in features if f in df.columns] 
    if not existing_features: 
        st.warning("No se encontraron propiedades de papel para este gráfico.")
        return

    n_features = len(existing_features)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 5 * n_features), sharex=True)
    if n_features == 1: axes = [axes]

    for i, feature in enumerate(existing_features):
        sns.lineplot(x='REEL', y=feature, data=df, ax=axes[i], marker='o', label=feature, color='darkblue')
        axes[i].set_title(f'Variación de {feature} a lo largo de los REELs')
        axes[i].set_ylabel(feature)
        axes[i].grid(axis='y', linestyle='--')

    axes[-1].set_xlabel('REEL')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_correlation_matrix(df, features):
    """Genera un mapa de calor (heatmap) para visualizar la matriz de correlación."""
    st.subheader("2. Matriz de Correlación Ampliada (Propiedades y Proceso)")
    existing_features = [f for f in features if f in df.columns and df[f].nunique() > 1]
    if len(existing_features) < 2: 
        st.warning("No hay suficientes variables para calcular la correlación.")
        return

    corr_matrix = df[existing_features].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black', 
                cbar_kws={'label': 'Coeficiente de Correlación'}, ax=ax)
    ax.set_title('Matriz de Correlación Ampliada entre las Propiedades del Papel')
    st.pyplot(fig)
    plt.close(fig)

def plot_scatter_relationships(df, x_col, y_cols, plot_number, custom_title=None):
    """Genera gráficos de dispersión para analizar la relación entre x_col y varias y_cols."""
    
    if x_col not in df.columns: return
    existing_y_cols = [y for y in y_cols if y in df.columns and df[y].nunique() > 1]
    if not existing_y_cols: return

    title = custom_title if custom_title else f"Análisis de Propiedades vs. {x_col}"
    st.subheader(f"{plot_number}. {title}")
    
    n_plots = len(existing_y_cols)
    cols = 2
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, y_col in enumerate(existing_y_cols):
        sns.scatterplot(x=x_col, y=y_col, data=df, ax=axes[i], alpha=0.7, color='teal')
        # Añadir línea de regresión
        if df[x_col].nunique() > 1:
            try:
                sns.regplot(x=x_col, y=y_col, data=df, ax=axes[i], scatter=False, color='red', line_kws={'linestyle':'--'})
            except:
                pass 
        axes[i].set_title(f'{y_col} vs. {x_col}')
        axes[i].grid(True, linestyle=':')

    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_histograms(df, features, plot_number):
    """Genera histogramas de distribución de propiedades."""
    existing_features = [f for f in features if f in df.columns]
    if not existing_features: 
        st.warning("No se encontraron propiedades para generar los histogramas.")
        return

    st.subheader(f"{plot_number}. Distribución de las Propiedades del Papel (Histogramas)")
    # Crear la figura para el histograma
    fig = plt.figure(figsize=(15, 10))
    # Usamos ax=fig.gca() para asegurar que hist funcione en Streamlit
    df[existing_features].hist(ax=fig.gca(), bins=10, edgecolor='black', color='skyblue') 
    plt.suptitle('Distribución de Frecuencia de las Propiedades del Papel', y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --- Función de Regresión Múltiple (Añadiendo ALMIDÓN) ---

def run_regression_analysis(df):
    """Ejecuta y muestra el análisis de regresión múltiple para SCT."""
    st.header("📊 Análisis de Regresión Múltiple: Impacto en la Resistencia (SCT) 📉")
    
    # Ahora incluye ALMIDÓN en las variables del modelo
    model_cols = ['SCT', 'DOSIFICACIÓN', 'VELOCIDAD', 'GRAMAJE', 'ALMIDÓN']
    model_cols_present = [col for col in model_cols if col in df.columns]
    
    if 'SCT' not in model_cols_present:
        st.error("Error: La variable dependiente 'SCT' no está disponible para la regresión.")
        return

    model_df = df[model_cols_present].dropna()

    if len(model_cols_present) < 2 or model_df.empty:
        st.warning("No hay suficientes datos limpios o variables (DOSIFICACIÓN, VELOCIDAD, GRAMAJE, ALMIDÓN) para ejecutar la regresión múltiple.")
        return

    # Definimos la fórmula del modelo: SCT ~ DOSIFICACIÓN + VELOCIDAD + GRAMAJE + ALMIDÓN
    formula_components = [c for c in model_cols_present if c != 'SCT']
    formula = 'SCT ~ ' + ' + '.join(formula_components)

    try:
        model = ols(formula, data=model_df).fit()
        
        st.subheader("Resumen Estadístico del Modelo")
        st.code(model.summary().as_text())

        st.subheader("Interpretación de Coeficientes Clave")

        # Coeficiente R-cuadrado ajustado
        r_squared_adj = model.rsquared_adj
        st.metric(label="R-cuadrado Ajustado", value=f"{r_squared_adj:.3f}", 
                  help=f"El {r_squared_adj*100:.1f}% de la variación en SCT es explicada por el modelo.")
        st.markdown("---")

        # Conclusiones individuales para cada predictor
        for var in formula_components:
            coef = model.params.get(var)
            p_val = model.pvalues.get(var)

            if pd.isna(coef) or pd.isna(p_val):
                st.info(f"**Efecto de {var}:** Coeficiente o P-valor no disponible.")
                continue

            st.markdown(f"**Efecto de {var}:**")
            col1, col2 = st.columns(2)
            col1.metric("Coeficiente", f"{coef:.4f}")
            col2.metric("P-valor", f"{p_val:.4f}")

            if p_val < 0.05:
                st.success(f"La **{var}** tiene un impacto estadísticamente **significativo** en el SCT.")
                st.markdown(f"*(Interpretación: Por cada unidad de aumento en {var}, el SCT varía en **{coef:.4f}** unidades, manteniendo las otras variables constantes.)*")
            else:
                st.info(f"La **{var}** NO tiene un impacto estadísticamente significativo en el SCT (P > 0.05) en este modelo.")
            
            st.markdown("---")
            
    except Exception as e:
        st.error(f"⚠️ Error al ejecutar el modelo de regresión. Asegúrese de que las columnas tengan datos válidos. Detalle: {e}")


# --- Cuerpo Principal de la Aplicación ---

def main():
    st.title(" Análisis de Calidad y Proceso de Fabricación de Papel - RC+5183")
    st.markdown("Cargue el archivo CSV de pruebas.")

    # 1. Carga de Archivo
    uploaded_file = st.sidebar.file_uploader("Sube el archivo CSV con separador ';'", type="csv")

    if uploaded_file is not None:
        df_analisis = load_and_preprocess_data(uploaded_file)
        
        if df_analisis is None:
            return

        st.success("✅ Datos cargados y preprocesados con éxito.")

        # Variables de interés
        propiedades_papel = ['GRAMAJE', 'SCT', 'CMT', 'COBB', 'POROSIDAD']
        variables_proceso = ['DOSIFICACIÓN', 'VELOCIDAD', 'ALMIDÓN'] # ALMIDÓN incluido
        todas_las_variables = propiedades_papel + variables_proceso

        # 2. Análisis Exploratorio de Datos (EDA)
        st.header("🔍 Análisis Exploratorio de Datos (EDA)")
        
        # Mostrar datos
        if st.checkbox("Mostrar Primeras Filas del DataFrame Limpio y Dimensiones"):
            cols_to_show = [c for c in todas_las_variables + ['REEL'] if c in df_analisis.columns]
            st.dataframe(df_analisis[cols_to_show].head())
            st.write(f"Filas totales en el dataset de análisis: **{len(df_analisis)}**")
            st.markdown("---")


        # --- Generación de Gráficos (Manteniendo el orden y añadiendo ALMIDÓN) ---

        # 1. Variación de las propiedades clave respecto al REEL (Original 1)
        plot_variation_vs_reel(df_analisis, propiedades_papel)
        st.markdown("---")

        # 2. Matriz de correlación ampliada (incluye proceso y ALMIDÓN) (Original 2)
        plot_correlation_matrix(df_analisis, todas_las_variables)
        st.markdown("---")

        # 3. Análisis de las variables de Proceso vs. Propiedades (DOSIFICACIÓN) (Original 3)
        plot_scatter_relationships(df_analisis, 'DOSIFICACIÓN', propiedades_papel, "3. Análisis de Propiedades vs. DOSIFICACIÓN")
        st.markdown("---")

        # 4. Análisis de las variables de Proceso vs. Propiedades (VELOCIDAD) (Original 4)
        plot_scatter_relationships(df_analisis, 'VELOCIDAD', propiedades_papel, "4. Análisis de Propiedades vs. VELOCIDAD")
        st.markdown("---")

        # 5. Impacto de ALMIDÓN vs. Propiedades de Calidad (NUEVO GRÁFICO)
        plot_scatter_relationships(df_analisis, 'ALMIDÓN', propiedades_papel, "5. Impacto de ALMIDÓN vs. Propiedades de Calidad")
        st.markdown("---")


        # 6a. Gráficos de dispersión: GRAMAJE vs... (Parte del Original 5/6)
        plot_scatter_relationships(df_analisis, 'GRAMAJE', ['SCT', 'CMT', 'COBB'], "6a. Relación de GRAMAJE con Resistencia y Absorción")
        # 6b. Gráficos de dispersión: SCT vs... (Parte del Original 5/6)
        plot_scatter_relationships(df_analisis, 'SCT', ['CMT', 'POROSIDAD'], "6b. Relación de SCT con CMT y Porosidad")
        st.markdown("---")

        # 7. Histograma de distribución de propiedades (Original 7)
        plot_histograms(df_analisis, propiedades_papel, "7")
        st.markdown("---")


        # 3. Análisis de Regresión Múltiple (Actualizado con ALMIDÓN)
        run_regression_analysis(df_analisis)

    else:
        st.info("Esperando la carga del archivo CSV. Por favor, suba el archivo en la barra lateral izquierda para comenzar el análisis.")

if __name__ == "__main__":
    main()
