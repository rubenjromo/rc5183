import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import streamlit as st
import os

# --- 0. Configuración Inicial y Constantes ---
st.set_page_config(layout="wide", page_title="Análisis RC 5183: Papel")
sns.set_style("whitegrid")

COL_INTERES = ['REEL', 'PESO', 'SCT', 'CMT', 'COBB', 'POROSIDAD',
               'DOSIFICACIÓN', 'VELOCIDAD', 'ALMIDÓN',
               'LABIO', 'CHORRO', 'COLUMNA']

# --- 1. Funciones de Carga y Preprocesamiento ---

def make_columns_unique_and_clean(df_input):
    """Limpia y garantiza la unicidad de los nombres de columna."""
    df_output = df_input.copy()
    df_output.columns = df_output.columns.astype(str).str.strip().str.replace(r'[\n\r]', '', regex=True)
    
    new_cols_dict = {}
    for i, name in enumerate(df_output.columns):
        if name.startswith('Unnamed:') or name == '':
            new_cols_dict[name] = f'Unnamed_{i}'
    df_output.rename(columns=new_cols_dict, inplace=True)
    
    cols = df_output.columns.tolist()
    seen = {}
    new_cols = []
    for col in cols:
        temp_col = col
        if temp_col in seen:
            seen[temp_col] += 1
            temp_col = f'{col}_{seen[temp_col]}'
        else:
            seen[temp_col] = 0
            
        # Limpieza para que sean nombres válidos en Python (sin espacios ni puntos)
        new_cols.append(temp_col.replace(' ', '_').replace('.', '_').replace('-', '_')) 
    df_output.columns = new_cols
    return df_output

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """Carga y aplica todo el preprocesamiento de datos desde un archivo subido."""
    
    # 1. Detección de encabezado ("REEL")
    try:
        uploaded_file.seek(0)
        temp_df = pd.read_csv(uploaded_file, encoding='latin1', header=None, sep=';', skip_blank_lines=False)
        uploaded_file.seek(0)
    except Exception:
        st.error("Error al intentar leer el archivo con codificación 'latin1' y separador ';'.")
        return pd.DataFrame() 
    
    header_row_index = -1
    for i in range(10):
        if temp_df.iloc[i].astype(str).str.contains('REEL', case=False, na=False).any():
            header_row_index = i
            break

    if header_row_index == -1:
        st.warning("No se pudo encontrar la fila de encabezado que contiene 'REEL' en las primeras 10 filas.")
        return pd.DataFrame()

    # 2. Carga final
    uploaded_file.seek(0) 
    df = pd.read_csv(uploaded_file, encoding='latin1', header=header_row_index, sep=';')
    
    # 3. Limpieza de columnas
    df = make_columns_unique_and_clean(df)
    df = df.dropna(axis=1, how='all')

    # 4. Renombres y eliminaciones específicas
    if 'GRAMAJE' in df.columns:
        df.drop(columns=['GRAMAJE'], inplace=True)
        
    if 'HORA' in df.columns:
        df.rename(columns={'HORA': 'ALMIDÓN'}, inplace=True)
    
    df = make_columns_unique_and_clean(df) # Segunda limpieza

    # 5. Conversión numérica (coma a punto y forzar float)
    df_limpio = df.copy()
    for col in COL_INTERES:
        if col in df_limpio.columns:
            col_series = pd.Series(df_limpio[col].values.astype(str).flatten())
            col_series = col_series.str.replace(',', '.', regex=False).str.strip()
            df_limpio.loc[:, col] = pd.to_numeric(col_series, errors='coerce')

    # 6. Limpieza final de filas y filtrado
    df_limpio = df_limpio.dropna(subset=['REEL']).copy()
    
    # Relleno de NaN con la media (solo en columnas que existen)
    for col in [c for c in COL_INTERES if c in df_limpio.columns]:
        mean_val = df_limpio[col].mean()
        df_limpio.loc[:, col] = df_limpio[col].fillna(mean_val) 

    df_analisis = df_limpio[df_limpio['REEL'] > 0].copy()
            
    return df_analisis

# --- 2. Funciones de Visualización ---

def plot_variation_vs_reel(df, features):
    st.subheader("1. Variación de Propiedades vs. REEL")
    existing_features = [f for f in features if f in df.columns] 
    if not existing_features: return

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
    st.subheader("2. Matriz de Correlación Ampliada 🌡️")
    existing_features = [f for f in features if f in df.columns and df[f].nunique() > 1]
    if len(existing_features) < 2: 
        st.warning("No hay suficientes variables con variación para calcular la correlación.")
        return
        
    corr_matrix = df[existing_features].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black', 
                cbar_kws={'label': 'Coeficiente de Correlación'}, ax=ax)
    ax.set_title('Matriz de Correlación entre Propiedades y Variables de Proceso')
    st.pyplot(fig)
    plt.close(fig)
    

def plot_scatter_relationships(df, x_col, y_cols):
    existing_y_cols = [y for y in y_cols if y in df.columns and df[y].nunique() > 1]
    if x_col not in df.columns or not existing_y_cols: return
        
    st.subheader(f"Gráficos de Dispersión: {x_col} vs. Propiedades")
    n_plots = len(existing_y_cols)
    cols = 2
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, y_col in enumerate(existing_y_cols):
        sns.scatterplot(x=x_col, y=y_col, data=df, ax=axes[i], alpha=0.7, color='teal')
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

def plot_histograms(df, features, title):
    st.subheader(title)
    existing_features = [f for f in features if f in df.columns]
    if not existing_features: return
    
    fig = plt.figure(figsize=(15, 5))
    df[existing_features].hist(ax=fig.gca(), bins=10, edgecolor='black', color='skyblue')
    plt.suptitle(title, y=1.05)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)
    plt.close(fig)


# --- 3. Función de Regresión ---

def run_regression_model(df, target_col, predictors):
    
    st.header(f"📈 Modelo de Regresión Múltiple para: **{target_col}**")
    
    model_cols = [target_col] + predictors
    model_cols_present = [col for col in model_cols if col in df.columns]
    
    if target_col not in df.columns:
        st.error(f"La columna objetivo '{target_col}' no se encontró en los datos.")
        return
        
    model_df = df[model_cols_present].dropna().copy()
    model_df = model_df.loc[:, (model_df.nunique() > 1)]

    target_col_clean = [col for col in model_df.columns if col == target_col][0]
    formula_components = [c for c in model_df.columns if c != target_col_clean]
    
    if not formula_components:
        st.warning(f"ADVERTENCIA: No hay suficientes variables predictoras (todas tienen un solo valor) para ejecutar la regresión para {target_col}.")
        return

    formula = f'{target_col_clean} ~ ' + ' + '.join(formula_components)
    st.markdown(f"**Fórmula del modelo:** `{formula}`")

    try:
        model = ols(formula, data=model_df).fit()

        st.subheader("Resumen Estadístico del Modelo")
        st.text(model.summary()) 

        st.markdown("### Interpretación de Resultados Clave")
        
        # Interpretación R-cuadrado
        r_squared_adj = model.rsquared_adj
        st.info(f"**R-cuadrado Ajustado:** **{r_squared_adj:.3f}**\n\n-> Esto significa que el **{r_squared_adj*100:.1f}%** de la variación en **{target_col_clean}** es explicada por las variables del modelo.")

        st.markdown("---")
        
        st.subheader("Efecto de cada Variable Predictora:")
        
        for var in formula_components:
            coef = model.params.get(var)
            p_val = model.pvalues.get(var)

            if pd.isna(coef) or pd.isna(p_val) or p_val > 0.999:
                st.markdown(f"**{var}:** Coeficiente o P-valor no disponible (Posible multicolinealidad o cero varianza).")
                continue

            col1, col2 = st.columns([1, 4])
            with col1:
                st.metric(label=f"Coeficiente de {var}", value=f"{coef:.4f}")
            with col2:
                if p_val < 0.05:
                    st.success(f"**IMPACTO SIGNIFICATIVO (P-valor: {p_val:.4f})**")
                    st.write(f"Por cada unidad que aumentes la **{var}**, el **{target_col_clean}** varía en **{coef:.4f}** unidades (manteniendo las otras variables constantes).")
                else:
                    st.warning(f"**NO SIGNIFICATIVO (P-valor: {p_val:.4f})**")
                    st.write(f"La {var} NO muestra un impacto estadísticamente significativo en {target_col_clean} en este modelo.")

    except Exception as e:
        st.error(f"ERROR al ejecutar el modelo de regresión: {e}")


# --- 4. Streamlit App Layout (Función Principal) ---

def main():
    st.title("Análisis de Propiedades del Papel y Variables de Proceso - RC+5183")
    
    # 🌟 Permitir subir archivo (CORRECCIÓN CLAVE)
    uploaded_file = st.sidebar.file_uploader("Cargar archivo csv", type=["csv"])

    if uploaded_file is not None:
        st.sidebar.success("Archivo cargado correctamente. Procesando...")
        df_analisis = load_and_preprocess_data(uploaded_file)
    else:
        st.info("Esperando carga de base de datos...")
        return 

    if df_analisis.empty:
        st.error("El DataFrame resultante está vacío o el encabezado no se pudo detectar correctamente.")
        st.stop()
        
    # Definición de variables
    propiedades_papel = ['PESO', 'SCT', 'CMT', 'COBB', 'POROSIDAD']
    variables_proceso_base = ['DOSIFICACIÓN', 'VELOCIDAD', 'ALMIDÓN']
    variables_nuevas = ['LABIO', 'CHORRO', 'COLUMNA']
    todas_las_variables = propiedades_papel + variables_proceso_base + variables_nuevas
    
    st.sidebar.header("Opciones de Análisis")
    
    # Mostrar datos
    if st.sidebar.checkbox("Mostrar Datos Preprocesados"):
        st.header("🔍 Vista Previa de los Datos")
        st.dataframe(df_analisis[[c for c in todas_las_variables if c in df_analisis.columns]].head(10))
        st.write(f"Total de filas para análisis: {len(df_analisis)}")
        st.write(f"Columnas detectadas y limpiadas: {', '.join(df_analisis.columns.tolist())}")

    # --- Sección de Gráficos ---
    st.sidebar.markdown("---")
    st.sidebar.header("Visualizaciones")

    with st.container():
        st.header("Gráficos de Variación, Correlación y Distribución")

        plot_variation_vs_reel(df_analisis, propiedades_papel)
        
        plot_correlation_matrix(df_analisis, todas_las_variables)

        plot_histograms(df_analisis, propiedades_papel, '3. Distribución de Frecuencia de las Propiedades del Papel')
      
        plot_histograms(df_analisis, variables_nuevas, '4. Distribución de Frecuencia de LABIO, CHORRO y COLUMNA')

    # --- Gráficos de Dispersión Dinámicos ---
    st.header("Relaciones de Dispersión entre Variables")
    
    col_x = st.selectbox("Seleccionar Variable Independiente (Eje X) para Dispersión:", 
                         [c for c in todas_las_variables if c in df_analisis.columns])
    plot_scatter_relationships(df_analisis, col_x, propiedades_papel)


    # --- Sección de Regresión ---
    st.sidebar.markdown("---")
    st.sidebar.header("Modelos de Regresión")
    
    reg_option = st.sidebar.selectbox(
        "Seleccionar Variable Dependiente (Y):",
        ['SCT', 'CMT', 'COBB']
    )
    
    predictors = [c for c in ['DOSIFICACIÓN', 'VELOCIDAD', 'ALMIDÓN', 'PESO', 'LABIO', 'CHORRO', 'COLUMNA'] if c in df_analisis.columns]
    
    run_regression_model(df_analisis, reg_option, predictors)

if __name__ == "__main__":
    main()
