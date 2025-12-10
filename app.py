import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# --- 0. Configuración y Funciones de Limpieza ---

# Configuración de estilo para Matplotlib/Seaborn
sns.set_style("whitegrid")

# Columnas de interés (definidas globalmente)
COLS_INTERES = ['REEL', 'PESO', 'SCT', 'CMT', 'COBB', 'POROSIDAD', 'DOSIFICACIÓN', 'VELOCIDAD', 'ALMIDÓN']

def make_columns_unique_and_clean(df_input):
    """Limpia agresivamente los nombres de columna y garantiza unicidad."""
    df_output = df_input.copy()
    
    # 1. Limpieza agresiva: Eliminar espacios y caracteres invisibles
    df_output.columns = df_output.columns.astype(str).str.strip().str.replace(r'[\n\r]', '', regex=True)

    # 2. Manejo de columnas sin nombre (Unnamed)
    new_cols_dict = {}
    for i, name in enumerate(df_output.columns):
        if name.startswith('Unnamed:') or name == '':
            new_cols_dict[name] = f'Unnamed_{i}'
    df_output.rename(columns=new_cols_dict, inplace=True)
    
    # 3. Desduplicación forzada
    cols = df_output.columns.tolist()
    seen = {}
    new_cols = []
    for col in cols:
        temp_col = col
        if temp_col in seen:
            seen[temp_col] += 1
            temp_col = f'{temp_col}_{seen[temp_col]}'
        else:
            seen[temp_col] = 0
        new_cols.append(temp_col)
    df_output.columns = new_cols
    
    return df_output

@st.cache_data
def load_and_preprocess_data(file_name):
    """Carga y aplica toda la lógica de preprocesamiento y limpieza al DataFrame."""
    try:
        # Carga inicial para detectar encabezado
        temp_df = pd.read_csv(file_name, encoding='latin1', header=None, sep=';', skip_blank_lines=False)
    except Exception:
        temp_df = pd.read_csv(file_name, encoding='utf-8', header=None, sep=';', skip_blank_lines=False)

    # Detección de encabezado
    header_row_index = -1
    for i in range(10):
        if temp_df.iloc[i].astype(str).str.contains('REEL', case=False, na=False).any():
            header_row_index = i
            break

    if header_row_index == -1:
        st.error("No se pudo encontrar la fila de encabezado que contiene 'REEL' en las primeras 10 filas.")
        return pd.DataFrame()

    # Carga final del DataFrame
    df = pd.read_csv(file_name, encoding='latin1', header=header_row_index, sep=';')
    
    # 1. Limpieza de columnas (Desduplicación inicial)
    df = make_columns_unique_and_clean(df)
    df = df.dropna(axis=1, how='all')

    # 2. Renombres y eliminación de columnas no deseadas (Priorizando PESO)
    if 'GRAMAJE' in df.columns:
        df.drop(columns=['GRAMAJE'], inplace=True)
        st.info("Columna 'GRAMAJE' eliminada (se mantiene 'PESO').")
        
    if 'HORA' in df.columns:
        df.rename(columns={'HORA': 'ALMIDÓN'}, inplace=True)
        
    # 3. Segunda limpieza de columnas (Post-rename)
    df = make_columns_unique_and_clean(df)

    # 4. Conversión a numérico (manejo de comas)
    for col in COLS_INTERES:
        if col in df.columns:
            col_array_str = df[col].values.astype(str).flatten() 
            col_series = pd.Series(col_array_str) 
            col_series = col_series.str.replace(',', '.', regex=False).str.strip()
            df[col] = pd.to_numeric(col_series, errors='coerce')

    # 5. Limpieza y manejo de nulos
    df_limpio = df.dropna(subset=['REEL']).copy()
    
    for col in [c for c in COLS_INTERES if c != 'REEL']:
        if col in df_limpio.columns:
            mean_val = df_limpio[col].mean()
            df_limpio.loc[:, col] = df_limpio[col].fillna(mean_val) 

    # 6. Filtrado final y garantía de tipo 1D
    df_analisis = df_limpio[df_limpio['REEL'] > 0].copy()

    for col in df_analisis.columns:
        if col in COLS_INTERES:
            if isinstance(df_analisis[col].values, np.ndarray) and df_analisis[col].values.ndim == 2:
                df_analisis.loc[:, col] = df_analisis[col].values.flatten()
            elif isinstance(df_analisis[col], pd.Series) and df_analisis[col].empty == False:
                 df_analisis.loc[:, col] = df_analisis[col].astype(float)
                 
    return df_analisis

# --- 2. Funciones de Visualización ---

def plot_variation_vs_reel(df, features):
    """Genera gráficos de línea para ver la variación de las propiedades vs. REEL."""
    st.subheader("📈 Variación de Propiedades vs. REEL")
    existing_features = [f for f in features if f in df.columns] 
    if not existing_features: 
        st.warning("No hay características válidas para graficar.")
        return
    
    df_plot = df[['REEL'] + existing_features].copy()
    
    n_features = len(existing_features)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 5 * n_features), sharex=True)
    if n_features == 1: axes = [axes]

    for i, feature in enumerate(existing_features):
        sns.lineplot(x='REEL', y=feature, data=df_plot, ax=axes[i], marker='o', label=feature, color='darkblue')
        axes[i].set_title(f'Variación de {feature} a lo largo de los REELs')
        axes[i].set_ylabel(feature)
        axes[i].grid(axis='y', linestyle='--')

    axes[-1].set_xlabel('REEL')
    plt.tight_layout()
    st.pyplot(fig)
    

def plot_correlation_matrix(df, features):
    """Genera un mapa de calor (heatmap) para visualizar la matriz de correlación."""
    st.subheader("🔥 Matriz de Correlación de Propiedades")
    existing_features = [f for f in features if f in df.columns and df[f].nunique() > 1]
    if len(existing_features) < 2: 
        st.warning("Se requieren al menos dos variables que varíen para calcular la correlación.")
        return
        
    corr_matrix = df[existing_features].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black', 
                cbar_kws={'label': 'Coeficiente de Correlación'}, ax=ax)
    ax.set_title('Matriz de Correlación Ampliada entre las Propiedades del Papel')
    st.pyplot(fig) 

def plot_scatter_relationships(df, x_col, y_cols):
    """Genera gráficos de dispersión para analizar la relación entre una variable y otras."""
    if x_col not in df.columns: return
    existing_y_cols = [y for y in y_cols if y in df.columns and df[y].nunique() > 1]
    if not existing_y_cols: return
        
    st.subheader(f"scatterplot Gráficos de Dispersión vs. **{x_col}**")
    n_plots = len(existing_y_cols)
    cols = 2
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    df_plot = df[[x_col] + existing_y_cols].copy() 

    for i, y_col in enumerate(existing_y_cols):
        sns.scatterplot(x=x_col, y=y_col, data=df_plot, ax=axes[i], alpha=0.7, color='teal')
        # Añadir línea de regresión para visualizar la tendencia lineal
        if df_plot[x_col].nunique() > 1:
            try:
                sns.regplot(x=x_col, y=y_col, data=df_plot, ax=axes[i], scatter=False, color='red', line_kws={'linestyle':'--'})
            except:
                pass 
        axes[i].set_title(f'{y_col} vs. {x_col}')
        axes[i].grid(True, linestyle=':')

    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    st.pyplot(fig)

def plot_histograms(df, features):
    """Genera histogramas para la distribución de propiedades."""
    st.subheader("📊 Distribución de Frecuencia (Histogramas)")
    existing_features = [f for f in features if f in df.columns and df[f].nunique() > 1]
    
    if not existing_features:
        st.warning("No hay variables válidas para generar histogramas.")
        return

    df[existing_features].hist(figsize=(15, 10), bins=10, edgecolor='black', color='skyblue')
    plt.suptitle('Distribución de Frecuencia de las Propiedades del Papel', y=1.02)
    plt.tight_layout()
    st.pyplot(plt.gcf())

def run_regression_analysis(df, properties, process_vars):
    """Ejecuta y muestra el análisis de regresión múltiple."""
    st.header("🔬 Análisis de Regresión Múltiple: Impacto en la Resistencia (SCT)")
    
    model_cols = ['SCT'] + process_vars + [p for p in properties if p != 'SCT']
    model_cols_present = [col for col in model_cols if col in df.columns]
    
    # Usar solo filas sin NaN en las variables del modelo
    model_df = df[model_cols_present].dropna().copy() 
    
    if len(model_cols_present) < 2 or model_df.empty:
        st.warning("No hay suficientes datos limpios o variables para ejecutar la regresión múltiple.")
        return

    # Definir la fórmula del modelo:
    formula_components = [c for c in model_cols_present if c != 'SCT']
    formula = 'SCT ~ ' + ' + '.join(formula_components) 
    st.text(f"Fórmula del modelo: {formula}")

    try:
        # Ajustar el modelo de Mínimos Cuadrados Ordinarios (OLS)
        model = ols(formula, data=model_df).fit()

        st.markdown("### Resumen Estadístico")
        st.code(model.summary().as_text())

        st.markdown("### Interpretación de Coeficientes")
        r_squared_adj = model.rsquared_adj
        st.write(f"**R-cuadrado Ajustado:** `{r_squared_adj:.3f}`")
        st.write(f"  -> Esto significa que el **{r_squared_adj*100:.1f}%** de la variación en SCT es explicada por las variables del modelo.")

        st.markdown("---")

        for var in formula_components:
            coef = model.params.get(var)
            p_val = model.pvalues.get(var)

            if pd.isna(coef) or pd.isna(p_val):
                st.write(f"**Efecto de {var}:** Coeficiente o P-valor no disponible.")
                continue

            col1, col2 = st.columns([1, 2])
            col1.metric(label=f"Coeficiente de {var}", value=f"{coef:.4f}")
            col2.write(f"P-valor: `{p_val:.4f}`")

            if p_val < 0.05:
                st.success(f"**Conclusión:** La **{var}** tiene un impacto estadísticamente **significativo** en el SCT.") 
                st.write(f"  -> Por cada unidad que aumente la **{var}**, el SCT varía en **{coef:.4f}** unidades (manteniendo las otras variables constantes).")
            else:
                st.warning(f"**Conclusión:** La {var} NO tiene un impacto estadísticamente significativo en el SCT (P > 0.05) en este modelo.")

    except Exception as e:
        st.error(f"ERROR: No se pudo ejecutar el modelo de regresión: {e}")

# --- 3. Streamlit App Layout ---

st.title("📄 Análisis de Calidad y Proceso del Papel - RC+5183")
st.markdown("Dashboard Estadístico.")

uploaded_file = st.file_uploader("Sube tu archivo CSV (separado por ';')", type=["csv"])

if uploaded_file is not None:
    # Guardamos el archivo subido localmente para que la función @st.cache_data pueda usarlo.
    # En un entorno de producción, esto debería ser más robusto.
    file_path = "uploaded_data.csv"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    df_analisis = load_and_preprocess_data(file_path)

    if not df_analisis.empty:
        st.success("✅ Datos cargados y preprocesados correctamente.")

        # Definir las variables para los gráficos
        propiedades_papel = ['PESO', 'SCT', 'CMT', 'COBB', 'POROSIDAD']
        variables_proceso = ['DOSIFICACIÓN', 'VELOCIDAD', 'ALMIDÓN'] 
        todas_las_variables = propiedades_papel + variables_proceso

        st.markdown("---")
        
        # 1. Mostrar datos limpios
        st.subheader("Datos de Análisis (Primeras 5 Filas)")
        st.dataframe(df_analisis[[c for c in COLS_INTERES if c in df_analisis.columns]].head())
        
        st.markdown("---")
        st.header("Gráficos de Análisis Exploratorio (EDA)")

        # 2. Variación de las propiedades clave respecto al REEL
        plot_variation_vs_reel(df_analisis, propiedades_papel) 

        # 3. Matriz de correlación ampliada (incluye proceso)
        plot_correlation_matrix(df_analisis, todas_las_variables)

        # 4. Gráficos de dispersión: Proceso vs. Propiedades
        plot_scatter_relationships(df_analisis, 'DOSIFICACIÓN', propiedades_papel)
        plot_scatter_relationships(df_analisis, 'VELOCIDAD', propiedades_papel)
        if 'ALMIDÓN' in df_analisis.columns:
            plot_scatter_relationships(df_analisis, 'ALMIDÓN', propiedades_papel)
        else:
            st.warning("Gráficos de ALMIDÓN omitidos porque la columna no fue encontrada.")

        # 5. Gráficos de dispersión entre pares de propiedades (Peso vs. Propiedades)
        st.markdown("### Relaciones Clave entre Propiedades")
        plot_scatter_relationships(df_analisis, 'PESO', ['SCT', 'CMT', 'COBB']) 
        plot_scatter_relationships(df_analisis, 'SCT', ['CMT', 'POROSIDAD'])

        # 6. Histograma de distribución de propiedades
        plot_histograms(df_analisis, propiedades_papel)

        st.markdown("---")
        
        # 7. Análisis de Regresión Múltiple
        run_regression_analysis(df_analisis, propiedades_papel, variables_proceso)


    else:
        st.error("No se pudo generar el DataFrame de análisis. Por favor, revise el formato de su archivo CSV.")
