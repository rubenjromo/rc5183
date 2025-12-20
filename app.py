import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.formula.api import ols
from tabulate import tabulate

# Configuración inicial y estilo
st.set_page_config(layout="wide", page_title="Análisis de Propiedades del Papel")
sns.set_style("whitegrid")

# Variables globales
COLUMNS_OF_INTEREST = ['REEL', 'PESO', 'SCT', 'CMT', 'MULLEN', 'COBB', 'POROSIDAD',
                       'DOSIFICACIÓN', 'VELOCIDAD', 'ALMIDÓN',
                       'LABIO', 'CHORRO', 'COLUMNA', 'GRAMAJE']
PROPIEDADES_PAPEL = ['PESO', 'SCT', 'CMT', 'MULLEN', 'COBB', 'POROSIDAD']
VARIABLES_PROCESO = ['DOSIFICACIÓN', 'VELOCIDAD', 'ALMIDÓN']
VARIABLES_NUEVAS = ['LABIO', 'CHORRO', 'COLUMNA']
TODAS_LAS_VARIABLES = PROPIEDADES_PAPEL + VARIABLES_PROCESO + VARIABLES_NUEVAS

# ==============================================================================
# 1. FUNCIONES DE CARGA Y PREPROCESAMIENTO
# ==============================================================================

@st.cache_data
def make_columns_unique_and_clean(df_input):
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
            temp_col = f'{temp_col}_{seen[temp_col]}'
        else:
            seen[temp_col] = 0
        new_cols.append(temp_col)
    df_output.columns = new_cols
    return df_output

@st.cache_data(show_spinner="Cargando datos...")
def load_and_preprocess_data(uploaded_file):
    try:
        uploaded_file.seek(0)
        temp_df = pd.read_csv(uploaded_file, encoding='latin1', header=None, sep=';', nrows=10)
        header_row_index = -1
        for i in range(len(temp_df)):
            if not temp_df.iloc[i].isnull().all():
                if temp_df.iloc[i].iloc[:15].astype(str).str.contains('REEL', case=False, na=False).any():
                    header_row_index = i
                    break

        if header_row_index == -1:
            return None, None, None

        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin1', header=header_row_index, sep=';')
        df = make_columns_unique_and_clean(df)
        df = df.dropna(axis=1, how='all')

        if 'HORA' in df.columns:
            df.rename(columns={'HORA': 'ALMIDÓN'}, inplace=True)
        
        df = make_columns_unique_and_clean(df)

        # Conversión numérica
        for col in [c for c in COLUMNS_OF_INTEREST if c != 'GRAMAJE']:
            if col in df.columns:
                col_array_str = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                df[col] = pd.to_numeric(col_array_str, errors='coerce')

        df_limpio = df.dropna(subset=['REEL']).copy()
        df_limpio.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_analisis = df_limpio[df_limpio['REEL'] > 0].copy()

        # Limpieza de GRAMAJE para asegurar que el filtro funcione bien
        if 'GRAMAJE' in df_analisis.columns:
            df_analisis['GRAMAJE'] = df_analisis['GRAMAJE'].astype(str).str.strip()

        return df_analisis
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ==============================================================================
# FUNCIONES DE APOYO PARA PESTAÑAS (Mantenidas según código original)
# ==============================================================================

def plot_distribution_histograms(df):
    cols_for_hist = [c for c in PROPIEDADES_PAPEL + VARIABLES_NUEVAS if c in df.columns and df[c].dtype in ['float64', 'int64']]
    if cols_for_hist:
        num_cols = len(cols_for_hist)
        fig_cols = min(3, num_cols)
        fig_rows = int(np.ceil(num_cols / fig_cols))
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(4 * fig_cols, 4 * fig_rows)) 
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        df.loc[:, cols_for_hist].hist(bins=10, edgecolor='black', color='skyblue', ax=axes[:num_cols])
        for i in range(num_cols, len(axes)): fig.delaxes(axes[i])
        plt.suptitle('Distribución de Frecuencia', y=1.02, fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

def display_dataframe_tab(df_analisis):
    st.header("Base de Datos")
    col1, col2 = st.columns(2)
    cols_to_show = [c for c in COLUMNS_OF_INTEREST if c in df_analisis.columns]
    with col1: st.markdown("### Primeras Filas"); st.dataframe(df_analisis[cols_to_show].head(), use_container_width=True)
    with col2: st.markdown("### Últimas Filas"); st.dataframe(df_analisis[cols_to_show].tail(), use_container_width=True)
    st.markdown("---")
    plot_distribution_histograms(df_analisis)

def display_correlation_tab(df):
    st.header("Matriz de Correlación")
    existing_features = [f for f in TODAS_LAS_VARIABLES if f in df.columns and df[f].nunique() > 1]
    if len(existing_features) < 2: return
    corr_matrix = df[existing_features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    plt.close(fig)

def display_reel_vs_tab(df):
    st.header("Variación vs. REEL")
    existing_features = [f for f in PROPIEDADES_PAPEL if f in df.columns]
    n_features = len(existing_features)
    if n_features == 0: return
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features), sharex=True)
    if n_features == 1: axes = [axes]
    for i, feature in enumerate(existing_features):
        sns.lineplot(x='REEL', y=feature, data=df, ax=axes[i], marker='o', color='darkblue')
        axes[i].set_ylabel(feature)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_scatter_relationships_for_tab(df, x_col, y_cols):
    existing_y_cols = [y for y in y_cols if y in df.columns and df[y].nunique() > 1]
    if x_col not in df.columns or not existing_y_cols: return
    st.markdown(f"#### Dispersión: Propiedades vs. `{x_col}`")
    n_plots = len(existing_y_cols)
    cols = 2
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten()
    for i, y_col in enumerate(existing_y_cols):
        sns.scatterplot(x=x_col, y=y_col, data=df, ax=axes[i], color='teal')
        try: sns.regplot(x=x_col, y=y_col, data=df, ax=axes[i], scatter=False, color='red', line_kws={'linestyle':'--'})
        except: pass
        axes[i].set_title(f'{y_col} vs {x_col}')
    for i in range(n_plots, len(axes)): fig.delaxes(axes[i])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def display_scatter_tab(df):
    st.header("Gráficos de Dispersión")
    for var in ['DOSIFICACIÓN', 'VELOCIDAD', 'ALMIDÓN', 'LABIO', 'CHORRO', 'COLUMNA']:
        if var in df.columns: plot_scatter_relationships_for_tab(df, var, PROPIEDADES_PAPEL)
    plot_scatter_relationships_for_tab(df, 'PESO', ['SCT', 'CMT', 'MULLEN', 'COBB'])

def display_boxplots_tab(df):
    st.header("Boxplots por Gramaje")
    if 'GRAMAJE' not in df.columns: return
    props = ['MULLEN', 'SCT', 'CMT', 'POROSIDAD']
    for prop in [p for p in props if p in df.columns]:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x='GRAMAJE', y=prop, data=df, palette='viridis', ax=ax)
        sns.swarmplot(x='GRAMAJE', y=prop, data=df, color='black', size=3, alpha=0.5, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

def run_ols_analysis_clean(df, dependent_var):
    model_cols = [dependent_var, 'DOSIFICACIÓN', 'VELOCIDAD', 'PESO', 'ALMIDÓN', 'LABIO', 'CHORRO', 'COLUMNA']
    model_cols_present = [col for col in model_cols if col in df.columns and df[col].nunique() > 1]
    if len(model_cols_present) < 2: return
    formula = f'{dependent_var} ~ ' + ' + '.join([c for c in model_cols_present if c != dependent_var])
    try:
        model = ols(formula, data=df).fit()
        st.markdown(f"### Regresión: {dependent_var}")
        st.write(f"**R-cuadrado Adj:** {model.rsquared_adj:.3f} | **Observaciones:** {int(model.nobs)}")
        # Simplificación de la tabla para visualización rápida
        coeffs = pd.DataFrame({'Coef': model.params, 'P-valor': model.pvalues}).round(4)
        st.table(coeffs)
    except: st.warning(f"No se pudo calcular OLS para {dependent_var}")

def display_regression_tab(df):
    st.header("Modelos de Regresión (OLS)")
    for prop in ['SCT', 'CMT', 'MULLEN']:
        if prop in df.columns: run_ols_analysis_clean(df, prop)

def display_averages_tab(df):
    st.header("Promedios por Gramaje")
    if 'GRAMAJE' in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        avg_df = df.groupby('GRAMAJE')[numeric_cols].mean().round(2)
        st.dataframe(avg_df)

# ==============================================================================
# 3. FUNCIÓN PRINCIPAL CON FILTRO DE GRAMAJE
# ==============================================================================

def main():
    st.title("Análisis Exploratorio y Regresión - Propiedades del Papel")
    st.markdown("Pruebas con RC+5183")

    with st.sidebar:
        st.header("⚙️ Configuración")
        uploaded_file = st.file_uploader("Subir archivo CSV", type="csv")
    
    if uploaded_file:
        df_full = load_and_preprocess_data(uploaded_file)
        
        if df_full is not None:
            # --- NUEVA LÓGICA DE FILTRO POR GRAMAJE ---
            gramajes_disponibles = sorted(df_full['GRAMAJE'].unique())
            opciones_filtro = ["TODO"] + list(gramajes_disponibles)
            
            with st.sidebar:
                st.subheader("Filtro de Análisis")
                seleccion = st.selectbox("Seleccione el Gramaje a analizar:", opciones_filtro)
            
            # Aplicar filtro al DataFrame principal
            if seleccion == "TODO":
                df_filtrado = df_full.copy()
                st.sidebar.success(f"Analizando base completa: {len(df_filtrado)} filas.")
            else:
                df_filtrado = df_full[df_full['GRAMAJE'] == seleccion].copy()
                st.sidebar.success(f"Filtrado por Gramaje {seleccion}: {len(df_filtrado)} filas.")

            # Preparar datos para OLS y Scatter (Imputación por media)
            cols_to_impute = [c for c in COLUMNS_OF_INTEREST if c in df_filtrado.columns and c not in ['GRAMAJE', 'REEL']]
            df_procesado = df_filtrado.copy()
            for col in cols_to_impute:
                df_procesado[col] = df_procesado[col].fillna(df_procesado[col].mean())

            # --- TABS ---
            tabs = st.tabs(["📋 Base", "🔗 Correlación", "📈 Variación", "⚫ Dispersión", "📦 Boxplots", "🔬 OLS", "🔢 Promedios"])
            
            with tabs[0]: display_dataframe_tab(df_filtrado)
            with tabs[1]: display_correlation_tab(df_procesado)
            with tabs[2]: display_reel_vs_tab(df_filtrado)
            with tabs[3]: display_scatter_tab(df_procesado)
            with tabs[4]: display_boxplots_tab(df_full) # El boxplot usa el Full para comparar
            with tabs[5]: display_regression_tab(df_procesado)
            with tabs[6]: display_averages_tab(df_full)

    else:
        st.info("Favor subir el archivo CSV en la barra lateral.")

if __name__ == '__main__':
    main()
