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
        if header_row_index == -1: return None
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin1', header=header_row_index, sep=';')
        df = make_columns_unique_and_clean(df)
        if 'HORA' in df.columns: df.rename(columns={'HORA': 'ALMIDÓN'}, inplace=True)
        for col in [c for c in COLUMNS_OF_INTEREST if c != 'GRAMAJE']:
            if col in df.columns:
                col_array_str = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                df[col] = pd.to_numeric(col_array_str, errors='coerce')
        df_limpio = df.dropna(subset=['REEL']).copy()
        df_analisis = df_limpio[df_limpio['REEL'] > 0].copy()
        if 'GRAMAJE' in df_analisis.columns:
            df_analisis['GRAMAJE'] = df_analisis['GRAMAJE'].astype(str).str.strip()
        return df_analisis
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ==============================================================================
# 2. FUNCIONES DE REGRESIÓN (RESTAURADAS)
# ==============================================================================

def run_ols_analysis_clean(df, dependent_var):
    """Ejecuta el análisis OLS completo con formato detallado."""
    model_cols = [dependent_var, 'DOSIFICACIÓN', 'VELOCIDAD', 'PESO', 'ALMIDÓN',
                  'LABIO', 'CHORRO', 'COLUMNA']
    model_cols_present = [col for col in model_cols if col in df.columns and df[col].nunique() > 1]
    
    model_df = df[model_cols_present].dropna().copy()

    if len(model_cols_present) < 2 or model_df.empty:
        st.warning(f"No hay suficientes datos para la regresión múltiple de **{dependent_var}**.")
        return

    formula_components = [c for c in model_cols_present if c != dependent_var]
    formula = f'{dependent_var} ~ ' + ' + '.join(formula_components)

    try:
        model = ols(formula, data=model_df).fit()

        st.markdown("\n" + "-" * 30)
        st.markdown(f"## Análisis de Regresión Múltiple: {dependent_var} (OLS)")
        st.markdown(f"**Expresión:** `{formula}`")

        # 1. Resumen General
        st.markdown("### Resumen del Modelo")
        metrics = [
            ["**R-cuadrado Ajustado**", f"{model.rsquared_adj:.3f}", f"{model.rsquared_adj*100:.1f}% de la variación explicada."],
            ["Prob(F-statistic)", f"{model.f_pvalue:.4e}", "Significancia global (< 0.05 es bueno)."],
            ["Observaciones", f"{int(model.nobs)}", "Filas usadas."]
        ]
        st.markdown(tabulate(metrics, headers=["Métrica", "Valor", "Interpretación"], tablefmt="pipe"))

        # 2. Tabla de Coeficientes
        results = []
        results.append(["Intercepto", model.params['Intercept'], model.pvalues['Intercept'], 'N/A', 'N/A'])

        for var in formula_components:
            coef = model.params.get(var)
            p_val = model.pvalues.get(var)
            signif = "**SÍ**" if p_val < 0.05 else "NO"
            interp = f"Significativo. Delta de {coef:.4f} por unidad." if p_val < 0.05 else "No significativo."
            results.append([var, coef, p_val, signif, interp])

        st.markdown("\n### Coeficientes del Modelo")
        st.markdown(tabulate(results, headers=["Variable", "Coef", "P-valor", "Sig.", "Interpretación"], 
                             floatfmt=(".0f", ".4f", ".4f", "", ""), tablefmt="pipe"))

        # 3. Ecuación
        st.markdown("\n### Ecuación de Regresión")
        eq = f"**{dependent_var}** = {model.params['Intercept']:.4f}"
        for var in formula_components:
            coef = model.params.get(var)
            sign = "+" if coef >= 0 else "-"
            eq += f" {sign} {abs(coef):.4f} x **{var}**"
        st.code(eq)

    except Exception as e:
        st.error(f"Error en regresión de {dependent_var}: {e}")

# ==============================================================================
# 3. FUNCIONES DE VISUALIZACIÓN (Mantenidas)
# ==============================================================================

def display_dataframe_tab(df):
    st.header("Vista de Datos")
    cols = [c for c in COLUMNS_OF_INTEREST if c in df.columns]
    st.dataframe(df[cols].head(10), use_container_width=True)

def display_correlation_tab(df):
    st.header("Matriz de Correlación")
    feats = [f for f in TODAS_LAS_VARIABLES if f in df.columns and df[f].nunique() > 1]
    if len(feats) < 2: return
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[feats].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig); plt.close(fig)

def display_reel_vs_tab(df):
    st.header("Variación por REEL")
    feats = [f for f in PROPIEDADES_PAPEL if f in df.columns]
    fig, axes = plt.subplots(len(feats), 1, figsize=(12, 3*len(feats)), sharex=True)
    if len(feats) == 1: axes = [axes]
    for i, f in enumerate(feats):
        sns.lineplot(x='REEL', y=f, data=df, ax=axes[i], marker='o')
    st.pyplot(fig); plt.close(fig)

def display_scatter_tab(df):
    st.header("Gráficos de Dispersión")
    for x in ['DOSIFICACIÓN', 'VELOCIDAD', 'PESO']:
        if x in df.columns:
            ys = [p for p in PROPIEDADES_PAPEL if p in df.columns and p != x]
            fig, axes = plt.subplots(1, len(ys), figsize=(16, 4))
            if len(ys) == 1: axes = [axes]
            for i, y in enumerate(ys):
                sns.scatterplot(x=x, y=y, data=df, ax=axes[i])
                sns.regplot(x=x, y=y, data=df, ax=axes[i], scatter=False, color='red')
            st.pyplot(fig); plt.close(fig)

def display_boxplots_tab(df_full):
    st.header("Distribución por Gramaje")
    for p in ['SCT', 'CMT', 'MULLEN']:
        if p in df_full.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x='GRAMAJE', y=p, data=df_full, palette='Set2', ax=ax)
            st.pyplot(fig); plt.close(fig)

def display_averages_tab(df_full):
    st.header("Promedios Históricos")
    nums = df_full.select_dtypes(include=[np.number]).columns
    st.dataframe(df_full.groupby('GRAMAJE')[nums].mean().round(2))

# ==============================================================================
# 4. MAIN
# ==============================================================================

def main():
    st.title("Análisis de Propiedades del Papel")
    with st.sidebar:
        file = st.file_uploader("Subir CSV", type="csv")
    
    if file:
        df_full = load_and_preprocess_data(file)
        if df_full is not None:
            gramajes = sorted(df_full['GRAMAJE'].unique())
            with st.sidebar:
                sel = st.selectbox("Filtrar por Gramaje:", ["TODO"] + list(gramajes))
            
            df_filtro = df_full.copy() if sel == "TODO" else df_full[df_full['GRAMAJE'] == sel].copy()
            
            # Imputación para OLS y Correlación
            df_ols = df_filtro.copy()
            numeric_cols = df_ols.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df_ols[col] = df_ols[col].fillna(df_ols[col].mean())

            tabs = st.tabs(["📋 Datos", "🔗 Corr.", "📈 Reel", "⚫ Dispersión", "📦 Boxplots", "🔬 OLS", "🔢 Prom."])
            
            with tabs[0]: display_dataframe_tab(df_filtro)
            with tabs[1]: display_correlation_tab(df_ols)
            with tabs[2]: display_reel_vs_tab(df_filtro)
            with tabs[3]: display_scatter_tab(df_ols)
            with tabs[4]: display_boxplots_tab(df_full)
            with tabs[5]: 
                for p in ['SCT', 'CMT', 'MULLEN']:
                    if p in df_ols.columns: run_ols_analysis_clean(df_ols, p)
            with tabs[6]: display_averages_tab(df_full)

if __name__ == '__main__':
    main()
