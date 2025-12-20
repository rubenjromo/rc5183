import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.formula.api import ols
from tabulate import tabulate

# Configuración inicial
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
# 1. FUNCIONES DE CARGA Y LIMPIEZA
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

@st.cache_data(show_spinner="Procesando datos...")
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
# 2. SECCIONES DE VISUALIZACIÓN (RESTAURADAS AL 100%)
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
        plt.suptitle('Distribución de Frecuencia de Propiedades y Variables', y=1.02, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        st.pyplot(fig); plt.close(fig)

def display_dataframe_tab(df):
    st.header("Base de Datos")
    cols_to_show = [c for c in COLUMNS_OF_INTEREST if c in df.columns]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Primeras 5 Filas")
        st.dataframe(df[cols_to_show].head(), use_container_width=True)
    with col2:
        st.markdown("### Últimas 5 Filas")
        st.dataframe(df[cols_to_show].tail(), use_container_width=True)
    st.markdown("### Distribución de Frecuencia")
    plot_distribution_histograms(df)

def plot_scatter_relationships_for_tab(df, x_col, y_cols):
    existing_y_cols = [y for y in y_cols if y in df.columns and df[y].nunique() > 1]
    if x_col not in df.columns or not existing_y_cols: return
    st.markdown(f"#### Dispersión: Propiedades vs. `{x_col}`")
    n_plots = len(existing_y_cols)
    cols = 2
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    df_plot = df[[x_col] + existing_y_cols].dropna().copy()
    if x_col in VARIABLES_PROCESO + VARIABLES_NUEVAS:
        df_plot = df_plot[df_plot[x_col] != 0]
    
    if df_plot.empty:
        st.warning(f"No hay datos para graficar vs {x_col}")
        return

    for i, y_col in enumerate(existing_y_cols):
        sns.scatterplot(x=x_col, y=y_col, data=df_plot, ax=axes[i], alpha=0.7, color='teal')
        try: sns.regplot(x=x_col, y=y_col, data=df_plot, ax=axes[i], scatter=False, color='red', line_kws={'linestyle':'--'})
        except: pass
        axes[i].set_title(f'{y_col} vs. {x_col}')
    for i in range(n_plots, len(axes)): fig.delaxes(axes[i])
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

def display_scatter_tab(df):
    st.header("Gráficos de Dispersión")
    for var in ['DOSIFICACIÓN', 'VELOCIDAD', 'ALMIDÓN', 'LABIO', 'CHORRO', 'COLUMNA']:
        if var in df.columns: plot_scatter_relationships_for_tab(df, var, PROPIEDADES_PAPEL)
    plot_scatter_relationships_for_tab(df, 'PESO', ['SCT', 'CMT', 'MULLEN', 'COBB'])
    plot_scatter_relationships_for_tab(df, 'SCT', ['CMT', 'MULLEN', 'POROSIDAD'])

# ==============================================================================
# 3. REGRESIÓN OLS (RESTABLECIDA CON DETALLES)
# ==============================================================================

def run_ols_analysis_clean(df, dependent_var):
    model_cols = [dependent_var, 'DOSIFICACIÓN', 'VELOCIDAD', 'PESO', 'ALMIDÓN', 'LABIO', 'CHORRO', 'COLUMNA']
    model_cols_present = [col for col in model_cols if col in df.columns and df[col].nunique() > 1]
    model_df = df[model_cols_present].dropna().copy()
    if len(model_cols_present) < 2 or model_df.empty: return

    formula_components = [c for c in model_cols_present if c != dependent_var]
    formula = f'{dependent_var} ~ ' + ' + '.join(formula_components)

    try:
        model = ols(formula, data=model_df).fit()
        st.markdown("-" * 60)
        st.markdown(f"## Análisis de Regresión Múltiple: {dependent_var} (OLS)")
        st.markdown(f"**Expresión:** `{formula}`")
        
        # Tabla resumen
        metrics = [["**R-cuadrado Ajustado**", f"{model.rsquared_adj:.3f}", f"{model.rsquared_adj*100:.1f}% explicado."],
                   ["Prob(F-statistic)", f"{model.f_pvalue:.4e}", "Significancia global."],
                   ["Observaciones", f"{int(model.nobs)}", "Total filas."]]
        st.markdown(tabulate(metrics, headers=["Métrica", "Valor", "Interpretación"], tablefmt="pipe"))

        # Coeficientes
        results = [["Intercepto", model.params['Intercept'], model.pvalues['Intercept'], 'N/A', 'N/A']]
        for var in formula_components:
            coef = model.params.get(var)
            p_val = model.pvalues.get(var)
            signif = "**SÍ**" if p_val < 0.05 else "NO"
            interp = f"Significativo. Delta {coef:.4f}." if p_val < 0.05 else "No significativo."
            results.append([var, coef, p_val, signif, interp])
        
        st.markdown("### Coeficientes del Modelo")
        st.markdown(tabulate(results, headers=["Variable", "Coef", "P-valor", "Sig.", "Interpretación"], floatfmt=(".0f", ".4f", ".4f", "", ""), tablefmt="pipe"))

        # Ecuación
        eq = f"**{dependent_var}** = {model.params['Intercept']:.4f}"
        for var in formula_components:
            coef = model.params.get(var)
            sign = "+" if coef >= 0 else "-"
            eq += f" {sign} {abs(coef):.4f} x **{var}**"
        st.code(eq)
    except: pass

# ==============================================================================
# 4. RESTO DE FUNCIONES (CORRELACIÓN, REEL, BOXPLOT, AVG)
# ==============================================================================

def display_correlation_tab(df):
    st.header("Matriz de Correlación")
    feats = [f for f in TODAS_LAS_VARIABLES if f in df.columns and df[f].nunique() > 1]
    if len(feats) < 2: return
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df[feats].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig); plt.close(fig)

def display_reel_vs_tab(df):
    st.header("Variación vs. REEL")
    feats = [f for f in PROPIEDADES_PAPEL if f in df.columns]
    fig, axes = plt.subplots(len(feats), 1, figsize=(12, 4 * len(feats)), sharex=True)
    if len(feats) == 1: axes = [axes]
    for i, f in enumerate(feats):
        sns.lineplot(x='REEL', y=f, data=df, ax=axes[i], marker='o', color='darkblue')
        axes[i].set_ylabel(f)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

def display_boxplots_tab(df_full):
    st.header("Boxplots por Gramaje")
    props = ['MULLEN', 'SCT', 'CMT', 'POROSIDAD']
    for p in [x for x in props if x in df_full.columns]:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='GRAMAJE', y=p, data=df_full, palette='viridis', ax=ax)
        sns.swarmplot(x='GRAMAJE', y=p, data=df_full, color='black', size=3, alpha=0.6, ax=ax)
        st.pyplot(fig); plt.close(fig)

def display_averages_tab(df_full):
    st.header("Promedios por Gramaje")
    nums = df_full.select_dtypes(include=[np.number]).columns.drop('REEL', errors='ignore')
    st.dataframe(df_full.groupby('GRAMAJE')[nums].mean().round(2))

# ==============================================================================
# 5. MAIN CON FILTRO
# ==============================================================================

def main():
    st.title("Análisis de Datos Exploratorio - Propiedades del Papel")
    with st.sidebar:
        file = st.file_uploader("Subir archivo CSV", type="csv")
    
    if file:
        df_full = load_and_preprocess_data(file)
        if df_full is not None:
            gramajes = sorted(df_full['GRAMAJE'].unique())
            with st.sidebar:
                sel = st.selectbox("Seleccione el Gramaje:", ["TODO"] + list(gramajes))
            
            df_filtro = df_full.copy() if sel == "TODO" else df_full[df_full['GRAMAJE'] == sel].copy()
            
            # Preparar datos (Imputación por media para gráficos de dispersión y OLS)
            df_prep = df_filtro.copy()
            for col in [c for c in COLUMNS_OF_INTEREST if c in df_prep.columns and c not in ['GRAMAJE', 'REEL']]:
                df_prep[col] = df_prep[col].fillna(df_prep[col].mean())

            tabs = st.tabs(["📋 Datos", "🔗 Correlación", "📈 Reel", "⚫ Dispersión", "📦 Boxplots", "🔬 OLS", "🔢 Promedios"])
            
            with tabs[0]: display_dataframe_tab(df_filtro)
            with tabs[1]: display_correlation_tab(df_prep)
            with tabs[2]: display_reel_vs_tab(df_filtro)
            with tabs[3]: display_scatter_tab(df_prep)
            with tabs[4]: display_boxplots_tab(df_full)
            with tabs[5]: 
                for p in ['SCT', 'CMT', 'MULLEN']:
                    if p in df_prep.columns: run_ols_analysis_clean(df_prep, p)
            with tabs[6]: display_averages_tab(df_full)

if __name__ == '__main__':
    main()
