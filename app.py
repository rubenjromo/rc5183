import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from tabulate import tabulate
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Análisis de Calidad y Proceso de Papel")

# --- VARIABLES GLOBALES ---
COLS_INTERES = ['REEL', 'GRAMAJE', 'PESO', 'SCT', 'CMT', 'COBB', 'POROSIDAD',
                'DOSIFICACIÓN', 'VELOCIDAD', 'ALMIDÓN',
                'LABIO', 'CHORRO', 'COLUMNA']
sns.set_style("whitegrid")

# ==============================================================================
# === 1. FUNCIÓN DE CARGA Y LIMPIEZA DE DATOS (CON CACHÉ) ===
# ==============================================================================

@st.cache_data
def load_and_clean_data(uploaded_file):
    """Carga y limpia el DataFrame con detección de encabezado y limpieza de 'Unnamed_'."""
    if uploaded_file is None:
        return pd.DataFrame(), "Cargue un archivo CSV para comenzar el análisis."

    try:
        # 1. Detección de encabezado
        file_data = uploaded_file.getvalue()
        # Usamos io.StringIO para leer el contenido del archivo cargado en memoria
        temp_df = pd.read_csv(io.StringIO(file_data.decode('latin1')), header=None, sep=';', skip_blank_lines=False)
        
        header_row_index = -1
        for i in range(5):
            if temp_df.iloc[i].astype(str).str.contains('REEL', case=False, na=False).any():
                header_row_index = i
                break
                
        if header_row_index == -1:
            return pd.DataFrame(), "Error: No se pudo encontrar la fila de encabezado que contiene 'REEL'."

        df = pd.read_csv(io.StringIO(file_data.decode('latin1')), header=header_row_index, sep=';')

        # --- Lógica de Limpieza ---
        def make_columns_unique_and_clean(df_input):
            df_output = df_input.copy()
            df_output.columns = df_output.columns.astype(str).str.strip().str.replace(r'[\n\r]', '', regex=True)
            new_cols = []
            seen = {}
            for i, col in enumerate(df_output.columns):
                temp_col = col
                if temp_col.startswith('Unnamed:') or temp_col == '':
                    temp_col = f'Unnamed_{i}'
                
                if temp_col in seen:
                    seen[temp_col] += 1
                    temp_col = f'{temp_col}_{seen[temp_col]}'
                else:
                    seen[temp_col] = 0
                new_cols.append(temp_col)
            df_output.columns = new_cols
            return df_output

        df = make_columns_unique_and_clean(df)

        # 2. ELIMINACIÓN AUTOMÁTICA DE COLUMNAS 'UNNAMED_'
        columns_to_drop = [col for col in df.columns if col.startswith('Unnamed_')]
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
            
        # 3. Mapeo de columnas (Asumir HORA es ALMIDÓN si no existe ALMIDÓN)
        if 'HORA' in df.columns and 'ALMIDÓN' not in df.columns:
            df.rename(columns={'HORA': 'ALMIDÓN'}, inplace=True)

        # 4. Conversión a numérico (manejo de coma como decimal)
        df = df.dropna(axis=1, how='all')
        
        for col in [c for c in COLS_INTERES if c in df.columns and c not in ['GRAMAJE']]:
            col_series = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
            df[col] = pd.to_numeric(col_series, errors='coerce')

        df_limpio = df.dropna(subset=['REEL']).copy()
        
        # 5. Imputación de NaN con la media y filtrado
        for col in [c for c in COLS_INTERES if c in df_limpio.columns and c not in ['REEL', 'GRAMAJE']]:
            mean_val = df_limpio[col].mean()
            df_limpio.loc[:, col] = df_limpio[col].fillna(mean_val)

        df_analisis = df_limpio[df_limpio['REEL'] > 0].copy()
        
        return df_analisis, "Datos cargados y limpiados con éxito."

    except Exception as e:
        return pd.DataFrame(), f"Error al procesar el archivo: {e}"

# ==============================================================================
# === 2. FUNCIONES DE VISUALIZACIÓN ===
# ==============================================================================

def plot_variation_vs_reel(df, features):
    """Genera gráficos de línea para ver la variación de las propiedades vs. REEL."""
    existing_features = [f for f in features if f in df.columns]
    if not existing_features: return None

    n_features = len(existing_features)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 5 * n_features), sharex=True)
    if n_features == 1: axes = [axes]

    for i, feature in enumerate(existing_features):
        sns.lineplot(x='REEL', y=feature, data=df, ax=axes[i], marker='o', color='darkblue')
        axes[i].set_title(f'Variación de {feature} a lo largo de los REELs', fontsize=14)
        axes[i].set_ylabel(feature)
        axes[i].grid(axis='y', linestyle='--')

    axes[-1].set_xlabel('REEL')
    plt.tight_layout()
    return fig

def plot_correlation_matrix(df, features):
    """Genera un mapa de calor (heatmap) para visualizar la matriz de correlación."""
    existing_features = [f for f in features if f in df.columns and df[f].nunique() > 1]
    if len(existing_features) < 2: return None

    corr_matrix = df[existing_features].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black',
                cbar_kws={'label': 'Coeficiente de Correlación'}, ax=ax)
    ax.set_title('Matriz de Correlación Ampliada entre las Propiedades', fontsize=16)
    plt.tight_layout()
    return fig

def plot_scatter_relationships(df, x_col, y_cols):
    """Genera gráficos de dispersión para analizar la relación entre una variable (x) y varias (y)."""
    if x_col not in df.columns: return None
    existing_y_cols = [y for y in y_cols if y in df.columns and df[y].nunique() > 1]
    if not existing_y_cols: return None

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
        axes[i].set_title(f'{y_col} vs. {x_col}', fontsize=12)
        axes[i].grid(True, linestyle=':')

    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    return fig

def plot_histograms(df, features, title_suffix):
    """Genera histogramas de distribución."""
    existing_features = [f for f in features if f in df.columns and df[f].nunique() > 1]
    if not existing_features: return None

    n_plots = len(existing_features)
    cols = 3
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(existing_features):
        df[feature].hist(ax=axes[i], bins=10, edgecolor='black', color='skyblue')
        axes[i].set_title(feature, fontsize=12)
        axes[i].set_ylabel('Frecuencia')

    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(f'Distribución de Frecuencia de {title_suffix}', y=1.02, fontsize=16)
    plt.tight_layout()
    return fig

# ==============================================================================
# === 3. FUNCIÓN DE REGRESIÓN OLS ===
# ==============================================================================

def run_ols_analysis_clean(df, dependent_var):
    """Ejecuta el análisis OLS y formatea los resultados para Streamlit."""
    st.markdown(f"\n" + "=" * 60)
    st.markdown(f"## ⚙️ Análisis de Regresión Múltiple: {dependent_var} (OLS)")
    
    model_cols = [dependent_var, 'DOSIFICACIÓN', 'VELOCIDAD', 'PESO', 'ALMIDÓN',
                  'LABIO', 'CHORRO', 'COLUMNA']
    model_cols_present = [col for col in model_cols if col in df.columns]
    model_df = df[model_cols_present].dropna().copy()

    if len(model_cols_present) < 2 or model_df.empty:
        st.warning(f"ADVERTENCIA: No hay suficientes datos limpios o variables para ejecutar la regresión múltiple para **{dependent_var}**.")
        return

    formula_components = [c for c in model_cols_present if c != dependent_var]
    formula = f'{dependent_var} ~ ' + ' + '.join(formula_components)
    st.markdown(f"**Fórmula:** `{formula}`")
    st.markdown("=" * 60)
    
    try:
        model = ols(formula, data=model_df).fit()
        
        # --- 1. Resumen General ---
        metrics = [
            ["**R-cuadrado Ajustado**", f"{model.rsquared_adj:.3f}", f"{model.rsquared_adj*100:.1f}% de la variación en {dependent_var} es explicada por el modelo."],
            ["Prob(F-statistic)", f"{model.f_pvalue:.4e}", "Nivel de significancia del modelo global (< 0.05 es significativo)."],
            ["Observaciones", f"{int(model.nobs)}", "Total de filas usadas para el modelo."]
        ]
        
        st.subheader("Métricas Clave del Modelo")
        st.table(pd.DataFrame(metrics, columns=["Métrica", "Valor", "Interpretación"]))

        # --- 2. Tabla de Coeficientes ---
        results = []
        
        results.append(["Intercepto", model.params['Intercept'], model.pvalues['Intercept'], 'N/A', 'N/A'])

        for var in formula_components:
            coef = model.params.get(var)
            p_val = model.pvalues.get(var)
            signif = "**SÍ**" if p_val < 0.05 else "NO"
            
            if p_val < 0.05:
                interpretation = f"Significativo. Varía en {coef:.4f} por cada unidad de {var}."
            else:
                interpretation = "No significativo (P > 0.05)."
                
            results.append([var, coef, p_val, signif, interpretation])

        headers = ["Variable", "Coeficiente", "P-valor", "Significativo", "Interpretación"]
        st.subheader("Coeficientes del Modelo: Impacto Individual")
        
        coef_df = pd.DataFrame(results, columns=headers)
        coef_df['Coeficiente'] = coef_df['Coeficiente'].apply(lambda x: f"{x:.4f}")
        coef_df['P-valor'] = coef_df['P-valor'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(coef_df, hide_index=True)


        # --- 3. Ecuación de Regresión (RENDERIZADO EN LATEX) ---
        st.subheader("📐 Ecuación de Regresión") 
        
        equation_latex = r"\mathbf{" + dependent_var + "} = " + f"{model.params['Intercept']:.4f}"
        for var in formula_components:
            coef = model.params.get(var)
            sign = "+" if coef >= 0 else "-"
            equation_latex += f" {sign} {abs(coef):.4f} \cdot \mathbf{{{var}}}"
        
        st.latex(equation_latex)
        
    except Exception as e:
        st.error(f"Error al ejecutar el modelo de regresión para **{dependent_var}**: {e}")
    st.markdown("=" * 60)

# ==============================================================================
# === 4. FUNCIÓN DE PROMEDIOS POR GRAMAJE ===
# ==============================================================================

def calculate_averages_by_gramaje(df):
    """Calcula y muestra los promedios agrupados por GRAMAJE."""
    st.markdown("\n" + "=" * 80)
    st.markdown("## 📋 Promedios de Variables por Gramaje")
    st.markdown("---")
    
    if 'GRAMAJE' not in df.columns or df['GRAMAJE'].isnull().all():
        st.warning("ADVERTENCIA: La columna 'GRAMAJE' no se encontró o está vacía. Análisis omitido.")
        return

    df_temp = df.copy() 
    df_temp['GRAMAJE'] = df_temp['GRAMAJE'].astype(str).str.strip().str.upper()

    numeric_cols = df_temp.drop(columns=['GRAMAJE', 'REEL'], errors='ignore').select_dtypes(include=['float64', 'int64']).columns
    
    if numeric_cols.empty:
        st.warning("ADVERTENCIA: No se encontraron columnas numéricas para promediar.")
        return
        
    averages_df = df_temp.groupby('GRAMAJE')[numeric_cols].mean().reset_index()
    averages_df = averages_df.round(2)
    
    st.markdown("Se muestra el valor promedio de cada propiedad y variable de proceso para cada tipo de **gramaje** presente en el conjunto de datos.")
    st.dataframe(averages_df)
    
# ==============================================================================
# === 5. APLICACIÓN STREAMLIT PRINCIPAL ===
# ==============================================================================

def main():
    st.title("📊 Análisis Exploratorio y Regresión de Calidad de Papel")
    st.markdown("Pruebas con RC+5183")

    # --- Sidebar para Carga de Archivo ---
    with st.sidebar:
        st.header("⚙️ Configuración")
        uploaded_file = st.file_uploader("Subir Archivo CSV (Separador: ;)", type=["csv"])
        st.markdown("---")

    # Cargar y limpiar datos
    df_analisis, status_message = load_and_clean_data(uploaded_file)
    st.sidebar.info(status_message)

    if df_analisis.empty:
        st.stop()
        
    # --- Pestañas de Análisis ---
    tab1, tab2, tab3, tab4 = st.tabs(["Data Limpia", "📈 Visualizaciones", "🔍 Regresión OLS", "📋 Promedios"])

    # --- PESTAÑA 1: DATA LIMPIA ---
    with tab1:
        st.header("1. Datos Limpios y Preprocesados")
        st.markdown(f"**Filas:** {df_analisis.shape[0]}, **Columnas:** {df_analisis.shape[1]}")
        st.dataframe(df_analisis.head(10))
      
    # --- PESTAÑA 2: VISUALIZACIONES ---
    with tab2:
        st.header("2. Visualizaciones Clave")
        
        propiedades_papel = ['PESO', 'SCT', 'CMT', 'COBB', 'POROSIDAD']
        variables_proceso = ['DOSIFICACIÓN', 'VELOCIDAD', 'ALMIDÓN']
        variables_nuevas = ['LABIO', 'CHORRO', 'COLUMNA']
        todas_las_variables = propiedades_papel + variables_proceso + variables_nuevas
        
        # 2.1 Correlación y Variación
        col_sec1, col_sec2 = st.columns(2)
        
        with col_sec1:
            st.subheader("2.1 Matriz de Correlación")
            fig_corr = plot_correlation_matrix(df_analisis, todas_las_variables)
            if fig_corr: st.pyplot(fig_corr)
            
        with col_sec2:
            st.subheader("2.2 Variación de Propiedades por REEL")
            fig_reel = plot_variation_vs_reel(df_analisis, propiedades_papel)
            if fig_reel: st.pyplot(fig_reel)

        st.markdown("---")

        # 2.3 Gráficos de Dispersión (Variables de Proceso vs. Propiedades de Calidad)
        st.subheader("2.3 Gráficos de Dispersión")
        
        col_disp1, col_disp2, col_disp3 = st.columns(3)
        
        # FILA 1: DOSIFICACIÓN, VELOCIDAD, ALMIDÓN
        with col_disp1:
            st.markdown("##### Relación con **DOSIFICACIÓN**")
            fig_dosif = plot_scatter_relationships(df_analisis, 'DOSIFICACIÓN', propiedades_papel)
            if fig_dosif: st.pyplot(fig_dosif)
        
        with col_disp2:
            st.markdown("##### Relación con **VELOCIDAD**")
            fig_vel = plot_scatter_relationships(df_analisis, 'VELOCIDAD', propiedades_papel)
            if fig_vel: st.pyplot(fig_vel)
        
        with col_disp3:
            st.markdown("##### Relación con **ALMIDÓN**")
            fig_almidon = plot_scatter_relationships(df_analisis, 'ALMIDÓN', propiedades_papel)
            if fig_almidon: st.pyplot(fig_almidon)
            
        st.markdown("---")

        # FILA 2: LABIO, CHORRO, COLUMNA
        st.subheader("2.4 Gráficos de Dispersión - Propiedades de Calidad")
        col_disp4, col_disp5, col_disp6 = st.columns(3)

        with col_disp4:
            st.markdown("##### Relación con **LABIO**")
            fig_labio = plot_scatter_relationships(df_analisis, 'LABIO', propiedades_papel)
            if fig_labio: st.pyplot(fig_labio)
        
        with col_disp5:
            st.markdown("##### Relación con **CHORRO**")
            fig_chorro = plot_scatter_relationships(df_analisis, 'CHORRO', propiedades_papel)
            if fig_chorro: st.pyplot(fig_chorro)
        
        with col_disp6:
            st.markdown("##### Relación con **COLUMNA**")
            fig_columna = plot_scatter_relationships(df_analisis, 'COLUMNA', propiedades_papel)
            if fig_columna: st.pyplot(fig_columna)
            
        st.markdown("---")

        # FILA 3: DISPERSIÓN ENTRE PROPIEDADES 
        st.subheader("2.5 Dispersión entre Propiedades de Calidad")
        col_disp7, col_disp8 = st.columns(2)

        with col_disp7:
            st.markdown("##### **PESO** vs. SCT/CMT/COBB")
            fig_peso = plot_scatter_relationships(df_analisis, 'PESO', ['SCT', 'CMT', 'COBB'])
            if fig_peso: st.pyplot(fig_peso)

        with col_disp8:
            st.markdown("##### **SCT** vs. CMT/POROSIDAD")
            fig_sct = plot_scatter_relationships(df_analisis, 'SCT', ['CMT', 'POROSIDAD'])
            if fig_sct: st.pyplot(fig_sct)

        st.markdown("---")

        # 2.6 Histogramas
        st.subheader("2.6 Distribución de Variables")
        
        col_hist1, col_hist2 = st.columns(2)

        with col_hist1:
            fig_hist_prop = plot_histograms(df_analisis, propiedades_papel, "Propiedades del Papel")
            if fig_hist_prop: st.pyplot(fig_hist_prop)
        
        with col_hist2:
            fig_hist_proc = plot_histograms(df_analisis, variables_nuevas, "Variables de Proceso (LABIO, CHORRO, COLUMNA)")
            if fig_hist_proc: st.pyplot(fig_hist_proc)

    # --- PESTAÑA 3: REGRESIÓN OLS ---
    with tab3:
        st.header("3. Regresión Múltiple por Mínimos Cuadrados Ordinarios (OLS)")
        
        run_ols_analysis_clean(df_analisis, 'SCT')
        run_ols_analysis_clean(df_analisis, 'CMT')

    # --- PESTAÑA 4: PROMEDIOS AGRUPADOS ---
    with tab4:
        st.header("4. Análisis de Promedios por Gramaje")
        calculate_averages_by_gramaje(df_analisis)

if __name__ == "__main__":
    main()
