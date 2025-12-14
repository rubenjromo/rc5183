import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.formula.api import ols
from tabulate import tabulate

# Configuración inicial y estilo de la aplicación
st.set_page_config(layout="wide", page_title="Análisis de Propiedades del Papel")
sns.set_style("whitegrid")
# La configuración de plt.rcParams['figure.figsize'] se maneja dentro de cada función de graficación.

# Variables globales para referencia
COLUMNS_OF_INTEREST = ['REEL', 'PESO', 'SCT', 'CMT', 'MULLEN', 'COBB', 'POROSIDAD',
                       'DOSIFICACIÓN', 'VELOCIDAD', 'ALMIDÓN',
                       'LABIO', 'CHORRO', 'COLUMNA', 'GRAMAJE']
PROPIEDADES_PAPEL = ['PESO', 'SCT', 'CMT', 'MULLEN', 'COBB', 'POROSIDAD']
VARIABLES_PROCESO = ['DOSIFICACIÓN', 'VELOCIDAD', 'ALMIDÓN']
VARIABLES_NUEVAS = ['LABIO', 'CHORRO', 'COLUMNA']
TODAS_LAS_VARIABLES = PROPIEDADES_PAPEL + VARIABLES_PROCESO + VARIABLES_NUEVAS

# ==============================================================================
# 1. FUNCIONES DE CARGA Y PREPROCESAMIENTO DE DATOS (CON CACHÉ)
# ==============================================================================

@st.cache_data
def make_columns_unique_and_clean(df_input):
    """Limpia y desduplica los nombres de columna."""
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

@st.cache_data(show_spinner="Cargando y preprocesando datos...")
def load_and_preprocess_data(uploaded_file):
    """Carga, detecta el encabezado y preprocesa el DataFrame."""
    try:
        # 1. Detección de encabezado
        uploaded_file.seek(0)
        temp_df = pd.read_csv(uploaded_file, encoding='latin1', header=None, sep=';', nrows=10, skip_blank_lines=False)
        header_row_index = -1
        for i in range(len(temp_df)):
            if not temp_df.iloc[i].isnull().all():
                if temp_df.iloc[i].iloc[:15].astype(str).str.contains('REEL', case=False, na=False).any():
                    header_row_index = i
                    break

        if header_row_index == -1:
            st.error("Error de Carga: No se pudo encontrar la fila de encabezado que contiene 'REEL' en las primeras 10 filas.")
            return None, None, None

        # 2. Carga final con el encabezado detectado
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin1', header=header_row_index, sep=';')
        
        # 3. Limpieza y estandarización
        df = make_columns_unique_and_clean(df)
        df = df.dropna(axis=1, how='all')

        if 'HORA' in df.columns:
            df.rename(columns={'HORA': 'ALMIDÓN'}, inplace=True)
        
        df = make_columns_unique_and_clean(df) # Segunda desduplicación

        # 4. Conversión a numérico (manejo de comas)
        for col in [c for c in COLUMNS_OF_INTEREST if c != 'GRAMAJE']:
            if col in df.columns:
                col_array_str = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                df[col] = pd.to_numeric(col_array_str, errors='coerce')

        df_limpio = df.dropna(subset=['REEL']).copy()
        df_limpio.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_analisis = df_limpio[df_limpio['REEL'] > 0].copy()

        # 5. Imputación de Media (para OLS y Scatter)
        df_analisis_ols = df_analisis.copy()
        df_analisis_scatter = df_analisis.copy()
        cols_to_impute = [c for c in COLUMNS_OF_INTEREST if c in df_analisis.columns and c != 'GRAMAJE' and c != 'REEL']

        for col in cols_to_impute:
            mean_val = df_analisis[col].mean()
            df_analisis_ols.loc[:, col] = df_analisis_ols[col].fillna(mean_val)
            df_analisis_scatter.loc[:, col] = df_analisis_scatter[col].fillna(mean_val)
            
        return df_analisis, df_analisis_ols, df_analisis_scatter

    except Exception as e:
        st.error(f"Ocurrió un error grave durante la carga/preprocesamiento. Error: {e}")
        return None, None, None

# ==============================================================================
# 2. SECCIONES (TABS) DEL ANÁLISIS
# ==============================================================================

def display_dataframe_tab(df_analisis):
    st.header("DataFrame de Datos Limpios")
    st.info("Vista de las primeras y últimas filas del conjunto de datos.")
    
    cols_to_show = [c for c in COLUMNS_OF_INTEREST if c in df_analisis.columns]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Primeras 5 Filas")
        st.dataframe(df_analisis[cols_to_show].head(), use_container_width=True)
    with col2:
        st.markdown("### Últimas 5 Filas")
        st.dataframe(df_analisis[cols_to_show].tail(5), use_container_width=True)
    
    st.markdown("### Distribución de Frecuencia de Variables")
    plot_distribution_histograms(df_analisis)
    
    
def plot_distribution_histograms(df):
    """Genera histogramas de distribución para variables clave."""
    cols_for_hist = [c for c in PROPIEDADES_PAPEL + VARIABLES_NUEVAS if c in df.columns and df[c].dtype in ['float64', 'int64']]

    if cols_for_hist:
        num_cols = len(cols_for_hist)
        fig_cols = min(3, num_cols)
        fig_rows = int(np.ceil(num_cols / fig_cols))
        
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(4 * fig_cols, 4 * fig_rows)) 
        
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        df.loc[:, cols_for_hist].hist(bins=10, edgecolor='black', color='skyblue', ax=axes)
        
        for i in range(num_cols, len(axes)):
            fig.delaxes(axes[i])
            
        plt.suptitle('Distribución de Frecuencia de Propiedades del Papel y Variables de Proceso', y=1.02, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.98]) 
        st.pyplot(fig) 
        plt.close(fig)
    else:
        st.warning("No hay columnas numéricas válidas para generar los histogramas.")

def display_correlation_tab(df_analisis_ols):
    st.header("Matriz de Correlación")
    st.info("El coeficiente de correlación (r) indica la fuerza y la dirección de la relación lineal entre dos variables. Valores cercanos a +1 o -1 indican una relación fuerte.")
    
    df_corr = df_analisis_ols.copy()
    existing_features = [f for f in TODAS_LAS_VARIABLES if f in df_corr.columns and df_corr[f].nunique() > 1 and df_corr[f].dropna().shape[0] > 1]
    
    if len(existing_features) < 2: 
        st.warning("No hay suficientes variables únicas para generar la matriz de correlación.")
        return

    corr_matrix = df_corr[existing_features].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black',
                cbar_kws={'label': 'Coeficiente de Correlación'}, ax=ax)
    ax.set_title('Matriz de Correlación Ampliada entre las Propiedades y Variables')
    st.pyplot(fig) 
    plt.close(fig)
    # 

def display_reel_vs_tab(df_analisis):
    st.header("Gráficos de Variación vs. REEL")
    st.info("Estos gráficos de línea muestran cómo varían las propiedades clave a lo largo de los diferentes reels, ayudando a identificar tendencias o inestabilidad en el proceso.")

    existing_features = [f for f in PROPIEDADES_PAPEL if f in df_analisis.columns]
    if not existing_features: 
        st.warning("No hay suficientes propiedades de papel válidas para este gráfico.")
        return

    n_features = len(existing_features)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features), sharex=True)
    if n_features == 1: axes = [axes]

    for i, feature in enumerate(existing_features):
        sns.lineplot(x='REEL', y=feature, data=df_analisis, ax=axes[i], marker='o', label=feature, color='darkblue')
        axes[i].set_title(f'Variación de {feature} a lo largo de los REELs')
        axes[i].set_ylabel(feature)
        axes[i].grid(axis='y', linestyle='--')

    axes[-1].set_xlabel('REEL')
    plt.tight_layout()
    st.pyplot(fig) 
    plt.close(fig)

def plot_scatter_relationships_for_tab(df, x_col, y_cols):
    """Genera un conjunto de gráficos de dispersión para una variable X dada."""
    existing_y_cols = [y for y in y_cols if y in df.columns and df[y].nunique() > 1]
    if x_col not in df.columns or not existing_y_cols: 
        return

    st.markdown(f"#### Dispersión: Propiedades vs. `{x_col}`")
    n_plots = len(existing_y_cols)
    cols = 2
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    df_plot = df[[x_col] + existing_y_cols].copy()
    
    if x_col in VARIABLES_PROCESO + VARIABLES_NUEVAS:
        df_plot = df_plot[df_plot[x_col] != 0].copy()

    df_plot.dropna(subset=[x_col] + existing_y_cols, inplace=True) 

    if df_plot.empty or df_plot[x_col].nunique() < 2:
        st.warning(f"No hay suficientes datos válidos para graficar las propiedades vs. `{x_col}`.")
        plt.close(fig)
        return

    for i, y_col in enumerate(existing_y_cols):
        if i < len(axes) and df_plot[y_col].dropna().shape[0] > 1 and df_plot[x_col].dropna().shape[0] > 1:
            sns.scatterplot(x=x_col, y=y_col, data=df_plot, ax=axes[i], alpha=0.7, color='teal')
            try:
                sns.regplot(x=x_col, y=y_col, data=df_plot, ax=axes[i], scatter=False, color='red', line_kws={'linestyle':'--'})
            except:
                pass
            axes[i].set_title(f'{y_col} vs. {x_col}')
            axes[i].grid(True, linestyle=':')
        elif i < len(axes):
            fig.delaxes(axes[i])


    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    st.pyplot(fig) 
    plt.close(fig)


def display_scatter_tab(df_analisis_scatter):
    st.header("Gráficos de Dispersión")
    st.info("Estos gráficos muestran la relación causa - efecto entre una variable explicativa (eje X) y las propiedades del papel (eje Y). La línea punteada roja indica la tendencia lineal.")
    
    st.subheader("1. Variables de Proceso vs. Propiedades")
    plot_scatter_relationships_for_tab(df_analisis_scatter, 'DOSIFICACIÓN', PROPIEDADES_PAPEL)
    plot_scatter_relationships_for_tab(df_analisis_scatter, 'VELOCIDAD', PROPIEDADES_PAPEL)
    if 'ALMIDÓN' in df_analisis_scatter.columns:
        plot_scatter_relationships_for_tab(df_analisis_scatter, 'ALMIDÓN', PROPIEDADES_PAPEL)
    
    st.subheader("2. Variables Nuevas vs. Propiedades")
    if 'LABIO' in df_analisis_scatter.columns:
        plot_scatter_relationships_for_tab(df_analisis_scatter, 'LABIO', PROPIEDADES_PAPEL)
    if 'CHORRO' in df_analisis_scatter.columns:
        plot_scatter_relationships_for_tab(df_analisis_scatter, 'CHORRO', PROPIEDADES_PAPEL)
    if 'COLUMNA' in df_analisis_scatter.columns:
        plot_scatter_relationships_for_tab(df_analisis_scatter, 'COLUMNA', PROPIEDADES_PAPEL)

    st.subheader("3. Propiedades Clave vs. otras Propiedades (Pares)")
    plot_scatter_relationships_for_tab(df_analisis_scatter, 'PESO', ['SCT', 'CMT', 'MULLEN', 'COBB']) 
    plot_scatter_relationships_for_tab(df_analisis_scatter, 'SCT', ['CMT', 'MULLEN', 'POROSIDAD'])

def display_boxplots_tab(df_analisis):
    st.header("Boxplots por Gramaje")
    st.info("Los boxplots muestran la distribución (media, cuartiles y atípicos) de las propiedades clave en función del gramaje. Un menor rango intercuartílico (caja más pequeña) indica mayor estabilidad.")
    
    if 'GRAMAJE' not in df_analisis.columns:
        st.warning("La columna 'GRAMAJE' es requerida para estos gráficos.")
        return

    properties_to_plot = ['MULLEN', 'SCT', 'CMT']
    
    fig, axes = plt.subplots(len(properties_to_plot), 1, figsize=(10, 5 * len(properties_to_plot)))
    
    if len(properties_to_plot) == 1: axes = [axes]
    
    for i, prop in enumerate(properties_to_plot):
        if prop in df_analisis.columns:
            df_plot = df_analisis.copy()
            df_plot['GRAMAJE'] = df_plot['GRAMAJE'].astype(str).str.strip().str.upper()
            df_plot.dropna(subset=[prop, 'GRAMAJE'], inplace=True)
            
            if df_plot.empty:
                axes[i].set_title(f"Distribución de {prop} (Datos insuficientes)")
                continue

            sns.boxplot(x='GRAMAJE', y=prop, data=df_plot, palette='viridis', hue='GRAMAJE', dodge=False, ax=axes[i])
            sns.swarmplot(x='GRAMAJE', y=prop, data=df_plot, color='black', size=3, alpha=0.6, dodge=False, ax=axes[i])
            axes[i].set_title(f'Distribución de {prop} por Gramaje de Papel')
            axes[i].set_xlabel('GRAMAJE')
            axes[i].set_ylabel(prop)
            axes[i].grid(axis='y', linestyle='--')
        else:
            fig.delaxes(axes[i])

    plt.tight_layout()
    st.pyplot(fig) 
    plt.close(fig)

def run_ols_analysis_clean(df, dependent_var):
    """Ejecuta el análisis OLS y formatea los resultados para Streamlit."""
    model_cols = [dependent_var, 'DOSIFICACIÓN', 'VELOCIDAD', 'PESO', 'ALMIDÓN',
                  'LABIO', 'CHORRO', 'COLUMNA']
    model_cols_present = [col for col in model_cols if col in df.columns]
    
    model_df = df[model_cols_present].dropna().copy()

    if len(model_cols_present) < 2 or model_df.empty:
        st.warning(f"ADVERTENCIA: No hay suficientes datos limpios o variables para ejecutar la regresión múltiple para **{dependent_var}**.")
        return

    formula_components = [c for c in model_cols_present if c != dependent_var]
    formula = f'{dependent_var} ~ ' + ' + '.join(formula_components)

    try:
        model = ols(formula, data=model_df).fit()

        st.markdown("\n" + "=" * 60)
        st.markdown(f"## Análisis de Regresión Múltiple: {dependent_var} (OLS)")
        st.markdown(f"**Fórmula:** `{formula}`")
        st.markdown("=" * 60)

        # 1. Resumen General
        st.markdown("### Resumen del Modelo")
        metrics = [
            ["**R-cuadrado Ajustado**", f"{model.rsquared_adj:.3f}", f"{model.rsquared_adj*100:.1f}% de la variación en {dependent_var} es explicada por el modelo."],
            ["Prob(F-statistic)", f"{model.f_pvalue:.4e}", "Nivel de significancia del modelo global (< 0.05 es significativo)."],
            ["Observaciones", f"{int(model.nobs)}", "Total de filas usadas para el modelo."]
        ]
        
        st.markdown(tabulate(metrics, headers=["Métrica", "Valor", "Interpretación"], tablefmt="pipe"))
        st.markdown("-" * 60)

        # 2. Tabla de Coeficientes
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
        st.markdown("\n### Coeficientes del Modelo: Impacto Individual ###")
        st.markdown(tabulate(results, headers=headers, floatfmt=(".0f", ".4f", ".4f", "", ""), tablefmt="pipe"))

        # 3. Ecuación de Regresión (Texto Plano)
        st.markdown("\n### Ecuación de Regresión (Estimada) ###")
        equation_str = f"**{dependent_var}** = {model.params['Intercept']:.4f}"
        for var in formula_components:
            coef = model.params.get(var)
            sign = "+" if coef >= 0 else "-"
            equation_str += f" {sign} {abs(coef):.4f} x **{var}**"

        st.code(equation_str)
        st.markdown("=" * 60)

    except Exception as e:
        st.error(f"ERROR al ejecutar el modelo de regresión para {dependent_var}: {e}")

def display_regression_tab(df_analisis_ols):
    st.header("Modelos de Regresión Múltiple (OLS)")
    st.info("La Regresión de Mínimos Cuadrados Ordinarios (OLS) modela la relación lineal entre una variable dependiente (propiedad del papel) y múltiples variables independientes (proceso y otras propiedades).")
    
    # Regresión SCT
    run_ols_analysis_clean(df_analisis_ols, 'SCT')

    # Regresión CMT
    run_ols_analysis_clean(df_analisis_ols, 'CMT')

    # Regresión MULLEN
    if 'MULLEN' in df_analisis_ols.columns:
        run_ols_analysis_clean(df_analisis_ols, 'MULLEN')
    else:
        st.warning("Regresión OLS para MULLEN omitida porque la columna no fue encontrada.")

def display_averages_tab(df_analisis):
    st.header("Promedios de Variables por Gramaje")
    st.info("Esta tabla muestra el valor promedio de cada propiedad y variable de proceso para cada tipo de **GRAMAJE** presente en sus datos.")
    
    if 'GRAMAJE' not in df_analisis.columns or df_analisis['GRAMAJE'].isnull().all():
        st.warning("La columna 'GRAMAJE' no se encontró o está completamente vacía. No se puede realizar la agrupación.")
        return

    df_temp = df_analisis.copy()
    df_temp['GRAMAJE'] = df_temp['GRAMAJE'].astype(str).str.strip().str.upper()
    numeric_cols = df_temp.drop(columns=['GRAMAJE', 'REEL'], errors='ignore').select_dtypes(include=['float64', 'int64']).columns

    if numeric_cols.empty:
        st.warning("No se encontraron columnas numéricas (excepto REEL) para promediar.")
        return

    averages_df = df_temp.groupby('GRAMAJE')[numeric_cols].mean().reset_index().round(2)
    
    # Reemplazar NaN con '-' para visualización
    display_df = averages_df.fillna('-')
    st.dataframe(display_df)
    
    # Botón de descarga
    csv = averages_df.to_csv(index=False).encode('latin1')
    st.download_button(
        label="Descargar Promedios por Gramaje (CSV)",
        data=csv,
        file_name='promedios_por_gramaje.csv',
        mime='text/csv',
    )


# ==============================================================================
# 3. FUNCIÓN PRINCIPAL DE LA APLICACIÓN
# ==============================================================================

def main():
    st.title("Análisis de Datos Exploratorio y Regresión de Propiedades del Papel")
    st.markdown("Pruebas con RC+5183")

    # --- BARRA LATERAL PARA CARGA ---
    with st.sidebar:
        st.header("⚙️ Carga de Datos")
        st.info("Sube tu archivo CSV.")
        uploaded_file = st.file_uploader("Subir archivo CSV", type="csv")
        
        if uploaded_file is None:
            st.warning("Esperando la carga de un archivo para comenzar el análisis.")

    # --- CONTENIDO PRINCIPAL ---
    
    if uploaded_file is None:
        st.markdown("### ¡Bienvenido!")
        st.markdown("Por favor, sube tu archivo CSV en la barra lateral izquierda para iniciar el análisis completo.")
        return

    df_analisis, df_analisis_ols, df_analisis_scatter = load_and_preprocess_data(uploaded_file)

    if df_analisis is not None:
        st.success(f"¡Archivo cargado y preprocesado con éxito! Total de filas válidas: {len(df_analisis)}")
        
        # --- CREACIÓN DE PESTAÑAS ---
        tab_df, tab_corr, tab_reel, tab_scatter, tab_boxplots, tab_reg, tab_avg = st.tabs([
            "📋 DataFrame / Distribución", 
            "🔗 Correlación", 
            "📈 Variación vs. REEL", 
            "⚫ Dispersión", 
            "📦 Boxplots por Gramaje", 
            "🔬 Regresiones (OLS)", 
            "🔢 Promedios por Gramaje"
        ])
        
        # Pestaña 1: DataFrame y Distribución
        with tab_df:
            display_dataframe_tab(df_analisis)

        # Pestaña 2: Correlación
        with tab_corr:
            display_correlation_tab(df_analisis_ols)

        # Pestaña 3: Variación vs. REEL
        with tab_reel:
            display_reel_vs_tab(df_analisis)

        # Pestaña 4: Dispersión
        with tab_scatter:
            display_scatter_tab(df_analisis_scatter)

        # Pestaña 5: Boxplots
        with tab_boxplots:
            display_boxplots_tab(df_analisis)

        # Pestaña 6: Regresiones
        with tab_reg:
            display_regression_tab(df_analisis_ols)

        # Pestaña 7: Promedios
        with tab_avg:
            display_averages_tab(df_analisis)

if __name__ == '__main__':
    main()
