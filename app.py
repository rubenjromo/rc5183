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
        uploaded_file.seek(0)
        temp_df = pd.read_csv(uploaded_file, encoding='latin1', header=None, sep=';', nrows=10, skip_blank_lines=False)
        header_row_index = -1
        for i in range(len(temp_df)):
            if not temp_df.iloc[i].isnull().all():
                if temp_df.iloc[i].iloc[:15].astype(str).str.contains('REEL', case=False, na=False).any():
                    header_row_index = i
                    break

        if header_row_index == -1:
            st.error("Error de Carga: No se pudo encontrar la fila de encabezado que contiene 'REEL'.")
            return None

        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin1', header=header_row_index, sep=';')
        df = make_columns_unique_and_clean(df)
        df = df.dropna(axis=1, how='all')

        if 'HORA' in df.columns:
            df.rename(columns={'HORA': 'ALMIDÓN'}, inplace=True)
        
        df = make_columns_unique_and_clean(df)

        for col in [c for c in COLUMNS_OF_INTEREST if c != 'GRAMAJE']:
            if col in df.columns:
                col_array_str = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                df[col] = pd.to_numeric(col_array_str, errors='coerce')

        df_limpio = df.dropna(subset=['REEL']).copy()
        df_limpio.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_analisis = df_limpio[df_limpio['REEL'] > 0].copy()

        if 'GRAMAJE' in df_analisis.columns:
            df_analisis['GRAMAJE'] = df_analisis['GRAMAJE'].astype(str).str.strip().str.upper()

        return df_analisis

    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ==============================================================================
# 2. SECCIONES (TABS) DEL ANÁLISIS
# ==============================================================================

def display_dataframe_tab(df_analisis):
    st.header("Base de Datos")
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
    cols_for_hist = [c for c in PROPIEDADES_PAPEL + VARIABLES_NUEVAS if c in df.columns and df[c].dtype in ['float64', 'int64']]
    if cols_for_hist:
        num_cols = len(cols_for_hist)
        fig_cols = min(3, num_cols)
        fig_rows = int(np.ceil(num_cols / fig_cols))
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(4 * fig_cols, 4 * fig_rows)) 
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        df.loc[:, cols_for_hist].hist(bins=10, edgecolor='black', color='skyblue', ax=axes[:num_cols])
        for i in range(num_cols, len(axes)): fig.delaxes(axes[i])
        plt.suptitle('Distribución de Frecuencia de Propiedades del Papel y Variables de Proceso', y=1.02, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        st.pyplot(fig); plt.close(fig)

def display_correlation_tab(df_corr):
    st.header("Matriz de Correlación")
    st.info("El coeficiente de correlación (r) indica la fuerza y la dirección de la relación lineal entre dos variables. Valores cercanos a +1 o -1 indican una relación fuerte.")
    existing_features = [f for f in TODAS_LAS_VARIABLES if f in df_corr.columns and df_corr[f].nunique() > 1]
    if len(existing_features) < 2: return
    corr_matrix = df_corr[existing_features].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black', ax=ax)
    ax.set_title('Matriz de Correlación Ampliada')
    st.pyplot(fig); plt.close(fig)

def display_reel_vs_tab(df_analisis):
    st.header("Gráficos de Variación vs. REEL")
    st.info("Variabilidad a lo largo de los reelsX.")
    
    existing_features = [f for f in PROPIEDADES_PAPEL if f in df_analisis.columns]
    if not existing_features: return
    
    n_features = len(existing_features)
    # CAMBIO: sharex=False para que cada gráfica tenga su propia leyenda en X
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features), sharex=False)
    if n_features == 1: axes = [axes]
    
    # Rango completo de REELs para mantener los huecos
    all_reels = pd.DataFrame({'REEL': range(int(df_analisis['REEL'].min()), int(df_analisis['REEL'].max()) + 1)})

    for i, feature in enumerate(existing_features):
        df_tendencia = pd.merge(all_reels, df_analisis[['REEL', feature]], on='REEL', how='left')
        
        # Graficamos
        axes[i].plot(df_tendencia['REEL'], df_tendencia[feature], 
                     marker='o', markersize=4, linestyle='-', color='darkblue', linewidth=1.5)
        
        axes[i].set_title(f'Tendencia de {feature} por REEL')
        axes[i].set_ylabel(feature)
        axes[i].set_xlabel('REEL') # Forzamos la etiqueta en cada eje
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Opcional: Si tienes muchos REELs, podemos espaciar las etiquetas del eje X
        # axes[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

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
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    df_plot = df[[x_col] + existing_y_cols].dropna().copy()
    if x_col in VARIABLES_PROCESO + VARIABLES_NUEVAS:
        df_plot = df_plot[df_plot[x_col] != 0]
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
    st.info("Estos gráficos muestran la relación causa - efecto entre una variable explicativa (eje X) y las propiedades del papel (eje Y). La línea punteada roja indica la tendencia lineal.")

    propiedades_sin_peso = [p for p in PROPIEDADES_PAPEL if p != 'PESO']

    variables_eje_x = ['DOSIFICACIÓN', 'VELOCIDAD', 'ALMIDÓN', 'LABIO', 'CHORRO', 'COLUMNA']
    for var in variables_eje_x:
      if var in df.columns:
        plot_scatter_relationships_for_tab(df, var, propiedades_sin_peso)
        
    st.markdown("---")
    st.markdown("### Relaciones Específicas entre Propiedades")
    plot_scatter_relationships_for_tab(df, 'PESO', ['SCT', 'CMT', 'MULLEN', 'COBB'])
    plot_scatter_relationships_for_tab(df, 'SCT', ['CMT', 'MULLEN', 'POROSIDAD'])

def display_boxplots_tab(df_full):
    st.header("Boxplots por Gramaje")
    st.info("🔴 Rojo: Valor Mínimo de Calidad | 🟢 Verde: Valor Estándar de Laboratorio")
    
    with st.expander("ℹ️ ¿Cómo interpretar este gráfico?"):
        st.markdown("""
        * **Caja central**: Representa el Rango Intercuartílico (RIC), donde se encuentra el 50% de los datos centrales.
        * **Línea interna**: Es la **Mediana**.
        * **Puntos Negros (Swarm)**: Datos reales de cada REEL.
        """)
    
    # --- DICCIONARIO DE VALORES LABORATORIO (ACTUALIZADO) ---
    referencias_calidad = {
        146: {'SCT': {'min': 2.40, 'std': 2.60}, 'CMT': {'min': 29, 'std': 32}},
        160: {'SCT': {'min': 2.80, 'std': 3.00}, 'CMT': {'min': 33, 'std': 36}},
        195: {'SCT': {'min': 3.20, 'std': 3.40}, 'CMT': {'min': 35, 'std': 39}},
        170: {'SCT': {'min': 2.74, 'std': 2.98}, 'MULLEN': {'min': 68, 'std': 74}},
        205: {'SCT': {'min': 3.30, 'std': 3.59}, 'MULLEN': {'min': 74, 'std': 80}},
        230: {'SCT': {'min': 3.70, 'std': 4.16}, 'MULLEN': {'min': 72, 'std': 76}},
        270: {'SCT': {'min': 4.35, 'std': 4.73}, 'MULLEN': {'min': 90, 'std': 95}},
        120: {
            'SCT': {'min': 1.97, 'std': 2.28},
            'CMT': {'min': 22.0, 'std': 26.5},
            'MULLEN': {'min': 48, 'std': 53},
        },
        150: {
            'SCT': {'min': 2.90, 'std': 3.10},
            'CMT': {'min': 32.0, 'std': 36.0},
            'MULLEN': {'min': 62, 'std': 70},
        }
    }

    properties_to_plot = ['MULLEN', 'SCT', 'CMT', 'POROSIDAD']
    
    for prop in properties_to_plot:
        if prop in df_full.columns:
            if df_full[prop].dropna().empty:
                continue
                
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 1. Dibujar Boxplot y Swarmplot
            # Es vital obtener el orden exacto de las etiquetas del eje X
            sns.boxplot(x='GRAMAJE', y=prop, data=df_full, palette='viridis', ax=ax, width=0.5)
            sns.swarmplot(x='GRAMAJE', y=prop, data=df_full, color='black', size=3, alpha=0.4, ax=ax)
            
            # OBTENEMOS LAS ETIQUETAS ACTUALES DEL GRÁFICO (tal cual aparecen en pantalla)
            etiquetas_x = [t.get_text() for t in ax.get_xticklabels()]
            
            # 2. Dibujar los puntos de referencia basados en la POSICIÓN REAL
            for i, label_x in enumerate(etiquetas_x):
                try:
                    # Convertimos la etiqueta del eje X a número para buscar en el diccionario
                    g_int = int(float(label_x))
                    
                    if g_int in referencias_calidad and prop in referencias_calidad[g_int]:
                        val_min = referencias_calidad[g_int][prop]['min']
                        val_std = referencias_calidad[g_int][prop]['std']
                        
                        # Dibujamos usando 'i' que es la posición exacta en el eje horizontal
                        ax.scatter(i, val_min, color='red', s=180, edgecolors='white', 
                                   linewidth=1.5, zorder=10)
                        ax.text(i + 0.12, val_min, f'{val_min}', color='red', 
                                fontweight='bold', va='center', zorder=11, fontsize=9)
                        
                        ax.scatter(i, val_std, color='green', s=180, edgecolors='white', 
                                   linewidth=1.5, zorder=10)
                        ax.text(i + 0.12, val_std, f'{val_std}', color='green', 
                                fontweight='bold', va='center', zorder=11, fontsize=9)
                except (ValueError, TypeError):
                    continue 

            ax.set_title(f'Análisis de Distribución: {prop} por Gramaje')
            ax.set_ylabel(f'Valor de {prop}')
            ax.set_xlabel('Gramaje (g/m²)')
            ax.margins(y=0.18) 
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

def run_ols_analysis_clean(df, dependent_var):
    model_cols = [dependent_var, 'DOSIFICACIÓN', 'VELOCIDAD', 'PESO', 'ALMIDÓN', 'LABIO', 'CHORRO', 'COLUMNA']
    model_cols_present = [col for col in model_cols if col in df.columns and df[col].nunique() > 1]
    model_df = df[model_cols_present].dropna().copy()
    if len(model_cols_present) < 2 or model_df.empty: return

    formula = f'{dependent_var} ~ ' + ' + '.join([c for c in model_cols_present if c != dependent_var])
    try:
        model = ols(formula, data=model_df).fit()
        st.markdown("-" * 60)
        st.markdown(f"## Análisis de Regresión Múltiple: {dependent_var} (OLS)")
        st.markdown(f"**Expresión:** `{formula}`")
        
        st.markdown("### Resumen del Modelo")
        metrics = [["**R-cuadrado Ajustado**", f"{model.rsquared_adj:.3f}", f"{model.rsquared_adj*100:.1f}% explicado."],
                   ["Prob(F-statistic)", f"{model.f_pvalue:.4e}", "Nivel de significancia global."],
                   ["Observaciones", f"{int(model.nobs)}", "Total de filas usadas."]]
        st.markdown(tabulate(metrics, headers=["Métrica", "Valor", "Interpretación"], tablefmt="pipe"))

        results = [["Intercepto", model.params['Intercept'], model.pvalues['Intercept'], 'N/A', 'N/A']]
        for var in [c for c in model_cols_present if c != dependent_var]:
            coef = model.params.get(var)
            p_val = model.pvalues.get(var)
            signif = "**SÍ**" if p_val < 0.05 else "NO"
            results.append([var, coef, p_val, signif, f"Varía {coef:.4f} por unidad" if p_val < 0.05 else "No significativo"])
        
        st.markdown("### Coeficientes del Modelo")
        st.markdown(tabulate(results, headers=["Variable", "Coeficiente", "P-valor", "Significativo", "Interpretación"], floatfmt=(".0f", ".4f", ".4f", "", ""), tablefmt="pipe"))

        eq = f"**{dependent_var}** = {model.params['Intercept']:.4f}"
        for var in [c for c in model_cols_present if c != dependent_var]:
            coef = model.params.get(var)
            eq += f" {'+' if coef >= 0 else '-'} {abs(coef):.4f} x **{var}**"
        st.code(eq)
    except: pass

def display_regression_tab(df):
    st.header("Modelos de Regresión Múltiple (OLS)")
    st.info("La Regresión de Mínimos Cuadrados Ordinarios (OLS) modela la relación lineal entre una propiedad y múltiples variables.")
    for p in ['SCT', 'CMT', 'MULLEN']:
        if p in df.columns: run_ols_analysis_clean(df, p)

def display_averages_tab(df_full):
    st.header("Promedios de Variables por Gramaje")
    st.info("Esta tabla muestra el valor promedio de cada propiedad para cada gramaje.")
    num_cols = df_full.select_dtypes(include=[np.number]).columns.drop('REEL', errors='ignore')
    avg_df = df_full.groupby('GRAMAJE')[num_cols].mean().round(2)
    st.dataframe(avg_df)

# ==============================================================================
# 3. FUNCIÓN PRINCIPAL
# ==============================================================================

def main():
    st.title("Análisis de Datos Exploratorio y Regresión de Propiedades del Papel")
    st.markdown("Pruebas con RC+5183")

    with st.sidebar:
        st.header("⚙️ Carga de Datos")
        uploaded_file = st.file_uploader("Subir archivo CSV", type="csv")
        if uploaded_file is None:
            st.warning("Esperando la carga de un archivo para comenzar el análisis.")

    if uploaded_file is None:
        st.markdown("Favor subir el archivo CSV en la barra lateral izquierda para iniciar el análisis.")
        return

    df_full = load_and_preprocess_data(uploaded_file)

    if df_full is not None:
        # --- FILTRO POR GRAMAJE ---
        gramajes_crudos = df_full['GRAMAJE'].unique()
        gramajes_enteros = sorted([int(float(g)) for g in gramajes_crudos if str(g).replace('.','').isdigit()])
      
        with st.sidebar:
            st.subheader("Filtro de Análisis")
            sel = st.selectbox("Seleccione el Gramaje a analizar:", ["TODO"] + gramajes_enteros)
        
        if sel == "TODO":
          df_filtro = df_full.copy()
        else:
          df_filtro = df_full[pd.to_numeric(df_full['GRAMAJE'], errors='coerce').fillna(-1).astype(int) == sel].copy()
        
        # Preparación de datos (Imputación por media para análisis estadístico)
        df_prep = df_filtro.copy()
        cols_impute = [c for c in COLUMNS_OF_INTEREST if c in df_prep.columns and c not in ['GRAMAJE', 'REEL']]
        for col in cols_impute:
            df_prep[col] = df_prep[col].fillna(df_prep[col].mean())

        st.success(f"Analizando: **{sel}** | Total filas: {len(df_filtro)}")

        # --- TABS ORIGINALES ---
        tab_df, tab_corr, tab_reel, tab_scatter, tab_boxplots, tab_reg, tab_avg = st.tabs([
            "📋 Datos / Distribución", "🔗 Correlación", "📈 Variación vs. REEL", 
            "⚫ Dispersión", "📦 Boxplots por Gramaje", "🔬 Regresiones (OLS)", "🔢 Promedios"
        ])
        
        with tab_df: display_dataframe_tab(df_filtro)
        with tab_corr: display_correlation_tab(df_prep)
        with tab_reel: display_reel_vs_tab(df_filtro)
        with tab_scatter: display_scatter_tab(df_prep)
        with tab_boxplots: display_boxplots_tab(df_full) # Mantiene comparación global
        with tab_reg: display_regression_tab(df_prep)
        with tab_avg: display_averages_tab(df_full)

if __name__ == '__main__':
    main()
