import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.linalg import pinv
import time

# --- 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS PREMIUM ---
st.set_page_config(page_title="Hopfield Unified Dashboard v5.1", layout="wide")

# CSS Avanzado para mantener la estética limpia y profesional
st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e2e8f0; }
    div[data-testid="stMetricValue"] { color: #00f0ff !important; font-family: 'Courier New', monospace; font-weight: bold; }
    /* Contenedor estético para la rejilla de botones */
    .canvas-container { background-color: #151922; padding: 20px; border-radius: 10px; border: 1px solid #232d3f; }
    div[data-testid="stHorizontalBlock"] { gap: 4px !important; }
    div.stToggle { margin-bottom: -10px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CLASE DE LA RED DE HOPFIELD ---
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.n_neurons = size * size
        self.W = np.zeros((self.n_neurons, self.n_neurons))

    def train_pseudoinverse(self, patterns):
        if len(patterns) == 0:
            return
        P = np.array([p.flatten() for p in patterns])
        P_plus = pinv(P)
        self.W = np.dot(P_plus, P)
        self.W = (self.W + self.W.T) / 2
        np.fill_diagonal(self.W, 0)
        self.W = self.W * 1.4 # Escala sináptica calibrada para transitorio suave

    def energy(self, x):
        v = np.tanh(x)
        return -0.5 * np.dot(v.T, np.dot(self.W, v))

# --- 3. PATRONES PREDEFINIDOS ---
def get_default_patterns(size):
    p1 = np.ones((size, size)) * -1
    np.fill_diagonal(p1, 1)
    
    p2 = np.ones((size, size)) * -1
    mid = size // 2
    p2[mid, :] = 1
    p2[:, mid] = 1
    
    p3 = np.ones((size, size)) * -1
    p3[0, :] = 1; p3[-1, :] = 1; p3[:, 0] = 1; p3[:, -1] = 1
                
    return {"Diagonal Limpia": p1, "Cruz Matemática": p2, "Marco Perimetral": p3}

# --- 4. ENCABEZADO PRINCIPAL ---
st.title("🧠 Hopfield Neuro-Dashboard v5.1")
st.markdown("Plataforma analítica para el estudio del estado transitorio en ecuaciones diferenciales neuro-asociativas.")

# Barra lateral de parámetros técnicos
with st.sidebar:
    st.header("⚙️ Parámetros del Sistema")
    grid_size = st.slider("Resolución de Rejilla (N x N)", 6, 10, 8)
    dt = st.slider("Diferencial de Tiempo (dt)", 0.01, 0.2, 0.04)
    n_steps = st.slider("Iteraciones Máximas", 50, 250, 120)
    noise_level = st.slider("Magnitud del Ruido (σ)", 0.5, 2.5, 1.4)
    st.divider()
    st.markdown("### 🔬 Ecuación de Estado")
    st.latex(r"\frac{dx_i}{dt} = -x_i + \sum_{j=1}^{N} w_{ij} \tanh(x_j)")

patterns_dict = get_default_patterns(grid_size)

# --- 📁 SECCIÓN 1: CONFIGURACIÓN DE MEMORIA Y MATRIZ W ---
st.header("1. Configuración de Memoria y Arquitectura Sináptica")
col_izq, col_der = st.columns([1, 1.2])

with col_izq:
    st.subheader("Patrón Objetivo (Recuerdo)")
    opcion_patron = st.radio(
        "Método de entrada del patrón:",
        ["Usar catálogo predefinido", "✏️ Diseñar en Lienzo Interactivo"],
        horizontal=True
    )
    
    target_pattern = np.ones((grid_size, grid_size)) * -1
    
    if opcion_patron == "Usar catálogo predefinido":
        target_name = st.selectbox("Selecciona del catálogo:", list(patterns_dict.keys()))
        target_pattern = patterns_dict[target_name]
        
        fig_target = px.imshow(target_pattern, color_continuous_scale='Greys')
        fig_target.update_layout(coloraxis_showscale=False, width=200, height=200, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_target, use_container_width=False)
        
    else:
        st.markdown("**Lienzo de un solo clic:**")
        
        if 'canvas_toggles' not in st.session_state or st.session_state.canvas_toggles.shape != (grid_size, grid_size):
            st.session_state.canvas_toggles = np.zeros((grid_size, grid_size), dtype=bool)
        
        if st.button("🧹 Limpiar Todo el Lienzo"):
            st.session_state.canvas_toggles = np.zeros((grid_size, grid_size), dtype=bool)
            st.rerun()
            
        st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
        for i in range(grid_size):
            cols = st.columns(grid_size)
            for j in range(grid_size):
                with cols[j]:
                    key = f"toggle_{i}_{j}"
                    valor_actual = st.session_state.canvas_toggles[i, j]
                    estado = st.toggle("", value=bool(valor_actual), key=key, label_visibility="collapsed")
                    st.session_state.canvas_toggles[i, j] = estado
        st.markdown('</div>', unsafe_allow_html=True)
                    
        target_pattern = np.where(st.session_state.canvas_toggles, 1, -1)

# Entrenar la red neuronal dinámicamente
hopfield = HopfieldNetwork(grid_size)
hopfield.train_pseudoinverse([target_pattern])

with col_der:
    st.subheader("Matriz de Pesos Sinápticos $W$")
    st.markdown(f"Conexiones calculadas mediante pseudoinversa ({grid_size**2} $\\times$ {grid_size**2} sinapsis):")
    
    fig_w = px.imshow(
        hopfield.W, 
        color_continuous_scale='Viridis',
        labels=dict(x="Neurona j", y="Neurona i")
    )
    fig_w.update_layout(
        width=450, height=360, 
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_dark"
    )
    st.plotly_chart(fig_w, use_container_width=True)

# --- 🚀 SECCIÓN 2: SIMULACIÓN Y DINÁMICA DE RELAJACIÓN ---
st.divider()
st.header("2. Dinámica de Relajación y Métricas de Rendimiento")
st.markdown("Presiona el botón para inyectar ruido Gaussiano al estado inicial y activar la ecuación diferencial.")

ejecutar = st.button("🚀 Iniciar Simulación Dinámica")

if ejecutar:
    # Capturar el estado ruidoso inicial
    initial_state = target_pattern.flatten() + np.random.normal(0, noise_level, grid_size**2)
    
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        st.markdown("**Evolución Morfológica Neuronal:**")
        plot_spot = st.empty()
    with col_sim2:
        st.markdown("**Monitoreo de Energía de Lyapunov:**")
        energy_spot = st.empty()
    
    # Estructura forense de almacenamiento temporal
    snapshots = {}
    snapshots[0] = initial_state.copy()
    
    x = initial_state.copy()
    history_energy = []
    pasos_clave = [1, 2, 4, 8, 15]

    # Bucle de integración de Euler
    for i in range(1, n_steps + 1):
        v = np.tanh(x)
        dx = -x + np.dot(hopfield.W, v)
        x += dx * dt
        
        history_energy.append(hopfield.energy(x))
        
        if i in pasos_clave:
            snapshots[i] = x.copy()
            
        if i % 2 == 0:
            fig_neurons = px.imshow(x.reshape(grid_size, grid_size), color_continuous_scale='RdBu_r', zmin=-1.5, zmax=1.5)
            fig_neurons.update_layout(coloraxis_showscale=False, width=320, height=320, margin=dict(l=0,r=0,t=0,b=0))
            plot_spot.plotly_chart(fig_neurons, use_container_width=True)
            
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Scatter(y=history_energy, mode='lines', line=dict(color='#00f0ff', width=3)))
            fig_energy.update_layout(xaxis_title="Pasos", yaxis_title="Energía (E)", template="plotly_dark", height=260, margin=dict(l=0,r=0,t=10,b=0))
            energy_spot.plotly_chart(fig_energy, use_container_width=True)
            time.sleep(0.02)

    snapshots["Final"] = x.copy()
    
    # --- PROCESAMIENTO ESTADÍSTICO ---
    final_binario = np.sign(x)
    objetivo_plano = target_pattern.flatten()
    pixeles_diferentes = np.sum(final_binario != objetivo_plano)
    porcentaje_error = (pixeles_diferentes / len(objetivo_plano)) * 100
    
    delta_energia = history_energy[-1] - history_energy[0]
    correlacion = np.corrcoef(final_binario, objetivo_plano)[0, 1]
    if np.isnan(correlacion): correlacion = 1.0 if pixeles_diferentes == 0 else 0.0

    # Determinar el tipo de convergencia
    if porcentaje_error == 0:
        status_text = "🎯 CONVERGENCIA EXITOSA: La red alcanzó el atractor global exacto."
        status_color = "green"
    elif porcentaje_error == 100:
        status_text = "🙃 ATRACTOR INVERTIDO: La red convergió al estado complementario simétrico."
        status_color = "orange"
    else:
        status_text = f"⚠️ ESTADO ESPURIO: El sistema quedó atrapado en un mínimo local o estado ruidoso parcial."
        status_color = "red"

    st.success("✅ Análisis temporal completado.")
    
    # Desplegar diagnóstico y métricas en tarjetas
    st.markdown(f"<div style='padding:15px; border-radius:8px; background-color:#1c2333; border-left:5px solid {status_color}; font-weight:bold;'>{status_text}</div>", unsafe_allow_html=True)
    st.write("")

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(label="❌ Desviación de Reconstrucción", value=f"{porcentaje_error:.1f} %", delta="Hamming Relativo")
    with m_col2:
        st.metric(label="📉 Estabilidad Dinámica (ΔE)", value=f"{delta_energia:.2f}", delta="Delta de Lyapunov")
    with m_col3:
        st.metric(label="📊 Fidelidad Morfológica (Pearson R)", value=f"{correlacion:.3f}", delta="Grado de Identidad")

    # --- DISECCIÓN CRONOLÓGICA DE LA TRANSICIÓN ---
    st.divider()
    st.subheader("⏱️ Evidencia Temporal Diseccionada (Análisis Transitorio)")
    st.markdown("Evolución paso a paso congelada en memoria para tu reporte científico:")
    
    pasos_cronologicos = [0, 1, 2, 4, 8, 15, "Final"]
    columnas_tiempo = st.columns(len(pasos_cronologicos))
    
    for idx, paso in enumerate(pasos_cronologicos):
        with columnas_tiempo[idx]:
            if paso == 0:
                st.caption("🔹 **Paso 0**<br>Ruido Puro", unsafe_allow_html=True)
            elif paso == "Final":
                st.caption("🎯 **Final**<br>Estacionario", unsafe_allow_html=True)
            else:
                st.caption(f"⏳ **Paso {paso}**<br>Transitorio", unsafe_allow_html=True)
                
            if paso in snapshots:
                fig_snap = px.imshow(snapshots[paso].reshape(grid_size, grid_size), color_continuous_scale='RdBu_r', zmin=-1.5, zmax=1.5)
                fig_snap.update_layout(coloraxis_showscale=False, width=135, height=135, margin=dict(l=2,r=2,t=2,b=2), xaxis_visible=False, yaxis_visible=False)
                st.plotly_chart(fig_snap, use_container_width=True)