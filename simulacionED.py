import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.linalg import pinv
import time

# --- 1. CONFIGURACIÓN DE PÁGINA Y ESTÁTICA MINIMALISTA PREMIUM ---
st.set_page_config(page_title="Simulador de Hopfield", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; color: #1e293b; }
    h1, h2, h3 { color: #0f172a; font-weight: 600; letter-spacing: -0.02em; }
    .canvas-container { 
        background-color: #ffffff; 
        padding: 24px; 
        border-radius: 8px; 
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stHorizontalBlock"] { gap: 6px !important; }
    div.stToggle { margin-bottom: -12px !important; }
    div[data-testid="stMetricValue"] { color: #1e40af !important; font-weight: 600; }
    div[data-testid="stMetricLabel"] { color: #64748b !important; }
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
        self.W = self.W * 1.4

    def energy(self, x):
        v = np.tanh(x)
        return -0.5 * np.dot(v.T, np.dot(self.W, v))

# --- 3. PATRONES EN MEMORIA ---
def get_default_patterns(size):
    p1 = np.ones((size, size)) * -1
    np.fill_diagonal(p1, 1)
    
    p2 = np.ones((size, size)) * -1
    mid = size // 2
    p2[mid, :] = 1
    p2[:, mid] = 1
    
    p3 = np.ones((size, size)) * -1
    p3[0, :] = 1
    p3[:, mid] = 1
    
    p4 = np.ones((size, size)) * -1
    p4[2:-2, 2:-2] = 1
    
    return {"Diagonal": p1, "Cruz": p2, "Letra T": p3, "Cuadrado Interno": p4}

# --- 4. INTERFAZ PRINCIPAL ---
st.title("Simulador Dinámico de Hopfield")
st.markdown("Estudio visual del estado transitorio en redes neuronales auto-asociativas continuas.")

# --- CONTROL DE PARÁMETROS EN LA BARRA LATERAL + ECUACIÓN GENERAL ---
with st.sidebar:
    st.header("Parámetros del Modelo")
    grid_size = st.slider("Tamaño de rejilla (N x N)", 6, 10, 8)
    dt = st.slider("Paso de tiempo (dt)", 0.01, 0.2, 0.04)
    n_steps = st.slider("Pasos máximos", 50, 200, 100)
    noise_level = st.slider("Nivel de ruido inicial", 0.5, 2.5, 1.4)
    
    st.divider()
    # Regresa aquí la ecuación general teórica
    st.markdown("### Ecuación General Teórica")
    st.latex(r"\frac{dx_i}{dt} = -x_i + \sum_{j=1}^{N} w_{ij} \tanh(x_j)")

# --- ECUACIONES ESTÁTICAS DINÁMICAS EN LA PÁGINA PRINCIPAL ---
n_total_neurons = grid_size * grid_size
st.write("")
col_eq1, col_eq2 = st.columns(2)

with col_eq1:
    st.markdown(f"**Ecuación Real de la Red Actual ({n_total_neurons} Neuronas)**")
    st.latex(rf"\frac{{dx_i}}{{dt}} = -x_i + \sum_{{j=1}}^{{{n_total_neurons}}} w_{{ij}} \tanh(x_j)")
    
with col_eq2:
    st.markdown(f"**Solución Numérica en Tiempo Real (Método de Euler, dt = {dt})**")
    st.latex(rf"x_i(t + {dt}) = x_i(t) + {dt} \cdot \left( -x_i(t) + \sum_{{j=1}}^{{{n_total_neurons}}} w_{{ij}} \tanh(x_j) \right)")

patterns_dict = get_default_patterns(grid_size)

# --- SECCIÓN 1: DEFINICIÓN DEL PATRÓN ---
st.divider()
col_izq, col_der = st.columns([1.1, 1])

with col_izq:
    st.subheader("1. Configuración del Recuerdo")
    opcion_patron = st.radio(
        "Origen del patrón objetivo:",
        ["Catálogo predefinido", "Dibujar manualmente en lienzo"],
        horizontal=True
    )
    
    target_pattern = np.ones((grid_size, grid_size)) * -1
    
    if opcion_patron == "Catálogo predefinido":
        target_name = st.selectbox("Selecciona un patrón:", list(patterns_dict.keys()))
        target_pattern = patterns_dict[target_name]
        
        fig_target = px.imshow(target_pattern, color_continuous_scale='Greys')
        fig_target.update_layout(coloraxis_showscale=False, width=160, height=160, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_target, use_container_width=False)
        
    else:
        if 'canvas_toggles' not in st.session_state or st.session_state.canvas_toggles.shape != (grid_size, grid_size):
            st.session_state.canvas_toggles = np.zeros((grid_size, grid_size), dtype=bool)
        
        if st.button("Limpiar lienzo completamente"):
            st.session_state.canvas_toggles = np.zeros((grid_size, grid_size), dtype=bool)
            for i in range(grid_size):
                for j in range(grid_size):
                    key = f"toggle_{i}_{j}"
                    st.session_state[key] = False
            st.rerun()
            
        st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
        for i in range(grid_size):
            cols = st.columns(grid_size)
            for j in range(grid_size):
                with cols[j]:
                    key = f"toggle_{i}_{j}"
                    valor_actual = st.session_state.get(key, False)
                    estado = st.toggle("", value=bool(valor_actual), key=key, label_visibility="collapsed")
                    st.session_state.canvas_toggles[i, j] = estado
        st.markdown('</div>', unsafe_allow_html=True)
                    
        target_pattern = np.where(st.session_state.canvas_toggles, 1, -1)

hopfield = HopfieldNetwork(grid_size)
hopfield.train_pseudoinverse([target_pattern])

with col_der:
    st.subheader("2. Detalles Técnicos")
    with st.expander("Ver Matriz de Pesos Sinápticos (W)"):
        st.markdown("Representación matemática de las conexiones:")
        fig_w = px.imshow(hopfield.W, color_continuous_scale='Blues')
        fig_w.update_layout(width=340, height=280, margin=dict(l=0, r=0, t=0, b=0), template="plotly_white")
        st.plotly_chart(fig_w, use_container_width=True)

# --- SECCIÓN 2: SIMULACIÓN ---
st.divider()
st.subheader("3. Ejecución de la Simulación")

if st.button("Lanzar Simulación Dinámica", type="primary"):
    initial_state = target_pattern.flatten() + np.random.normal(0, noise_level, grid_size**2)
    
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        st.caption("Evolución de la rejilla (Diverso-Color):")
        plot_spot = st.empty()
    with col_sim2:
        st.caption("Decrecimiento de la energía (Lyapunov):")
        energy_spot = st.empty()
    
    snapshots = {}
    snapshots[0] = initial_state.copy()
    
    x = initial_state.copy()
    history_energy = []
    pasos_clave = [1, 2, 4, 8, 15]

    for i in range(1, n_steps + 1):
        v = np.tanh(x)
        dx = -x + np.dot(hopfield.W, v)
        x += dx * dt
        
        history_energy.append(hopfield.energy(x))
        
        if i in pasos_clave:
            snapshots[i] = x.copy()
            
        if i % 2 == 0:
            fig_neurons = px.imshow(x.reshape(grid_size, grid_size), color_continuous_scale='balance', zmin=-1.5, zmax=1.5)
            fig_neurons.update_layout(coloraxis_showscale=False, width=280, height=280, margin=dict(l=0,r=0,t=0,b=0), template="plotly_white")
            plot_spot.plotly_chart(fig_neurons, use_container_width=True)
            
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Scatter(y=history_energy, mode='lines', line=dict(color='#2563eb', width=2)))
            fig_energy.update_layout(xaxis_title="Pasos", yaxis_title="Energía", template="plotly_white", height=240, margin=dict(l=0,r=0,t=10,b=0))
            energy_spot.plotly_chart(fig_energy, use_container_width=True)
            time.sleep(0.02)

    snapshots["Final"] = x.copy()
    
    final_binario = np.sign(x)
    objetivo_plano = target_pattern.flatten()
    pixeles_diferentes = np.sum(final_binario != objetivo_plano)
    porcentaje_error = (pixeles_diferentes / len(objetivo_plano)) * 100
    delta_energia = history_energy[-1] - history_energy[0]
    
    if porcentaje_error == 0:
        st.info("Estado de convergencia: Atractor global alcanzado con éxito.")
    else:
        st.warning("Estado de convergencia: Mínimo local o estado parcial.")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric(label="Error final de reconstrucción", value=f"{porcentaje_error:.1f} %")
    with m_col2:
        st.metric(label="Diferencial de Energía (ΔE)", value=f"{delta_energia:.2f}")

    # --- LÍNEA DE TIEMPO DE LA TRANSICIÓN ---
    st.write("")
    st.subheader("4. Análisis de la Transición Temporal")
    st.markdown("Disección del estado transitorio. Se observa con precisión matemática y color el paso del caos a la estructura:")
    
    pasos_cronologicos = [0, 1, 2, 4, 8, 15, "Final"]
    columnas_tiempo = st.columns(len(pasos_cronologicos))
    
    for idx, paso in enumerate(pasos_cronologicos):
        with columnas_tiempo[idx]:
            if paso == 0:
                st.caption("**Paso 0** (Ruido)")
            elif paso == "Final":
                st.caption("**Final** (Estable)")
            else:
                st.caption(f"**Paso {paso}**")
                
            if paso in snapshots:
                fig_snap = px.imshow(
                    snapshots[paso].reshape(grid_size, grid_size),
                    color_continuous_scale='balance', 
                    zmin=-1.5, zmax=1.5
                )
                fig_snap.update_layout(
                    coloraxis_showscale=False, width=120, height=120,
                    margin=dict(l=2,r=2,t=2,b=2), xaxis_visible=False, yaxis_visible=False,
                    template="plotly_white"
                )
                st.plotly_chart(fig_snap, use_container_width=True)
