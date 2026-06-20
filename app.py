"""
app.py
======
Caso 2 · TSP exacto (depósito + 12 clientes) · Interfaz Streamlit
II-1122 · Modelos de Optimización Industrial · UCR Sede Alajuela

Esta app SOLO se encarga de mostrar la interfaz. Toda la lógica del
modelo matemático vive en tsp_solver.py.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tsp_solver import MatrizInvalidaError, ResultadoTSP, resolver_tsp, validar_matriz

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="Caso 2 · TSP exacto",
    page_icon="🚚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Estilos mínimos (tipografía y tarjetas de métricas)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stMetric { background-color: #F4F6F8; border-radius: 10px; padding: 12px 16px; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .bloque-formula {
        background-color: #0E1117;
        color: #E8EAED;
        padding: 18px 22px;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        font-size: 0.92rem;
        line-height: 1.55;
        white-space: pre-wrap;
    }
    .nota-caja {
        background-color: #FFF8E1;
        border-left: 4px solid #F2B705;
        padding: 10px 16px;
        border-radius: 4px;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

@st.cache_data
def cargar_datos():
    with open(DATA_DIR / "distancias.json", encoding="utf-8") as f:
        dist = json.load(f)
    with open(DATA_DIR / "coordenadas.json", encoding="utf-8") as f:
        coords_raw = json.load(f)
    coords = {int(k): tuple(v) for k, v in coords_raw.items()}
    return dist["puntos"], dist["matriz"], coords


@st.cache_data
def resolver_cacheado(puntos: list[int], matriz_tuple: tuple, deposito_index: int):
    matriz = [list(fila) for fila in matriz_tuple]
    return resolver_tsp(puntos, matriz, deposito_index)


puntos, matriz_base, coords = cargar_datos()
n_puntos = len(puntos)

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------

st.title("🚚 Caso 2 · Camino más corto + TSP exacto")
st.caption("II-1122 · Modelos de Optimización Industrial · UCR Sede Alajuela · Bloque 3, Clase 15")

st.markdown(
    """
Una empresa de distribución sale de su **depósito (punto 0)**, visita a sus **12 clientes**
exactamente una vez cada uno, y regresa al depósito — recorriendo la **menor distancia total posible**.

La red vial real tiene cientos de calles. Por eso el problema se resuelve en **dos etapas**:
primero se calculan los caminos más cortos entre cada par de puntos relevantes (eso ya viene
resuelto en la matriz de abajo), y luego se resuelve el **TSP exacto** sobre esos 13 puntos —
que es el paso que hace esta aplicación.
    """
)

tab_datos, tab_modelo, tab_resultado, tab_escenarios = st.tabs(
    ["📊 1. Datos", "🧮 2. Modelo matemático", "✅ 3. Solución óptima", "🔄 4. Escenarios"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Datos
# ---------------------------------------------------------------------------
with tab_datos:
    st.subheader("Matriz de distancias (km)")
    st.markdown(
        "13 puntos: el **punto 0 es el depósito**, los otros 12 son clientes. "
        "La matriz es **simétrica** (la distancia de ida es igual a la de vuelta) "
        "y la diagonal es cero (distancia de un punto a sí mismo)."
    )

    df_matriz = pd.DataFrame(matriz_base, index=puntos, columns=puntos)
    st.dataframe(
        df_matriz.style.background_gradient(cmap="Blues").format("{:.0f}"),
        use_container_width=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Puntos totales", n_puntos)
        st.metric("Clientes a visitar", n_puntos - 1)
    with col_b:
        st.metric("Distancia mínima entre dos puntos", f"{min(v for row in matriz_base for v in row if v > 0):.0f} km")
        st.metric("Distancia máxima entre dos puntos", f"{max(max(row) for row in matriz_base):.0f} km")

    with st.expander("✔️ Verificación de coherencia de los datos"):
        try:
            validar_matriz(puntos, matriz_base)
            st.success(
                "La matriz es cuadrada, simétrica, con diagonal en cero y sin distancias "
                "negativas. Es una matriz válida para un TSP simétrico."
            )
        except MatrizInvalidaError as e:
            st.error(f"Problema detectado en los datos: {e}")

        st.markdown(
            """
<div class="nota-caja">
<b>Nota técnica:</b> al venir de caminos más cortos sobre una red vial real (no de un plano
euclidiano), la matriz puede tener pequeñas violaciones de la desigualdad triangular
(d(i,k) &gt; d(i,j) + d(j,k) en algunos tríos). Esto es normal — refleja restricciones reales
de las calles (sentidos, rotondas, etc.) — y no afecta la validez del modelo TSP, que solo
necesita que la matriz sea simétrica y no negativa.
</div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Mapa relativo de los 13 puntos")
    st.caption(
        "Las posiciones se generaron a partir de las distancias de la matriz (no son "
        "coordenadas GPS reales), de forma que la separación visual entre dos puntos "
        "es proporcional a la distancia real entre ellos."
    )
    fig_mapa = go.Figure()
    xs = [coords[p][0] for p in puntos]
    ys = [coords[p][1] for p in puntos]
    colors = ["#D7263D" if p == 0 else "#1B998B" for p in puntos]
    sizes = [22 if p == 0 else 16 for p in puntos]
    fig_mapa.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="markers+text",
            text=[f"{'Depósito' if p == 0 else p}" for p in puntos],
            textposition="top center",
            marker=dict(size=sizes, color=colors, line=dict(width=1, color="white")),
            hovertemplate="Punto %{text}<extra></extra>",
        )
    )
    fig_mapa.update_layout(
        height=420, showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_mapa, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 — Modelo matemático
# ---------------------------------------------------------------------------
with tab_modelo:
    st.subheader("Formulación: TSP simétrico con eliminación de subciclos (MTZ)")

    st.markdown("**Conjuntos y parámetros**")
    st.markdown(
        f"""
- `N = {{0, 1, ..., {n_puntos - 1}}}` → los {n_puntos} puntos (índice 0 = depósito)
- `d[i][j]` → distancia entre el punto *i* y el punto *j* (dato, viene de la matriz)
        """
    )

    st.markdown("**Variables de decisión**")
    st.markdown(
        """
- `x[i][j] ∈ {0,1}` para *i ≠ j* → vale 1 si la ruta va directo de *i* a *j*, 0 si no
- `u[i] ∈ [1, n-1]` para *i ≠ 0* → posición de *i* en el orden de visita (variable
  auxiliar; solo sirve para impedir subrutas, no representa nada físico por sí sola)
        """
    )

    st.markdown("**Función objetivo**")
    st.markdown(
        '<div class="bloque-formula">min  Σᵢ Σⱼ d[i][j] · x[i][j]      (i ≠ j)\n\n'
        "Minimizar la distancia total recorrida en el ciclo completo.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Restricciones**")

    st.markdown("*(R1) Salida única — de cada punto sale exactamente un arco*")
    st.markdown(
        '<div class="bloque-formula">Σⱼ x[i][j] = 1     para todo i ∈ N  (j ≠ i)</div>',
        unsafe_allow_html=True,
    )

    st.markdown("*(R2) Entrada única — a cada punto entra exactamente un arco*")
    st.markdown(
        '<div class="bloque-formula">Σᵢ x[i][j] = 1     para todo j ∈ N  (i ≠ j)</div>',
        unsafe_allow_html=True,
    )

    st.markdown("*(R3) Eliminación de subciclos (MTZ)*")
    st.markdown(
        '<div class="bloque-formula">u[i] - u[j] + n·x[i][j] ≤ n - 1     '
        "para todo i,j ≠ depósito, i ≠ j</div>",
        unsafe_allow_html=True,
    )

    st.markdown("*(R4) Rango de la variable auxiliar de orden*")
    st.markdown(
        '<div class="bloque-formula">1 ≤ u[i] ≤ n - 1     para todo i ≠ depósito</div>',
        unsafe_allow_html=True,
    )

    with st.expander("¿Por qué R1 + R2 no son suficientes? (la parte que más se presta a confusión)"):
        st.markdown(
            """
R1 y R2 por sí solas solo obligan a que **cada punto tenga una entrada y una salida**.
Eso es necesario, pero **no garantiza una sola ruta conectada**: el solver podría
"hacer trampa" y armar varios ciclos pequeños y separados que, sumados, también
cumplen R1 y R2 — por ejemplo un ciclo `0 → 3 → 0` y otro aparte
`21 → 35 → 22 → 21`, completamente desconectados entre sí.

Matemáticamente esas soluciones serían más baratas (menos distancia), pero **no
sirven para la empresa**: el repartidor necesita un único recorrido que pase por
los 13 puntos en una sola vuelta, no varias rutas sueltas.

La restricción R3 (MTZ) impide esto: obliga a que la variable auxiliar `u` crezca
de forma estrictamente ordenada a lo largo de cualquier arco usado entre dos
clientes. Un subciclo que no toque el depósito generaría una contradicción en
ese orden, así que el solver no puede formarlo. Es la pieza que convierte
"un conjunto de ciclos válidos por separado" en **una única ruta cerrada**.
            """
        )

    st.markdown("**Tamaño del modelo para esta instancia (13 puntos)**")
    n = n_puntos
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Variables binarias x", n * (n - 1))
    col2.metric("Variables continuas u", n - 1)
    col3.metric("Restricciones R1+R2", 2 * n)
    col4.metric("Restricciones MTZ (R3)", (n - 1) * (n - 2))

    st.markdown(
        """
<div class="nota-caja">
Esta instancia es pequeña (156 variables binarias, ~170 restricciones en total),
por eso corre sin problema en cualquier solver MIP de uso libre/educativo
(CBC, incluido en esta app) o en versiones sin licencia comercial de AMPL.
Una red vial completa con cientos de nodos generaría miles de variables y
decenas de miles de restricciones MTZ — fuera de ese rango.
</div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# TAB 3 — Solución óptima
# ---------------------------------------------------------------------------
with tab_resultado:
    st.subheader("Resolver el modelo")
    st.caption("Resuelve el TSP exacto sobre los 13 puntos con el solver CBC (libre, sin licencia).")

    if st.button("▶️ Ejecutar optimización", type="primary"):
        with st.spinner("Resolviendo el modelo (puede tardar unos segundos)..."):
            resultado: ResultadoTSP = resolver_cacheado(
                puntos, tuple(tuple(row) for row in matriz_base), 0
            )
        st.session_state["resultado"] = resultado

    resultado: ResultadoTSP | None = st.session_state.get("resultado")

    if resultado is None:
        st.info("Presiona el botón para calcular la ruta óptima.")
    elif not resultado.factible:
        st.error(
            f"El modelo no encontró solución factible (estado: {resultado.status}). "
            "Esto indicaría un problema en los datos de entrada, no en la formulación."
        )
    else:
        st.success(f"Solución óptima encontrada en {resultado.tiempo_resolucion_seg:.2f} segundos.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Distancia total óptima", f"{resultado.distancia_total:.0f} km")
        c2.metric("Puntos visitados", n_puntos)
        c3.metric("Estado del solver", resultado.status)

        col_izq, col_der = st.columns([3, 2])

        with col_izq:
            st.markdown("**Mapa de la ruta óptima**")
            fig = go.Figure()

            # arcos de la ruta
            seq = resultado.secuencia_puntos
            for a, b in zip(seq[:-1], seq[1:]):
                xa, ya = coords[a]
                xb, yb = coords[b]
                fig.add_trace(
                    go.Scatter(
                        x=[xa, xb], y=[ya, yb], mode="lines",
                        line=dict(width=2.5, color="#1B998B"),
                        hoverinfo="skip", showlegend=False,
                    )
                )

            xs = [coords[p][0] for p in puntos]
            ys = [coords[p][1] for p in puntos]
            colors = ["#D7263D" if p == 0 else "#2462A6" for p in puntos]
            sizes = [24 if p == 0 else 18 for p in puntos]
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="markers+text",
                    text=[f"{'Depósito' if p == 0 else p}" for p in puntos],
                    textposition="top center",
                    marker=dict(size=sizes, color=colors, line=dict(width=1, color="white")),
                    hovertemplate="Punto %{text}<extra></extra>",
                    showlegend=False,
                )
            )
            fig.update_layout(
                height=460,
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_der:
            st.markdown("**Secuencia de visita**")
            secuencia_str = " → ".join(
                "Depósito" if p == 0 else str(p) for p in resultado.secuencia_puntos
            )
            st.markdown(f"`{secuencia_str}`")

            st.markdown("**Arcos seleccionados**")
            df_arcos = pd.DataFrame(resultado.arcos, columns=["Desde", "Hasta", "Distancia (km)"])
            st.dataframe(df_arcos, use_container_width=True, hide_index=True)

        st.markdown("**Tamaño del modelo resuelto**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Variables binarias", resultado.n_variables_binarias)
        c2.metric("Variables u", resultado.n_variables_u)
        c3.metric("Restricciones asignación", resultado.n_restricciones_asignacion)
        c4.metric("Restricciones MTZ", resultado.n_restricciones_mtz)

# ---------------------------------------------------------------------------
# TAB 4 — Escenarios
# ---------------------------------------------------------------------------
with tab_escenarios:
    st.subheader("¿Qué pasa si cambian los datos?")
    st.caption(
        "Modifica la matriz para simular un cierre de calle (aumento de distancia) "
        "y vuelve a resolver. Útil para responder la Parte III del laboratorio."
    )

    st.markdown("**Simular cierre temporal de una vía entre dos puntos**")
    col1, col2, col3 = st.columns(3)
    with col1:
        origen = st.selectbox("Punto A", puntos, index=0, key="origen_sel")
    with col2:
        destino_opciones = [p for p in puntos if p != origen]
        destino = st.selectbox("Punto B", destino_opciones, index=0, key="destino_sel")
    with col3:
        i_o, i_d = puntos.index(origen), puntos.index(destino)
        distancia_actual = matriz_base[i_o][i_d]
        nueva_distancia = st.number_input(
            "Nueva distancia (km)",
            min_value=0.0,
            value=float(distancia_actual),
            step=1.0,
            help=f"Distancia actual entre {origen} y {destino}: {distancia_actual} km",
        )

    if st.button("🔁 Recalcular con el cambio"):
        matriz_mod = [row[:] for row in matriz_base]
        matriz_mod[i_o][i_d] = nueva_distancia
        matriz_mod[i_d][i_o] = nueva_distancia  # se mantiene simétrica

        try:
            validar_matriz(puntos, matriz_mod)
            with st.spinner("Resolviendo escenario modificado..."):
                resultado_mod = resolver_tsp(puntos, matriz_mod, 0)
            st.session_state["resultado_mod"] = resultado_mod
            st.session_state["matriz_mod_info"] = (origen, destino, distancia_actual, nueva_distancia)
        except MatrizInvalidaError as e:
            st.error(f"Cambio no válido: {e}")

    resultado_mod: ResultadoTSP | None = st.session_state.get("resultado_mod")
    resultado_base: ResultadoTSP | None = st.session_state.get("resultado")

    if resultado_mod is not None and resultado_mod.factible:
        o, d_, antes, despues = st.session_state["matriz_mod_info"]
        st.markdown(f"Cambio aplicado: distancia entre **{o}** y **{d_}**: {antes} km → **{despues} km**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ruta original**")
            if resultado_base is not None and resultado_base.factible:
                st.write(" → ".join("Dep." if p == 0 else str(p) for p in resultado_base.secuencia_puntos))
                st.metric("Distancia original", f"{resultado_base.distancia_total:.0f} km")
            else:
                st.info("Ejecuta primero la optimización en la pestaña 3 para comparar.")
        with col2:
            st.markdown("**Ruta con el cambio**")
            st.write(" → ".join("Dep." if p == 0 else str(p) for p in resultado_mod.secuencia_puntos))
            st.metric("Distancia nueva", f"{resultado_mod.distancia_total:.0f} km")
            if resultado_base is not None and resultado_base.factible:
                delta = resultado_mod.distancia_total - resultado_base.distancia_total
                if delta == 0:
                    st.caption("La ruta óptima no cambió: el arco modificado no formaba parte de la solución óptima.")
                else:
                    st.caption(f"Cambio en distancia total: {delta:+.0f} km")
                if resultado_mod.secuencia_puntos != resultado_base.secuencia_puntos:
                    st.caption("⚠️ La secuencia completa de visita cambió, no solo el tramo modificado.")

    st.markdown("---")
    st.markdown("**Simular un cliente nuevo**")
    st.caption(
        "No agrega un punto real (no tenemos sus distancias), pero muestra cómo "
        "crece el tamaño del modelo — la base de la pregunta 8 del laboratorio."
    )
    n_actual = n_puntos
    n_nuevo = n_actual + 1
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Con {n_actual} puntos (actual)**")
        st.write(f"Variables binarias: {n_actual*(n_actual-1)}")
        st.write(f"Restricciones MTZ: {(n_actual-1)*(n_actual-2)}")
    with c2:
        st.markdown(f"**Con {n_nuevo} puntos (+1 cliente)**")
        st.write(f"Variables binarias: {n_nuevo*(n_nuevo-1)}  "
                  f"({n_nuevo*(n_nuevo-1) - n_actual*(n_actual-1):+d})")
        st.write(f"Restricciones MTZ: {(n_nuevo-1)*(n_nuevo-2)}  "
                  f"({(n_nuevo-1)*(n_nuevo-2) - (n_actual-1)*(n_actual-2):+d})")

st.markdown("---")
st.caption(
    "Repositorio del Caso 2 · Modelo TSP con formulación MTZ · "
    "Solver: PuLP + CBC (libre, sin licencia) · UCR Sede Alajuela"
)
