"""
tsp_solver.py
=============
Caso 2 · TSP exacto (depósito + 12 clientes)
II-1122 · Modelos de Optimización Industrial · UCR Sede Alajuela

Este módulo contiene SOLO el modelo matemático y su resolución.
La interfaz (app.py) lo importa y solo se encarga de mostrar resultados.

------------------------------------------------------------------
MODELO: TSP simétrico, ciclo cerrado, con eliminación de subciclos
         mediante restricciones MTZ (Miller–Tucker–Zemlin)
------------------------------------------------------------------

Conjuntos
    N = {0, 1, ..., n-1}           índices de los 13 puntos (0 = depósito)

Parámetros
    d[i][j] >= 0                   distancia entre el punto i y el punto j
                                    (simétrica: d[i][j] = d[j][i], diagonal = 0)

Variables de decisión
    x[i][j] ∈ {0,1}   para i ≠ j   1 si la ruta va directo de i a j
    u[i] ∈ [1, n-1]   para i ≠ 0   posición de i en el orden de visita
                                    (variable auxiliar, no representa nada
                                     físico por sí sola: solo sirve para
                                     impedir subciclos)

Función objetivo
    min  Σ_i Σ_j d[i][j] · x[i][j]            (i ≠ j)

Restricciones
    (R1) Σ_j x[i][j] = 1   ∀ i              sale exactamente un arco de cada punto
    (R2) Σ_i x[i][j] = 1   ∀ j              entra exactamente un arco a cada punto
    (R3) u[i] - u[j] + n·x[i][j] ≤ n-1      ∀ i,j ≠ 0, i ≠ j   (MTZ, anti-subciclo)
    (R4) 1 ≤ u[i] ≤ n-1                     ∀ i ≠ 0

Por qué R1+R2 NO bastan
    R1 y R2 por sí solas solo garantizan que cada punto tenga "grado 2"
    (una entrada, una salida). Eso permite soluciones inválidas como dos
    ciclos separados, por ejemplo {0-3-0} y {10-21-22-...-10}, que
    cumplen R1/R2 con menor distancia total pero NO son una ruta única
    que visite los 13 puntos. R3 (MTZ) elimina esa posibilidad: obliga
    a que el orden u[i] sea estrictamente creciente a lo largo de
    cualquier arco usado entre no-depósito, lo cual es imposible si
    existe un subciclo que no pasa por el depósito.

Coherencia de los datos (se valida en validar_matriz):
    - La matriz debe ser cuadrada (13x13 para este caso)
    - Simétrica: d[i][j] = d[j][i]   (es una red de doble vía)
    - Diagonal en cero: d[i][i] = 0   (distancia de un punto a sí mismo)
    - No negativa: d[i][j] >= 0       (una distancia no puede ser negativa)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pulp


# ---------------------------------------------------------------------------
# Validación de datos
# ---------------------------------------------------------------------------

class MatrizInvalidaError(ValueError):
    """Se lanza cuando la matriz de distancias no es coherente para un TSP."""


def validar_matriz(puntos: list[int], d: list[list[float]]) -> None:
    """
    Verifica que la matriz de distancias sea coherente para resolver un TSP
    simétrico real. Si algo no cuadra, lanza un error explicando exactamente
    qué está mal y dónde, para evitar resultados sin sentido.
    """
    n = len(puntos)

    if len(d) != n:
        raise MatrizInvalidaError(
            f"La matriz tiene {len(d)} filas pero hay {n} puntos. Deben coincidir."
        )

    for i, fila in enumerate(d):
        if len(fila) != n:
            raise MatrizInvalidaError(
                f"La fila {i} (punto {puntos[i]}) tiene {len(fila)} columnas, "
                f"se esperaban {n}."
            )

    for i in range(n):
        if d[i][i] != 0:
            raise MatrizInvalidaError(
                f"La diagonal debe ser 0 (distancia de un punto a sí mismo). "
                f"d[{puntos[i]}][{puntos[i]}] = {d[i][i]}."
            )
        for j in range(n):
            if d[i][j] < 0:
                raise MatrizInvalidaError(
                    f"Distancia negativa entre {puntos[i]} y {puntos[j]}: {d[i][j]}. "
                    f"Una distancia no puede ser negativa."
                )
            if d[i][j] != d[j][i]:
                raise MatrizInvalidaError(
                    f"La matriz no es simétrica entre {puntos[i]} y {puntos[j]}: "
                    f"d[{puntos[i]}][{puntos[j]}]={d[i][j]} pero "
                    f"d[{puntos[j]}][{puntos[i]}]={d[j][i]}. "
                    f"En este caso la red se asume de doble vía, por lo que "
                    f"la distancia debe ser igual en ambos sentidos."
                )

    if n < 3:
        raise MatrizInvalidaError(
            "Se necesitan al menos 3 puntos (depósito + 2 clientes) para que "
            "el TSP tenga sentido como recorrido cerrado."
        )


# ---------------------------------------------------------------------------
# Resultado del modelo
# ---------------------------------------------------------------------------

@dataclass
class ResultadoTSP:
    status: str
    factible: bool
    distancia_total: float | None
    secuencia_puntos: list[int] = field(default_factory=list)
    arcos: list[tuple[int, int, float]] = field(default_factory=list)  # (origen, destino, distancia)
    n_variables_binarias: int = 0
    n_variables_u: int = 0
    n_restricciones_asignacion: int = 0
    n_restricciones_mtz: int = 0
    tiempo_resolucion_seg: float = 0.0


# ---------------------------------------------------------------------------
# Modelo y resolución
# ---------------------------------------------------------------------------

def resolver_tsp(
    puntos: list[int],
    d: list[list[float]],
    deposito_index: int = 0,
) -> ResultadoTSP:
    """
    Construye y resuelve el modelo TSP (formulación MTZ) descrito arriba.

    Parameters
    ----------
    puntos : lista de etiquetas de los puntos, en el mismo orden que las
             filas/columnas de d. puntos[deposito_index] es el depósito.
    d      : matriz de distancias (ya validada con validar_matriz).
    deposito_index : índice (no etiqueta) del depósito dentro de `puntos`.

    Returns
    -------
    ResultadoTSP con la solución óptima, la secuencia de visita y el
    tamaño del modelo (útil para justificar por qué corre sin licencia
    comercial de AMPL/solvers).
    """
    import time

    validar_matriz(puntos, d)

    n = len(puntos)
    idx = list(range(n))

    prob = pulp.LpProblem("TSP_Caso2", pulp.LpMinimize)

    # Variables de decisión
    x = pulp.LpVariable.dicts(
        "x", [(i, j) for i in idx for j in idx if i != j], cat="Binary"
    )
    u = pulp.LpVariable.dicts(
        "u", [i for i in idx if i != deposito_index],
        lowBound=1, upBound=n - 1, cat="Continuous",
    )

    # Función objetivo: minimizar distancia total recorrida
    prob += pulp.lpSum(d[i][j] * x[(i, j)] for i in idx for j in idx if i != j)

    # R1: salida única de cada punto
    for i in idx:
        prob += pulp.lpSum(x[(i, j)] for j in idx if j != i) == 1, f"SalidaUnica_{i}"

    # R2: entrada única a cada punto
    for j in idx:
        prob += pulp.lpSum(x[(i, j)] for i in idx if i != j) == 1, f"EntradaUnica_{j}"

    # R3: eliminación de subciclos (MTZ), no involucra al depósito
    for i in idx:
        if i == deposito_index:
            continue
        for j in idx:
            if j == deposito_index or i == j:
                continue
            prob += (
                u[i] - u[j] + n * x[(i, j)] <= n - 1,
                f"MTZ_{i}_{j}",
            )

    n_mtz = (n - 1) * (n - 2)

    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    tiempo = time.time() - t0

    status = pulp.LpStatus[prob.status]
    factible = status == "Optimal"

    if not factible:
        return ResultadoTSP(
            status=status,
            factible=False,
            distancia_total=None,
            n_variables_binarias=n * (n - 1),
            n_variables_u=n - 1,
            n_restricciones_asignacion=2 * n,
            n_restricciones_mtz=n_mtz,
            tiempo_resolucion_seg=tiempo,
        )

    # Reconstruir arcos seleccionados (x[i,j] = 1)
    arcos_idx = [
        (i, j) for i in idx for j in idx
        if i != j and pulp.value(x[(i, j)]) > 0.5
    ]
    arcos = [(puntos[i], puntos[j], d[i][j]) for i, j in arcos_idx]

    # Reconstruir la secuencia de visita siguiendo los arcos desde el depósito
    siguiente = {i: j for i, j in arcos_idx}
    secuencia_idx = [deposito_index]
    actual = deposito_index
    for _ in range(n - 1):
        actual = siguiente[actual]
        secuencia_idx.append(actual)
    secuencia_idx.append(deposito_index)

    return ResultadoTSP(
        status=status,
        factible=True,
        distancia_total=pulp.value(prob.objective),
        secuencia_puntos=[puntos[i] for i in secuencia_idx],
        arcos=arcos,
        n_variables_binarias=n * (n - 1),
        n_variables_u=n - 1,
        n_restricciones_asignacion=2 * n,
        n_restricciones_mtz=n_mtz,
        tiempo_resolucion_seg=tiempo,
    )
