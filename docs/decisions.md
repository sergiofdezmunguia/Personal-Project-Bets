# Decisiones

Registro de lo decidido y de lo que falta por decidir. Añade arriba lo más reciente.

## Abiertas

- **Visibilidad del repo:** seguir público o separar código público y resultados privados.
- **Dónde corre el recolector de cuotas:** debe funcionar con el PC apagado.
- **Proveedor de cuotas en vivo:** The Odds API o API de Betfair; verificar cuota gratuita, casas y acceso desde España.
- **Presupuesto máximo** si algún día se llega a la fase 5.
- **xG mientras Understat esté en mantenimiento:** esperar o evaluar otra fuente. Bloquea la unión
  de datos de la fase 1 y la primera ejecución con datos reales.
- **Renombres de equipos (fase 1):** `team_id` ya existe (ver 2026-11-14); falta revisar si algún
  club cambió de nombre en football-data entre 2012 y 2026.
- **Hora de decisión (antes de las fases 3 y 4):** `shift(1)` da por conocido el partido anterior y su
  xG. Decidiendo en T-48 h, con poco descanso entre partidos, ese partido aún no se habría jugado.
- **Registro de experimentos en `train_xgboost.py`:** el registro ya existe (ver 2026-11-15), pero el
  script antiguo sigue sobrescribiendo su JSON. Conectarlo al registro cuando se rehaga en la fase 3.
- **Normalización del margen:** la referencia usa la proporcional. Probar Shin o potencia como
  variantes registradas antes de calcular CLV en la fase 4.
- **Forma entre temporadas:** la ventana cruza temporadas, así que un equipo que vuelve tras un año
  usa partidos de hace dos temporadas. No es una fuga, pero está sin evaluar.

## Tomadas

### 2026-11-15: fase 2, modelos de referencia y su validación

- **Registro de experimentos:** `experiments/registro.jsonl`, una línea JSON por variante evaluada
  (fecha, variante, fase, parámetros, métricas agregadas, hash de datos y commit). Solo se abre en
  modo añadir y cada fila lleva el hash de la anterior: `src/experiments.verify` detecta filas
  borradas o editadas, y `log` se niega a escribir sobre un registro roto. Se sube al repo (solo
  agregados). **Motivo:** el recuento de variantes es lo que permite juzgar un resultado positivo.
- **Calendario de temporadas:** 2012/13 solo arranca ratings; 2013/14–2016/17 es el ajuste de
  hiperparámetros (rejilla fijada antes de mirar, todas las variantes registradas, se elige por
  log loss); 2017/18–2024/25 es la evaluación por ventanas. `load_history` ni siquiera abre los CSV
  de 2025/26 o posteriores. **Motivo:** elegir hiperparámetros en las mismas temporadas que se
  informan sesgaría la comparación a favor de los modelos.
- **Ventanas:** Elo y pi-ratings se actualizan al cerrar cada día natural (los partidos del mismo día
  no se ven entre sí, porque no hay hora antes de 2019/20); el paso de rating a 1X2 es un logit
  ordenado reajustado cada temporada con las temporadas previas. Dixon-Coles se reajusta cada lunes
  con partidos anteriores a ese lunes (máximo 4 años) y peso exp(−ξ·días), un modelo por país
  (las divisiones inglesas y españolas se enlazan por ascensos). **Motivo:** es la información que
  existiría el día del partido.
- **Equipos nuevos:** empiezan en la media de su liga menos un desplazamiento (Elo −100, pi −0,2);
  en Dixon-Coles los llegados hace menos de un año se encogen hacia la media de su liga desplazada
  (ataque −δ, defensa +δ, δ en la rejilla). En la primera temporada Elo usa 1500 − 150 por categoría.
  **Motivo:** los ascendidos desde una división no descargada suelen ser más débiles que la media.
- **Métricas e intervalos:** log loss, RPS, Brier y ECE (10 tramos) en el conjunto común de partidos
  con probabilidad de todos los modelos. IC 95 % por bootstrap de semanas ISO (2.000 remuestras) y
  diferencias emparejadas frente al cierre sin margen. **Motivo:** los partidos de una jornada
  comparten noticias y condiciones; remuestrear partidos sueltos estrecharía los intervalos.
- **Apertura de Pinnacle como variante:** se evalúa también la apertura sin margen, para tener una
  escala de cuánto mejora el mercado hasta el cierre.

### 2026-11-14: identidad de equipo y columnas `PP*` de 2026/27

`team_id` es el nombre de football-data normalizado (`src/football_data.py`). En las 8 ligas
descargadas ningún nombre se repite entre países y el mismo club conserva el nombre al subir o
bajar, así que el historial no se corta. Los renombres se resolverían con `TEAM_ALIASES` (sin
revisar aún). Las columnas `PP*` que aparecen en 2026/27 **no son Pinnacle**: su margen medio es
del 8,3 % frente al 3,4 % de `PS*` en 2024/25 (probablemente Paddy Power). Se guardan como `pp_*`
y nunca sirven de referencia.
**Motivo:** confundirlas habría dado por buena una referencia con un margen de casa recreativa.

### 2026-11-06: división de desarrollo por día y 2025/26 bloqueada en el código

`split_by_date` corta por día natural, redondeando a la baja hasta el último día completo que cabe
en 70/15/15. `temporal_split` rechaza cualquier fila desde el 1-jul-2025. Su `test` es de desarrollo,
no la prueba final.
**Motivo:** un corte por filas parte una jornada, y la regla de la prueba final debe fallar de forma
visible en vez de depender de la disciplina. Es un corte provisional: la validación por ventanas
llegará en la fase 3.

### 2026-11-06: entorno reproducible

Python 3.11. Dependencias directas en `requirements.in` y versiones exactas en `requirements.txt`,
generado con `uv pip compile`. XGBoost es obligatorio, sin modelo alternativo, y usa `n_jobs=4`.
**Motivo:** sin versiones fijas no se reproduce un resultado. El modelo alternativo cambiaba de
algoritmo en silencio, y con todos los núcleos el ajuste tardaba 53 s frente a 0,04 s en WSL, con
predicciones idénticas.

### 2026-11-06: referencia de mercado tras Pinnacle

Pinnacle hasta enero de 2026 y Betfair Exchange de cierre con la media del mercado desde entonces.
**Motivo:** las columnas de Pinnacle desaparecen de football-data a mitad de la 2025/26.
**Propuesta pendiente de validar** comparando ambas en 2024/25.

### 2026-11-06: empezar simulando, con el mercado como referencia

Sin dinero real ni apuestas automáticas. Criterios de avance en `docs/plan-modelo.md`.
**Motivo:** la evidencia revisada no demuestra una ventaja persistente.
