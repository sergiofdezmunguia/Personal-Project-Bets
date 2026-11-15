# Decisiones

Registro de lo decidido y de lo que falta por decidir. Añade arriba lo más reciente.

## Abiertas

- **Visibilidad del repo:** seguir público o separar código público y resultados privados.
- **Dónde corre el recolector de cuotas:** debe funcionar con el PC apagado. Propuesta (fase 4):
  cron o Programador de tareas cada 6 h para football-data; equipo siempre encendido si llega la
  clave de The Odds API. GitHub Actions no, porque obligaría a subir datos al repo público.
- **Proveedor de cuotas en vivo:** football-data `fixtures.csv` ya funciona sin clave (2 instantáneas
  por semana). The Odds API (plan gratuito, ~2 pasadas al día en 8 ligas) necesita registro:
  pendiente de autorización. La API de Betfair queda descartada porque exige una cuenta en una casa.
- **Presupuesto máximo** si algún día se llega a la fase 5.
- **xG mientras Understat esté en mantenimiento:** esperar o evaluar otra fuente. Bloquea la unión
  de datos de la fase 1 y la primera ejecución con datos reales.
- **Renombres de equipos (fase 1):** `team_id` ya existe (ver 2026-11-14); falta revisar si algún
  club cambió de nombre en football-data entre 2012 y 2026.
- **Hora de decisión (fase 4):** resuelta para el histórico en la fase 3 (anclas de saque y apertura,
  ver 2026-11-15). Falta fijarla con cuotas propias con marca de tiempo.
- **Registro de experimentos en `train_xgboost.py`:** la fase 3 usa `scripts/fase3_ml.py`, que sí
  registra. El script antiguo sigue sin conectar: retirarlo o archivarlo.
- **Forma entre temporadas:** la ventana cruza temporadas, así que un equipo que vuelve tras un año
  usa partidos de hace dos temporadas. No es una fuga, pero está sin evaluar.

## Tomadas

### 2026-11-15: fase 4, normalización del margen, cuotas propias y CLV

- **Normalización de referencia: potencia** (`src/margin.py`). Se eligió con una regla fijada antes
  de mirar: el menor log loss del cierre de Pinnacle en el ajuste 2013/14–2016/17 (potencia 0,99413,
  Shin 0,99418, proporcional 0,99435). En la evaluación, Shin y potencia son indistinguibles y ambas
  mejoran a la proporcional en unas 2·10⁻⁴ y en calibración. **Motivo:** la proporcional reparte mal
  el margen entre el favorito y la sorpresa, y eso fabricaba la mejora del residuo de la fase 3. Las
  fases 2 y 3 se quedan con la proporcional, que sigue siendo el valor por defecto.
- **Residuos de la fase 3 reevaluados sin reajustar** (mismos hiperparámetros, apertura con
  potencia): LightGBM −0,00008 [−0,00015, +0,00000] frente a la apertura con potencia. La auditoría
  lo considera real, pero diminuto, concentrado (2017/18–2019/20; D1 y E2) y sin resistir la
  multiplicidad. **No es una ventaja.**
- **CLV:** `clv_ev` = cuota efectiva · p_cierre sin margen (potencia) − 1, con IC por semanas. La
  decisión solo ve las cuotas con `captured_utc` ≤ hora de decisión. Hay apuestas planas cuando el EV
  supera un umbral fijado antes de mirar (0, 2 % y 5 %). En el histórico, los residuos casi no
  apuestan (13–14 apuestas, porque su corrección es menor que el margen) y Dixon-Coles tiene un CLV
  del −4,2 %. Nadie se acerca a la puerta de la fase 4.
- **Recolector:** SQLite fuera de git (`data/odds/`), con clave única e `INSERT OR IGNORE`.
  football-data se pide de forma condicional; sin `Last-Modified`, la versión es la huella del
  contenido. The Odds API está listo, pero sin clave. **Motivo:** repetir una captura nunca debe
  cambiar los datos ni la hora en que vimos una cuota.
- **Robustez:** el recorte «quitar el 1 % más favorable» se sustituye por un recorte simétrico más el
  asimétrico frente a un nulo de signo aleatorio (auditoría). **Motivo:** el recorte asimétrico sube la
  media de cualquier diferencia con colas gruesas, incluso la del cierre frente a la apertura.
- **Auditoría (`backtest-auditor`): aprobada con reservas, sin fugas.** Las reservas están corregidas o
  documentadas en `docs/clv.md`.

### 2026-11-15: fase 3, ML y residuos sobre la apertura

- **Dos horas de decisión.** *Saque*: el partido del día D ve los resultados de días < D (como la fase
  2). *Apertura*: football-data recoge las cuotas previas «el viernes por la tarde para los partidos de
  fin de semana y el martes por la tarde para los de entre semana» (notes.txt). El ancla es el martes
  para mar–jue y el viernes para vie–lun, y se ven los resultados de días < ancla. `run_online` y
  Dixon-Coles aceptan el ancla; con la de la fase 2 el resultado es idéntico. **Motivo:** el modelo de
  residuos usa la apertura y no puede saber nada que la apertura no supiera.
- **Modelos.** LightGBM y CatBoost (saque, sin cuotas) sobre Elo, pi, Dixon-Coles, forma 5/20 por
  equipo, descanso y liga. Residuos sobre la apertura (LightGBM con `init_score` y softmax con
  desplazamiento L2). Un modelo por temporada entrenado con 2013/14…S−1. Ajuste en 2014/15–2016/17
  (2013/14 solo entrena), en dos rondas registradas: la segunda, solo hacia menos complejidad, porque
  la primera elegía el borde. Evaluación registrada una vez, en 2017/18–2024/25.
- **El calendario es público.** El descanso usa las fechas de los partidos anteriores aunque sean
  posteriores al ancla: las fechas se conocen de antemano.
- **Navidad.** Del 20-dic al 5-ene football-data recoge la apertura antes del ancla supuesta (lo
  detectó la auditoría). Se informa una tabla de sensibilidad sin esas jornadas; en la fase 4 la hora
  la darán las cuotas propias.
- **Corrección heredada de la fase 2.** En la primera temporada de cada liga, los equipos que debutan
  tarde ya no heredan la media del día menos el desplazamiento, sino el valor de su categoría. La fase 2
  se reevaluó (Elo y pi cambian como mucho 0,0001 de log loss) y conserva su selección de
  hiperparámetros.
- **Resultado (auditado):** nada gana al cierre. LightGBM +0,0147 [+0,0128, +0,0167], CatBoost +0,0155
  y residuos +0,0027. El residuo LightGBM mejora la apertura proporcional en −0,0001, pero no la
  apertura con Shin (+0,00003 [−0,00010, +0,00017]). Ver `docs/ml.md`. **Motivo para documentarlo
  así:** la mejora sale de la normalización, no de las variables.
- **Siguiente:** Shin o potencia como normalización de referencia (pasa a ser prioritaria) y CLV con
  cuotas propias.

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
- **Auditoría (`backtest-auditor`, 2026-11-15): aprobado con reservas**, sin fugas ni errores de
  métricas. Reservas corregidas: (1) el registro real exige `src/` y `scripts/` commiteados (la
  primera rejilla y la primera evaluación quedaron con `+cambios`; se repitieron con el código
  commiteado y las filas antiguas se conservan); (2) tests del corte semanal de Dixon-Coles en
  cualquier día de la semana; (3) tests antifuga con regresión entre temporadas y con un equipo
  nuevo; (4) gradiente de Dixon-Coles a cero donde τ se recorta; (5) la referencia de la fase 2 es
  **solo Pinnacle** (`pin_close_*`), sin recurrir a Betfair ni a la media.
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
