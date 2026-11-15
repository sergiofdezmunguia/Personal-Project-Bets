# Fase 4: margen, cuotas propias y CLV

Investigación y código del 2026-11-15. Solo simulación: nada coloca apuestas ni usa cuentas de casas.
Cifras generadas en [`fase4-resultados.md`](fase4-resultados.md) (`scripts/fase4_margen_clv.py`);
aquí, la lectura, el diseño y la auditoría.

## 1. Normalización del margen: Shin y potencia

`src/margin.py` añade dos métodos a la proporcional de las fases 2 y 3. Ninguno tiene parámetros
libres: Shin resuelve su z y potencia su k partido a partido (bisección, contrastada con `brentq` en
los tests).

**Selección (registrada antes de evaluar).** Regla fijada de antemano: la normalización con menor
log loss del cierre de Pinnacle en las temporadas de ajuste 2013/14–2016/17 pasa a ser la de
referencia. Resultado (`experiments/seleccion_fase4.json`):

| Normalización | Log loss cierre (ajuste) | Log loss apertura (ajuste) |
|---|---|---|
| proporcional | 0.99435 | 0.99652 |
| Shin | 0.99418 | 0.99633 |
| **potencia** | **0.99413** | **0.99626** |

**Evaluación 2017/18–2024/25** (26.660 partidos; Δ log loss emparejada, IC 95 % por semanas;
Δ > 0 es peor):

| Comparación | Δ log loss |
|---|---|
| Cierre potencia vs cierre proporcional | −0,00018 [−0,00037, +0,00002] |
| Cierre Shin vs cierre proporcional | −0,00016 [−0,00030, −0,00002] |
| Apertura potencia vs apertura proporcional | −0,00018 [−0,00039, +0,00003] |
| Apertura Shin vs apertura potencia | +0,00001 [−0,00006, +0,00008] |
| Apertura potencia vs cierre potencia | +0,00280 [+0,00180, +0,00378] |
| Residuo LightGBM (fase 3, proporcional) vs apertura proporcional | −0,00013 [−0,00022, −0,00005] (reproduce la fase 3) |
| Residuo LightGBM (fase 3, proporcional) vs apertura potencia | +0,00004 [−0,00015, +0,00023] |
| Residuo LightGBM reentrenado con potencia vs apertura potencia | −0,00008 [−0,00015, +0,00000] |
| Residuo logit reentrenado con potencia vs apertura potencia | −0,00005 [−0,00014, +0,00005] |
| Residuo LightGBM con potencia vs cierre potencia | +0,00272 [+0,00173, +0,00370] |

Lectura:

- **Shin y potencia mejoran la referencia por igual** (diferencia entre ellas indistinguible de 0) y
  bajan el ECE del cierre de 0,0067 a 0,0053. La elección de potencia en el ajuste es casi ruido
  (5·10⁻⁵), pero la regla estaba fijada antes y en la evaluación dan lo mismo.
- **La «mejora» de la fase 3 era la normalización:** frente a la apertura con potencia, el residuo
  de la fase 3 queda en +0,00004.
- **Reentrenado con la apertura con potencia**, sin tocar hiperparámetros, el residuo LightGBM queda
  en −0,00008 con el IC tocando 0. La auditoría lo da por real pero diminuto: no es calibración (un
  recalibrado solo de interceptos da +0,00003), aguanta el recorte simétrico (−0,00009) y el
  asimétrico queda por debajo de su nulo; pero se concentra en 2017/18–2019/20 y en D1 y E2, ronda 0
  desde 2020/21 y, con 6 variantes de residuo frente a 4 referencias en datos de evaluación ya
  mirados en la fase 3, no sobrevive a una corrección por multiplicidad. **No es una ventaja.**
- Nada gana al cierre: el mejor residuo queda a +0,0027 del cierre con potencia.

## 2. Cuotas propias con hora: fuentes

Para medir CLV en vivo hacen falta cuotas vistas a una hora conocida (la de la decisión) y el cierre.
Revisado el 2026-11-15 (términos y `robots.txt`):

| Fuente | Acceso | Condiciones | Qué da | Veredicto |
|---|---|---|---|---|
| **football-data `fixtures.csv`** | Libre, sin clave ni registro | `robots.txt`: `User-agent: *` sin `Disallow` (bloquea solo rastreadores de IA por nombre). El aviso legal no restringe la descarga; el proyecto ya usa sus CSV | Próximos partidos de muchas ligas con cuotas de bet365, Betfred, BetVictor, bwin, Paddy Power, Sky Bet, máximo, media y Betfair Exchange (1X2 y más/menos de 2,5). Se recogen el martes y el viernes por la tarde (notes.txt); `Last-Modified` da la versión | **Viable ya.** Dos instantáneas por semana, no T-1 h. El cierre llega después en el CSV de la temporada (`AvgC*`, `BFEC*`) con la misma clave de partido |
| **The Odds API** (v4) | Plan gratuito de 500 créditos/mes, **clave por correo tras registrarse** | Términos: prohíbe revender o redistribuir los datos como producto; uso propio permitido. Aviso de juego responsable si se promocionan casas | 1X2 (`h2h`, más `h2h_lay` en exchanges) por región, con `last_update` por casa. 1 crédito por liga con una región y un mercado | **Bloqueada: requiere registro.** Código listo y probado con respuestas inventadas |
| API de Betfair | Cuenta de Betfair y clave de aplicación | Requiere cuenta en una casa de apuestas | Precio de intercambio real, cierre exacto | Descartada: el encargo prohíbe abrir cuentas en casas |
| Raspar webs de casas | — | Lo prohíben sus términos | — | Descartada |

**Presupuesto de The Odds API gratis.** Con la región `eu` y solo `h2h`, una pasada por las 8 ligas
cuesta 8 créditos: 500/8 ≈ 62 pasadas al mes, unas 2 al día. Da para T-24 h y T-2 h de los partidos
del fin de semana en las 8 ligas, o para más horas en menos ligas, pero no para seguir el cierre a
T-10 min en todas. Las claves de deporte (`soccer_spain_la_liga`…) están en
`src/odds_collector.TOA_SPORTS` y faltan por verificar con la clave.

Fuentes: [football-data, notas](https://www.football-data.co.uk/notes.txt),
[aviso legal de football-data](https://www.football-data.co.uk/disclaimer.php),
[The Odds API, guía v4](https://the-odds-api.com/liveapi/guides/v4/),
[The Odds API, términos](https://the-odds-api.com/terms-and-conditions.html),
[límites del plan gratuito (blog de un competidor, posible sesgo)](https://oddspapi.io/blog/the-odds-api-free-tier-limits).

## 3. Recolector

`src/odds_collector.py` y `scripts/recolectar_cuotas.py`:

- **Instantánea:** fuente, partido (`event_key`), liga, saque en UTC, equipos, mercado, línea, casa,
  resultado, tipo (a favor o en contra), cuota, hora de la fuente y **`captured_utc`**, la hora UTC a
  la que la vimos por primera vez. Para decidir solo cuenta `captured_utc`: una cuota publicada antes
  pero vista después no estaba a nuestro alcance.
- **Idempotente:** SQLite (`data/odds/cuotas.sqlite`, excluido de git; o `BETS_ODDS_DIR`) con clave
  única (fuente, partido, mercado, línea, casa, resultado, tipo, versión de la fuente) e
  `INSERT OR IGNORE`. Repetir una captura no duplica filas y conserva la primera hora. football-data se
  pide con `If-Modified-Since`: sin cambios, la respuesta es un 304 vacío. Cada intento queda en la
  tabla `captures`.
- **Respeto a las fuentes:** agente de usuario propio, pausa de 2 s entre peticiones a The Odds API,
  parada si quedan menos de 20 créditos, y la clave nunca se guarda en la base.
- **Probado en real** el 2026-11-15 con football-data: 1.763 instantáneas de 51 partidos en la primera
  captura y 304 sin filas nuevas en la segunda. Los datos se quedan en local.
- **Hipótesis:** la hora de `fixtures.csv` es la del Reino Unido (notes.txt no lo dice); se convierte
  a UTC con `Europe/London`.

**Dónde corre (decisión abierta).** Con el PC apagado se pierden capturas. Para football-data el coste
es pequeño (cambia dos veces por semana y el cierre llega luego en su CSV): basta una tarea cada 6 h
en el Programador de tareas de Windows o cron en WSL. Para The Odds API, si llega la clave, conviene un
equipo siempre encendido; GitHub Actions no sirve para guardar datos sin subirlos a un repo público.

## 4. CLV

`src/clv.py`, con el mismo código para el histórico y para las cuotas propias:

1. **Precio al decidir:** la última instantánea con `captured_utc ≤ hora de decisión` y anterior al
   saque.
2. **Decisión:** 1 unidad al resultado de mayor valor esperado p_modelo · cuota_efectiva − 1, si
   supera un umbral fijado antes de mirar (0, 2 % y 5 %). La cuota efectiva descuenta la comisión
   del exchange sobre la ganancia: 1 + (cuota − 1)(1 − c).
3. **Cierre:** la última instantánea de la casa de referencia antes del saque (`max_close_gap_min`
   descarta «cierres» vistos demasiado pronto).
4. **CLV:** `clv_ev` = cuota_efectiva · p_cierre sin margen − 1 (la métrica de la puerta de la fase 4)
   y `clv_log` = ln(cuota tomada / cuota de cierre). IC 95 % por bootstrap de semanas.

**Antifuga (tests en `tests/test_clv.py`):** multiplicar por 1,7 todas las cuotas posteriores a la
decisión no cambia ninguna apuesta (solo el CLV); un mutante que toma la última cuota sin mirar la hora
lo detecta una guarda de `simulate`; las predicciones hechas al saque o después se descartan; las
cuotas en juego nunca hacen de cierre. Controles: la apertura sin margen usada como modelo no apuesta
nunca (paga el margen) y el cierre usado como modelo (fuga deliberada) da CLV positivo.

### CLV histórico (2017/18–2024/25, apuestas planas a la apertura de Pinnacle)

Reloj sintético (`clv.history_snapshots`): la apertura se ve a las 00:00 UTC del día ancla de la fase 3
(martes o viernes) y el cierre a las 23:58 del día del partido. La predicción existe en el ancla. CLV
frente al cierre de Pinnacle con potencia; sin comisión (casa, no exchange). Pinnacle no opera en
España: mide la calidad del precio, no una estrategia ejecutable.

| Modelo | Umbral | Apuestas | CLV [IC 95 %] | ROI plano [IC 95 %] |
|---|---|---|---|---|
| Residuo LightGBM, potencia | 0 % | 13 | −1,44 % [−5,52 %, +2,05 %] | −32,8 % [−87,4 %, +38,8 %] |
| Residuo logit, potencia | 0 % | 14 | −3,50 % [−5,52 %, −1,55 %] | −22,6 % [−71,4 %, +39,3 %] |
| Residuos | 2 % y 5 % | 0 | — | — |
| Dixon-Coles (ancla de apertura) | 0 % | 24.930 | −4,23 % [−4,40 %, −4,05 %] | −6,8 % [−8,9 %, −4,6 %] |
| Dixon-Coles (ancla de apertura) | 2 % | 22.750 | −4,32 % [−4,50 %, −4,13 %] | −7,0 % [−9,4 %, −4,7 %] |
| Dixon-Coles (ancla de apertura) | 5 % | 19.270 | −4,46 % [−4,66 %, −4,25 %] | −7,1 % [−9,7 %, −4,5 %] |

- **Los residuos casi nunca apuestan**, y es por construcción: corrigen la apertura en décimas de
  punto, y el margen medio es del 3,2 %. Con umbral 0 hay 13 y 14 apuestas; con 2 % y 5 %, ninguna.
  La auditoría midió que su EV máximo, en el percentil 99,9, es −0,14 %. **No es una prueba negativa
  sobre su información**, sino una falta de potencia. La señal más sensible es la correlación entre su
  corrección y el movimiento apertura→cierre: +0,03, +0,04 y +0,05 (local, empate y visitante),
  positiva en las 8 temporadas. Indica información real, pero muy pequeña.
- **Dixon-Coles cree ver valor en casi todos los partidos y paga el margen entero** (CLV −4,2 %, el
  ROI va en la misma dirección). Con más exigencia de EV, el CLV empeora: sus «valores» grandes son
  errores suyos, no del mercado.
- **Controles de la tubería (auditoría):** si se adelanta la decisión hasta después del cierre, apuesta
  al cierre, y con el cierre como oráculo el CLV sale +5,6 %. Ninguno de los 79.980 precios usados al
  decidir es el cierre.
- **Puerta a dinero real** (plan, fase 4): al menos 1.000 apuestas con el IC del CLV por encima de 0
  tras comisiones. Ningún modelo se acerca: o no apuesta o tiene un CLV claramente negativo.

## 5. Auditoría (`backtest-auditor`, 2026-11-15): aprobado con reservas, sin fugas

Comprobado: Shin y potencia contra `brentq` (error de 2·10⁻¹³ en 300 cierres); con el objetivo
barajado, la ventaja desaparece (Δ de +0,004); al meter el cierre como apertura, la fuga se detecta;
el reloj sintético no deja ver el cierre; 34/34 tests en verde. Reservas y qué se hizo con ellas:

1. **«Quitar el 1 % más favorable» es un estadístico sesgado.** Con colas gruesas sube la media
   aunque no haya diferencia: también el cierre parecía frágil. **Corregido:** recorte simétrico más
   el asimétrico frente a un nulo de signo aleatorio (`robustez`, tabla regenerada en
   `fase4-resultados.md`).
2. **El CLV de los residuos no sirve para sacar conclusiones** (no hay muestra). Documentado arriba.
   Para la fase siguiente, la señal es la correlación con el movimiento del mercado o un objetivo
   directo sobre ese movimiento (escalón 6 del plan).
3. **El −0,00008 está concentrado y decae** (2017/18–2019/20; D1 y E2), y no sobrevive a la
   multiplicidad. Documentado arriba.
4. **Recolector sin `Last-Modified`:** la versión pasaba a ser la hora de captura y se rompía la
   idempotencia. **Corregido:** se usa la huella del contenido, que nunca se manda como
   `If-Modified-Since`. Tiene test.
5. **El recorte del CLV con muestras pequeñas** (n = 13 quitaba el 7,7 %). **Corregido:** ahora es
   simétrico y solo se aplica con 200 apuestas o más.
6. **No comprobado:** la hora real de recogida de la apertura en football-data, The Odds API con
   datos reales (no hay clave) y la reconstrucción de la caché de variables desde cero.

## 6. Qué falta y qué bloquea

- **Bloquea (P7):** la clave de The Odds API. Hace falta registrarse, y el encargo prohíbe hacerlo
  sin autorización. Sin ella, las únicas cuotas propias son dos instantáneas a la semana de
  football-data.
- **Decisión abierta (P8):** dónde corre el recolector. La propuesta es cron o el Programador de
  tareas cada 6 h.
- **Simulación en vivo:** cuando haya capturas, `clv.simulate` sirve para las predicciones con hora
  (`decision_utc`). El cierre se toma de la última instantánea, con `max_close_gap_min`, o del CSV de
  temporada de football-data (`AvgC*`, `BFEC*`) con el mismo `event_key`. Con Betfair Exchange hay
  que aplicar la comisión real.
