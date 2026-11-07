# Plan del modelo de apuestas

Investigación del 2026-11-06. Etiquetas de evidencia:

- **[comprobado]**: verificado en los ficheros o servicios reales ese día.
- **[fuente]**: lo dice una fuente citada; no lo hemos reproducido.
- **[hipótesis]**: criterio propio pendiente de probar.

## La pregunta correcta

La pregunta no es qué modelo acierta más, sino **si el modelo encuentra precios que el mercado
corrige después**. La ventaja, si existe, dependerá más del mercado elegido, del momento de la
apuesta y de la disciplina de evaluación que del algoritmo.

Ninguna fuente revisada demuestra una ventaja estable frente a las cuotas de cierre:

- Winkelmann et al. (2023), con 14 temporadas: las ineficiencias aparecen en temporadas sueltas, no
  de forma persistente, y simulando mercados eficientes también salen «efectos» en temporadas
  aisladas. **[fuente]**
- Kaunitz et al. (2017) ganaron explotando diferencias entre casas, pero algunas limitaron sus
  cuentas a las pocas semanas de apostar con dinero real. **[fuente]**

Por tanto, «no hay ventaja explotable» es un resultado posible y válido para el portfolio.

## Estado actual del código [comprobado]

| Hallazgo | Consecuencia |
|---|---|
| `train_xgboost.py` no usa cuotas | No puede medir si se gana al mercado |
| `rolling` sin agrupar por equipo | **Corregido en la fase 0**, con test de regresión |
| División 70/15/15 por filas | **Corregido en la fase 0:** ahora corta por día natural |
| Solo 2 ligas y 2 temporadas | Muestra insuficiente |
| La constitución describe BeSoccer y el código usa soccerdata | Documentación desfasada |

## Fuentes de datos

### Núcleo: fútbol

Comprobado sobre La Liga, LaLiga 2, Premier League y League One en football-data.co.uk. La descarga
es gratuita.

| Dato | Disponibilidad [comprobado] |
|---|---|
| Resultados, tiros y córners | Sí en las temporadas revisadas; LaLiga 2 no trae tiros ni córners en 2013/14, y sí en 2019/20 |
| Pinnacle apertura y cierre, 1X2 (`PSH`, `PSCH`) | Desde 2012/13 |
| Cierre medio y máximo (`AvgCH`, `MaxCH`) | Sí desde 2019/20; ausentes en 2013/14; intermedias sin revisar |
| Over/Under 2,5 y hándicap asiático de cierre | Sí desde 2019/20; temporadas anteriores sin revisar |
| Betfair Exchange de cierre (`BFECH`, `BFEC>2.5`, `BFECAHH`) | Desde 2024/25 |

**Hallazgo clave:** el cierre de Pinnacle deja de publicarse a mitad de la 2025/26 (último partido
entre el 25-nov-2025 y el 12-ene-2026, según la liga) y las columnas no existen en la 2026/27.
Consecuencias:

- Hasta enero de 2026 la referencia histórica puede ser Pinnacle.
- Desde entonces hay que usar **Betfair Exchange de cierre** y la media de cierre del mercado.
- La 2024/25 y parte de la 2025/26 tienen ambas: sirven para medir cuánto difieren antes de cambiar.
- Betfair cubre entre el 92 % y el 95 % de los partidos revisados en 2025/26 y prácticamente todos en
  2026/27. Falta medir su liquidez en divisiones bajas.

Además, la última fila de La Liga 2026/27 era del 2-nov al consultar el 6-nov. Sirve como histórico,
**no para predecir antes del partido**.

### xG y calidad de juego

| Fuente | Estado | Uso |
|---|---|---|
| Understat | **En mantenimiento el 6-nov-2026:** responde HTTP 200 con la página «We will come back soon», fechada el 3-nov, también en sus URL de datos **[comprobado]**. Cobertura amplia solo en grandes ligas **[fuente]** | xG desde 2014/15 en las grandes ligas, cuando vuelva |
| FBref | Perdió xG y métricas avanzadas de Opta en enero de 2026 **[fuente]** | No depender de ella. Ya bloqueaba `soccerdata` |
| FotMob | Ofrece xG en muchas competiciones **[fuente]**; condiciones sin revisar | Candidata a evaluar |
| StatsBomb Open Data | Solo competiciones seleccionadas **[fuente]** | Aprender eventos; no para producción |
| ClubElo | La API dio 502 y la web 403 **[comprobado]** | No depender de ella: calcular nuestro propio Elo |

### Cuotas en tiempo real

El histórico gratuito solo da aperturas y cierres. Para apostar en un momento concreto hay que
**recoger nuestras propias cuotas desde ya**, porque ese histórico no se puede recuperar gratis.

| Fuente | Notas |
|---|---|
| The Odds API | 1 crédito por región y mercado; su histórico es de pago **[fuente]**. Falta verificar la cuota gratuita y qué casas cubre |
| API de Betfair | Precio de intercambio real. Falta verificar acceso desde España y condiciones |
| Betfair histórico | Gratis a 1 minuto solo con último precio y sin volumen; de pago con más detalle; desde 2016 **[fuente]** |

### Otros deportes

- **Tenis:** tennis-data.co.uk descarga 2025 y 2026 con Pinnacle, Bet365, máximo y media; 2026 añade
  Betfair Exchange. Falta medir cobertura y si son de cierre. **[comprobado]**
- El repositorio `JeffSackmann/tennis_atp`, muy citado, devuelve 404 hoy. **[comprobado]** Hay que
  buscar alternativa para estadísticas detalladas.

## Qué mercados atacar

Orden propuesto. **[hipótesis]**, salvo lo citado.

| # | Mercado | Por qué | Pega |
|---|---|---|---|
| 1 | **Fútbol, Over/Under 2,5 y 1X2, en grandes ligas** | Mejor dato gratuito. Es el laboratorio donde validar el método | Probablemente el más eficiente |
| 2 | **Fútbol, segundas y terceras divisiones** | Menos competencia y dinero **[fuente: planteamiento de Elaad et al. 2020]** | No comprobamos si los precios son peores; menos liquidez |
| 3 | **Hándicap asiático** | Margen típicamente menor; se deriva del mismo modelo de goles | Cuotas menos disponibles en casas españolas: verificar |
| 4 | **Tenis, ganador del partido** | Sin empate, muchos partidos al año, modelo de jugador claro | Lesiones y retiradas; hay que verificar datos |
| 5 | **Córners, tarjetas y jugadores** | Plausiblemente menos eficientes | **Sin histórico gratuito de cierre: hoy no se pueden validar.** Solo tras acumular cuotas propias |

## Qué modelos probar

Cada escalón debe ganar al anterior fuera de muestra.

1. **Mercado sin margen:** la referencia. Probablemente ganará a todo lo demás en log loss.
2. **Elo y pi-ratings:** referencia simple. Los pi-ratings superaron a Elo y un estudio mostró
   beneficio en cinco temporadas de Premier, sin reproducción posterior que hayamos encontrado. **[fuente]**
3. **Dixon-Coles o Poisson bivariante:** modela goles, así que da 1X2, totales y hándicaps coherentes
   entre sí. Funciona con pocos datos.
4. **CatBoost o LightGBM con ratings y xG:** árboles como CatBoost sobre pi-ratings fueron los
   mejores en datos solo de goles, según un trabajo de 2023. **[fuente]**
5. **Modelo de residuos del mercado:** parte de la cuota de apertura sin margen y aprende solo la
   corrección. **[hipótesis]** Es el candidato con más sentido para encontrar ventaja real, porque no
   intenta redescubrir lo que el mercado ya sabe.
6. **Predecir el movimiento de la cuota hacia el cierre:** objetivo directo sobre CLV. **[hipótesis]**
   Requiere cuotas propias con marca de tiempo.

**Elección del modelo:** por calibración y log loss, no por accuracy. Walsh y Joshi (2024) obtuvieron
+34,69 % de ROI medio eligiendo por calibración frente a −35,17 % por accuracy, en una sola temporada
de NBA: es indicio, no ley. **[fuente]**

**Tamaño de apuesta en simulación:** fracción de Kelly con tope por apuesta y por jornada. Aplicar
siempre la comisión real del exchange.

## Métricas

| Capa | Métrica | Para qué |
|---|---|---|
| Predicción | Log loss, RPS, Brier | Frente al mercado sin margen |
| Calibración | Curva de fiabilidad, ECE | Que un 60 % ocurra un 60 % de las veces |
| Precio | **CLV** frente al cierre | Señal con menos ruido que el resultado. Según Buchdahl, el beneficio puede exigir miles de apuestas para ser significativo **[fuente, no independiente]** |
| Negocio | ROI, caída máxima, intervalo por bootstrap | Tras comisiones y solo a precios realmente disponibles |

## Desde cuándo y cada cuánto

### Profundidad histórica

| Uso | Periodo | Motivo |
|---|---|---|
| Entrenamiento | Desde 2014/15, con más peso a lo reciente | Coincide con xG de Understat. Sin xG se puede ir a 2012/13 |
| Validación por ventanas | 2021/22 a 2024/25 | Cierre de Pinnacle completo. Revisar aparte 2019/20 y 2020/21: partidos sin público |
| Prueba final | 2025/26, una sola vez | Sin tocar hasta congelar el modelo |
| Simulación en vivo | 2026/27 en adelante | Con cuotas propias |

### Calendario de cargas [hipótesis]

| Tarea | Frecuencia |
|---|---|
| Cuotas | Cada 12 h hasta T-48 h; luego T-24 h, T-6 h, T-1 h y T-10 min |
| Resultados y xG | Cada mañana y tras cada jornada |
| Histórico football-data | Semanal, para conciliar |
| Ratings | Tras cada jornada |
| Predicciones | T-24 h y T-2 h, con registro inmutable |
| Reentrenamiento | Mensual, y solo se promueve si gana al modelo vigente |

El recolector de cuotas no puede depender de que el PC esté encendido, porque perdería los cierres.
Hay que decidir dónde se ejecuta: GitHub Actions es gratuito en repos públicos, pero sus tareas
programadas pueden retrasarse, justo cuando importa el T-10 min.

## Fases

| Fase | Entregable | Criterio para avanzar |
|---|---|---|
| **0. Base** | Corregir el error de `rolling`, división por fecha, tests antifuga | Tests verdes; reproducible desde cero |
| **1. Datos** | Histórico unificado y conciliado de football-data + Understat | Informe de calidad: cobertura, duplicados, nombres de equipo |
| **2. Referencia** | Mercado sin margen, Elo, pi-ratings, Dixon-Coles | Tabla comparativa con intervalos |
| **3. ML** | CatBoost/LightGBM y modelo de residuos, por ventanas | Auditoría superada; prueba final una sola vez |
| **4. Simulación** | Recolector de cuotas y apuestas ficticias en vivo | **Puerta a dinero real [propuesta]:** al menos 1.000 apuestas simuladas, CLV positivo con el límite inferior del IC 95 % por encima de 0 tras comisiones, y precios disponibles en un operador legal |
| **5. Real (opcional)** | Solo con aprobación expresa, presupuesto cerrado y dinero que se pueda perder | Se detiene si incumple límites prefijados |

Para el portfolio bastan las fases 0 a 4: cuentan con rigor una historia de ML con validación
temporal, datos reales y un resultado honesto, aunque sea negativo.

## Marco legal y fiscal en España

Información orientativa; confirmar con un asesor antes de la fase 5.

- Pinnacle no tiene licencia española y cerró cuentas de clientes en España en 2012 **[fuente secundaria]**.
- Betfair figura en el buscador de operadores de la DGOJ **[fuente]**. Confirmar la licencia vigente
  y si cubre el intercambio directamente en el registro.
- Hacienda: los premios se declaran en la base imponible general (apartado F1 de la renta). Las
  pérdidas solo se compensan hasta el importe de las ganancias del juego del mismo año **[fuente: AEAT]**.
- Las casas pueden limitar a quien gana de forma consistente **[fuente]**. Si ocurre, el ROI simulado
  no se trasladaría al real.

## Fuentes

- football-data.co.uk: [notas](https://www.football-data.co.uk/notes.txt) y CSV por temporada.
- [Winkelmann et al. 2023](https://pub.uni-bielefeld.de/record/2983980)
- [Kaunitz et al. 2017](https://pdf.arxiv.org/pdf/1710.02824)
- Elaad, Reade y Singleton 2020, *Finance Research Letters* (leído solo el resumen de la búsqueda; localizar el artículo).
- [Walsh y Joshi 2024](https://researchportal.bath.ac.uk/en/publications/machine-learning-for-sports-betting-should-model-selection-be-bas/)
- [Constantinou y Fenton 2013, pi-ratings](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2012-0036/html)
- [Árboles potenciados sobre pi-ratings, arXiv 2023](https://arxiv.org/pdf/2309.14807v1.pdf)
- [Buchdahl sobre CLV, reproducido por terceros](https://pod-d614f9k2t-pod-team.vercel.app/blog/closing-line-value--clv-demystified-by-expert-joseph-buchdahl)
- [The Odds API](https://the-odds-api.com/liveapi/guides/v4/)
- [Betfair, datos históricos](https://betfair-datascientists.github.io/data/usingHistoricDataSite)
- [FBref tras perder datos de Opta](https://footyapps.com/guide/fbref-alternatives)
- [DGOJ, buscador de operadores](https://www.ordenacionjuego.es/en/operadores-juego/operadores-licencia/operadores)
- [Pinnacle en España, fuente secundaria](https://miscasasdeapuestas.com/articulos/pinnacle-apuestas-opiniones)
- [AEAT, premios en el IRPF](https://sede.agenciatributaria.gob.es/Sede/ayuda/manuales-videos-folletos/manuales-practicos/irpf-2024/c11-ganancias-perdidas-patrimoniales/determinacion-importe-ganancias-perdidas-patrimon-generales/no-derivadas-transmisiones-elementos-patrimoniales.html)
