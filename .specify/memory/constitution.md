# Football Betting ML Pipeline Constitution

## Core Principles

### I. Modular Pipeline Architecture
El pipeline de datos y modelos debe estar claramente separado en módulos con responsabilidades únicas:
- **api_client**: Comunicación con la API de BeSoccer (URLs, parámetros, peticiones HTTP)
- **data_ingestion**: Transformación de JSON crudos a estructuras tabulares en data/processed
- **preprocessing**: Feature engineering respetando restricciones temporales
- **train**: División de datos, entrenamiento de modelos, evaluación y guardado de artefactos en models
- **predict**: Carga de modelos entrenados y generación de predicciones

Cada módulo debe ser independiente y tener interfaces claras. No mezclar responsabilidades.

### II. Temporal Integrity (NON-NEGOTIABLE)
No deben usarse datos futuros para construir features de partidos pasados:
- Splits de entrenamiento/validación por tiempo (temporadas antiguas → recientes), no aleatorios
- Features calculadas únicamente con información disponible antes del partido objetivo
- Rachas, clasificaciones y estadísticas acumuladas siempre previas a la fecha del partido
- Validación estricta en cada cambio del pipeline de datos

### III. Traceability & Reproducibility
Garantizar trazabilidad completa de datos y experimentos:
- Toda llamada significativa a la API debe dejar volcado en data/raw con nombres que incluyan fecha y contexto (liga, año, tipo)
- Guardar datasets procesados en data/processed con convenciones claras
- Fijar seeds (random_state) en splits y modelos
- Resultados de evaluación guardados de forma reproducible (JSON/texto en models o reports)
- Configuración y secretos exclusivamente en .env, nunca hardcodeados

### IV. Incremental & Reversible Development
Priorizar cambios pequeños, documentados y fáciles de revertir:
- Mini-pasos verificables: datos de una temporada → target simple → modelo básico → añadir features
- Evitar refactors masivos sin necesidad clara
- No romper interfaces públicas sin analizar impacto
- Documentar decisiones importantes (estrategia de split, métricas, features) en comentarios breves

### V. BeSoccer API Best Practices
Centralización, throttling y validación obligatorios:
- Todas las llamadas HTTP centralizadas en BeSoccerClient, sin requests.get sueltos
- Siempre incluir: key (API key), format=json, tz=Europe/Madrid
- Throttling simple basado en tiempo mínimo entre llamadas
- Reutilizar respuestas guardadas en data/raw antes que llamar repetidamente
- Validar que claves y estructuras esenciales están presentes; fallar con error claro si estructura inesperada

## Technology Stack

**Lenguaje base**: Python 3 (compatible ≥3.6)

**Librerías de referencia**:
- requests, python-dotenv (API y configuración)
- pandas, numpy (manejo de datos)
- scikit-learn (modelos baseline), xgboost (cuando el entorno lo permita)
- joblib (serialización de modelos)

**Estructura de proyecto**:
- `src/` — Paquete Python principal
- `data/raw` — Datos crudos de API
- `data/processed` — Datasets limpios y con features
- `models/` — Modelos entrenados y artefactos
- `notebooks/` — Exploración y prototipado

**Entorno**:
- Uso obligatorio de entornos virtuales (.venv/)
- Dependencias centralizadas en requirements.txt
- Evitar dependencias globales implícitas

## Code Quality Standards

**Nombres y estructura**:
- snake_case en funciones, variables y módulos
- Funciones pequeñas con una responsabilidad clara
- Comentarios solo donde aporten claridad, no reiteren código obvio

**Manejo de errores**:
- Errores de red/API manejados explícitamente con try/except cuando proceda
- No ocultar fallos críticos silenciosamente
- Mensajes de error claros con endpoint y parámetros clave cuando fallen llamadas API

**Verificación continua**:
- Tras generar/modificar código ejecutable: razonar sobre imports, rutas y dependencias
- Tras cambios en pipeline de datos: verificar que columnas críticas (IDs, fecha, goles, target) existen con tipos correctos
- Tras cambios en modelos: validar que entrenan con subconjunto de datos y artefactos se pueden guardar/cargar

## Model Evaluation & Metrics

Para clasificación multiclase (1X2):
- Usar métricas apropiadas: accuracy, log-loss u otras acordadas
- Guardar resultados de forma reproducible (JSON/texto junto al modelo)
- Documentar estrategia de evaluación (temporadas train/valid/test)

Extensiones futuras (tarjetas, corners):
- Definir targets específicos (regresión o clasificación por bins)
- Aplicar mismos principios de temporalidad y reproducibilidad

## Behavior Under Uncertainty

**Explicitar supuestos**:
- Comentar claramente suposiciones sobre estructuras de respuestas de API
- Facilitar ajustes futuros dejando código fácil de adaptar

**Proponer alternativas**:
- Si xgboost no se puede instalar: usar RandomForest/GradientBoosting de scikit-learn
- Si API no da cierta métrica: aproximar con datos derivados o ajustar alcance

**No ocultar errores graves**:
- Comunicar claramente: fallo de conexión, corrupción de datos, imposibilidad de entrenar modelo
- Sugerir pasos concretos para solucionarlo

## Interaction Style

**Conciso y orientado a acción**:
- Proponer cambios con información suficiente: qué archivo, qué función, qué efecto en pipeline
- Evitar texto redundante, usar listas/bloques cuando clarifiquen

**Foco en objetivo global**:
- Construir pipeline robusto para datos de fútbol
- Modelo útil para apuestas (resultado, tarjetas, corners)
- No desviarse a complejidades innecesarias mientras partes básicas no funcionen

**Priorizar mini-pasos verificables**:
- Obtener datos de una temporada → construir target simple → probar modelo básico → añadir features y complejidad

## Governance

Esta constitución supersede todas las demás prácticas del proyecto. Toda modificación de código, adición de features o cambios en el pipeline debe verificar cumplimiento con estos principios.

Complejidad debe estar justificada. Simplicidad por defecto (YAGNI).

**Version**: 1.0.0 | **Ratified**: 2026-02-15 | **Last Amended**: 2026-02-15
