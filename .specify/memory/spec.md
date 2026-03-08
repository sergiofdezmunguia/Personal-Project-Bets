# Football Betting ML Pipeline - Initial Specification

## Overview
Sistema de machine learning para predicción de resultados de partidos de fútbol y otros mercados de apuestas utilizando datos históricos de la API de BeSoccer.

## Goals
- **Primary**: Predecir resultado 1X2 (local/empate/visitante) de partidos de fútbol
- **Secondary** (future): Predecir tarjetas amarillas/rojas, corners, over/under goles

## Data Source
- **API**: BeSoccer (http://apiclient.besoccerapps.com/scripts/api/api.php)
- **API Key**: Configurada en `.env` (500 llamadas gratuitas disponibles)
- **Key Endpoints**:
  - `req=matchs&league=X` - Partidos por liga
  - `req=tables&league=X&group=Y` - Clasificación
  - `req=matchsday` - Partidos del día
  - `req=league_status&id=X&year=Y` - Estado de temporada

## Technical Stack
- Python 3.6+
- pandas, numpy (data manipulation)
- scikit-learn (baseline models)
- xgboost (optional, for production models)
- requests, python-dotenv (API & config)

## Architecture

### Module Structure
```
src/
├── api_client.py      # BeSoccer API communication
├── data_ingestion.py  # JSON → DataFrames
├── preprocessing.py   # Feature engineering
├── train.py          # Model training
└── predict.py        # Predictions
```

### Data Flow
```
API (BeSoccer) 
  → data/raw/*.json (raw responses)
  → data_ingestion 
  → data/processed/*.csv (normalized tables)
  → preprocessing (features + target)
  → train (model artifacts in models/)
  → predict (probabilities)
```

## Core Requirements

### R1: Temporal Integrity
- NO usar datos futuros para features de partidos pasados
- Splits de train/valid/test por tiempo (temporadas), NO aleatorios
- Features calculadas solo con información previa al partido

### R2: Data Traceability
- Toda llamada a API debe guardar respuesta en `data/raw/` con timestamp y contexto
- Datasets procesados en `data/processed/` con nombres descriptivos
- Resultados de evaluación guardados reproduciblemente

### R3: API Resource Management
- Máximo 500 llamadas disponibles
- Throttling en cliente (rate limiting)
- Reutilizar datos guardados antes que llamar nuevamente

### R4: Modular & Testable
- Cada módulo con responsabilidad única
- Interfaces claras entre módulos
- Código verificable por pasos incrementales

## Current State

### Implemented ✅
- Project structure (src/, data/, models/, notebooks/)
- `.gitignore` (ignoring .venv, .env, __pycache__)
- `requirements.txt` (sin xgboost por problemas de instalación)
- `src/config.py` (carga .env, define rutas)
- `src/api_client.py` (BeSoccerClient con métodos básicos)
- `src/fetch_sample_data.py` (script para descargar 5 JSONs de ejemplo)
- `notebooks/01_exploracion.ipynb` (vacío)

### Blockers ⚠️
- Script `fetch_sample_data.py` no ejecutado exitosamente
- Problemas con `.venv` (rutas rotas)
- `.env` no cargando correctamente en algunos casos
- No tenemos JSONs reales para validar estructura de API

## Next Milestones

### Milestone 1: Data Acquisition & Validation
**Goal**: Obtener y validar datos reales de la API

**Tasks**:
1. Fix Python environment (`.venv`)
2. Verify `.env` configuration
3. Execute `fetch_sample_data.py` successfully
4. Analyze JSON structure in notebook
5. Document actual API response format

**Acceptance Criteria**:
- ✅ JSONs guardados en `data/raw/`
- ✅ Notebook con análisis de estructura (columnas, tipos)
- ✅ Documentación de campos clave identificados

### Milestone 2: Data Pipeline
**Goal**: Pipeline robusto de datos crudos a dataset entrenable

**Tasks**:
1. Implement `data_ingestion.py` (JSON → DataFrame)
2. Define schema for matches table
3. Define schema for standings table
4. Save processed data to `data/processed/`
5. Implement basic data quality checks

**Acceptance Criteria**:
- ✅ CSV de partidos normalizado en `data/processed/`
- ✅ CSV de clasificaciones normalizado
- ✅ Script ejecutable end-to-end
- ✅ Mínimo 1 temporada completa de LaLiga procesada

### Milestone 3: Feature Engineering & Target
**Goal**: Features útiles y target bien definido

**Tasks**:
1. Calculate `target_result` (1=home win, 0=draw, -1=away win)
2. Implement pre-match standings features (position, points, goal_diff)
3. Implement recent form features (last 5 matches)
4. Implement head-to-head features
5. Validate temporal integrity of features

**Acceptance Criteria**:
- ✅ Dataset con features + target
- ✅ No data leakage (verificado)
- ✅ Features documentadas en código

### Milestone 4: Baseline Model
**Goal**: Modelo funcional end-to-end

**Tasks**:
1. Implement `train.py` with temporal split
2. Train RandomForestClassifier (baseline)
3. Evaluate with accuracy + log-loss
4. Save model artifacts to `models/`
5. Implement `predict.py` to load and use model

**Acceptance Criteria**:
- ✅ Modelo entrenado y guardado
- ✅ Métricas de evaluación calculadas y guardadas
- ✅ Pipeline reproducible documentado

## Open Questions
1. ¿Estructura exacta de respuesta JSON de `req=matchs`? (nombre del array, campos disponibles)
2. ¿ID de LaLiga en BeSoccer? (asumir `league=1` hasta confirmar)
3. ¿Vienen tarjetas/corners en respuesta de partidos o endpoint separado?
4. ¿Parámetros para obtener jornadas específicas o rangos de fechas?
5. ¿Formato de temporada/season en los endpoints? (2023, 2023-24, etc.)

## Out of Scope (v1)
- Múltiples ligas simultáneas
- Cuotas de casas de apuestas
- Modelos avanzados (deep learning)
- Deployment/serving del modelo
- UI/dashboard

## Success Metrics
- **Data**: Mínimo 380 partidos (1 temporada LaLiga) procesados correctamente
- **Model**: Accuracy > baseline (predicción aleatoria = 33% para 1X2)
- **Reproducibility**: Cualquier persona puede ejecutar pipeline completo desde cero
- **Code Quality**: Cumple constitution (modular, temporal integrity, trazabilidad)
