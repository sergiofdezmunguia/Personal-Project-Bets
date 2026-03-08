---
agent: speckit.project
---

# Football Betting ML Pipeline - Project Overview

## Project Goal

Desarrollar un sistema completo de machine learning para predicción de resultados de partidos de fútbol y otros mercados de apuestas (tarjetas, corners) utilizando datos históricos de la API de BeSoccer.

## Domain Context

**Objetivo principal**: Predecir resultados de partidos de fútbol (1X2: local/empate/visitante) mediante modelos de ML entrenados con datos históricos.

**Objetivos secundarios** (futuro):
- Predicción de tarjetas amarillas/rojas (local y visitante)
- Predicción de corners totales
- Predicción de over/under goles, corners, etc.

**Fuente de datos**: API de BeSoccer (http://apiclient.besoccerapps.com/scripts/api/api.php)
- 500 llamadas gratuitas disponibles
- Endpoints clave:
  - `req=matchs&league=X` - Partidos por liga/jornada
  - `req=tables&league=X&group=Y` - Clasificación de liga
  - `req=matchsday` - Partidos del día
  - `req=league_status&id=X&year=Y` - Estado/estructura de temporada
  - `req=categories&filter=my_leagues` - Ligas disponibles

## Current State

**Estructura creada**:
```
bets/
├── .env (con BESOCCER_API_KEY configurada)
├── .gitignore (ignora .venv, .env, __pycache__, data temporal)
├── requirements.txt (sin xgboost por ahora, resto de deps básicas)
├── src/
│   ├── __init__.py
│   ├── config.py (carga .env, define rutas)
│   ├── api_client.py (BeSoccerClient adaptado a la API real)
│   └── fetch_sample_data.py (script para descargar JSONs de ejemplo)
├── data/
│   ├── raw/ (para JSONs de la API)
│   └── processed/ (para CSVs limpios)
├── models/ (para modelos entrenados)
└── notebooks/
    └── 01_exploracion.ipynb (notebook inicial, aún vacío)
```

**Código implementado**:
- `BeSoccerClient` con métodos para:
  - `get_my_leagues()`, `get_league_status()`, `get_matches_day()`, `get_matches_by_league()`, `get_league_table()`
- `fetch_sample_data.py` que hace 5 llamadas de ejemplo y guarda JSONs en `data/raw/`

**Bloqueadores actuales**:
- El script `fetch_sample_data.py` no se ha podido ejecutar con éxito aún (problemas de entorno: .venv roto, .env no cargado correctamente)
- No tenemos JSONs reales para analizar estructura de respuestas
- No sabemos exactamente qué campos devuelve la API en partidos y clasificación

## Next Immediate Steps

1. **Ejecutar `fetch_sample_data.py` con éxito**:
   - Verificar que `.env` tiene `BESOCCER_API_KEY` y `BESOCCER_BASE_URL` correctos
   - Recrear `.venv` limpio si hace falta
   - Ejecutar: `python -m src.fetch_sample_data` desde raíz con `.venv` activado
   - Confirmar que se crean JSONs en `data/raw/`

2. **Análisis de estructura de datos** (en `01_exploracion.ipynb`):
   - Cargar JSONs de `matchs_league_1_*.json` y `tables_league_1_group_1_*.json`
   - Usar `pd.json_normalize` para ver columnas disponibles
   - Identificar campos clave:
     - Para partidos: IDs, fecha, equipos, goles, jornada, tarjetas, corners (si existen)
     - Para clasificación: team_id, posición, puntos, PJ, PG, PE, PP, GF, GC, racha

3. **Crear módulo `data_ingestion.py`**:
   - Funciones para normalizar JSONs de la API a DataFrames
   - Guardar datasets base en `data/processed/` (ej: `matches_laliga_2023.csv`)
   - Definir columnas estándar: match_id, league_id, date, round, home_team_id, away_team_id, home_goals, away_goals

4. **Definir target y primeras features** (en `preprocessing.py`):
   - Calcular `target_result` multiclase: 1 (gana local), 0 (empate), -1 (gana visitante)
   - Features básicas de clasificación previa al partido:
     - home_position, away_position, home_points, away_points
     - home_goal_diff, away_goal_diff
     - Diferencias: points_diff, pos_diff

5. **Prototipo de modelo baseline**:
   - Script `train.py` que:
     - Carga datos procesados
     - Aplica features
     - Split temporal (ej: 80% train, 20% valid por fecha)
     - Entrena RandomForestClassifier (baseline sin xgboost)
     - Evalúa con accuracy y log-loss
     - Guarda modelo en `models/`

## Key Constraints & Requirements

**Temporalidad estricta**:
- No usar datos futuros para features de partidos pasados
- Splits siempre por tiempo, no aleatorios
- Features basadas solo en info previa al partido

**Trazabilidad**:
- Toda llamada significativa a API → volcado en `data/raw/`
- Datasets procesados en `data/processed/` con nombres claros
- Resultados de evaluación guardados reproduciblemente

**API BeSoccer**:
- Máximo 500 llamadas (no desperdiciar)
- Throttling en cliente para respetar rate limits
- Reutilizar JSONs guardados antes que volver a llamar

**Código limpio**:
- Módulos separados por responsabilidad
- Nombres descriptivos (snake_case)
- Manejo explícito de errores de API/red
- Secretos solo en .env

## Success Criteria (MVP)

**Milestone 1** - Datos verificados:
- ✅ JSONs de al menos 1 temporada de LaLiga en `data/raw/`
- ✅ DataFrame de partidos normalizado en `data/processed/`
- ✅ Notebook con análisis exploratorio básico

**Milestone 2** - Modelo baseline funcionando:
- ✅ Target 1X2 definido y calculado
- ✅ Al menos 5 features básicas (posición, puntos, racha)
- ✅ Modelo entrenado (RandomForest o similar)
- ✅ Métricas de evaluación calculadas (accuracy, log-loss)
- ✅ Pipeline reproducible de principio a fin

**Milestone 3** - Extensión a otros mercados:
- ✅ Targets para tarjetas y corners definidos
- ✅ Features específicas calculadas
- ✅ Modelos multi-target o separados por mercado

## Questions & Uncertainties

1. **Estructura exacta de respuestas de API BeSoccer**:
   - ¿Qué nombre tiene la lista de partidos en la respuesta? (`matchs`, `data`, `match`?)
   - ¿Vienen tarjetas y corners en el mismo endpoint de partidos o hay que llamar a eventos separados?
   - ¿Cómo se identifican jornadas/temporadas?

2. **ID de LaLiga en BeSoccer**:
   - Asumimos `league=1` pero hay que confirmarlo con `get_my_leagues()` o `get_competitions_detail()`

3. **Parámetros para histórico completo**:
   - ¿`get_matches_by_league()` admite parámetros de jornada (`round=X`) o rango de fechas?
   - ¿Cuántas jornadas tiene LaLiga según `league_status`?

4. **Formato de cuotas de casas de apuestas**:
   - ¿La API de BeSoccer incluye cuotas o hay que obtenerlas de otra fuente?
   - De momento no priorizar, enfocarnos en probabilidades del modelo

## Resources & References

- **API Documentation**: https://documenter.getpostman.com/view/6414020/2s93JwM1t4
- **API Key**: Configurada en `.env` (no compartir en el repo)
- **Constitution**: `.specify/memory/constitution.md` (principios y reglas del proyecto)

## Notes

- El proyecto está en fase inicial de setup; el código base existe pero no se ha validado ejecutándolo con datos reales aún.
- La prioridad inmediata es conseguir datos de la API para poder diseñar el pipeline con información concreta, no supuestos.
- El entorno Python tiene problemas (`.venv` roto o mal configurado); requiere limpieza antes de avanzar.
