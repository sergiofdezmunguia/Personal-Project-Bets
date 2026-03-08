# Proyecto de apuestas de fútbol con soccerdata + Machine Learning

Este proyecto utiliza la librería **soccerdata** para descargar datos de fútbol de múltiples fuentes (FBref, Understat, WhoScored, etc.), construye un conjunto de datos de partidos, genera características (features) y entrena modelos de machine learning (por ejemplo, XGBoost) para predecir resultados de partidos (local/empate/visitante) y, potencialmente, otros mercados como tarjetas o corners.

## Fuente de Datos: soccerdata

**[soccerdata](https://github.com/probberechts/soccerdata)** es una librería que proporciona:
- 📊 Datos de múltiples fuentes: FBref, Understat, WhoScored, Sofascore, etc.
- ⚽ Estadísticas detalladas: partidos, equipos, jugadores, xG, eventos, etc.
- 🏆 Ligas soportadas: La Liga, Premier League, Bundesliga, Serie A, Ligue 1, y más
- 🐼 DataFrames de pandas con columnas estandarizadas
- 💾 Caché automático de datos

## Estructura básica

- `src/` — Código fuente del proyecto.
  - `extract_pipeline.py` — Extracción unificada (Understat + ClubElo)
  - `train_xgboost.py` — Entrenamiento baseline Over 2.5 con validación temporal
  - `explore_understat.py` — Exploración detallada de Understat
  - `explore_soccerdata.py` — Script de exploración general
  - `config.py` — Configuración y rutas del proyecto
- `data/raw/` — Datos crudos descargados
- `data/processed/` — Datos ya transformados/listos para entrenar
- `models/` — Modelos entrenados y artefactos (scalers, etc.)
- `notebooks/` — Notebooks para exploración de datos y prototipado
- `archive/scripts/` — Scripts legacy/experimentales archivados
- `.specify/memory/` — Documentación Speckit (constitution, spec)

## Configuración inicial

1. Crear entorno virtual y activarlo:

```bash
# Ubuntu/WSL
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Extraer datos base:

```bash
python -m src.extract_pipeline
```

4. Entrenar baseline ML (Over 2.5):

```bash
python -m src.train_xgboost
```

5. Explorar datos disponibles de forma manual (opcional):

```bash
python -m src.explore_soccerdata
```

## Estado actual recomendado para push

- Codigo de extraccion operativo: Understat y ClubElo
- Codigo de entrenamiento baseline operativo: `src/train_xgboost.py`
- Scripts antiguos movidos a `archive/scripts/`
- `gitignore` ajustado para no subir entornos, datos crudos, binarios y locks

## Próximos pasos

- ✅ Explorar soccerdata y sus capacidades
- 🔄 Implementar ingesta completa de datos históricos (varias temporadas)
- 🔄 Diseñar y codificar features (forma reciente, Elo, H2H, xG, etc.)
- ⏳ Entrenar modelo multiclase (local/empate/visitante)
- ⏳ Extender el pipeline para otros mercados (amarillas, corners, goles, etc.)
