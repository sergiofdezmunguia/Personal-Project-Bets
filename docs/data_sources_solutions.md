# Soluciones para Obtener Datos de Fútbol

## Problema Actual

Los sitios web (FBref, Football-Data.co.uk) están bloqueando las peticiones automáticas de `soccerdata` con errores 403/503. Esto es común en web scraping debido a medidas anti-bot.

## Opciones Disponibles

### ✅ Opción 1: Descargar datos manualmente (RECOMENDADO)

**Football-Data.co.uk** (datos históricos con odds):
1. Ir a: https://www.football-data.co.uk/data.php
2. Descargar CSVs de las ligas deseadas
3. Guardar en `data/raw/`
4. Ejecutar: `python -m src.load_manual_data`

**Ventajas:**
- ✅ Datos inmediatos
- ✅ Incluye odds de casas de apuestas
- ✅ Datos históricos de múltiples temporadas
- ✅ No hay problemas de bloqueo

### ⚙️ Opción 2: Configurar `soccerdata` con delay/proxy

Si queremos usar `soccerdata`, necesitamos configurar delays más largos o usar un proxy:

```python
import soccerdata as sd

# Opción A: Con proxy Tor (requiere tener Tor instalado y corriendo)
fbref = sd.FBref(
    leagues=['ESP-La Liga'],
    seasons=['2324'],
    proxy='tor'  # Tor en localhost:9050
)

# Opción B: Con proxy personalizado
fbref = sd.FBref(
    leagues=['ESP-La Liga'],
    seasons=['2324'],
    proxy={
        'http': 'http://proxy-server:port',
        'https': 'http://proxy-server:port'
    }
)

# Opción C: Esperar más tiempo entre peticiones (modificar internamente)
# Esto requiere modificar rate_limit en el código de soccerdata
```

**Limitaciones:**
- ❌ Requiere configuración adicional (Tor/Proxy)
- ❌ Puede seguir siendo bloqueado
- ❌ Más lento

### 📊 Opción 3: Usar datasets de Kaggle

Buscar datasets ya preparados en Kaggle:
- [European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer)
- [Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores)
- [International football results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

**Ventajas:**
- ✅ Datos listos para usar
- ✅ Sin problemas de scraping
- ✅ Datasets curados

**Limitaciones:**
- ❌ Pueden estar desactualizados
- ❌ No incluyen temporada actual

### ⏰ Opción 4: Reintentar más tarde

A veces los sitios bloquean temporalmente. Esperar unas horas y reintentar puede funcionar.

## Recomendación

Para este proyecto, **recomiendo la Opción 1** (descarga manual de Football-Data.co.uk):

1. Los datos incluyen **odds de casas de apuestas** (útil para betting)
2. Datos históricos completos
3. Sin problemas técnicos
4. Podemos empezar a trabajar inmediatamente

Una vez tengamos el modelo funcionando, podemos explorar otras fuentes para datos más recientes.

## Próximos Pasos

1. Descargar CSVs manualmente de:
   - https://www.football-data.co.uk/spainm.php (La Liga)
   - https://www.football-data.co.uk/englandm.php (Premier League)

2. Guardar archivos como:
   - `data/raw/SP1.csv` (La Liga 2023-24)
   - `data/raw/E0.csv` (Premier League 2023-24)
   - Y temporadas anteriores si es necesario

3. Ejecutar script de procesamiento:
   ```bash
   python -m src.load_manual_data
   ```

4. ¡Empezar a construir features y modelos!
