"""
Script para probar las fuentes que deberían funcionar:
- ClubElo (ya confirmado)
- Understat (con nombre de liga corregido)
- WhoScored (ahora con Chrome instalado)
"""
import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import soccerdata as sd
from src.config import RAW_DATA_DIR

print("=== Probando fuentes funcionales ===\n")

# 1. ClubElo - Ya sabemos que funciona
print("1. ClubElo - Rankings ELO de equipos")
print("=" * 60)
try:
    clubelo = sd.ClubElo()
    data = clubelo.read_by_date('2024-01-01')
    print(f"✓ {len(data)} equipos con rankings ELO")
    print(f"Columnas: {list(data.columns)}")
    print("\nPrimeros 5 equipos:")
    print(data.head())
    print()
except Exception as e:
    print(f"✗ Error: {e}\n")

# 2. Understat - Arreglar nombre de liga
print("\n2. Understat - Estadísticas avanzadas (xG, xA, etc.)")
print("=" * 60)
print("Ligas disponibles:")
print(sd.Understat.available_leagues())
print()

try:
    # Usar el nombre correcto de la liga
    understat = sd.Understat(leagues=['ESP-La Liga'], seasons=['2023'])
    print("✓ Scraper creado")
    
    print("Obteniendo calendario de partidos...")
    matches = understat.read_schedule()
    print(f"✓ Calendario obtenido: {len(matches)} partidos")
    print(f"Columnas: {list(matches.columns)}")
    print("\nPrimeros 5 filas:")
    print(matches.head())
    print()
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# 3. WhoScored - Con Chrome instalado
print("\n3. WhoScored - Datos detallados de partidos")
print("=" * 60)
print("⚠️  WhoScored usa Selenium y puede tardar más...")
try:
    ws = sd.WhoScored(leagues=['ESP-La Liga'], seasons=['2324'])
    print("✓ Scraper creado (esto puede tardar, abrirá Chrome en modo headless)")
    
    # Intentar leer datos básicos
    schedule = ws.read_schedule()
    print(f"✓ Calendario obtenido: {len(schedule)} partidos")
    print(f"Columnas: {list(schedule.columns)}")
    print("\nPrimeros 3 partidos:")
    print(schedule.head(3))
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Pruebas completadas")
print("=" * 60)
