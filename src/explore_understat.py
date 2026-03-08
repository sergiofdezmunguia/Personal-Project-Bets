"""
Script para obtener datos de Understat con el nombre correcto de liga
"""
import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import soccerdata as sd
from src.config import RAW_DATA_DIR

print("=== Explorando Understat ===\n")

# Understat usa 'ESP-La Liga', no 'La_Liga'
print("Creando scraper de Understat para La Liga y Premier League...")
try:
    understat = sd.Understat(
        leagues=['ESP-La Liga', 'ENG-Premier League'], 
        seasons=['2023', '2024']
    )
    print("✓ Scraper creado\n")
    
    # 1. Calendario y resultados
    print("1. Obteniendo calendario y resultados...")
    schedule = understat.read_schedule()
    print(f"✓ {len(schedule)} partidos")
    print(f"Columnas: {list(schedule.columns)}")
    print("\nPrimeros 5 partidos:")
    print(schedule.head())
    print()
    
    # 2. Estadísticas de equipos por partido
    print("2. Obteniendo estadísticas de equipos por partido...")
    team_match_stats = understat.read_team_match_stats()
    print(f"✓ {len(team_match_stats)} filas de estadísticas")
    print(f"Columnas: {list(team_match_stats.columns)}")
    print("\nPrimeras 3 filas:")
    print(team_match_stats.head(3))
    print()
    
    # 3. Estadísticas de jugadores
    print("3. Obteniendo estadísticas de jugadores por temporada...")
    players = understat.read_player_season_stats()
    print(f"✓ {len(players)} jugadores")
    print(f"Columnas: {list(players.columns)}")
    print("\nTop 5 jugadores por xG:")
    if 'xg' in players.columns:
        top_xg = players.nlargest(5, 'xg')[['position', 'xg', 'goals', 'assists']]
        print(top_xg)
    print()
    
    # Guardar datos
    output_dir = Path(RAW_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    schedule_file = output_dir / "understat_schedule.csv"
    schedule.to_csv(schedule_file)
    print(f"✓ Calendario guardado en: {schedule_file}")
    
    team_stats_file = output_dir / "understat_team_match_stats.csv"
    team_match_stats.to_csv(team_stats_file)
    print(f"✓ Stats de equipos guardadas en: {team_stats_file}")
    
    players_file = output_dir / "understat_players.csv"
    players.to_csv(players_file)
    print(f"✓ Jugadores guardados en: {players_file}")
    
    print("\n✅ ¡Understat funciona perfectamente!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
