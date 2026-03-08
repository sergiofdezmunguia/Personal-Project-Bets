"""
Script de exploración de soccerdata.

Este script demuestra cómo usar soccerdata para obtener datos de fútbol
de múltiples fuentes (FBref, Understat, etc.) para el análisis de apuestas.
"""

import soccerdata as sd
import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_DIR

def main():
    """Explorar las capacidades de soccerdata."""
    
    print("=== Explorando soccerdata ===\n")
    
    # 1. Ver ligas disponibles
    print("1. Ligas disponibles en FBref:")
    print(sd.FBref.available_leagues())
    print()
    
    # 2. Crear scraper para La Liga y Premier League temporadas recientes
    print("2. Creando scraper para La Liga y Premier League (2022-2023, 2023-2024)...")
    fbref = sd.FBref(
        leagues=['ESP-La Liga', 'ENG-Premier League'],
        seasons=['2223', '2324']
    )
    print("✓ Scraper creado\n")
    
    # 3. Obtener calendario de partidos
    print("3. Obteniendo calendario de partidos...")
    schedule = fbref.read_schedule()
    print(f"✓ Calendario obtenido: {len(schedule)} partidos")
    print("\nPrimeros 5 partidos:")
    print(schedule.head())
    print(f"\nColumnas disponibles: {list(schedule.columns)}")
    print()
    
    # 4. Guardar calendario
    output_file = RAW_DATA_DIR / "schedule_laliga_epl_2223_2324.csv"
    schedule.to_csv(output_file)
    print(f"✓ Calendario guardado en: {output_file}\n")
    
    # 5. Obtener estadísticas de equipos por temporada
    print("4. Obteniendo estadísticas de equipos (shooting)...")
    team_stats = fbref.read_team_season_stats(stat_type='shooting')
    print(f"✓ Estadísticas obtenidas: {len(team_stats)} equipos")
    print("\nPrimeros 5 equipos:")
    print(team_stats.head())
    print()
    
    # 6. Guardar estadísticas de equipos
    output_file = RAW_DATA_DIR / "team_shooting_stats_laliga_epl.csv"
    team_stats.to_csv(output_file)
    print(f"✓ Estadísticas de equipos guardadas en: {output_file}\n")
    
    # 7. Resumen de datos disponibles
    print("=== RESUMEN ===")
    print(f"Total de partidos: {len(schedule)}")
    print(f"Ligas: {schedule.index.get_level_values('league').unique().tolist()}")
    print(f"Temporadas: {schedule.index.get_level_values('season').unique().tolist()}")
    
    # Info de partidos con xG disponibles
    schedule_with_xg = schedule.dropna(subset=['home_xg', 'away_xg'])
    print(f"Partidos con xG: {len(schedule_with_xg)} ({len(schedule_with_xg)/len(schedule)*100:.1f}%)")
    
    print("\n=== PRÓXIMOS PASOS ===")
    print("1. Explorar más tipos de estadísticas:")
    print("   - read_team_season_stats(stat_type='passing')")
    print("   - read_team_season_stats(stat_type='defense')")
    print("   - read_player_season_stats(stat_type='standard')")
    print("2. Obtener datos de partidos específicos:")
    print("   - read_team_match_stats()")
    print("   - read_events() para goles, tarjetas, etc.")
    print("3. Combinar con otras fuentes:")
    print("   - Understat para xG más detallado")
    print("   - WhoScored para más estadísticas")

if __name__ == "__main__":
    main()
