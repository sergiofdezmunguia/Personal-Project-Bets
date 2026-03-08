"""
Script de exploración de soccerdata usando Football-Data.co.uk.

Esta fuente es más permisiva y no requiere web scraping agresivo.
Proporciona datos históricos de partidos con odds de casas de apuestas.
"""

import soccerdata as sd
import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_DIR

def main():
    """Explorar datos de Football-Data.co.uk."""
    
    print("=== Explorando Football-Data.co.uk ===\n")
    
    # 1. Ver ligas disponibles
    print("1. Ligas disponibles en Football-Data.co.uk:")
    available_leagues = sd.MatchHistory.available_leagues()
    print(available_leagues[:20])  # Mostrar las primeras 20
    print(f"... y {len(available_leagues) - 20} más\n")
    
    # 2. Crear scraper para La Liga y Premier League
    print("2. Creando scraper para La Liga y Premier League...")
    fduk = sd.MatchHistory(
        leagues=['ESP-La Liga', 'ENG-Premier League'],
        seasons=['2223', '2324', '2425']  # Últimas 3 temporadas
    )
    print("✓ Scraper creado\n")
    
    # 3. Obtener datos de partidos con odds
    print("3. Descargando datos de partidos...")
    try:
        games = fduk.read_games()
        print(f"✓ Datos descargados: {len(games)} partidos")
        print("\nPrimeros 5 partidos:")
        print(games.head())
        print(f"\nColumnas disponibles ({len(games.columns)}):")
        print(list(games.columns)[:15], "...")
        print()
        
        # 4. Guardar datos
        output_file = RAW_DATA_DIR / "football_data_laliga_epl_3seasons.csv"
        games.to_csv(output_file)
        print(f"✓ Datos guardados en: {output_file}\n")
        
        # 5. Análisis básico
        print("=== ANÁLISIS BÁSICO ===")
        print(f"Total de partidos: {len(games)}")
        print(f"\nPartidos por liga:")
        print(games.index.get_level_values('league').value_counts())
        print(f"\nPartidos por temporada:")
        print(games.index.get_level_values('season').value_counts())
        
        # Info de resultados
        if 'FTR' in games.columns:  # Full Time Result
            print(f"\nDistribución de resultados:")
            print(games['FTR'].value_counts())
            print(f"  H = Home win, D = Draw, A = Away win")
        
        # Info de odds disponibles
        odds_columns = [col for col in games.columns if 'avg' in col.lower() or 'max' in col.lower()]
        if odds_columns:
            print(f"\nColumnas de odds disponibles: {len(odds_columns)}")
            print(f"Ejemplos: {odds_columns[:5]}")
        
        print("\n=== PRÓXIMOS PASOS ===")
        print("1. Analizar las odds para detectar value bets")
        print("2. Crear features basadas en:")
        print("   - Forma reciente (últimos 5 partidos)")
        print("   - Head-to-head histórico")
        print("   - Goles marcados/recibidos promedio")
        print("   - Diferencia de odds entre casas")
        print("3. Entrenar modelo baseline con estos datos")
        
    except Exception as e:
        print(f"❌ Error al descargar datos: {e}")
        print("\nIntentando con una liga y temporada específica...")
        
        # Intentar con datos más recientes
        fduk_simple = sd.MatchHistory(leagues=['ENG-Premier League'], seasons=['2324'])
        games_simple = fduk_simple.read_games()
        print(f"✓ Descargados {len(games_simple)} partidos de Premier League 2023-24")
        print(games_simple.head())

if __name__ == "__main__":
    main()
