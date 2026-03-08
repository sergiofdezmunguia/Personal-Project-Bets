"""
Script para cargar datos descargados manualmente de football-data.co.uk

Instrucciones:
1. Ir a https://www.football-data.co.uk/data.php
2. Descargar CSVs de las ligas deseadas (La Liga, Premier League, etc.)
3. Guardar los archivos en data/raw/
4. Ejecutar este script para procesar los datos

Ejemplo de archivos a descargar:
- SP1.csv (La Liga 2023-24)
- E0.csv (Premier League 2023-24)
"""

import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_football_data_csv(filepath, league_name, season):
    """Cargar y procesar un CSV de football-data.co.uk."""
    df = pd.read_csv(filepath)
    
    # Agregar información de liga y temporada
    df['league'] = league_name
    df['season'] = season
    
    # Renombrar columnas principales a formato estándar
    column_mapping = {
        'Date': 'date',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FTHG': 'home_goals',  # Full Time Home Goals
        'FTAG': 'away_goals',  # Full Time Away Goals
        'FTR': 'result',  # Full Time Result (H/D/A)
        'HTHG': 'ht_home_goals',  # Half Time Home Goals
        'HTAG': 'ht_away_goals',  # Half Time Away Goals
        'HTR': 'ht_result',
        'HS': 'home_shots',
        'AS': 'away_shots',
        'HST': 'home_shots_target',
        'AST': 'away_shots_target',
        'HF': 'home_fouls',
        'AF': 'away_fouls',
        'HC': 'home_corners',
        'AC': 'away_corners',
        'HY': 'home_yellow',
        'AY': 'away_yellow',
        'HR': 'home_red',
        'AR': 'away_red',
    }
    
    # Renombrar solo las columnas que existen
    existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_cols)
    
    return df

def main():
    """Procesar archivos CSV descargados manualmente."""
    print("=== Procesando datos manuales de Football-Data.co.uk ===\n")
    
    # Definir archivos a buscar
    files_to_process = [
        {'file': 'SP1.csv', 'league': 'ESP-La Liga', 'season': '2324'},
        {'file': 'E0.csv', 'league': 'ENG-Premier League', 'season': '2324'},
        {'file': 'SP1_2223.csv', 'league': 'ESP-La Liga', 'season': '2223'},
        {'file': 'E0_2223.csv', 'league': 'ENG-Premier League', 'season': '2223'},
    ]
    
    all_data = []
    found_files = 0
    
    for file_info in files_to_process:
        filepath = RAW_DATA_DIR / file_info['file']
        
        if filepath.exists():
            print(f"✓ Procesando {file_info['file']}...")
            df = load_football_data_csv(
                filepath, 
                file_info['league'], 
                file_info['season']
            )
            all_data.append(df)
            found_files += 1
            print(f"  {len(df)} partidos cargados")
        else:
            print(f"✗ No encontrado: {file_info['file']}")
    
    if found_files == 0:
        print("\n❌ No se encontraron archivos CSV en data/raw/")
        print("\nPara descargar datos:")
        print("1. Ve a: https://www.football-data.co.uk/data.php")
        print("2. Descarga los CSVs de las ligas deseadas")
        print("3. Guárdalos en: data/raw/")
        print("\nArchivos esperados:")
        for file_info in files_to_process:
            print(f"  - {file_info['file']} ({file_info['league']} {file_info['season']})")
        return
    
    # Combinar todos los datos
    print(f"\n=== Combinando {found_files} archivos ===")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Información básica
    print(f"\nTotal de partidos: {len(combined_df)}")
    print(f"\nPartidos por liga:")
    print(combined_df['league'].value_counts())
    print(f"\nPartidos por temporada:")
    print(combined_df['season'].value_counts())
    
    if 'result' in combined_df.columns:
        print(f"\nDistribución de resultados:")
        print(combined_df['result'].value_counts())
        print("  H = Home win, D = Draw, A = Away win")
    
    # Guardar datos procesados
    output_file = PROCESSED_DATA_DIR / "all_matches_processed.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"\n✓ Datos guardados en: {output_file}")
    
    # Mostrar columnas disponibles
    print(f"\nColumnas disponibles ({len(combined_df.columns)}):")
    print(combined_df.columns.tolist())
    
    # Mostrar muestra de datos
    print("\n=== Primeros 3 partidos ===")
    display_cols = ['date', 'league', 'home_team', 'away_team', 'home_goals', 
                    'away_goals', 'result']
    available_display_cols = [col for col in display_cols if col in combined_df.columns]
    print(combined_df[available_display_cols].head(3))

if __name__ == "__main__":
    main()
