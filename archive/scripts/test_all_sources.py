"""
Script para probar todas las fuentes de datos de soccerdata
y ver cuáles funcionan.
"""
import soccerdata as sd
import sys

print("=== Probando todas las fuentes de datos de soccerdata ===\n")

# Lista de fuentes a probar
sources_to_test = [
    {
        'name': 'FBref',
        'class': sd.FBref,
        'method': 'read_leagues',
        'params': {'leagues': ['ESP-La Liga'], 'seasons': ['2324']}
    },
    {
        'name': 'Understat',
        'class': sd.Understat,
        'method': 'read_leagues',
        'params': {'leagues': ['La Liga'], 'seasons': ['2023']}
    },
    {
        'name': 'WhoScored',
        'class': sd.WhoScored,
        'method': 'read_leagues',
        'params': {'leagues': ['ESP-La Liga'], 'seasons': ['2324']}
    },
    {
        'name': 'ClubElo',
        'class': sd.ClubElo,
        'method': 'read_by_date',
        'params': {}
    },
    {
        'name': 'MatchHistory (Football-Data.co.uk)',
        'class': sd.MatchHistory,
        'method': 'read_games',
        'params': {'leagues': ['ESP-La Liga'], 'seasons': ['2324']}
    },
]

results = []

for source_config in sources_to_test:
    print(f"\n{'='*60}")
    print(f"Probando: {source_config['name']}")
    print('='*60)
    
    try:
        # Crear instancia
        print(f"  → Creando instancia...")
        if source_config['name'] == 'ClubElo':
            # ClubElo no necesita parámetros en el constructor
            scraper = source_config['class']()
        else:
            scraper = source_config['class'](**source_config['params'])
        print(f"  ✓ Instancia creada")
        
        # Intentar leer datos
        print(f"  → Intentando leer datos con {source_config['method']}()...")
        method = getattr(scraper, source_config['method'])
        
        if source_config['name'] == 'ClubElo':
            # ClubElo necesita una fecha
            data = method('2024-01-01')
        else:
            data = method()
        
        print(f"  ✓ ÉXITO: Datos obtenidos")
        print(f"  → Tamaño: {len(data)} registros")
        print(f"  → Columnas: {list(data.columns)[:5]}...")
        
        results.append({
            'source': source_config['name'],
            'status': '✓ FUNCIONA',
            'records': len(data)
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ ERROR: {error_msg[:100]}")
        
        results.append({
            'source': source_config['name'],
            'status': f'✗ FALLA',
            'error': error_msg[:80]
        })

# Resumen final
print(f"\n\n{'='*60}")
print("RESUMEN DE RESULTADOS")
print('='*60)

for result in results:
    status_symbol = '✓' if '✓' in result['status'] else '✗'
    print(f"\n{status_symbol} {result['source']}")
    print(f"   Estado: {result['status']}")
    if 'records' in result:
        print(f"   Registros: {result['records']}")
    if 'error' in result:
        print(f"   Error: {result['error']}")

# Contar fuentes funcionales
working_sources = [r for r in results if '✓' in r['status']]
print(f"\n{'='*60}")
print(f"Total fuentes probadas: {len(results)}")
print(f"Fuentes funcionando: {len(working_sources)}")
print(f"Fuentes con problemas: {len(results) - len(working_sources)}")
print('='*60)
