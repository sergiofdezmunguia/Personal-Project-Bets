# Fase 4: normalización del margen y CLV histórico (resultados generados)

Generado con `scripts/fase4_margen_clv.py evaluacion` el 2026-11-15. Solo agregados.
Hash de los datos: `3e3b9ea6a34b8bf2`. Filas de la fase 4 en el registro hasta hoy: 27. Lectura y auditoría en
[`clv.md`](clv.md).

## Ajuste (2013/14–2016/17): qué normalización usar

Regla fijada antes de mirar: la de menor log loss del cierre de Pinnacle. **Elegida: potencia.**

| Normalización | Log loss cierre | Log loss apertura |
|---|---|---|
| proporcional | 0.99435 | 0.99652 |
| shin | 0.99418 | 0.99633 |
| potencia | 0.99413 | 0.99626 |

## Evaluación 2017/18–2024/25 (26660 partidos, 350 semanas)

Δ log loss emparejada (IC 95 % por bootstrap de semanas); **Δ > 0 es peor que la referencia**.
Los residuos usan los hiperparámetros de la fase 3 sin reajustar; solo cambia la normalización de la
apertura que reciben (como desplazamiento y como variable).

| Modelo | Log loss [IC 95 %] | Δ vs Cierre, proporcional | Δ vs Cierre, potencia | Δ vs Apertura, proporcional | Δ vs Apertura, potencia | ECE |
|---|---|---|---|---|---|---|
| Cierre, proporcional | 0.99441 [0.9895, 0.9992] | — | +0.00018 [-0.00002, +0.00037] | -0.00280 [-0.00373, -0.00184] | -0.00262 [-0.00356, -0.00164] | 0.0067 |
| Cierre, Shin | 0.99426 [0.9892, 0.9992] | -0.00016 [-0.00030, -0.00002] | +0.00002 [-0.00005, +0.00008] | -0.00296 [-0.00392, -0.00194] | -0.00278 [-0.00373, -0.00179] | 0.0056 |
| Cierre, potencia | 0.99424 [0.9892, 0.9992] | -0.00018 [-0.00037, +0.00002] | — | -0.00297 [-0.00393, -0.00193] | -0.00280 [-0.00378, -0.00180] | 0.0053 |
| Apertura, proporcional | 0.99721 [0.9924, 1.0019] | +0.00280 [+0.00184, +0.00373] | +0.00297 [+0.00193, +0.00393] | — | +0.00018 [-0.00003, +0.00039] | 0.0071 |
| Apertura, Shin | 0.99705 [0.9921, 1.0019] | +0.00263 [+0.00166, +0.00357] | +0.00281 [+0.00180, +0.00378] | -0.00017 [-0.00032, -0.00002] | +0.00001 [-0.00006, +0.00008] | 0.0058 |
| Apertura, potencia | 0.99704 [0.9921, 1.0019] | +0.00262 [+0.00164, +0.00356] | +0.00280 [+0.00180, +0.00378] | -0.00018 [-0.00039, +0.00003] | — | 0.0059 |
| Dixon-Coles (ancla de apertura) | 1.01335 [1.0085, 1.0177] | +0.01894 [+0.01671, +0.02109] | +0.01911 [+0.01683, +0.02131] | +0.01614 [+0.01411, +0.01808] | +0.01631 [+0.01419, +0.01831] | 0.0080 |
| Residuo LightGBM, proporcional | 0.99708 [0.9922, 1.0019] | +0.00267 [+0.00174, +0.00361] | +0.00284 [+0.00183, +0.00381] | -0.00013 [-0.00022, -0.00005] | +0.00004 [-0.00015, +0.00023] | 0.0062 |
| Residuo logit, proporcional | 0.99711 [0.9923, 1.0019] | +0.00270 [+0.00175, +0.00363] | +0.00288 [+0.00186, +0.00383] | -0.00010 [-0.00023, +0.00004] | +0.00008 [-0.00011, +0.00026] | 0.0071 |
| Residuo LightGBM, Shin | 0.99696 [0.9920, 1.0018] | +0.00255 [+0.00161, +0.00349] | +0.00273 [+0.00174, +0.00370] | -0.00025 [-0.00043, -0.00007] | -0.00007 [-0.00016, +0.00003] | 0.0056 |
| Residuo logit, Shin | 0.99700 [0.9921, 1.0019] | +0.00259 [+0.00162, +0.00352] | +0.00276 [+0.00178, +0.00374] | -0.00021 [-0.00042, -0.00001] | -0.00003 [-0.00015, +0.00008] | 0.0057 |
| Residuo LightGBM, potencia | 0.99696 [0.9920, 1.0019] | +0.00254 [+0.00158, +0.00350] | +0.00272 [+0.00173, +0.00370] | -0.00025 [-0.00049, -0.00003] | -0.00008 [-0.00015, +0.00000] | 0.0057 |
| Residuo logit, potencia | 0.99699 [0.9920, 1.0019] | +0.00257 [+0.00160, +0.00351] | +0.00275 [+0.00176, +0.00372] | -0.00023 [-0.00046, +0.00000] | -0.00005 [-0.00014, +0.00005] | 0.0057 |

### Residuos frente a la apertura con potencia, por temporada

| Temporada | N | Residuo LightGBM, proporcional | Residuo logit, proporcional | Residuo LightGBM, Shin | Residuo logit, Shin | Residuo LightGBM, potencia | Residuo logit, potencia |
|---|---|---|---|---|---|---|---|
| 2017/18 | 3377 | -0.00007 [-0.00055, +0.00032] | -0.00011 [-0.00061, +0.00036] | -0.00017 [-0.00039, +0.00003] | -0.00016 [-0.00043, +0.00010] | -0.00014 [-0.00032, +0.00006] | -0.00012 [-0.00033, +0.00011] |
| 2018/19 | 3368 | -0.00017 [-0.00079, +0.00044] | -0.00012 [-0.00068, +0.00046] | -0.00023 [-0.00054, +0.00009] | -0.00022 [-0.00060, +0.00014] | -0.00022 [-0.00046, +0.00003] | -0.00026 [-0.00056, +0.00004] |
| 2019/20 | 3124 | -0.00053 [-0.00107, -0.00001] | -0.00047 [-0.00110, +0.00013] | -0.00026 [-0.00050, -0.00002] | -0.00018 [-0.00055, +0.00020] | -0.00022 [-0.00043, -0.00001] | -0.00012 [-0.00046, +0.00022] |
| 2020/21 | 3376 | -0.00011 [-0.00049, +0.00031] | +0.00038 [-0.00005, +0.00081] | -0.00004 [-0.00032, +0.00022] | +0.00039 [+0.00000, +0.00076] | -0.00003 [-0.00026, +0.00020] | +0.00039 [+0.00004, +0.00074] |
| 2021/22 | 3389 | +0.00020 [-0.00026, +0.00064] | +0.00017 [-0.00024, +0.00053] | +0.00009 [-0.00021, +0.00037] | +0.00006 [-0.00022, +0.00031] | +0.00000 [-0.00027, +0.00025] | -0.00004 [-0.00028, +0.00018] |
| 2022/23 | 3392 | +0.00022 [-0.00020, +0.00065] | -0.00001 [-0.00044, +0.00036] | +0.00010 [-0.00013, +0.00034] | -0.00003 [-0.00030, +0.00022] | +0.00013 [-0.00007, +0.00035] | +0.00003 [-0.00022, +0.00028] |
| 2023/24 | 3316 | +0.00033 [-0.00041, +0.00101] | +0.00035 [-0.00034, +0.00095] | -0.00005 [-0.00034, +0.00023] | -0.00005 [-0.00034, +0.00020] | -0.00009 [-0.00025, +0.00008] | -0.00013 [-0.00030, +0.00004] |
| 2024/25 | 3318 | +0.00044 [-0.00018, +0.00105] | +0.00041 [-0.00022, +0.00101] | -0.00003 [-0.00027, +0.00020] | -0.00010 [-0.00042, +0.00018] | -0.00008 [-0.00020, +0.00005] | -0.00017 [-0.00037, +0.00001] |

### Robustez: recortes de la Δ partido a partido

*Tabla regenerada tras la auditoría con `scripts/fase4_margen_clv.py robustez` (mismas predicciones,
deterministas; no se registra). La versión original solo quitaba el 1 % más favorable, un estadístico
sesgado con colas gruesas.* El recorte asimétrico solo indica fragilidad si queda **por encima** de su
valor bajo un nulo de signo aleatorio (200 permutaciones).

| Modelo | Δ vs Apertura, potencia | Δ recortada (0,5 % por cola) | Δ sin el 1 % más favorable | Lo mismo bajo el nulo |
|---|---|---|---|---|
| Cierre, proporcional | -0.00262 | -0.00246 | +0.00027 | +0.00277 |
| Cierre, Shin | -0.00278 | -0.00282 | -0.00004 | +0.00280 |
| Cierre, potencia | -0.00280 | -0.00290 | -0.00009 | +0.00280 |
| Apertura, proporcional | +0.00018 | +0.00066 | +0.00121 | +0.00079 |
| Apertura, Shin | +0.00001 | +0.00020 | +0.00039 | +0.00027 |
| Dixon-Coles (ancla de apertura) | +0.01631 | +0.01683 | +0.02224 | +0.00554 |
| Residuo LightGBM, proporcional | +0.00004 | +0.00047 | +0.00097 | +0.00069 |
| Residuo logit, proporcional | +0.00008 | +0.00048 | +0.00095 | +0.00066 |
| Residuo LightGBM, Shin | -0.00007 | +0.00005 | +0.00030 | +0.00030 |
| Residuo logit, Shin | -0.00003 | +0.00010 | +0.00040 | +0.00036 |
| Residuo LightGBM, potencia | -0.00008 | -0.00009 | +0.00014 | +0.00024 |
| Residuo logit, potencia | -0.00005 | -0.00009 | +0.00019 | +0.00029 |

## CLV histórico de apuestas simuladas a la apertura de Pinnacle

Una apuesta plana por partido, al resultado de mayor valor esperado p_modelo · cuota_apertura − 1,
si supera el umbral. CLV = cuota tomada · p_cierre sin margen − 1. Pinnacle no opera en España:
es una medida de calidad del precio, no una estrategia ejecutable. Sin comisión (casa, no exchange).

| Modelo | Umbral | Apuestas | CLV (cierre potencia) [IC 95 %] | CLV sin el 1 % mejor (versión original; con n = 13 quita 1 apuesta) | % CLV > 0 | CLV (cierre proporcional) | ROI plano [IC 95 %] |
|---|---|---|---|---|---|---|---|
| Residuo LightGBM, potencia | 0% | 13 | -0.0144 [-0.0552, +0.0205] | -0.0267 | 38.5% | -0.0167 | -0.3277 [-0.8740, +0.3884] |
| Residuo LightGBM, potencia | 2% | 0 | — | — | — | — | — |
| Residuo LightGBM, potencia | 5% | 0 | — | — | — | — | — |
| Residuo logit, potencia | 0% | 14 | -0.0350 [-0.0552, -0.0155] | -0.0388 | 28.6% | -0.0366 | -0.2264 [-0.7136, +0.3931] |
| Residuo logit, potencia | 2% | 0 | — | — | — | — | — |
| Residuo logit, potencia | 5% | 0 | — | — | — | — | — |
| Dixon-Coles (ancla de apertura) | 0% | 24930 | -0.0423 [-0.0440, -0.0405] | -0.0461 | 31.5% | -0.0301 | -0.0675 [-0.0894, -0.0461] |
| Dixon-Coles (ancla de apertura) | 2% | 22750 | -0.0432 [-0.0450, -0.0413] | -0.0470 | 31.4% | -0.0299 | -0.0703 [-0.0935, -0.0469] |
| Dixon-Coles (ancla de apertura) | 5% | 19270 | -0.0446 [-0.0466, -0.0425] | -0.0486 | 31.2% | -0.0295 | -0.0714 [-0.0972, -0.0451] |

CLV medio por temporada (apuestas):

| Modelo | Umbral | 2017/18 | 2018/19 | 2019/20 | 2020/21 | 2021/22 | 2022/23 | 2023/24 | 2024/25 |
|---|---|---|---|---|---|---|---|---|---|
| Residuo LightGBM, potencia | 0% | -0.0154 (6) | -0.0979 (2) | +nan (0) | +0.0328 (4) | +nan (0) | -0.0301 (1) | +nan (0) | +nan (0) |
| Residuo logit, potencia | 0% | -0.0436 (3) | +nan (0) | -0.0101 (2) | -0.0437 (8) | +0.0102 (1) | +nan (0) | +nan (0) | +nan (0) |
| Dixon-Coles (ancla de apertura) | 0% | -0.0282 (3241) | -0.0406 (3142) | -0.0423 (2917) | -0.0360 (3200) | -0.0425 (3179) | -0.0396 (3206) | -0.0515 (3064) | -0.0591 (2981) |
| Dixon-Coles (ancla de apertura) | 2% | -0.0288 (2973) | -0.0414 (2869) | -0.0430 (2669) | -0.0366 (2950) | -0.0438 (2919) | -0.0396 (2925) | -0.0536 (2756) | -0.0609 (2689) |
| Dixon-Coles (ancla de apertura) | 5% | -0.0295 (2557) | -0.0423 (2414) | -0.0433 (2270) | -0.0362 (2475) | -0.0461 (2491) | -0.0407 (2464) | -0.0566 (2335) | -0.0648 (2264) |
