# Decisiones

Registro de lo decidido y de lo que falta por decidir. Añade arriba lo más reciente.

## Abiertas

- **Visibilidad del repo:** seguir público o separar código público y resultados privados.
- **Dónde corre el recolector de cuotas:** debe funcionar con el PC apagado.
- **Proveedor de cuotas en vivo:** The Odds API o API de Betfair; verificar cuota gratuita, casas y acceso desde España.
- **Presupuesto máximo** si algún día se llega a la fase 5.
- **xG mientras Understat esté en mantenimiento:** esperar o evaluar otra fuente. Bloquea la unión
  de datos de la fase 1 y la primera ejecución con datos reales.
- **Identidad de equipo (fase 1):** las ventanas agrupan por nombre. Un nombre repetido en dos ligas
  mezclaría historiales y un cambio de nombre los cortaría. Decidir al unir fuentes si se usa `team_id`.
- **Hora de decisión (antes de las fases 3 y 4):** `shift(1)` da por conocido el partido anterior y su
  xG. Decidiendo en T-48 h, con poco descanso entre partidos, ese partido aún no se habría jugado.
- **Registro de experimentos (antes de usar datos reales):** cada ejecución calcula el `test` de
  desarrollo y sobrescribe el JSON sin dejar recuento.
- **Forma entre temporadas:** la ventana cruza temporadas, así que un equipo que vuelve tras un año
  usa partidos de hace dos temporadas. No es una fuga, pero está sin evaluar.

## Tomadas

### 2026-11-06: división de desarrollo por día y 2025/26 bloqueada en el código

`split_by_date` corta por día natural, redondeando a la baja hasta el último día completo que cabe
en 70/15/15. `temporal_split` rechaza cualquier fila desde el 1-jul-2025. Su `test` es de desarrollo,
no la prueba final.
**Motivo:** un corte por filas parte una jornada, y la regla de la prueba final debe fallar de forma
visible en vez de depender de la disciplina. Es un corte provisional: la validación por ventanas
llegará en la fase 3.

### 2026-11-06: entorno reproducible

Python 3.11. Dependencias directas en `requirements.in` y versiones exactas en `requirements.txt`,
generado con `uv pip compile`. XGBoost es obligatorio, sin modelo alternativo, y usa `n_jobs=4`.
**Motivo:** sin versiones fijas no se reproduce un resultado. El modelo alternativo cambiaba de
algoritmo en silencio, y con todos los núcleos el ajuste tardaba 53 s frente a 0,04 s en WSL, con
predicciones idénticas.

### 2026-11-06: referencia de mercado tras Pinnacle

Pinnacle hasta enero de 2026 y Betfair Exchange de cierre con la media del mercado desde entonces.
**Motivo:** las columnas de Pinnacle desaparecen de football-data a mitad de la 2025/26.
**Propuesta pendiente de validar** comparando ambas en 2024/25.

### 2026-11-06: empezar simulando, con el mercado como referencia

Sin dinero real ni apuestas automáticas. Criterios de avance en `docs/plan-modelo.md`.
**Motivo:** la evidencia revisada no demuestra una ventaja persistente.
