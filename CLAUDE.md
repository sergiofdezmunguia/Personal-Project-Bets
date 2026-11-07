# Personal Project Bets: guía para Claude Code

Proyecto personal y de portfolio: **medir si un modelo de ML puede ganar al mercado de apuestas**.
El objetivo inicial es investigar y simular. No se apuesta dinero real.

Responde siempre en español.

## Empieza por aquí

1. `docs/plan-modelo.md`: fuentes de datos, mercados, modelos, calendario de cargas y fases.
2. `docs/decisions.md`: decisiones tomadas y abiertas. Añade cada decisión nueva con su motivo.

`.specify/` y `.github/agents/` son de una etapa anterior basada en la API de BeSoccer. Están
desfasados, salvo el principio de integridad temporal de la constitución, que sigue vigente.

## Reglas que no se negocian

- **Integridad temporal.** Una variable de un partido solo usa información disponible antes de la
  hora en que se haría la apuesta. La división de datos siempre es temporal y por fecha de partido.
- **El mercado es la referencia.** Un modelo no se da por bueno por su accuracy o AUC. Debe compararse
  con la cuota de cierre sin margen: log loss, RPS, calibración y CLV.
- **Conjunto de prueba final intacto.** La temporada 2025/26 no se usa hasta congelar el modelo.
- **Todo experimento cuenta.** Registra cada variante probada, también las que fallan. Sin ese
  recuento, un resultado positivo puede ser solo fruto de probar muchas cosas.
- **Sin apuestas automáticas.** Nada coloca apuestas ni guarda credenciales de casas en el código.
  Los secretos van en `.env`.
- **Dinero real solo con aprobación de Sergio**, tras superar los criterios de la fase 4 del plan.
- **Pinnacle es solo un dato de referencia.** No tiene licencia en España. Apuesta real, si llega,
  solo en operadores que figuren en el registro de la DGOJ.
- **Respeta las fuentes:** caché local, pausas entre peticiones y sin saltarse bloqueos.

## Agentes del proyecto

- `sports-data-engineer`: ingesta, calidad de datos y unión de fuentes.
- `betting-modeler`: variables, modelos, calibración y comparación con el mercado.
- `backtest-auditor`: busca fugas de información y errores en los resultados. Solo lee y ejecuta
  comprobaciones; no modifica código.

Todo resultado que parezca una ventaja pasa por `backtest-auditor` antes de aceptarse.

## Git

Trabaja en ramas `feat/*` o `fix/*`. No hagas commit, push ni PR sin que Sergio lo autorice.
El repositorio es público: no subas datos descargados, modelos, claves ni apuestas personales.
