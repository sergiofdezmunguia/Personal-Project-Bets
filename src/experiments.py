"""Registro de experimentos: una fila por ejecución, sin sobrescribir nunca.

Cada ejecución añade una línea JSON a `experiments/registro.jsonl` con fecha, variante, fase
(`ajuste` o `evaluacion`), parámetros, métricas agregadas, hash de los datos y commit. El fichero
solo se abre en modo «añadir». Además, cada fila guarda el hash de la anterior (`prev_hash`), de
modo que borrar o editar una fila antigua rompe la cadena y `verify()` lo detecta.

Solo se registran agregados (nunca filas de partidos): el repositorio es público.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments" / "registro.jsonl"
GENESIS = "0" * 64


def _row_hash(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def data_hash(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Huella de los datos de entrada: cambia si cambia cualquier partido o resultado usado."""
    cols = columns or [c for c in ("match_id", "date", "home_goals", "away_goals",
                                    "ref_prob_home", "ref_prob_draw", "ref_prob_away") if c in df]  # fmt: skip
    part = df[cols].sort_values(cols[0], kind="stable")
    hashed = pd.util.hash_pandas_object(part, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "scripts"],
                               capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
        return out + ("+cambios" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"


def read(path: Path = REGISTRY) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def verify(path: Path = REGISTRY) -> None:
    """Falla si alguna fila se borró, se editó o se reordenó."""
    prev = GENESIS
    if not Path(path).exists():
        return
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if json.loads(line)["prev_hash"] != prev:
            raise ValueError(f"Registro alterado en la fila {n} de {path}")
        prev = _row_hash(line)


def log(variant: str, phase: str, params: dict, metrics: dict, data: str, *,
        path: Path = REGISTRY, notes: str = "") -> dict:  # fmt: skip
    """Añade una fila al registro. Nunca reescribe las anteriores.

    En el registro real (`REGISTRY`) exige que `src/` y `scripts/` estén commiteados, para que
    cada fila apunte a un código que se pueda recuperar."""
    if phase not in ("ajuste", "evaluacion", "prueba"):
        raise ValueError(f"Fase desconocida: {phase}")
    path = Path(path)
    commit = git_commit()
    if path.resolve() == REGISTRY.resolve() and commit.endswith("+cambios"):
        raise RuntimeError("Haz commit de src/ y scripts/ antes de registrar un experimento")
    verify(path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    row = {
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "variante": variant,
        "fase": phase,
        "parametros": params,
        "metricas": metrics,
        "hash_datos": data,
        "commit": commit,
        "notas": notes,
        "prev_hash": _row_hash(lines[-1]) if lines else GENESIS,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:  # solo añadir
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=float) + "\n")
    return row
