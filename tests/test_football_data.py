"""Unificación de football-data con CSV inventados (sin red ni datos reales)."""

from __future__ import annotations

import numpy as np
import pytest

from src import football_data as fd

OLD = (  # 2018/19: BbAv*, sin hora ni Betfair
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,PSH,PSD,PSA,PSCH,PSCD,PSCA,BbAvH,BbAvD,BbAvA\n"
    "SP1,17/08/2018,Real Equipo,Club Uno,2,1,H,10,8,2.0,3.4,4.0,1.95,3.5,4.2,1.9,3.3,3.9\n"
    "SP1,18/08/2018,Club Uno,Real Equipo,0,0,D,5,6,2.5,3.1,3.0,2.6,3.0,3.0,2.4,3.0,2.9\n"
)
NEW = (  # 2026/27: BOM, hora, sin Pinnacle (PP* no lo es), con Betfair y media
    "﻿Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,PPH,PPD,PPA,PPCH,PPCD,PPCA,"
    "AvgCH,AvgCD,AvgCA,BFECH,BFECD,BFECA\n"
    "SP2,15/08/2026,20:00,Real Equipo,Otro FC,1,3,A,2.1,3.2,3.3,2.0,3.2,3.4,2.05,3.25,3.5,2.1,3.4,3.6\n"
    "SP2,16/08/2026,18:00,Otro FC,Club Dos,1,1,H,1.8,3.5,4.5,1.8,3.5,4.6,1.85,3.5,4.4,,,\n"
)


@pytest.fixture
def raw_dir(tmp_path):
    (tmp_path / "SP1_1819.csv").write_text(OLD, encoding="utf-8")
    (tmp_path / "SP2_2627.csv").write_text(NEW, encoding="utf-8")
    return tmp_path


def test_columns_are_unified_across_seasons_and_bom(raw_dir):
    df = fd.load_all(raw_dir)
    assert len(df) == 4 and list(df["season"].unique()) == ["2018/19", "2026/27"]
    assert df["avg_open_home"].notna().sum() == 2  # BbAvH → avg_open_home
    assert df.loc[df.season == "2026/27", "kickoff"].notna().all()
    assert df["date"].is_monotonic_increasing


def test_pp_columns_are_never_pinnacle(raw_dir):
    df = fd.load_all(raw_dir)
    new = df[df.season == "2026/27"]
    assert new["pin_close_home"].isna().all() and new["pp_close_home"].notna().all()
    assert "pinnacle" not in set(new["ref_source"])


def test_reference_falls_back_pinnacle_betfair_media(raw_dir):
    df = fd.load_all(raw_dir).set_index("home_team")
    old = df[df.season == "2018/19"]
    assert set(old["ref_source"]) == {"pinnacle"}
    new = df[df.season == "2026/27"]
    assert new.loc["Real Equipo", "ref_source"] == "betfair"
    assert new.loc["Otro FC", "ref_source"] == "media"
    probs = df[["ref_prob_home", "ref_prob_draw", "ref_prob_away"]].sum(axis=1)
    assert np.allclose(probs, 1)


def test_team_identity_survives_promotion(raw_dir):
    df = fd.load_all(raw_dir)
    ids = set(df["home_team_id"]) | set(df["away_team_id"])
    assert "real-equipo" in ids and len(ids) == 4  # mismo id en SP1 y en SP2


def test_final_test_flag_and_quality_counts(raw_dir):
    df = fd.load_all(raw_dir)
    assert df.loc[df.season == "2026/27", "is_final_test"].all()
    assert not df.loc[df.season == "2018/19", "is_final_test"].any()
    q = fd.quality(df)
    assert q["duplicados"] == 0 and q["fechas_ilegibles"] == 0
    assert q["resultado_incoherente"] == 1  # 1-1 marcado como H en los datos inventados


def test_missing_directory_explains_how_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="descargar_football_data"):
        fd.load_all(tmp_path)
