"""Versioned output names must never land on an existing file."""
from archery.versioning import next_versioned


def test_first_and_next(tmp_path):
    assert next_versioned(tmp_path, "R", ".html").name == "R_v01.html"
    (tmp_path / "R_v01.html").write_text("x")
    assert next_versioned(tmp_path, "R", ".html").name == "R_v02.html"


def test_case_differs_from_existing(tmp_path):
    # The same athlete typed in a different case on a later run.
    (tmp_path / "Archery_Report_Dhar_Ath_20261001_PARTIAL_v01.html").write_text("old")
    got = next_versioned(tmp_path, "Archery_Report_dhar_ath_20261001_PARTIAL", ".html")
    assert got.name == "Archery_Report_dhar_ath_20261001_PARTIAL_v02.html"
    assert not got.exists()
