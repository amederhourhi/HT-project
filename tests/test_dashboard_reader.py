"""
Dashboard reader regression test.

Why this file exists
--------------------
`src/dashboard/app.py` was rewritten from a launch hub into a single triage
reader, and a rewrite that has never been executed is a rewrite that has never
been tested. The other test files in this repository check the rule engine
against dictionaries it is handed; none of them check that the page a clinician
actually looks at says what the engine decided.

This runs the real app in Streamlit's headless test runtime against real
records and asserts on the rendered text. It is deliberately narrow: it checks
that the page states the right tier for the right reason, and that it does not
imply safety where the rules found nothing. It does not check styling.

Records are selected by their position in the catalogue, which the app builds
from the cohort manifest filtered to waveforms present on disk. The test
recomputes that index rather than hardcoding it, so adding records to the
cohort does not break it.

Run:  python tests/test_dashboard_reader.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
from streamlit.testing.v1 import AppTest

APP = str(project_root / "src" / "dashboard" / "app.py")
COHORT_DIR = project_root / "data" / "cohort"
MANIFEST_CSV = COHORT_DIR / "cohort_manifest.csv"

# Tier headlines as the presentation layer renders them.
TIER_1 = "Tier 1 - Routine follow-up"
TIER_2 = "Tier 2 - Clinician review"
TIER_3 = "Tier 3 - Emergency escalation"


def check(label, condition, detail=""):
    assert condition, f"FAILED: {label}" + (f"  ->  {detail}" if detail else "")
    print(f"  ok  {label}")


def catalogue_index_for(ecg_id: int) -> int:
    """
    Position of a cohort record in the picker.

    Mirrors `load_catalogue`: manifest order, records without waveforms skipped.
    """
    manifest = pd.read_csv(MANIFEST_CSV)
    position = 0
    for row in manifest.itertuples(index=False):
        base = COHORT_DIR / row.filename_lr
        if not (base.with_suffix(".hea").exists() and base.with_suffix(".dat").exists()):
            continue
        if int(row.ecg_id) == ecg_id:
            return position
        position += 1
    raise AssertionError(f"record {ecg_id} is not in the on-disk catalogue")


def rendered_text(at: AppTest) -> str:
    """Every markdown string the page emitted, joined for substring assertions."""
    return "\n".join(str(el.value) for el in at.markdown)


def open_record(ecg_id: int) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert not at.exception, f"app raised on first run: {[e.value for e in at.exception]}"
    at.selectbox[0].set_value(catalogue_index_for(ecg_id)).run()
    assert not at.exception, f"app raised for record {ecg_id}: {[e.value for e in at.exception]}"
    return at


if __name__ == "__main__":
    print("\n" + "=" * 74)
    print("DASHBOARD READER - rendered output against real records")
    print("=" * 74)

    # ---------------------------------------------------------------
    # 1. The reader boots, offers records, and does not raise.
    # ---------------------------------------------------------------
    print("\n[1] Cold start")
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    check("boots without an exception", not at.exception,
          str([e.value for e in at.exception]))
    check("offers a record picker", len(at.selectbox) >= 1)
    text = rendered_text(at)
    check("states that it is adjunctive, not a diagnosis",
          "not a diagnosis" in text)
    check("names the commit it is running", "commit" in text and "triage_rules.py" in text)

    # ---------------------------------------------------------------
    # 2. Contiguous elevation escalates and the evidence is on screen.
    # ---------------------------------------------------------------
    print("\n[2] Record 544 - anterior elevation, 67M")
    at = open_record(544)
    text = rendered_text(at)
    check("renders the escalation tier", TIER_3 in text)
    check("names the rule that fired", "G1_CONTIGUOUS_STE" in text)
    check("shows the per-lead measurement grid", "ST deviation at the J point" in text)
    check("shows the demographic cut-point it applied to V2",
          "men &gt;= 40 y in V2: &gt;= 0.20 mV" in text or "men >= 40 y in V2: >= 0.20 mV" in text)
    check("lists the qualifying pair table", "V3 / V4" in text and "V1 / V2" in text)
    # NB: whether that markup renders as a table or leaks as literal text cannot
    # be checked here. `at.markdown` returns the source the app passed in, not a
    # DOM, so a blank line inside the HTML block - the failure that made the pair
    # table print as source - is invisible to this runtime. That is a browser
    # check, and it was done in one.
    check("reports every lead as measured", "12 of 12" in text)

    # ---------------------------------------------------------------
    # 3. A clean record says so without claiming normality.
    # ---------------------------------------------------------------
    print("\n[3] Record 10038 - routine")
    at = open_record(10038)
    text = rendered_text(at)
    check("renders the routine tier", TIER_1 in text)
    check("says no guardrail fired", "No guardrail fired" in text)
    check("refuses to read that as a normal ECG",
          "not a claim that the ECG is normal" in text)
    check("reports no qualifying contiguous pair", "No contiguous pair meets" in text)

    # ---------------------------------------------------------------
    # 4. Escalation without ST criteria must not imply ST criteria.
    # ---------------------------------------------------------------
    # 10231 escalates on rate with a wide QRS and no contiguous elevation. If the
    # reader showed only the tier, a reviewer could not tell an occlusion from a
    # perfusion-rate problem, and the ST panel would be the thing that tells them.
    print("\n[4] Record 10231 - escalates without ST criteria")
    at = open_record(10231)
    text = rendered_text(at)
    check("renders the escalation tier", TIER_3 in text)
    check("attributes it to the rate rule", "G2_SEVERE_TACHYCARDIA" in text)
    check("still reports no qualifying contiguous pair", "No contiguous pair meets" in text)
    check("carries the conduction caveat", "G3_WIDE_QRS" in text)

    # ---------------------------------------------------------------
    # 5. Unknown demographics tighten V2-V3 instead of relaxing them.
    # ---------------------------------------------------------------
    # The sample records carry no age or sex, so opening one exercises the
    # fallback on the page rather than only in a unit test. They are appended
    # after the cohort, so scan backwards until the picker shows one.
    print("\n[5] Sample record with no demographics on file")
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    options = at.selectbox[0].options
    found = None
    for candidate in range(len(options) - 1, max(len(options) - 5, -1), -1):
        at.selectbox[0].set_value(candidate).run()
        if "no demographics on file" in rendered_text(at):
            found = candidate
            break
    check("a demographic-free record is offered", found is not None,
          "the sample records are missing from data/samples")
    if found is not None:
        text = rendered_text(at)
        check("states the fallback on the page", "age/sex unknown" in text)
        check("holds V2 to the most sensitive cut-point",
              "most sensitive cut-point &gt;= 0.15 mV" in text
              or "most sensitive cut-point >= 0.15 mV" in text)

    # ---------------------------------------------------------------
    # 6. Overriding demographics moves the threshold, live.
    # ---------------------------------------------------------------
    print("\n[6] Age override on record 544")
    at = open_record(544)
    before = rendered_text(at)
    at.number_input[0].set_value(35).run()
    after = rendered_text(at)
    check("the page re-runs without raising", not at.exception)
    check("a younger man raises the V2/V3 cut-point",
          "0.25" in after and "0.25" not in before,
          f"before had 0.20 only; after: {'0.25' in after}")

    print("\n" + "=" * 74)
    print("Dashboard reader checks passed.")
    print("=" * 74 + "\n")
