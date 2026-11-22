import numpy as np
import pytest
from src.features.fiducials_qrs_v2 import _candidate_boundaries, analyze_12lead_features


def test_rejects_nonfinite():
    a=np.zeros((1000,12)); a[2,0]=np.nan
    assert analyze_12lead_features(a,[300,700],['x']*12,100)['status']=='rejected'


def test_rejects_nonmonotonic_peaks():
    a=np.zeros((1000,12))
    assert analyze_12lead_features(a,[400,400],['x']*12,100)['status']=='rejected'


def test_refuses_unsupported_rate():
    with pytest.raises(ValueError):
        analyze_12lead_features(np.zeros((1000,12)),[100,500],['x']*12,250)


def test_no_signal_has_no_bounds():
    entries, pair=_candidate_boundaries(np.zeros((12,350)),125,500)
    assert entries==[] and pair==[]


def test_single_uncorroborated_lead_is_not_an_accepted_global_boundary():
    # No positive energy in any other lead -> no consensus.
    beats=np.zeros((12,350))
    beats[0,120:126] = np.array([0,.5,1,.5,0,0])
    entries, bounds=_candidate_boundaries(beats,125,500)
    assert bounds==[]
