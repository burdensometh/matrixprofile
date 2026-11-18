#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import io
import os
import tempfile

import numpy as np
import pytest

from matrixprofile.algorithms.mpx import mpx
from matrixprofile.algorithms.skimp import skimp
from matrixprofile.algorithms.scrimp import scrimp_plus_plus
from matrixprofile.discover import motifs, discords
from matrixprofile.io import to_json, from_json, to_disk, from_disk
from matrixprofile.revision import (
    revise,
    revision_history,
    is_stale,
    distance_profile_at,
    MAX_LOCAL_DIRTY_FRACTION,
)


def _series(n=60, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n).cumsum()


def test_revise_matches_full_exact_mpx_single_point():
    ts = _series(80)
    w = 8
    profile = mpx(ts, w, n_jobs=1)

    point = 10
    revised = revise(profile, point, ts[point] + 5.0, source='sensor@v2')

    ts2 = ts.copy()
    ts2[point] += 5.0
    truth = mpx(ts2, w, n_jobs=1)

    np.testing.assert_allclose(revised['mp'], truth['mp'])
    np.testing.assert_array_equal(revised['pi'], truth['pi'])


def test_revise_identifies_affected_windows_and_reverse_dependencies():
    ts = _series(80, seed=42)
    w = 8
    profile = mpx(ts, w, n_jobs=1)

    point = 10
    revised = revise(profile, point, ts[point] + 5.0, source='s@v2')
    info = revised['last_revision']

    # every window covering the point (p - w + 1 .. p) is affected
    expected_affected = list(range(max(0, point - w + 1), point + 1))
    assert info['affected_windows'] == expected_affected

    # reverse dependencies are entries whose old 1NN points at an affected
    # window and which are not themselves affected
    pi = profile['pi']
    affected = set(expected_affected)
    expected_reverse = [
        i for i in range(len(pi))
        if i not in affected and pi[i] in affected
    ]
    assert info['reverse_dependent'] == expected_reverse
    assert set(info['dirty_windows']) == affected | set(expected_reverse)


def test_revised_windows_are_candidates_for_clean_entries():
    # A clean entry may acquire a revised window as its new 1NN.  The result
    # must equal the full recomputation even at such entries.
    rng = np.random.RandomState(11)
    ts = rng.randn(70).cumsum()
    w = 7
    profile = mpx(ts, w, n_jobs=1)

    points = np.array([5, 6])
    values = ts[points] + np.array([6.0, -6.0])
    revised = revise(profile, points, values, source='s@v2')

    ts2 = ts.copy()
    ts2[points] = values
    truth = mpx(ts2, w, n_jobs=1)

    np.testing.assert_allclose(revised['mp'], truth['mp'])
    np.testing.assert_array_equal(revised['pi'], truth['pi'])


def test_revise_batch_points_and_sequential_versions():
    ts = _series(50, seed=3)
    w = 6
    profile = mpx(ts, w, n_jobs=1)

    first = revise(profile, [4, 5], [1.0, 2.0], source='s@v2')
    assert first['data_version'] == 1

    second = revise(first, [6], [3.0], source='s@v3')
    assert second['data_version'] == 2

    history = revision_history(second)
    assert [h['data_version'] for h in history] == [1, 2]
    assert [h['source'] for h in history] == ['s@v2', 's@v3']
    assert history[0]['indices'] == [4, 5]

    ts2 = ts.copy()
    ts2[[4, 5, 6]] = [1.0, 2.0, 3.0]
    truth = mpx(ts2, w, n_jobs=1)
    np.testing.assert_allclose(second['mp'], truth['mp'])
    np.testing.assert_array_equal(second['pi'], truth['pi'])


def test_input_profile_is_not_mutated():
    ts = _series(40)
    profile = mpx(ts, 5, n_jobs=1)
    original_ts = profile['data']['ts'].copy()
    original_mp = profile['mp'].copy()

    revise(profile, 10, 99.0, source='s@v2')

    np.testing.assert_array_equal(profile['data']['ts'], original_ts)
    np.testing.assert_array_equal(profile['mp'], original_mp)
    assert 'data_version' not in profile


def test_revision_invalidates_masks_and_discovery_caches_together():
    ts = _series(60, seed=5)
    profile = discords(motifs(mpx(ts, 8, n_jobs=1)))
    assert 'motifs' in profile and 'discords' in profile

    revised = revise(profile, 12, ts[12] + 4, source='s@v2')

    # masks track the data change
    assert revised['valid_mask'].dtype == bool
    assert revised['imputed_mask'].dtype == bool
    # rebuilt caches are stamped with the new version -> not stale
    assert 'motifs' in revised and 'discords' in revised
    assert revised['cache_data_version'] == revised['data_version']
    assert not is_stale(revised)

    # disabling discovery leaves no cache at all
    no_discover = revise(profile, 12, ts[12] + 4, source='s@v2',
                         discover=False)
    assert 'motifs' not in no_discover
    assert 'discords' not in no_discover


def test_nan_revision_marks_invalid_and_imputation_flag():
    ts = _series(40, seed=6)
    profile = mpx(ts, 6, n_jobs=1)

    invalid = revise(profile, 5, np.nan, source='s@v2')
    assert invalid['valid_mask'][5] is np.bool_(False) or \
        invalid['valid_mask'][5] is False
    assert not invalid['imputed_mask'][5]

    imputed = revise(profile, 5, 7.5, source='s@v3', imputed=True)
    assert imputed['valid_mask'][5]
    assert imputed['imputed_mask'][5]

    # batch imputation flags
    batch = revise(profile, [5, 6], [7.5, 8.5], source='s@v4',
                   imputed=[True, False])
    assert batch['imputed_mask'][5] and not batch['imputed_mask'][6]


def test_imputed_revision_matches_exact_recompute():
    ts = _series(40, seed=8)
    w = 6
    profile = mpx(ts, w, n_jobs=1)
    revised = revise(profile, 10, 42.0, source='s@v2', imputed=True)

    ts2 = ts.copy()
    ts2[10] = 42.0
    truth = mpx(ts2, w, n_jobs=1)
    np.testing.assert_allclose(revised['mp'], truth['mp'])
    np.testing.assert_array_equal(revised['pi'], truth['pi'])


def test_forced_full_fallback_is_exact():
    ts = _series(40, seed=9)
    w = 6
    profile = mpx(ts, w, n_jobs=1)

    revised = revise(profile, 5, ts[5] + 3, source='s@v2',
                     allow_local=False)
    assert revised['revision_log'][-1]['strategy'] == 'full_exact'

    ts2 = ts.copy()
    ts2[5] += 3
    truth = mpx(ts2, w, n_jobs=1)
    np.testing.assert_allclose(revised['mp'], truth['mp'])


def test_approximate_profile_falls_back_to_exact():
    ts = _series(50, seed=4)
    approximate = scrimp_plus_plus(ts, 8, sample_pct=0.2, n_jobs=1)
    revised = revise(approximate, 5, ts[5] + 3, source='s@v2')

    assert revised['revision_log'][-1]['strategy'] == 'full_exact'
    ts2 = ts.copy()
    ts2[5] += 3
    truth = mpx(ts2, 8, n_jobs=1)
    np.testing.assert_allclose(revised['mp'], truth['mp'])


def test_similarity_join_fallback_preserves_query():
    rng = np.random.RandomState(2)
    ts = rng.randn(30).cumsum()
    query = rng.randn(15).cumsum()
    profile = mpx(ts, 5, query=query, n_jobs=1)

    revised = revise(profile, 7, ts[7] + 4, source='s@v2')
    assert revised['revision_log'][-1]['strategy'] == 'full_exact'

    ts2 = ts.copy()
    ts2[7] += 4
    truth = mpx(ts2, 5, query=query, n_jobs=1)
    np.testing.assert_allclose(revised['mp'], truth['mp'])
    np.testing.assert_array_equal(revised['pi'], truth['pi'])
    np.testing.assert_allclose(revised['data']['query'], query)


def test_dirty_threshold_triggers_conservative_fallback():
    # revising enough points to dirty over MAX_LOCAL_DIRTY_FRACTION of the
    # profile must trigger the exact fallback instead of a local sweep
    ts = _series(60, seed=12)
    w = 6
    profile = mpx(ts, w, n_jobs=1)

    points = np.arange(0, 45)  # covers almost every window
    values = ts[points] + 10.0
    revised = revise(profile, points, values, source='s@v2')

    assert revised['last_revision']['strategy'] == 'full_exact'
    ts2 = ts.copy()
    ts2[points] = values
    truth = mpx(ts2, w, n_jobs=1)
    np.testing.assert_allclose(revised['mp'], truth['mp'])


def test_pmp_revision_updates_every_window_row():
    ts = _series(50, seed=7)
    windows = [8, 10, 12]
    pmp = skimp(ts, windows=windows, sample_pct=1, n_jobs=1)

    revised = revise(pmp, 20, ts[20] + 4, source='pmp@v1')
    ts2 = ts.copy()
    ts2[20] += 4

    for row, window in enumerate(windows):
        truth = mpx(ts2, window, n_jobs=1)
        length = len(truth['mp'])
        np.testing.assert_allclose(revised['pmp'][row, :length], truth['mp'])
        np.testing.assert_array_equal(revised['pmpi'][row, :length],
                                      truth['pi'])

    assert revised['data_version'] == 1
    assert len(revised['valid_mask']) == len(ts)
    assert 'motifs' in revised and 'discords' in revised


def test_constant_window_revision():
    rng = np.random.RandomState(5)
    ts = rng.randn(40).cumsum()
    w = 6
    profile = mpx(ts, w, n_jobs=1)

    points = np.arange(10, 16)
    revised = revise(profile, points, np.full(6, 3.0), source='s@v2')

    ts2 = ts.copy()
    ts2[points] = 3.0
    truth = mpx(ts2, w, n_jobs=1)
    np.testing.assert_allclose(revised['mp'], truth['mp'])
    np.testing.assert_array_equal(revised['pi'], truth['pi'])


def test_distance_profile_matches_mpx_convention():
    ts = _series(30, seed=13)
    w = 5
    profile = mpx(ts, w, n_jobs=1)
    dp = distance_profile_at(ts, w, 0)

    assert len(dp) == len(ts) - w + 1
    # the minimum over the legal (exclusion-zone respecting) candidates is
    # exactly the MPX matrix profile entry for window 0
    min_lag = int(np.ceil(w / 4.0))
    dp[max(0, 0 - min_lag):min(len(dp), 0 + min_lag + 1)] = np.inf
    assert np.isclose(dp.min(), profile['mp'][0])


def test_source_version_is_required():
    profile = mpx(_series(30), 5, n_jobs=1)
    with pytest.raises(ValueError):
        revise(profile, 3, 1.0, source='')
    with pytest.raises(ValueError):
        revise(profile, 3, 1.0, source=None)


def test_index_validation():
    profile = mpx(_series(30), 5, n_jobs=1)
    with pytest.raises(IndexError):
        revise(profile, 30, 1.0, source='s@v2')
    with pytest.raises(ValueError):
        revise(profile, [1, 2], [1.0], source='s@v2')


def test_json_round_trip_preserves_revision_metadata():
    ts = _series(40, seed=3)
    profile = mpx(ts, 6, n_jobs=1)
    revised = revise(profile, 10, ts[10] + 2, source='io@v2')

    restored = from_json(io.StringIO(to_json(revised)))

    np.testing.assert_allclose(restored['mp'], revised['mp'])
    assert restored['data_version'] == 1
    assert restored['revision_log'][0]['source'] == 'io@v2'
    np.testing.assert_array_equal(np.asarray(restored['valid_mask'], dtype=bool),
                                  revised['valid_mask'])
    np.testing.assert_array_equal(np.asarray(restored['imputed_mask'], dtype=bool),
                                  revised['imputed_mask'])


def test_disk_round_trip():
    ts = _series(40, seed=4)
    revised = revise(mpx(ts, 6, n_jobs=1), 10, 5.0, source='io@v2')

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'profile.json')
        to_disk(revised, path)
        restored = from_disk(path)

    np.testing.assert_allclose(restored['mp'], revised['mp'])
    assert restored['data_version'] == revised['data_version']


def test_legacy_profile_without_revision_keys_is_revisable():
    ts = _series(40, seed=5)
    legacy = mpx(ts, 6, n_jobs=1)
    restored = from_json(io.StringIO(to_json(legacy)))

    assert 'data_version' not in restored
    revised = revise(restored, 5, 1.0, source='legacy@v1')

    assert revised['data_version'] == 1
    assert len(revised['valid_mask']) == len(ts)
    assert not revised['imputed_mask'].any()
    truth = mpx(np.r_[ts[:5], 1.0, ts[6:]], 6, n_jobs=1)
    np.testing.assert_allclose(revised['mp'], truth['mp'])


def test_stale_detection_after_manual_cache_tampering():
    ts = _series(40, seed=6)
    revised = revise(mpx(ts, 6, n_jobs=1), 5, 1.0, source='s@v2')
    assert not is_stale(revised)

    # simulate an old cache surviving a data change
    revised['data_version'] = 3
    assert is_stale(revised)


@pytest.mark.parametrize('seed', range(10))
def test_randomized_revisions_always_match_full_mpx(seed):
    rng = np.random.RandomState(seed)
    n = rng.randint(25, 100)
    w = rng.randint(4, max(5, n // 3))
    ts = rng.randn(n).cumsum()
    profile = mpx(ts, w, n_jobs=1)

    n_revisions = rng.randint(1, 4)
    points = rng.choice(n, size=n_revisions, replace=False)
    values = ts[points] + rng.randn(n_revisions) * 5

    revised = revise(profile, points, values, source='r@%d' % seed)

    ts2 = ts.copy()
    ts2[points] = values
    truth = mpx(ts2, w, n_jobs=1)
    np.testing.assert_allclose(revised['mp'], truth['mp'])
    np.testing.assert_array_equal(revised['pi'], truth['pi'])
