# -*- coding: utf-8 -*-
"""
Source-versioned revision support for MatrixProfile / Pan-MatrixProfile
objects.

A historical observation may be corrected (revised) after a profile has been
computed.  Revising a single data point does *not* only change the matrix
profile value at the matching subsequence index:

1. Every window that covers the revised point (``max(0, p - w + 1) .. p``)
   changes content, so its own 1NN entry is stale.
2. Any other window whose recorded nearest neighbor points at one of those
   changed windows loses the legality of its recorded optimum ("old neighbor
   dependency").  Its new nearest neighbor may be anywhere, including one of
   the revised windows.
3. Consequently every query must consider the revised windows as *new*
   candidates, not merely recompute the revised indices themselves.

This module implements an incremental, exact local update that follows those
rules.  When the local update cannot be proven exact (unsupported algorithm,
similarity join, approximate profile, too many dirty windows, ...) it
conservatively falls back to a full exact recomputation.

The update is published atomically: the time series, validity/imputation
masks, matrix profile / profile indices and every derived cache (motifs,
discords and other discovery results) are invalidated together and only the
fully rebuilt profile is returned.  Old on-disk versions stay readable
because all bookkeeping is attached via additive keys.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import copy

import numpy as np

from matrixprofile import core
from matrixprofile.algorithms.mpx import mpx
from matrixprofile.algorithms.top_k_motifs import top_k_motifs
from matrixprofile.algorithms.top_k_discords import top_k_discords


# Keys that cache discovery/interpolation results derived from mp/pi.  They
# are invalidated on every revision because they cannot survive a data change.
DERIVED_CACHE_KEYS = (
    'motifs',
    'discords',
    'cmp',
    'cpi',
    'av',
)

# Algorithms whose mp/pi are produced by an exact self-join with the same
# z-normalized Euclidean convention as MPX and therefore support a local
# update.
LOCAL_UPDATE_ALGORITHMS = ('mpx',)

# When the dirty fraction of a single profile exceeds this ratio, the local
# update degenerates to an O(L^2) sweep without any advantage over a fresh
# exact MPX run, so we conservatively recompute from scratch.
MAX_LOCAL_DIRTY_FRACTION = 0.5

# Constant-subsequence threshold mirrored from mass2/mpx behavior.
CONSTANT_THRESHOLD = 1e-10


def _is_nan_inf(value):
    """Scalar nan/inf test that never raises on ordinary floats."""
    return bool(np.isnan(value) or np.isinf(value))


def _window_stats(ts, w):
    """
    Returns the per-window mean and inverse standard deviation using the same
    Welford-based statistics routine (``cycore.muinvn``) as MPX.
    """
    from matrixprofile import cycore
    mu, inv_sig = cycore.muinvn(ts.astype('d'), int(w))
    return np.asarray(mu), np.asarray(inv_sig)


def distance_profile_at(ts, w, query_index, mu=None, inv_sig=None):
    """
    Exact z-normalized Euclidean distance profile of the length-``w``
    subsequence at ``query_index`` against every length-``w`` window of
    ``ts``.

    The metric matches MPX exactly:

        d(i, j) = sqrt(2 * w * (1 - pearson(i, j)))

    Constant windows follow the same convention as ``mass2``: two constant
    windows have distance 0, a constant query against a non-constant window
    has the maximal distance ``sqrt(2 * w)``.

    Parameters
    ----------
    ts : np.ndarray
        (Revised) time series.
    w : int
        Window size.
    query_index : int
        Start index of the query subsequence.
    mu, inv_sig : np.ndarray, optional
        Precomputed window means and inverse standard deviations.

    Returns
    -------
    np.ndarray :
        Distance profile of length ``len(ts) - w + 1``.
    """
    ts = core.to_np_array(ts).astype('d')
    n = len(ts)
    profile_len = n - w + 1
    if mu is None or inv_sig is None:
        mu, inv_sig = _window_stats(ts, w)

    roll = np.lib.stride_tricks.sliding_window_view(ts, w)
    query = roll[query_index]
    q_mu = mu[query_index]
    q_inv_sig = inv_sig[query_index]

    centered_q = query - q_mu
    # sum_j (x_j - mu_i) * (q_j - mu_q) for every window i
    cov_sum = (roll - mu[:, None]) @ centered_q
    corr = cov_sum * inv_sig * q_inv_sig
    corr = np.clip(corr, -1.0, 1.0)

    max_distance = np.sqrt(2.0 * w)
    dist = np.sqrt(np.clip(2.0 * w * (1.0 - corr), 0.0, None))

    # Zero-sigma windows show up as non-finite (infinite) inverse sigmas.
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma = np.where(np.isfinite(inv_sig), 1.0 / inv_sig, 0.0)
        q_sigma = 1.0 / q_inv_sig if np.isfinite(q_inv_sig) else 0.0
    constant_windows = ~np.isfinite(inv_sig) | (sigma < CONSTANT_THRESHOLD)
    q_constant = q_sigma < CONSTANT_THRESHOLD
    if q_constant:
        dist[:] = max_distance
        dist[constant_windows] = 0.0

    return dist


def _excluded(index, other, min_lag, is_join):
    """MPX self-join exclusion test (min lag = ceil(w / 4))."""
    if is_join:
        return False
    return abs(int(index) - int(other)) <= min_lag


def _affected_windows(point_indices, w, profile_len):
    """
    All window start indices whose content covers at least one revised point.
    """
    affected = np.zeros(profile_len, dtype=bool)
    for point in point_indices:
        start = max(0, int(point) - w + 1)
        stop = min(profile_len - 1, int(point))
        if start <= stop:
            affected[start:stop + 1] = True
    return np.flatnonzero(affected)


def _identify_dirty(point_indices, mp, pi, w):
    """
    Identify every matrix-profile entry that cannot be trusted after the
    points in ``point_indices`` change.

    Two sets are unioned:

    * ``A``: windows covering a revised point (their subsequence content
      changed, so their own row is stale);
    * ``R``: windows whose recorded 1NN index points into ``A`` (the old
      neighbor dependency).  The recorded optimum of such an entry is no
      longer a legal/evaluated candidate.

    Returns
    -------
    tuple : (affected, reverse_dependent, dirty) boolean masks
    """
    profile_len = len(mp)
    affected_mask = np.zeros(profile_len, dtype=bool)
    affected_idx = _affected_windows(point_indices, w, profile_len)
    affected_mask[affected_idx] = True

    reverse_mask = np.zeros(profile_len, dtype=bool)
    pi_arr = np.asarray(pi)
    valid_pi = np.isfinite(pi_arr.astype('d')) & (pi_arr >= 0)
    reverse_mask[valid_pi & np.isin(pi_arr, affected_idx)] = True

    dirty_mask = affected_mask | reverse_mask
    return affected_mask, reverse_mask, dirty_mask


def _local_update_mp(ts, w, mp, pi, point_indices, is_join=False):
    """
    Exact incremental update of a self-join matrix profile.

    Returns ``(mp, pi, info)`` where ``info`` explains whether the local
    path was used or why a full recomputation is required.

    Correctness argument
    --------------------
    Let A be the windows covering revised points and D = A ∪ R the dirty
    set (R = entries whose recorded 1NN lies in A).

    * Entries outside D keep their old recorded neighbor j, which is outside
      A, so the pair distance is unchanged.  Their only possible improvement
      is a *new* candidate from A; the distance profile of every a ∈ A is
      therefore evaluated against every entry and merged in.
    * Entries inside D have no trustworthy recorded optimum.  Their exact
      1NN is recomputed by evaluating their distance profile over the whole
      series, honoring the exclusion zone.  Distance profiles computed for
      A are reused via the symmetric cache.
    """
    ts = core.to_np_array(ts).astype('d')
    n = len(ts)
    profile_len = n - w + 1
    min_lag = int(np.ceil(w / 4.0))

    affected_mask, reverse_mask, dirty_mask = _identify_dirty(
        point_indices, mp, pi, w
    )
    affected_idx = np.flatnonzero(affected_mask)
    dirty_idx = np.flatnonzero(dirty_mask)

    info = {
        'strategy': 'local',
        'affected_windows': affected_idx.tolist(),
        'reverse_dependent': np.flatnonzero(reverse_mask).tolist(),
        'dirty_windows': dirty_idx.tolist(),
    }

    # Conservative fallback: when most of the profile is dirty the O(|D| * L)
    # local sweep is more expensive than the optimized MPX implementation and
    # offers no guarantee benefit.
    if profile_len == 0 or float(len(dirty_idx)) / profile_len > MAX_LOCAL_DIRTY_FRACTION:
        info['strategy'] = 'full_exact'
        info['reason'] = 'dirty fraction exceeded local update threshold'
        return _full_exact_mp(ts, w, is_join=is_join), info

    mu, inv_sig = _window_stats(ts, w)

    new_mp = np.array(mp, dtype='d', copy=True)
    new_pi = np.array(pi, dtype='int64', copy=True)

    # Cache of distance profiles keyed by query index.  Every profile is
    # symmetric (d(i, j) == d(j, i)), so a profile computed for one dirty
    # window answers queries for all of its targets.
    profile_cache = {}

    def get_profile(idx):
        idx = int(idx)
        if idx not in profile_cache:
            profile_cache[idx] = distance_profile_at(
                ts, w, idx, mu=mu, inv_sig=inv_sig
            )
        return profile_cache[idx]

    # 1) Revised windows become new candidates for *every* query, including
    #    the clean entries whose old optimum may now be beaten.
    for a in affected_idx:
        dp = get_profile(a)
        for target in range(profile_len):
            if _excluded(a, target, min_lag, is_join):
                continue
            candidate = dp[target]
            if not np.isfinite(candidate):
                continue
            if not np.isfinite(new_mp[target]) or candidate < new_mp[target]:
                new_mp[target] = candidate
                new_pi[target] = a

    # 2) Dirty entries (including reverse-dependency entries outside A) must
    #    re-find a legal optimum over the whole series.
    for d in dirty_idx:
        dp = get_profile(d)
        best_dist = np.inf
        best_idx = -1
        for candidate_idx in range(profile_len):
            if _excluded(d, candidate_idx, min_lag, is_join):
                continue
            value = dp[candidate_idx]
            if np.isfinite(value) and value < best_dist:
                best_dist = value
                best_idx = candidate_idx
        if best_idx >= 0:
            new_mp[d] = best_dist
            new_pi[d] = best_idx
        else:
            new_mp[d] = np.inf
            new_pi[d] = -1

    return (new_mp, new_pi), info


def _full_exact_mp(ts, w, is_join=False, query=None):
    """
    Conservative fallback: recompute the exact profile with MPX, the same
    algorithm that produced the supported initial profiles.  When the
    original profile was a similarity join, the original query is retained.
    """
    if is_join and isinstance(query, np.ndarray):
        profile = mpx(ts, w, query=query, cross_correlation=False, n_jobs=1)
    else:
        profile = mpx(ts, w, query=None, cross_correlation=False, n_jobs=1)
    return profile['mp'], profile['pi']


def _supports_local_update(profile):
    """
    A profile may be updated locally only when every assumption of the
    incremental routine holds.  Anything else triggers the exact fallback.
    """
    if not core.is_mp_obj(profile):
        return False, 'pan-matrix-profile is re-joined row by row'

    if profile.get('join', False):
        return False, 'similarity joins are recomputed exactly'

    if profile.get('metric', 'euclidean') != 'euclidean':
        return False, 'non-euclidean metric requires exact recomputation'

    algorithm = profile.get('algorithm', None)
    if algorithm not in LOCAL_UPDATE_ALGORITHMS:
        return False, 'algorithm {} has no local update rule'.format(algorithm)

    sample_pct = profile.get('sample_pct', 1)
    try:
        if float(sample_pct) < 1.0:
            return False, 'approximate profile is promoted to an exact recomputation'
    except (TypeError, ValueError):
        return False, 'unknown sample_pct {}'.format(sample_pct)

    data = profile.get('data', {})
    if not isinstance(data, dict) or not isinstance(data.get('ts'), np.ndarray):
        return False, 'profile has no embedded time series'

    return True, None


def _ensure_revision_bookkeeping(profile):
    """
    Attach additive, backwards compatible revision metadata to a profile.
    Existing on-disk profiles without these keys load normally and get the
    bookkeeping lazily on the first revision.
    """
    ts = profile['data']['ts']
    n = len(ts)
    if 'valid_mask' not in profile:
        profile['valid_mask'] = np.ones(n, dtype=bool)
    if 'imputed_mask' not in profile:
        profile['imputed_mask'] = np.zeros(n, dtype=bool)
    if 'data_version' not in profile:
        profile['data_version'] = 0
    if 'revision_log' not in profile:
        profile['revision_log'] = []
    return profile


def _invalidate_derived_caches(profile):
    """
    Drop every cache derived from mp/pi together.  Leaving a stale motif or
    interpolation cache next to a fresh ``data.ts`` would silently serve
    results that no longer hold, so none of them may survive a revision.
    """
    for key in DERIVED_CACHE_KEYS:
        profile.pop(key, None)


def _rebuild_discovery(profile, discover):
    """
    Re-run discovery (motifs/discords) after the profile is updated.  Only
    invoked once the new profile is internally consistent, and the rebuilt
    cache is explicitly stamped with the data version it was built from.
    """
    if not discover:
        return profile
    profile = top_k_motifs(profile)
    profile = top_k_discords(profile)
    profile['cache_data_version'] = int(profile.get('data_version', 0))
    return profile


def revise(profile, indices, values, source, imputed=False, discover=True,
           allow_local=True):
    """
    Apply a source-versioned revision of historical observations to an
    existing MatrixProfile and return the updated profile.

    The revision is atomic: either the full update succeeds or the caller's
    profile is untouched.  The returned object carries a monotonically
    increasing ``data_version``, per-point ``valid_mask`` / ``imputed_mask``
    and a ``revision_log`` entry describing the source of the change.

    Parameters
    ----------
    profile : dict
        A ``MatrixProfile`` data structure (e.g. output of ``mpx`` or
        ``compute`` with a single window).
    indices : int or array_like
        One or more time-series positions to revise.
    values : float or array_like
        The corrected observations, aligned with ``indices``.  Use
        ``np.nan`` / ``np.inf`` to mark an observation invalid; the point is
        then flagged in ``valid_mask`` and (optionally) imputed.
    source : str
        Identifier/version of the data source that issued the correction,
        e.g. ``'sensor-a@v3'``.  Stored verbatim in the revision log.
    imputed : bool or array_like, default False
        True when the supplied value is an interpolation rather than an
        authoritative observation.  Imputed points are flagged in
        ``imputed_mask`` and the covering windows are treated identically to
        revised windows.
    discover : bool, default True
        Recompute top motifs and discords on the updated profile.  Cached
        discovery results are always invalidated regardless of this flag.
    allow_local : bool, default True
        Allow the exact incremental local update.  Set to False to force the
        conservative full exact recomputation.

    Returns
    -------
    dict :
        The updated profile (a deep copy; the input is never mutated).

    Notes
    -----
    Pan-MatrixProfile objects are supported by re-running each window row
    through the same revision rule; rows that cannot be updated locally fall
    back to MPX for that window.  Approximate profiles (SCRIMP++, STOMP, ...)
    and similarity joins always fall back to a full exact recomputation.
    """
    if not core.is_mp_or_pmp_obj(profile):
        raise ValueError('Expecting a MatrixProfile or PMP data structure!')

    if source is None or not str(source):
        raise ValueError('a non-empty source version is required for revisions!')

    indices = np.atleast_1d(np.asarray(indices, dtype='int64'))
    values = np.atleast_1d(np.asarray(values, dtype='d'))
    if indices.shape != values.shape:
        raise ValueError('indices and values must have the same length!')

    imputed_arr = np.atleast_1d(np.asarray(imputed, dtype=bool))
    if imputed_arr.size == 1 and indices.size > 1:
        imputed_arr = np.repeat(imputed_arr, indices.size)
    if imputed_arr.shape != indices.shape:
        raise ValueError('imputed flag shape must match indices!')

    work = copy.deepcopy(profile)
    work = _ensure_revision_bookkeeping(work)

    ts = work['data']['ts']
    n = len(ts)
    if np.any(indices < 0) or np.any(indices >= n):
        raise IndexError('revision index outside of the time series range!')

    if core.is_pmp_obj(work):
        work = _revise_pmp(work, indices, values, source, imputed_arr,
                           discover, allow_local)
    else:
        work = _revise_mp(work, indices, values, source, imputed_arr,
                          discover, allow_local)

    return work


def _apply_observation_updates(work, indices, values, imputed_arr):
    """
    Write corrected observations and maintain the validity / imputation
    masks.  The masks move in lockstep with ``data.ts``.
    """
    ts = work['data']['ts']
    for idx, value, was_imputed in zip(indices, values, imputed_arr):
        ts[int(idx)] = value
        if _is_nan_inf(value):
            work['valid_mask'][int(idx)] = False
            # An invalid observation cannot simultaneously be a usable
            # imputation; clear any stale imputation flag.
            work['imputed_mask'][int(idx)] = False
        else:
            work['valid_mask'][int(idx)] = True
            work['imputed_mask'][int(idx)] = bool(was_imputed)


def _append_revision_log(work, indices, values, source, imputed_arr,
                         strategy, reason=None):
    work['data_version'] = int(work.get('data_version', 0)) + 1
    work['revision_log'].append({
        'data_version': work['data_version'],
        'source': str(source),
        'indices': np.asarray(indices, dtype='int64').tolist(),
        'values': np.asarray(values, dtype='d').tolist(),
        'imputed': np.asarray(imputed_arr, dtype=bool).tolist(),
        'strategy': strategy,
        'reason': reason,
    })


def _revise_mp(work, indices, values, source, imputed_arr, discover,
               allow_local):
    ts = work['data']['ts']
    w = int(work['w'])
    old_mp = work['mp']
    old_pi = work['pi']
    is_join = bool(work.get('join', False))

    supported, reason = _supports_local_update(work)
    use_local = allow_local and supported
    if not allow_local and supported:
        reason = 'local update explicitly disabled by caller'

    # Write the corrected observations first so whichever strategy runs
    # sees the revised series.
    _apply_observation_updates(work, indices, values, imputed_arr)
    revised_ts = work['data']['ts']

    strategy = 'full_exact'
    fallback_reason = None
    info = None
    query = work.get('data', {}).get('query', None)
    if use_local:
        (new_mp, new_pi), info = _local_update_mp(
            revised_ts, w, old_mp, old_pi, indices, is_join=is_join
        )
        strategy = info['strategy']
        if strategy == 'full_exact':
            fallback_reason = info.get('reason')
    else:
        fallback_reason = reason
        new_mp, new_pi = _full_exact_mp(revised_ts, w, is_join=is_join,
                                        query=query)
        info = {
            'strategy': 'full_exact',
            'reason': fallback_reason,
            'affected_windows': _affected_windows(
                indices, w, len(old_mp)).tolist(),
            'reverse_dependent': [],
            'dirty_windows': [],
        }

    # Publish only after every component is consistent.
    work['mp'] = new_mp
    work['pi'] = new_pi
    _invalidate_derived_caches(work)
    _append_revision_log(work, indices, values, source, imputed_arr,
                         strategy, fallback_reason)
    work['last_revision'] = info
    work = _rebuild_discovery(work, discover)
    return work


def _revise_pmp(work, indices, values, source, imputed_arr, discover,
                allow_local):
    """
    Pan-MatrixProfile revision: every window row is an independent self-join
    profile.  A row whose local preconditions fail is recomputed with MPX
    for that window only, keeping the blast radius minimal while remaining
    exact.
    """
    _apply_observation_updates(work, indices, values, imputed_arr)
    revised_ts = work['data']['ts']

    windows = np.asarray(work['windows']).ravel()
    pmp = work['pmp']
    pmpi = work['pmpi']
    row_strategies = []

    for row, w in enumerate(windows):
        w = int(w)
        profile_len = len(revised_ts) - w + 1
        mp_row = np.asarray(pmp[row, :profile_len], dtype='d')
        pi_row = np.asarray(pmpi[row, :profile_len], dtype='int64')

        use_local = allow_local and work.get('metric', 'euclidean') == 'euclidean'
        reason = None
        if use_local:
            (new_mp, new_pi), info = _local_update_mp(
                revised_ts, w, mp_row, pi_row, indices, is_join=False
            )
            row_strategy = info['strategy']
            reason = info.get('reason')
        else:
            reason = 'pmp local update disabled or non-euclidean metric'
            new_mp, new_pi = _full_exact_mp(revised_ts, w, is_join=False)
            row_strategy = 'full_exact'

        pmp[row, :profile_len] = new_mp
        pmpi[row, :profile_len] = new_pi
        row_strategies.append({'window': w, 'strategy': row_strategy,
                               'reason': reason})

    _invalidate_derived_caches(work)
    overall = 'local' if all(r['strategy'] == 'local' for r in row_strategies) \
        else 'full_exact'
    _append_revision_log(work, indices, values, source, imputed_arr,
                         overall,
                         'row-wise revision of pan-matrix-profile')
    work['last_revision'] = {'rows': row_strategies}
    work = _rebuild_discovery(work, discover)
    return work


def revision_history(profile):
    """
    Return the list of revisions applied to a profile, oldest first.  Empty
    for a profile that has never been revised.
    """
    if not isinstance(profile, dict):
        return []
    return list(profile.get('revision_log', []))


def is_stale(profile):
    """
    True when a profile carries derived caches (motifs/discords/...) that
    predate its current ``data_version``.  A correctly published revision
    never leaves such caches behind; this helper is mainly useful after
    manual manipulation or when loading profiles produced by third parties.
    """
    if not isinstance(profile, dict) or profile.get('data_version', 0) == 0:
        return False
    has_discovery_cache = any(key in profile for key in ('motifs', 'discords'))
    if not has_discovery_cache:
        return False
    cached_version = profile.get('cache_data_version', None)
    if cached_version is None:
        # Legacy profile with revisions but no cache version marker: treat
        # any surviving discovery cache as stale to stay conservative.
        return True
    return cached_version < profile.get('data_version', 0)
