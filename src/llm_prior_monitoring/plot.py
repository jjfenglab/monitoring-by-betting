# DEPRECATED: 
# Only used in earlier synthetic Gaussian experiments, but exp_fakenotes
"""Plotting helpers for SR-comparison harness outputs.

Three figures, all driven by the per-cell `curves.pkl` and the master
`all_summary.csv` produced by `aggregate_cell.py` / `aggregate_results.py`:

* `plot_curves`         — ARL0 (x) vs conditional EDD (y) for each detector.
* `plot_summary_rows`   — across settings, mean J_rho width and mean feature
                          overlap for the `shared` and `j_specific` rows.
* `plot_thresholdVsARL` — threshold (x, linear) vs ARL0 / EDD, one subplot per cell.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import nanmean_or_nan


# Method identity = the bet (e_process) + its null source; colors keyed on the
# combined label from `method_label`. Old-name equivalences: symmetric (fixed
# pool) -> symmetric_fixed, symmetric_random -> symmetric_random, and the
# per-step oracle is symmetric_per_step.
METHOD_COLORS: dict[str, str] = {
    "permutation": "C4",
    "symmetric_per_step": "C3",
    "symmetric_random": "C5",
    "symmetric_fixed": "C6",
    # j_specific: re-add when re-integrated
    # "j_specific": "C2",
}


def method_label(e_process: str, null_source: str | None) -> str:
    """Combined method name: e_process alone when null_source is none/None."""
    return e_process if null_source in (None, "none") else f"{e_process}_{null_source}"

CHANGEPOINT_LEARNER_MARKERS: dict[str, str] = {
    "mean_shift": "d",
    "kernel_cosine": "p",
    "kernel_rbf": "^",
    "sw_stored": "o",
    "sw_random": "s",
}

CONCEPT_LEARNER_LINESTYLES: dict[str, str] = {
    "identity": "-",
    "mock": "--",
}

# Short labels for the fields facet grids are split on; used in the subplot titles.
FACET_FIELD_ABBREV: dict[str, str] = {
    "annotation_rate": "ar",
    "noise_rate": "nr",
}


def detector_style(
    e_process: str,
    null_source: str | None,
    changepoint_learner: str,
    concept_learner_type: str = "identity",
    cp_feature_source: str = "embedding",
) -> tuple[str, str, str]:
    """Return (color, linestyle+marker, markerfacecolor) for a detector combination.

    color = method (e_process+null_source), linestyle = concept_learner_type,
    marker = changepoint_learner. cp_feature_source is encoded as marker *fill* so
    embedding vs concept share color/linestyle/marker but stay distinguishable:
    'embedding' -> filled (color), 'concept' -> hollow ('none').
    """
    color = METHOD_COLORS.get(method_label(e_process, null_source), "C7")
    line = CONCEPT_LEARNER_LINESTYLES.get(concept_learner_type, "-")
    marker = CHANGEPOINT_LEARNER_MARKERS.get(changepoint_learner, "")
    facecolor = color if cp_feature_source == "embedding" else "none"
    return color, line + marker, facecolor



def filter_curves_for_figure(curves: dict, key_fields: tuple[str, ...], variant: str,
                             classifier: str | None = None, removal: str | None = None) -> dict:
    """Sub-dict of `curves` for one (imputer_variant, classifier, removal) figure.

    Keeps the matching mock cells, plus the untunable `identity` baseline for the
    same classifier (identity has imputer_variant='off' and removal='none', so it is drawn
    in every figure of its classifier to anchor the comparison). `classifier`/`removal` are
    ignored when absent from `key_fields` (older pickles) or passed as None.
    """
    i_var = key_fields.index("imputer_variant")
    i_clt = key_fields.index("concept_learner_type")
    i_clf = key_fields.index("classifier") if classifier and "classifier" in key_fields else None
    i_rm = (key_fields.index("concept_removal_variant")
            if removal and "concept_removal_variant" in key_fields else None)

    def clf_ok(k):
        return i_clf is None or k[i_clf] == classifier

    def keep(k):
        target = (k[i_var] == variant and clf_ok(k)
                  and (i_rm is None or k[i_rm] == removal))
        return target or (k[i_clt] == "identity" and clf_ok(k))

    return {k: v for k, v in curves.items() if keep(k)}


def _facet_axis(curves: dict, key_fields: tuple[str, ...] | None,
                field: str | None, i_clt: int | None):
    """(values, index) for one facet axis, or ([None], None) to collapse it to a
    single row/column — used when `field` is None or absent from `key_fields`.

    Values are the sorted uniques present in the curve keys, so a build that does
    not sweep the field degrades to a single slice. `identity` curves are excluded
    from the uniques: they carry pinned placeholder rates (ar=1.0, nr=0.0) that
    would otherwise manufacture a facet containing only the baseline.
    """
    if not key_fields or field is None or field not in key_fields:
        return [None], None
    i = key_fields.index(field)
    keys = [k for k in curves if i_clt is None or k[i_clt] != "identity"] or list(curves)
    return sorted({k[i] for k in keys}), i


def _draw_facet_grid(
    label: str, curves: dict, thresholds: np.ndarray, key_fields: tuple[str, ...] | None,
    out_path: Path, *, xsel, ysel, xlabel: str, ylabel: str,
    target_arls: list[float] | None = None, diag: bool = False, n_pre: float | None = None,
    xlim: tuple[float, float] | None = None, ylim: tuple[float, float] | None = None,
    row_field: str | None = "annotation_rate", col_field: str | None = "noise_rate",
) -> None:
    """Facet by `row_field` (rows) × `col_field` (cols); within each facet overlay
    one line per detector triple (styled by `detector_style`). Either axis degrades
    to a single slice when its field is None or absent from `key_fields` — the ARL0
    figure passes `col_field=None` because ARL0 is noise_rate-invariant. `xsel(curve,
    thresholds)` selects the x array; `ysel(curve)` the y array. When `xlim`/`ylim`
    are given they are enforced on every facet so scale is uniform across facets
    and across separately-drawn figures.

    `key_fields` must be the tuple `curves` is actually keyed by — fields are
    resolved positionally, so passing the shift tuple with null-keyed curves would
    read the wrong column."""
    i_clt = (key_fields.index("concept_learner_type")
             if key_fields and "concept_learner_type" in key_fields else None)
    rows_vals, i_row = _facet_axis(curves, key_fields, row_field, i_clt)
    cols_vals, i_col = _facet_axis(curves, key_fields, col_field, i_clt)
    nrows, ncols = len(rows_vals), len(cols_vals)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )

    def in_facet(key, row_val, col_val) -> bool:
        # identity has no concepts, so its pinned rates are meaningless — broadcast
        # it into every facet as the common comparison anchor.
        if i_clt is not None and key[i_clt] == "identity":
            return True
        return ((i_row is None or key[i_row] == row_val)
                and (i_col is None or key[i_col] == col_val))

    for r, row_val in enumerate(rows_vals):
        for cc, col_val in enumerate(cols_vals):
            ax = axes[r][cc]
            for key, curve in curves.items():
                if not in_facet(key, row_val, col_val):
                    continue
                x, y = xsel(curve, thresholds), ysel(curve)
                mask = ~np.isnan(y)
                # cp_feature_source is appended to key_fields (absent in older
                # pickles); it drives marker fill + a label suffix so embedding vs
                # concept series are distinguishable, not just same-styled duplicates.
                has_cpf = bool(key_fields) and "cp_feature_source" in key_fields
                cp_feat = str(key[key_fields.index("cp_feature_source")]) if has_cpf else "embedding"
                color, style, facecolor = detector_style(*key[:4], cp_feat)
                extra = [cp_feat] if has_cpf else []
                ax.plot(x[mask], y[mask], style, ms=6, markeredgewidth=1.2, color=color,
                        markerfacecolor=facecolor,
                        label="/".join([method_label(key[0], key[1]), *map(str, key[2:4]), *extra]))
            for arl in (target_arls or []):
                ax.axvline(arl, color="0.85", ls=":", lw=1)
            title = ", ".join(f"{FACET_FIELD_ABBREV.get(f, f)}={v:g}"
                              for f, v in ((row_field, row_val), (col_field, col_val))
                              if v is not None)
            if title:
                ax.set_title(title, fontsize=8)
    # Reference lines span the final x-range (from xlim if given, else autoscale)...
    if diag or n_pre is not None:
        xr = np.array(xlim if xlim is not None else axes[0][0].get_xlim())
        ref = xr if diag else xr - n_pre
        for row in axes:
            for ax in row:
                ax.plot(xr, ref, "k--", lw=1)
    # ...then enforce uniform scale last (shared axes → set once) so nothing rescales.
    if xlim is not None:
        axes[0][0].set_xlim(xlim)
    if ylim is not None:
        axes[0][0].set_ylim(ylim)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for row in axes:
        row[0].set_ylabel(ylabel)
    fig.suptitle(label, fontsize=10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=4, fontsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot -> {out_path}")


def compute_range(curve_dicts: list[dict], field: str, pad_frac: float = 0.05) -> tuple[float, float]:
    """Global (lo, hi) of `curve[field]` over finite values across every curve in
    every dict, padded by `pad_frac`. Used to share one scale across facets/figures."""
    vals = np.concatenate([c[field] for cd in curve_dicts for c in cd.values()])
    finite = vals[np.isfinite(vals)]
    assert finite.size, f"no finite values for {field!r}"
    lo, hi = float(finite.min()), float(finite.max())
    pad = pad_frac * (hi - lo) or 1.0
    return lo - pad, hi + pad


def plot_curves(
    label: str, curves: dict, target_arls: list[float], out_path: Path,
    n_pre: float | None = None, key_fields: tuple[str, ...] | None = None,
    xlim: tuple[float, float] | None = None, ylim: tuple[float, float] | None = None,
) -> None:
    """ARL0 (x) vs conditional EDD (y), one line per detector, faceted by
    annotation_rate × noise_rate (single panel when those are not swept).
    Vertical guides at each target ARL0; slope-1 line `EDD = ARL0 - n_pre`.
    Pass `xlim`/`ylim` to share one scale across the per-contamination figures."""
    _draw_facet_grid(
        label, curves, np.asarray([]), key_fields, out_path,
        xsel=lambda curve, _t: curve["arl0"], ysel=lambda curve: curve["edd"],
        xlabel="censored ARL0 lower bound", ylabel="conditional EDD",
        target_arls=target_arls, n_pre=n_pre, xlim=xlim, ylim=ylim,
    )




def plot_thresholdVsARL(
    cells: list[tuple[str, dict, np.ndarray]],
    arl0_path: Path,
    edd_paths: list[Path],
    null_curves: dict,
    null_key_fields: tuple[str, ...],
    key_fields: tuple[str, ...] | None = None,
    n_pre: float | None = None,
    thr_xlim: tuple[float, float] | None = None,
    edd_ylim: tuple[float, float] | None = None,
    arl0_ylim: tuple[float, float] | None = None,
) -> None:
    """Threshold (x, linear) vs ARL0 / EDD (y), detectors overlaid within a facet.

    EDD facets by annotation_rate × noise_rate; ARL0 facets by annotation_rate
    only, since the null stream never reaches the noise_rate branch and ARL0 is
    therefore noise_rate-invariant. Each degrades to a single slice when a field
    is not swept.

    `cells`: list of (label, curves, thresholds) from each contamination's
    `curves.pkl`, keyed by `key_fields`. `null_curves` holds the null-derived ARL0
    keyed by `null_key_fields`; it is cell-independent, so one figure is drawn from
    it → `arl0_path`, while EDD is per-cell → `edd_paths[i]`. When `n_pre` is given
    the EDD facets get the slope-1 line `EDD = THRESHOLD - n_pre`.
    Pass `thr_xlim`/`edd_ylim`/`arl0_ylim` to share scales across separately-drawn
    figures (e.g. per imputer_variant); when None they are computed from `cells`.
    """
    assert len(edd_paths) == len(cells), (
        f"edd_paths ({len(edd_paths)}) must match cells ({len(cells)})"
    )
    assert key_fields is None or len(null_key_fields) == len(key_fields) - 1, (
        f"null_key_fields ({null_key_fields}) must be key_fields ({key_fields}) "
        "minus noise_rate; fields are resolved positionally"
    )
    # Shared scales: threshold x-range is common; EDD y-range spans all cells so
    # every per-contamination figure is directly comparable. Precomputed ranges
    # (across variants) win, so OFF/ON figures share one scale.
    all_curve_dicts = [curves for _, curves, _ in cells]
    if thr_xlim is None:
        thr_xlim = (float(cells[0][2].min()), float(cells[0][2].max()))
    if edd_ylim is None:
        edd_ylim = compute_range(all_curve_dicts, "edd")
    if arl0_ylim is None:
        arl0_ylim = compute_range([null_curves], "arl0")

    # ARL0: null-derived, so contamination- and noise_rate-independent → one figure,
    # one line per detector, faceted by annotation_rate alone.
    thresholds0 = cells[0][2]
    _draw_facet_grid(
        "no shift (ARL0)", null_curves, thresholds0, null_key_fields, arl0_path,
        xsel=lambda _curve, thr: thr, ysel=lambda curve: curve["arl0"],
        xlabel="threshold", ylabel="ARL0", diag=True,
        xlim=thr_xlim, ylim=arl0_ylim, col_field=None,
    )
    # EDD: one faceted figure per contamination cell (shared threshold x + EDD y).
    for (label, curves, thresholds), edd_path in zip(cells, edd_paths):
        _draw_facet_grid(
            label, curves, thresholds, key_fields, edd_path,
            xsel=lambda _curve, thr: thr, ysel=lambda curve: curve["edd"],
            xlabel="threshold", ylabel="conditional EDD", n_pre=n_pre,
            xlim=thr_xlim, ylim=edd_ylim,
        )
