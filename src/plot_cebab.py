"""ARL-vs-EDD figures for exp_cebab: declare baselines, facet, draw.

Three stages, in order:

1. declare  -- --baseline COL=V1,V2 (repeatable, specs UNION) splits off the rows that
   belong in every panel: the CTM family, and the arms that never annotate so no
   annotation-cost level owns them. So plotted = OR(baselines) u rest.
2. facet    -- --facet-row / --facet-col cut the rest into a grid.
3. draw     -- compared series solid, baselines dashed in every panel.

Baselines are declared, never inferred: unlike its predecessor this script gives NaN no
special meaning, so a row that is neither a baseline nor faceted is an error, not a row
that silently multiplies across the grid.

--style paper (the default) names each arm from semisynthetic_display_names.json's
series_short_names and draws one legend below the panels, dashed iff the arm is a CTM;
--style diagnostic keeps the long "Field: Value, ..." labels, the Comparators legend and
the subtitle.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns

from llm_prior_monitoring.display_names import (display_assignment, name_series_short,
                                                order_series_names)

# Config columns eligible to appear in a facet, a hue label, a --baseline or a --filter.
# LGM and CTM rows each carry only their own block; the other block is NaN after the
# concat, which is why a row's own NaN columns are skipped when its label is built.
LGM_LABEL_COLS = ["cp_feature_source", "concept_learner_type", "changepoint_learner",
                  "e_process", "classifier_type", "annotation_rate",
                  "scl_variant", "budget_profile", "budget",
                  "imputer_variant", "removal_variant", "oracle_variant",
                  "search_all_sizes", "llm_model", "history_sample_size"]
CTM_LABEL_COLS = ["scorer", "variant", "jump_rates"]
LABEL_COLS = ["family"] + LGM_LABEL_COLS + CTM_LABEL_COLS

# Both groups share one palette, so colour identifies the series and linestyle its role.
# The guide line is dotted so it does not read as one of the dashed baselines.
SERIES_STYLE = (0, (5, 1)) #"--" #(0, (3, 1, 1, 1, 1, 1)) #
BASELINE_STYLE = (0, (1, 1)) #"-."
GUIDE_STYLE = (0, (1, 3)) #":"
# Seed-bootstrap band per --band: (column infix, legend label). Simultaneous is sup-t
# (covers the whole curve); pointwise covers each threshold separately.
BAND_KINDS = {
    "simultaneous": ("sim", "simultaneous 95% band (seed bootstrap)"),
    "pointwise": ("pw", "pointwise 95% band (seed bootstrap)"),
}
BAND_ALPHA = 0.2
# Paper style: one fixed Okabe-Ito (seaborn "colorblind") colour per paper name, so an arm
# keeps its colour across experiments whichever other arms share the figure. Arms that
# play the same role in different grids (LLM / LLM+BERTopic, the oracle variants) share a
# colour; build_paper_palette asserts no two arms in one figure collide.
PAPER_COLORS = {
    "LLM adaptive": "#0173b2",
    "LLM+BERTopic adaptive": "#0173b2",
    "SAE adaptive": "#de8f05",
    "Embedding adaptive": "#029e73",
    "oracle adaptive": "#d55e00",
    "oracle-all8 adaptive": "#d55e00",
    "oracle-pool adaptive": "#d55e00",
    "Ensemble adaptive": "#cc78bc",
    "SAE-online adaptive": "#fbafe4",
    "simulated adaptive": "#ece133",
    "Oracle": "#ca9161",
    "Informed-shift CTM": "#56b4e9",
    "Unrelated-shift CTM": "#949494",
    "Centroid CTM": "#000000",
}
PAPER_PANEL_SIZE = (3.3, 2.4)
PAPER_RC = {"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
            "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
            "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42}
GUIDE_COLOR = "0.35"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARL vs EDD figures for exp_cebab.")
    p.add_argument("--csv", type=Path, required=True, help="all_summary.csv")
    p.add_argument("--ctm-csv", type=Path, default=None,
                   help="all_ctm_summary.csv; when given, CTM curves join the same axes.")
    p.add_argument("--n-burn-in", type=int, required=True,
                   help="Burn-in excluded from every run length; sets the EDD reference "
                        "line offset to n_pre - n_burn_in. Stated in the subtitle when --ctm-csv is "
                        "given: the CTM bets from t=0 while the SCL burns in, a gap that "
                        "is invisible on shared axes yet dominates EDD.")
    p.add_argument("--out-nominal", type=Path, required=True,
                   help="Output figure: nominal ARL vs EDD")
    p.add_argument("--out-calibration", type=Path, required=True,
                   help="Output figure: empirical vs nominal ARL")
    p.add_argument("--baseline", type=str, action="append", default=None,
                   dest="baselines", metavar="COL=VAL[,VAL...]",
                   help="Rows drawn dashed in every panel and exempt from --filter. "
                        "Values within a spec union; repeated specs union. "
                        "Default: no baselines.")
    p.add_argument("--n-pre", type=int, default=None,
                   help="If set, draw the guide line EDD = ARL - n_pre")
    p.add_argument("--threshold-min", type=float, default=0,
                   help="Shared x-axis min for the two ARL-vs-EDD figures "
                        "(the sweep's min threshold). Default: data-derived limits.")
    p.add_argument("--threshold-max", type=float, default=None,
                   help="Shared x-axis max for the two ARL-vs-EDD figures.")
    p.add_argument("--threshold-min-null", type=float, default=None,
                   help="x-axis min for the calibration figure, which sweeps the same "
                        "nominal ARL but usually wants its own range. "
                        "Default: data-derived limits.")
    p.add_argument("--threshold-max-null", type=float, default=None,
                   help="x-axis max for the calibration figure.")
    p.add_argument("--facet-row", type=str, default=None,
                   help="Config column whose values become rows of the subplot grid.")
    p.add_argument("--facet-col", type=str, default=None,
                   help="Config column whose values become columns of the subplot grid.")
    p.add_argument("--facet-row-order", type=str, nargs="+", default=None,
                   help="Explicit panel order for --facet-row values; values not "
                        "listed are appended alphabetically. Default: alphabetical.")
    p.add_argument("--facet-col-order", type=str, nargs="+", default=None,
                   help="Explicit panel order for --facet-col values (see "
                        "--facet-row-order).")
    p.add_argument("--hue-cols", type=str, nargs="+", default=None,
                   help="Config columns forming the series label. Default: every label "
                        "column that varies and is not already spent on a facet.")
    p.add_argument("--title-prefix", type=str, default=None,
                   help="Prefix prepended to every figure title (e.g., shift_aspect name).")
    p.add_argument("--band", choices=list(BAND_KINDS), default="simultaneous",
                   help="Which seed-bootstrap band to shade around each curve.")
    p.add_argument("--style", choices=["paper", "diagnostic"], default="paper",
                   help="paper: short series names, one legend, print rcParams. "
                        "diagnostic: the long label chains, the Comparators legend and "
                        "the subtitle -- use it when a sweep's arms are not all named.")
    p.add_argument("--log", type=Path, required=True)
    return p.parse_args()


def load_summaries(csv: Path, ctm_csv: Path | None) -> pd.DataFrame:
    """Concat the LGM and CTM summaries, stamping `family` as the arm's provenance.

    family is a filter key (--baseline family=CTM), so it stays a bare token; the burn-in
    difference it used to spell out goes in the subtitle. Each family's config columns
    are NaN on the other's rows -- compute_summary emits identical metric columns for
    both, which is what makes the concat meaningful.
    """
    # history_sample_size stays text ("100", not 100.0): NaN rows would otherwise make
    # pandas re-infer the column as float and the legend would print "100.0".
    df = pd.read_csv(csv, dtype={"history_sample_size": str})
    df["family"] = "LGM"
    logging.info("loaded %d rows from %s", len(df), csv)
    if ctm_csv is not None:
        ctm = pd.read_csv(ctm_csv)
        ctm["family"] = "CTM"
        df = pd.concat([df, ctm], ignore_index=True)
        logging.info("loaded %d CTM rows from %s", len(ctm), ctm_csv)
    # empirical_arl is deliberately NOT required: an arm run without null streams has NaN
    # there and still belongs on the EDD figures. seaborn skips its rows on the two
    # figures that plot it.
    return df.dropna(subset=["edd", "nominal_arl"])


def parse_specs(df: pd.DataFrame, specs: list[str] | None,
                strict: bool = True) -> list[tuple[str, list[str]]]:
    """Split "COL=V1,V2" specs into (column, values), validating both against df.

    When strict=True (filters), asserts that EACH listed value matches at least one row,
    so a renamed budget profile fails loudly instead of quietly narrowing the figure.
    When strict=False (baselines), warns and drops values/specs that match no rows, since
    baselines are optional comparators that may not exist in every experiment configuration.
    """
    parsed = []
    for spec in specs or []:
        assert "=" in spec, f"expected COL=VAL[,VAL...], got {spec!r}"
        col, joined = spec.split("=", 1)
        assert col in df.columns, f"column {col!r} not in {sorted(df.columns)}"
        values = joined.split(",")
        as_str = df[col].astype(str)
        matched_values = []
        for value in values:
            if (as_str == value).any():
                matched_values.append(value)
            elif strict:
                raise AssertionError(f"{col}={value} matched no rows")
            else:
                logging.warning("baseline spec %s=%s matched no rows, skipping", col, value)
        if matched_values:
            parsed.append((col, matched_values))
    return parsed


def build_mask(df: pd.DataFrame, specs: list[tuple[str, list[str]]],
               mode: str, default: bool) -> pd.Series:
    """Boolean mask from parsed specs: values within a spec union, specs reduced by mode.

    Comparison is on str(value), so 0.5 in the CSV matches the CLI's "0.5". `default` is
    the mask returned when there are no specs -- True for a filter (keep everything),
    False for baselines (nothing is a baseline unless declared).
    """
    assert mode in ("all", "any"), f"unknown mode {mode!r}"
    if not specs:
        return pd.Series(default, index=df.index)
    masks = pd.concat([df[col].astype(str).isin(values) for col, values in specs], axis=1)
    return masks.all(axis=1) if mode == "all" else masks.any(axis=1)


def split_baselines(df: pd.DataFrame,
                    specs: list[tuple[str, list[str]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition into (baselines, rest). Baseline specs union: any match is a baseline."""
    is_baseline = build_mask(df, specs, mode="any", default=False)
    baselines, rest = df[is_baseline], df[~is_baseline]
    logging.info("baselines: %d rows from %s | rest: %d rows",
                 len(baselines), specs, len(rest))
    return baselines, rest


def label_series(df: pd.DataFrame, hue_cols: list[str]) -> pd.Series:
    """Build one paper-facing method label per row, skipping the row's NaN columns."""
    return df.apply(
        lambda row: ", ".join(
            display_assignment(c, row[c]) for c in hue_cols if pd.notna(row[c])
        ),
        axis=1,
    )


def order_values(values, order: list[str] | None) -> list:
    """Panel order for one facet axis: listed values first (those present, in the given
    order, compared as strings), the rest appended alphabetically."""
    if order is None:
        return sorted(values)
    by_str = {str(v): v for v in values}
    listed = [by_str[o] for o in order if o in by_str]
    rest = sorted(v for v in values if str(v) not in set(order))
    return listed + rest


def build_paper_palette(methods: list[str]) -> dict:
    """Fixed colour per paper name from PAPER_COLORS; no two series in a figure may share one."""
    missing = [m for m in methods if m not in PAPER_COLORS]
    assert not missing, f"no PAPER_COLORS entry for {missing}"
    palette = {m: PAPER_COLORS[m] for m in methods}
    assert len(set(palette.values())) == len(palette), f"colour collision in {palette}"
    return palette


def pad_limits(lo: float, hi: float, frac: float = 0.05) -> tuple[float, float]:
    """Axis limits with a margin, so markers on the extremes are not clipped."""
    span = hi - lo
    return lo - frac * span, hi + frac * span


def draw_series(ax, sub: pd.DataFrame, x_col: str, y_col: str, palette: dict,
                hue_order: list[str], linestyle: str, band_prefix: str | None) -> None:
    """Line through each method's threshold sweep, over its bootstrap bands.

    With `band_prefix` (e.g. "edd_sim"), `{prefix}_{lo,hi}` is shaded per method; NaN
    band entries (censored thresholds) leave gaps.
    The per-threshold markers are commented out, not deleted: with many series they
    crowd the panel and make solid-vs-dashed hard to read. Uncomment to get them back.
    """
    if len(sub) == 0:
        return
    if band_prefix is not None:
        for method, rows in sub.sort_values(x_col).groupby("method"):
            ax.fill_between(rows[x_col], rows[f"{band_prefix}_lo"],
                            rows[f"{band_prefix}_hi"], color=palette[method],
                            alpha=BAND_ALPHA, linewidth=0)
    shared = dict(data=sub, x=x_col, y=y_col, hue="method", hue_order=hue_order,
                  palette=palette, legend=False, ax=ax)
    sns.lineplot(**{**shared, "data": sub.sort_values(x_col), "linestyle": linestyle,
                    "errorbar": None})
    # sns.scatterplot(**shared, s=80)


def draw_facet_grid(series_df: pd.DataFrame, baseline_df: pd.DataFrame,
                    x_col: str, y_col: str, x_label: str, y_label: str,
                    title: str, subtitle: str, out_path: Path,
                    facet_row: str | None, facet_col: str | None,
                    palette: dict, hue_order: list[str], ref_offset: int | None,
                    limits_df: pd.DataFrame, x_limits: tuple[float, float] | None,
                    facet_row_order: list[str] | None,
                    facet_col_order: list[str] | None,
                    band_prefix: str | None, band_label: str, style: str) -> None:
    """Grid of x_col vs y_col panels, one per (facet_row, facet_col) value pair.

    baseline_df is drawn dashed into every panel; series_df is cut by the facets and
    drawn solid. With neither facet given this is a single panel. Axis limits come from
    limits_df rather than from the panel, so every panel shares one scale. x_limits, when
    given, overrides the data-derived x range. ref_offset draws the guide y = x - offset.
    """
    col_values = (order_values(series_df[facet_col].unique(), facet_col_order)
                  if facet_col else [None])
    row_values = (order_values(series_df[facet_row].unique(), facet_row_order)
                  if facet_row else [None])
    panel_width, panel_height = (PAPER_PANEL_SIZE if style == "paper" else (6, 5))
    fig, axes = plt.subplots(
        len(row_values), len(col_values),
        figsize=(panel_width * len(col_values), panel_height * len(row_values)),
        sharex=True, sharey=True, squeeze=False,
        # Constrained layout is what puts the paper legend below the panels
        # ("outside lower center"); tight_layout cannot place an outside legend.
        **({"layout": "constrained"} if style == "paper" else {}),
    )

    x_min, x_max = limits_df[x_col].min(), limits_df[x_col].max()
    if x_limits is not None:
        x_min, x_max = x_limits
    y_min, y_max = limits_df[y_col].min(), limits_df[y_col].max()
    if ref_offset is not None:
        # The guide line spans the full x range; keep it inside the axes.
        y_min = min(y_min, x_min - ref_offset)
        y_max = max(y_max, x_max - ref_offset)

    for i, row_value in enumerate(row_values):
        for j, col_value in enumerate(col_values):
            ax = axes[i][j]
            sub = series_df
            if facet_row:
                sub = sub[sub[facet_row] == row_value]
            if facet_col:
                sub = sub[sub[facet_col] == col_value]
            # References first, so the arms being compared are drawn over them.
            draw_series(ax, baseline_df, x_col, y_col, palette, hue_order, BASELINE_STYLE,
                        band_prefix)
            draw_series(ax, sub, x_col, y_col, palette, hue_order, SERIES_STYLE,
                        band_prefix)
            if ref_offset is not None:
                ax.plot([x_min, x_max], [x_min - ref_offset, x_max - ref_offset],
                        color=GUIDE_COLOR if style == "paper" else "k",
                        linestyle=GUIDE_STYLE, lw=1)

            panel = ", ".join(
                display_assignment(c, v)
                for c, v in [(facet_row, row_value), (facet_col, col_value)]
                if c
            )
            ax.set_title(panel)
            ax.set_xlabel(x_label if i == len(row_values) - 1 else "")
            ax.set_ylabel(y_label if j == 0 else "")
            ax.set_xlim(*pad_limits(x_min, x_max))
            ax.set_ylim(*pad_limits(y_min, y_max))

    # A method with no finite (x, y) row draws nothing in this figure -- an arm with no
    # empirical ARL on the calibration figure, say. Keep it out of the legend.
    def drawn_methods(df):
        drawn = set(df[df[[x_col, y_col]].notna().all(axis=1)]["method"])
        # Paper legends follow the JSON order that `hue_order` carries; diagnostic ones
        # have no meaningful order of their own, so they stay alphabetical.
        return ([m for m in hue_order if m in drawn] if style == "paper"
                else sorted(drawn))

    build_legends(fig, palette, drawn_methods(series_df), drawn_methods(baseline_df),
                  ref_offset, band_label, style)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if style == "paper":
        fig.savefig(out_path, dpi=300)
    else:
        fig.suptitle(f"{title}\n{subtitle}" if subtitle else title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_legends(fig, palette: dict, series_labels: list[str],
                  baseline_labels: list[str], ref_offset: int | None,
                  band_label: str, style: str) -> None:
    """One frameless legend below the panels (paper), or two stacked ones (diagnostic).

    Diagnostic draws the compared series solid and the baselines dashed under
    "Comparators"; paper puts every arm in one legend, dashed iff it is a CTM.
    Handles are built by hand rather than scraped: the panels share a palette and hue
    order, but any one panel may be missing methods.
    """
    if style == "paper":
        labels = list(series_labels) + list(baseline_labels)
        handles = [plt.Line2D([], [], color=palette[m],
                              linestyle=BASELINE_STYLE if m in baseline_labels
                              else SERIES_STYLE)
                   for m in labels]
        if ref_offset is not None:
            handles.append(plt.Line2D([], [], color=GUIDE_COLOR, linestyle=GUIDE_STYLE,
                                      lw=1))
            labels.append(f"$y = x - {ref_offset}$" if ref_offset else "$y = x$")
        handles.append(Patch(color="grey", alpha=BAND_ALPHA))
        # The full label ("... (seed bootstrap)") overruns a column-width legend.
        labels.append(band_label.split(" (")[0])
        fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False,
                   prop={"size": 7}, columnspacing=1.2, handlelength=2.2)
        return

    # Labels are "col=val, col=val" chains, so a legend easily grows wider than the grid
    # it sits under. ~12 characters per inch at the default font size.
    fig_chars = 12 * fig.get_figwidth()

    def fit_ncol(labels: list[str]) -> int:
        return max(1, min(len(labels), int(fig_chars // (max(map(len, labels)) + 4))))

    def build_handles(labels: list[str], linestyle: str) -> list:
        # marker="o" dropped with the panel markers: on a short legend segment the dot
        # is most of the handle, which is where solid-vs-dashed is hardest to tell apart.
        # return [plt.Line2D([], [], color=palette[m], marker="o", linestyle=linestyle)
        #         for m in labels]
        return [plt.Line2D([], [], color=palette[m], linestyle=linestyle)
                for m in labels]

    handles, labels = build_handles(series_labels, SERIES_STYLE), list(series_labels)
    if ref_offset is not None:
        handles.append(plt.Line2D([], [], color="k", linestyle=GUIDE_STYLE, lw=1))
        labels.append(f"y = x - {ref_offset}" if ref_offset else "y = x")
    handles.append(Patch(color="grey", alpha=BAND_ALPHA))
    labels.append(band_label)
    main = fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.0), loc="upper center",
                      ncol=fit_ncol(labels))
    if baseline_labels:
        # The reference legend goes under the main one, whose height depends on how many
        # series survived the filter -- so measure it rather than guessing an offset.
        fig.canvas.draw()
        below = main.get_window_extent().transformed(fig.transFigure.inverted()).y0
        fig.legend(build_handles(baseline_labels, BASELINE_STYLE), baseline_labels,
                   bbox_to_anchor=(0.5, below - 0.02), loc="upper center",
                   ncol=fit_ncol(baseline_labels), title="Comparators")


def main() -> None:
    """Load, split baselines, label per --style, then write the two figures."""
    args = parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.log, level=logging.INFO, filemode="w",
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info("args: %s", vars(args))

    df = load_summaries(args.csv, args.ctm_csv)
    baselines, series = split_baselines(df, parse_specs(df, args.baselines, strict=False))

    facet_cols = [c for c in [args.facet_row, args.facet_col] if c]
    for col in facet_cols:
        assert col in df.columns, f"facet column {col!r} not in {sorted(df.columns)}"
        # NaN carries no meaning in this script: a plotted row must land in a panel.
        orphans = series[series[col].isna()]
        assert len(orphans) == 0, (
            f"{len(orphans)} filtered rows have no {col!r} value and so no panel to land "
            f"in -- declare them with --baseline or exclude them with --filter")

    # One frame for labelling and colouring, so a method keeps its identity whether it is
    # a compared series or a reference.
    plotted = pd.concat([series.assign(is_baseline=False),
                         baselines.assign(is_baseline=True)], ignore_index=True)
    if args.style == "paper":
        plotted["method"] = name_series_short(plotted, args.hue_cols)
        methods = order_series_names(plotted["method"])
        # Dashes mark the CTM family rather than baseline status: the paper figure has one
        # legend, so --baseline no longer decides which arms are drawn dashed.
        plotted["is_baseline"] = plotted["family"] == "CTM"
        palette = build_paper_palette(methods)
        plt.rcParams.update(PAPER_RC)
    else:
        plotted["method"] = label_series(plotted, args.hue_cols)
        methods = sorted(plotted["method"].unique())
        # tab10/tab20 recycle past 10/20 series, silently repeating colours.
        cmap = "tab10" if len(methods) <= 10 else "tab20" if len(methods) <= 20 else "husl"
        palette = dict(zip(methods, sns.color_palette(cmap, n_colors=len(methods))))
    series, baselines = plotted[~plotted["is_baseline"]], plotted[plotted["is_baseline"]]

    # Columns held fixed across the whole figure belong in the subtitle, not repeated in
    # every legend entry. The paper figure carries no title at all.
    if args.style == "paper":
        subtitle = ""
        if args.title_prefix:
            logging.info("--title-prefix %r ignored under --style paper",
                         args.title_prefix)
    else:
        fixed = [c for c in LABEL_COLS
                 if c in plotted.columns and c not in facet_cols
                 and c not in args.hue_cols and plotted[c].nunique(dropna=True) == 1]
        subtitle = ", ".join(
            display_assignment(c, plotted[c].dropna().iloc[0]) for c in fixed
        )
        subtitle = f"{subtitle} | burn-in={args.n_burn_in} (excluded from ARL/EDD)"

    logging.info("%d series (%d compared, %d reference) across %d rows", len(methods),
                 series["method"].nunique(), baselines["method"].nunique(), len(plotted))

    # Both ARL-vs-EDD figures share the specified sweep range on the x-axis, so they are
    # directly comparable. The calibration figure sweeps the same nominal ARL but takes
    # its range from its own pair of flags; either pair may be left data-derived.
    assert (args.threshold_min is None) == (args.threshold_max is None), \
        "--threshold-min and --threshold-max must be given together"
    arl_x_limits = (None if args.threshold_max is None
                    else (args.threshold_min, args.threshold_max))
    assert (args.threshold_min_null is None) == (args.threshold_max_null is None), \
        "--threshold-min-null and --threshold-max-null must be given together"
    calibration_x_limits = (None if args.threshold_max_null is None
                            else (args.threshold_min_null, args.threshold_max_null))

    # Run lengths are in post-burn-in process time, so the change lands at process step
    # n_pre - n_burn_in and that is where the zero-power line must sit. One offset now
    # serves both families, which only became true once the CTM took the same burn-in.
    ref_offset = args.n_pre - args.n_burn_in
    paper = args.style == "paper"
    arl_label = "Nominal ARL$_0$ ($1/\\alpha$)" if paper else "Nominal ARL (1/\u03b1)"
    edd_label = "EDD" if paper else "EDD (truncated at horizon)"
    empirical_label = "Empirical ARL$_0$" if paper else "Empirical ARL"
    # (x, y, x_label, y_label, title, output, guide-line offset, x limits, band prefix)
    figures = [
        ("nominal_arl", "edd", arl_label, edd_label,
         "Nominal ARL vs EDD", args.out_nominal, ref_offset, arl_x_limits, "edd"),
        ("nominal_arl", "empirical_arl", arl_label, empirical_label,
         "Calibration: Empirical vs Nominal ARL", args.out_calibration, 0,
         calibration_x_limits, "empirical_arl"),
    ]

    band_infix, band_label = BAND_KINDS[args.band]
    for (x_col, y_col, x_label, y_label, title, out_path, ref_offset, x_limits,
         band_prefix) in figures:
        full_title = f"{args.title_prefix}: {title}" if args.title_prefix else title
        try:
            draw_facet_grid(series, baselines, x_col, y_col, x_label, y_label,
                            full_title, "", out_path, args.facet_row, args.facet_col,
                            palette, methods, ref_offset, limits_df=plotted,
                            x_limits=x_limits, facet_row_order=args.facet_row_order,
                            facet_col_order=args.facet_col_order,
                            band_prefix=f"{band_prefix}_{band_infix}",
                            band_label=band_label, style=args.style)
            logging.info("wrote %s", out_path)
            print(f"wrote {out_path}")
        except Exception as e:
            logging.warning(e)
            print(f"WARNING '{title}' not plotting\n", e)


if __name__ == "__main__":
    main()
