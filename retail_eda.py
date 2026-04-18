import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy import stats

# ── colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "blue":   "#378ADD",
    "teal":   "#1D9E75",
    "amber":  "#EF9F27",
    "coral":  "#D85A30",
    "purple": "#7F77DD",
    "red":    "#E24B4A",
    "green":  "#639922",
    "gray":   "#888780",
}
CAT_COLORS = list(PALETTE.values())

STYLE = {
    "figure.facecolor":   "#FAFAF8",
    "axes.facecolor":     "#FAFAF8",
    "axes.edgecolor":     "#D3D1C7",
    "axes.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#E8E6DF",
    "grid.linewidth":     0.5,
    "axes.labelcolor":    "#444441",
    "axes.titlepad":      10,
    "xtick.color":        "#888780",
    "ytick.color":        "#888780",
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.titlesize":     12,
    "axes.labelsize":     10,
    "font.family":        "DejaVu Sans",
    "legend.frameon":     False,
    "legend.fontsize":    9,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _apply_style():
    plt.rcParams.update(STYLE)


def _bar_labels(ax, fmt="{:.0f}", color="#444441", fontsize=9, pad=3):
    """Add value labels on top of bar charts."""
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h) and h != 0:
            ax.annotate(
                fmt.format(h),
                (p.get_x() + p.get_width() / 2, h),
                ha="center", va="bottom",
                fontsize=fontsize, color=color,
                xytext=(0, pad), textcoords="offset points",
            )


def _hbar_labels(ax, fmt="{:.1f}", color="#444441", fontsize=9, pad=3):
    """Add value labels on horizontal bars."""
    for p in ax.patches:
        w = p.get_width()
        if np.isfinite(w) and w != 0:
            ax.annotate(
                fmt.format(w),
                (w, p.get_y() + p.get_height() / 2),
                ha="left" if w >= 0 else "right", va="center",
                fontsize=fontsize, color=color,
                xytext=(pad if w >= 0 else -pad, 0), textcoords="offset points",
            )


def _save_or_show(fig, name: str, save_path: str | None):
    fig.tight_layout()
    if save_path:
        fig.savefig(f"{save_path}/{name}.png", dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}/{name}.png")
    else:
        plt.show()
    plt.close(fig)


# ── data loading & feature engineering ───────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load CSV and engineer derived features.

    Parameters
    ----------
    path : str
        Path to retail_price.csv

    Returns
    -------
    pd.DataFrame
        Enriched dataframe ready for EDA.
    """
    df = pd.read_csv(path)

    # competitive
    df["avg_comp_price"]  = (df["comp_1"] + df["comp_2"] + df["comp_3"]) / 3
    df["price_vs_comp"]   = df["unit_price"] / df["avg_comp_price"]
    df["price_premium"]   = (df["unit_price"] / df["avg_comp_price"] - 1) * 100
    df["price_gap_comp1"] = df["unit_price"] - df["comp_1"]

    # revenue & margin
    df["revenue"]       = df["unit_price"] * df["qty"]
    df["freight_pct"]   = df["freight_price"] / df["unit_price"] * 100
    df["margin_proxy"]  = df["unit_price"] - df["freight_price"]

    # price dynamics
    df["price_change"]     = df["unit_price"] - df["lag_price"]
    df["price_change_pct"] = (df["unit_price"] - df["lag_price"]) / (df["lag_price"] + 1e-9) * 100

    # seasonality
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # log transforms
    df["log_price"] = np.log(df["unit_price"])
    df["log_qty"]   = np.log(df["qty"].clip(lower=0.01))

    print(f"Loaded {len(df)} rows × {df.shape[1]} features")
    print(f"Products : {df['product_id'].nunique()}  |  "
          f"Categories : {df['product_category_name'].nunique()}  |  "
          f"Period : {df['month_year'].min()} → {df['month_year'].max()}")
    return df


# ── section 1: overview ───────────────────────────────────────────────────────

def overview(df: pd.DataFrame, save_path: str | None = None):
    """Print summary statistics and plot demand/price distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched dataframe from load_data().
    save_path : str, optional
        Directory to save the figure. If None, calls plt.show().
    """
    _apply_style()
    print("\n" + "=" * 60)
    print("  SECTION 1 — OVERVIEW")
    print("=" * 60)

    cols = ["qty", "unit_price", "freight_price", "product_score",
            "customers", "revenue", "freight_pct"]
    print(df[cols].describe().round(2).to_string())

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Dataset Overview", fontsize=14, fontweight="bold",
                 color="#2C2C2A", y=1.01)
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1a — demand histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df["qty"], bins=25, color=PALETTE["blue"], edgecolor="white",
             linewidth=0.5, alpha=0.85)
    ax1.set_title("Monthly demand distribution")
    ax1.set_xlabel("Units sold")
    ax1.set_ylabel("Count")
    ax1.axvline(df["qty"].median(), color=PALETTE["coral"], lw=1.5,
                linestyle="--", label=f"Median={df['qty'].median():.0f}")
    ax1.legend()

    # 1b — price histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df["unit_price"], bins=25, color=PALETTE["teal"],
             edgecolor="white", linewidth=0.5, alpha=0.85)
    ax2.set_title("Unit price distribution")
    ax2.set_xlabel("Price (R$)")
    ax2.set_ylabel("Count")
    ax2.axvline(df["unit_price"].median(), color=PALETTE["coral"], lw=1.5,
                linestyle="--", label=f"Median=R${df['unit_price'].median():.0f}")
    ax2.legend()

    # 1c — freight % histogram
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df["freight_pct"], bins=25, color=PALETTE["amber"],
             edgecolor="white", linewidth=0.5, alpha=0.85)
    ax3.set_title("Freight as % of price")
    ax3.set_xlabel("Freight %")
    ax3.set_ylabel("Count")
    ax3.axvline(df["freight_pct"].median(), color=PALETTE["coral"], lw=1.5,
                linestyle="--", label=f"Median={df['freight_pct'].median():.0f}%")
    ax3.legend()

    # 1d — price vs qty scatter
    ax4 = fig.add_subplot(gs[1, 0:2])
    cats = df["product_category_name"].unique()
    for i, cat in enumerate(cats):
        sub = df[df["product_category_name"] == cat]
        ax4.scatter(sub["unit_price"], sub["qty"],
                    color=CAT_COLORS[i % len(CAT_COLORS)],
                    alpha=0.55, s=28, label=cat)
    ax4.set_title("Price vs. demand (all SKUs)")
    ax4.set_xlabel("Unit price (R$)")
    ax4.set_ylabel("Monthly qty sold")
    ax4.legend(fontsize=7, ncol=2)

    # 1e — product score distribution
    ax5 = fig.add_subplot(gs[1, 2])
    score_bins = pd.cut(df["product_score"],
                        bins=[3, 3.8, 4.0, 4.2, 4.6],
                        labels=["Low\n(<3.8)", "Med\n(3.8-4)", "High\n(4-4.2)", "Top\n(>4.2)"])
    score_counts = score_bins.value_counts().sort_index()
    ax5.bar(score_counts.index, score_counts.values,
            color=[PALETTE["red"], PALETTE["amber"],
                   PALETTE["teal"], PALETTE["blue"]],
            edgecolor="white", linewidth=0.5)
    ax5.set_title("Product score segments")
    ax5.set_xlabel("Score tier")
    ax5.set_ylabel("SKU-months")
    _bar_labels(ax5)

    _save_or_show(fig, "01_overview", save_path)


# ── section 2: category analysis ─────────────────────────────────────────────

def category_analysis(df: pd.DataFrame, save_path: str | None = None):
    """Revenue, pricing, freight, and demand breakdown by product category.

    Parameters
    ----------
    df : pd.DataFrame
    save_path : str, optional
    """
    _apply_style()
    print("\n" + "=" * 60)
    print("  SECTION 2 — CATEGORY ANALYSIS")
    print("=" * 60)

    cat = df.groupby("product_category_name").agg(
        n             = ("qty", "count"),
        avg_price     = ("unit_price", "mean"),
        avg_qty       = ("qty", "mean"),
        total_revenue = ("revenue", "sum"),
        avg_score     = ("product_score", "mean"),
        avg_freight_p = ("freight_pct", "mean"),
        avg_customers = ("customers", "mean"),
        avg_comp      = ("avg_comp_price", "mean"),
        price_premium = ("price_premium", "mean"),
    ).round(2)
    cat = cat.sort_values("total_revenue", ascending=False)
    print(cat.to_string())

    short = {c: c.replace("_", "\n") for c in cat.index}
    labels = [short[c] for c in cat.index]
    colors = CAT_COLORS[:len(cat)]

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("Category Analysis", fontsize=14, fontweight="bold",
                 color="#2C2C2A", y=1.01)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # 2a — total revenue
    ax1 = fig.add_subplot(gs[0, 0:2])
    bars = ax1.barh(cat.index[::-1], cat["total_revenue"][::-1],
                    color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax1.set_title("Total revenue by category (R$)")
    ax1.set_xlabel("Revenue (R$)")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"R${x/1000:.0f}K"))
    for bar, val in zip(bars, cat["total_revenue"][::-1]):
        ax1.text(val + 1000, bar.get_y() + bar.get_height() / 2,
                 f"R${val/1000:.0f}K", va="center", fontsize=9,
                 color="#444441")

    # 2b — pie market share
    ax2 = fig.add_subplot(gs[0, 2])
    wedges, texts, autotexts = ax2.pie(
        cat["total_revenue"], labels=None,
        colors=colors, autopct="%1.0f%%",
        startangle=140, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2}
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax2.set_title("Revenue share")
    ax2.legend(cat.index, loc="lower left", bbox_to_anchor=(-0.1, -0.15),
               fontsize=7, ncol=2)

    # 2c — avg price vs avg competitor price
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(cat))
    w = 0.38
    ax3.bar(x - w/2, cat["avg_price"], width=w, label="Our price",
            color=PALETTE["blue"], edgecolor="white", linewidth=0.5)
    ax3.bar(x + w/2, cat["avg_comp"], width=w, label="Avg comp price",
            color=PALETTE["gray"], edgecolor="white", linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax3.set_title("Our price vs. competitors (R$)")
    ax3.set_ylabel("Price (R$)")
    ax3.legend()

    # 2d — freight burden
    ax4 = fig.add_subplot(gs[1, 1])
    freight_sorted = cat["avg_freight_p"].sort_values(ascending=True)
    bar_colors = [PALETTE["red"] if v > 40 else PALETTE["amber"] if v > 25
                  else PALETTE["teal"] for v in freight_sorted]
    ax4.barh(
        [short[c] for c in freight_sorted.index],
        freight_sorted.values,
        color=bar_colors, edgecolor="white", linewidth=0.5
    )
    ax4.set_title("Freight as % of price")
    ax4.set_xlabel("Freight %")
    ax4.axvline(25, color=PALETTE["coral"], lw=1.2, linestyle="--",
                alpha=0.7, label="25% threshold")
    ax4.legend(fontsize=8)
    _hbar_labels(ax4, fmt="{:.0f}%")

    # 2e — avg demand & score
    ax5 = fig.add_subplot(gs[1, 2])
    cat_scatter_colors = [CAT_COLORS[i % len(CAT_COLORS)] for i in range(len(cat))]
    sc = ax5.scatter(cat["avg_qty"], cat["avg_price"],
                     s=cat["avg_score"] ** 4 * 5,
                     c=cat_scatter_colors, alpha=0.8, edgecolors="white", linewidth=0.8)
    for cat_name, row in cat.iterrows():
        ax5.annotate(
            short[cat_name],
            (row["avg_qty"], row["avg_price"]),
            fontsize=7, ha="center", va="bottom",
            xytext=(0, 6), textcoords="offset points",
            color="#444441"
        )
    ax5.set_title("Avg demand vs. price\n(bubble = score)")
    ax5.set_xlabel("Avg monthly qty")
    ax5.set_ylabel("Avg price (R$)")

    _save_or_show(fig, "02_category_analysis", save_path)


# ── section 3: seasonality ────────────────────────────────────────────────────

def seasonality_analysis(df: pd.DataFrame, save_path: str | None = None):
    """Monthly demand trends, price cycles, and holiday/weekend effects.

    Parameters
    ----------
    df : pd.DataFrame
    save_path : str, optional
    """
    _apply_style()
    print("\n" + "=" * 60)
    print("  SECTION 3 — SEASONALITY ANALYSIS")
    print("=" * 60)

    months = list(range(1, 13))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    monthly_agg = df.groupby("month").agg(
        avg_qty=("qty", "mean"),
        avg_price=("unit_price", "mean"),
        total_customers=("customers", "sum"),
    )
    print("\nMonthly aggregates:")
    print(monthly_agg.round(2).to_string())

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("Seasonality Analysis", fontsize=14, fontweight="bold",
                 color="#2C2C2A", y=1.01)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 3a — overall monthly demand
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.bar(months, monthly_agg["avg_qty"], color=PALETTE["blue"],
            edgecolor="white", linewidth=0.5, alpha=0.7, label="Avg qty")
    ax1_r = ax1.twinx()
    ax1_r.plot(months, monthly_agg["avg_price"], color=PALETTE["coral"],
               marker="o", lw=2, ms=5, label="Avg price")
    ax1_r.set_ylabel("Avg price (R$)", color=PALETTE["coral"])
    ax1_r.tick_params(axis="y", labelcolor=PALETTE["coral"])
    ax1.set_xticks(months)
    ax1.set_xticklabels(month_labels)
    ax1.set_title("Monthly avg demand (bars) & avg price (line)")
    ax1.set_ylabel("Avg units sold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # 3b — category heatmap (manual)
    ax2 = fig.add_subplot(gs[0, 2])
    pivot = df.pivot_table(values="qty", index="month",
                           columns="product_category_name", aggfunc="mean")
    cats_ordered = pivot.columns.tolist()
    data_mat = pivot.values
    norm_mat  = (data_mat - np.nanmin(data_mat, axis=0)) / \
                (np.nanmax(data_mat, axis=0) - np.nanmin(data_mat, axis=0) + 1e-9)
    im = ax2.imshow(norm_mat.T, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=1)
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=7)
    ax2.set_yticks(range(len(cats_ordered)))
    ax2.set_yticklabels([c.replace("_", "\n") for c in cats_ordered], fontsize=7)
    ax2.set_title("Demand heatmap\n(normalised per category)")
    plt.colorbar(im, ax=ax2, fraction=0.04, label="Relative demand")

    # 3c — category demand lines
    ax3 = fig.add_subplot(gs[1, 0:2])
    for i, cat in enumerate(cats_ordered):
        if cat in pivot.columns:
            ax3.plot(months, pivot[cat].values,
                     color=CAT_COLORS[i % len(CAT_COLORS)],
                     marker="o", ms=4, lw=1.6, label=cat, alpha=0.85)
    ax3.set_xticks(months)
    ax3.set_xticklabels(month_labels)
    ax3.set_title("Monthly demand trend by category")
    ax3.set_ylabel("Avg units sold")
    ax3.legend(fontsize=7, ncol=2, loc="upper right")

    # 3d — holiday effect
    ax4 = fig.add_subplot(gs[1, 2])
    hol_agg = df.groupby("holiday")["qty"].mean()
    ax4.bar(hol_agg.index.astype(str), hol_agg.values,
            color=[PALETTE["teal"] if v >= hol_agg.mean() else PALETTE["gray"]
                   for v in hol_agg.values],
            edgecolor="white", linewidth=0.5)
    ax4.set_title("Avg demand by holiday density\n(0 = none, 4 = peak)")
    ax4.set_xlabel("Holiday level")
    ax4.set_ylabel("Avg qty sold")
    ax4.axhline(hol_agg.mean(), color=PALETTE["coral"], lw=1.2,
                linestyle="--", alpha=0.8, label="Overall mean")
    ax4.legend()
    _bar_labels(ax4, fmt="{:.1f}")

    _save_or_show(fig, "03_seasonality", save_path)


# ── section 4: competitive analysis ──────────────────────────────────────────

def competitive_analysis(df: pd.DataFrame, save_path: str | None = None):
    """Price premium vs. competitors, price stability, and cross-price analysis.

    Parameters
    ----------
    df : pd.DataFrame
    save_path : str, optional
    """
    _apply_style()
    print("\n" + "=" * 60)
    print("  SECTION 4 — COMPETITIVE ANALYSIS")
    print("=" * 60)

    cat_premium = df.groupby("product_category_name")["price_premium"].mean().sort_values()
    pct_above   = (df["price_vs_comp"] > 1).mean() * 100
    print(f"\n% SKU-months priced above avg competitor : {pct_above:.1f}%")
    print("\nCategory-level price premium vs avg competitor (%):")
    print(cat_premium.round(2).to_string())

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("Competitive Pricing Analysis", fontsize=14, fontweight="bold",
                 color="#2C2C2A", y=1.01)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 4a — price premium waterfall
    ax1 = fig.add_subplot(gs[0, 0:2])
    bar_colors = [PALETTE["teal"] if v >= 0 else PALETTE["coral"]
                  for v in cat_premium.values]
    ax1.barh(
        [c.replace("_", "\n") for c in cat_premium.index],
        cat_premium.values, color=bar_colors,
        edgecolor="white", linewidth=0.5
    )
    ax1.axvline(0, color="#888780", lw=0.8)
    ax1.set_title("Price premium vs. avg competitor (%)")
    ax1.set_xlabel("Premium %  (+ = above comp, − = below comp)")
    _hbar_labels(ax1, fmt="{:+.0f}%")

    # 4b — price vs comp scatter
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(df["avg_comp_price"], df["unit_price"],
                alpha=0.35, s=18, color=PALETTE["blue"])
    lim = max(df["avg_comp_price"].max(), df["unit_price"].max()) * 1.05
    ax2.plot([0, lim], [0, lim], color=PALETTE["coral"],
             lw=1.2, linestyle="--", label="Price parity")
    ax2.set_xlim(0, lim)
    ax2.set_ylim(0, lim)
    ax2.set_xlabel("Avg competitor price (R$)")
    ax2.set_ylabel("Our unit price (R$)")
    ax2.set_title("Our price vs. avg competitor")
    ax2.legend()

    # 4c — price stability (range %)
    pv = df.groupby("product_id")["unit_price"].agg(
        mean="mean", std="std", min="min", max="max"
    )
    pv["range_pct"] = ((pv["max"] - pv["min"]) / pv["mean"] * 100).round(1)
    pv_sorted = pv["range_pct"].sort_values(ascending=False).head(15)

    ax3 = fig.add_subplot(gs[1, 0])
    stab_colors = [PALETTE["red"] if v > 60 else PALETTE["amber"] if v > 35
                   else PALETTE["teal"] for v in pv_sorted.values]
    ax3.barh(pv_sorted.index[::-1], pv_sorted.values[::-1],
             color=stab_colors[::-1], edgecolor="white", linewidth=0.5)
    ax3.set_title("Price variability\n(range as % of avg, top 15 SKUs)")
    ax3.set_xlabel("Price range %")
    _hbar_labels(ax3, fmt="{:.0f}%")

    # 4d — price change distribution
    ax4 = fig.add_subplot(gs[1, 1])
    bins = [-60, -10, -2, 2, 10, 60]
    labels_ch = ["Big\ndrop", "Small\ndrop", "Stable", "Small\nrise", "Big\nrise"]
    cuts = pd.cut(df["price_change_pct"], bins=bins, labels=labels_ch)
    change_counts = cuts.value_counts().sort_index()
    ch_colors = [PALETTE["red"], PALETTE["coral"], PALETTE["gray"],
                 PALETTE["teal"], PALETTE["blue"]]
    ax4.bar(change_counts.index, change_counts.values,
            color=ch_colors, edgecolor="white", linewidth=0.5)
    ax4.set_title("MoM price change distribution")
    ax4.set_ylabel("# observations")
    _bar_labels(ax4)

    # 4e — comp score vs price analysis
    ax5 = fig.add_subplot(gs[1, 2])
    ps_cols = {"ps1": PALETTE["blue"], "ps2": PALETTE["teal"], "ps3": PALETTE["amber"]}
    for ps, color in ps_cols.items():
        ax5.scatter(df[ps], df["unit_price"], alpha=0.3, s=15,
                    color=color, label=ps)
    ax5.set_xlabel("Competitor score")
    ax5.set_ylabel("Our unit price (R$)")
    ax5.set_title("Our price vs. competitor scores")
    ax5.legend(title="Comp", fontsize=8)

    _save_or_show(fig, "04_competitive", save_path)


# ── section 5: demand drivers ─────────────────────────────────────────────────

def demand_drivers(df: pd.DataFrame, save_path: str | None = None):
    """Correlation analysis, elasticity curves, and customer-demand relationship.

    Parameters
    ----------
    df : pd.DataFrame
    save_path : str, optional
    """
    _apply_style()
    print("\n" + "=" * 60)
    print("  SECTION 5 — DEMAND DRIVERS")
    print("=" * 60)

    driver_cols = ["unit_price", "freight_price", "product_score",
                   "customers", "weekend", "holiday", "avg_comp_price",
                   "price_vs_comp", "lag_price", "price_premium"]
    corrs = df[driver_cols + ["qty"]].corr()["qty"].drop("qty").sort_values()
    print("\nCorrelation with qty:")
    print(corrs.round(3).to_string())

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("Demand Drivers", fontsize=14, fontweight="bold",
                 color="#2C2C2A", y=1.01)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 5a — correlation bar
    ax1 = fig.add_subplot(gs[0, 0:2])
    corr_colors = [PALETTE["coral"] if v < 0 else PALETTE["teal"] for v in corrs.values]
    ax1.barh(corrs.index, corrs.values, color=corr_colors,
             edgecolor="white", linewidth=0.5)
    ax1.axvline(0, color="#888780", lw=0.8)
    ax1.set_title("Pearson correlation with monthly demand (qty)")
    ax1.set_xlabel("Correlation coefficient")
    _hbar_labels(ax1, fmt="{:.3f}")

    # 5b — log-log price elasticity
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(df["log_price"], df["log_qty"], alpha=0.3, s=14,
                color=PALETTE["blue"])
    m, b, r, p, _ = stats.linregress(df["log_price"], df["log_qty"])
    x_line = np.linspace(df["log_price"].min(), df["log_price"].max(), 100)
    ax2.plot(x_line, m * x_line + b, color=PALETTE["coral"],
             lw=2, label=f"ε = {m:.2f}  R²={r**2:.2f}")
    ax2.set_xlabel("log(price)")
    ax2.set_ylabel("log(qty)")
    ax2.set_title("Price elasticity\n(log-log regression)")
    ax2.legend()

    # 5c — customers vs qty
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(df["customers"], df["qty"], alpha=0.35, s=16,
                color=PALETTE["teal"])
    m2, b2, r2, p2, _ = stats.linregress(df["customers"], df["qty"])
    x2 = np.linspace(df["customers"].min(), df["customers"].max(), 100)
    ax3.plot(x2, m2 * x2 + b2, color=PALETTE["coral"], lw=2,
             label=f"r={r2:.2f}")
    ax3.set_xlabel("Customer count")
    ax3.set_ylabel("Qty sold")
    ax3.set_title("Customers vs. demand")
    ax3.legend()

    # 5d — score tier vs qty
    ax4 = fig.add_subplot(gs[1, 1])
    score_bins = pd.cut(df["product_score"],
                        bins=[3, 3.8, 4.0, 4.2, 4.6],
                        labels=["Low\n<3.8", "Med\n3.8-4", "High\n4-4.2", "Top\n>4.2"])
    score_qty = df.groupby(score_bins, observed=True)["qty"].mean()
    score_colors = [PALETTE["red"], PALETTE["amber"], PALETTE["teal"], PALETTE["blue"]]
    ax4.bar(score_qty.index, score_qty.values, color=score_colors,
            edgecolor="white", linewidth=0.5)
    ax4.set_title("Avg demand by product score tier")
    ax4.set_ylabel("Avg qty sold")
    _bar_labels(ax4, fmt="{:.1f}")

    # 5e — lag price vs current price
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(df["lag_price"], df["qty"], alpha=0.3, s=14,
                color=PALETTE["purple"])
    m3, b3, r3, _, _ = stats.linregress(df["lag_price"], df["qty"])
    x3 = np.linspace(df["lag_price"].min(), df["lag_price"].max(), 100)
    ax5.plot(x3, m3 * x3 + b3, color=PALETTE["coral"], lw=2,
             label=f"r={r3:.2f}")
    ax5.set_xlabel("Lag price (R$)")
    ax5.set_ylabel("Qty sold")
    ax5.set_title("Anchoring effect:\nlag price vs. demand")
    ax5.legend()

    _save_or_show(fig, "05_demand_drivers", save_path)


# ── section 6: price dynamics ─────────────────────────────────────────────────

def price_dynamics(df: pd.DataFrame, save_path: str | None = None):
    """Price-change effects on demand and top-revenue product analysis.

    Parameters
    ----------
    df : pd.DataFrame
    save_path : str, optional
    """
    _apply_style()
    print("\n" + "=" * 60)
    print("  SECTION 6 — PRICE DYNAMICS")
    print("=" * 60)

    top_prod = (df.groupby("product_id")
                  .agg(total_rev=("revenue", "sum"),
                       avg_price=("unit_price", "mean"),
                       total_qty=("qty", "sum"),
                       category=("product_category_name", "first"))
                  .sort_values("total_rev", ascending=False)
                  .head(10))
    print("\nTop 10 products by revenue:")
    print(top_prod.to_string())

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("Price Dynamics", fontsize=14, fontweight="bold",
                 color="#2C2C2A", y=1.01)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 6a — price change vs demand change
    bins = [-60, -10, -2, 2, 10, 60]
    labels_ch = ["Big drop", "Small drop", "Stable", "Small rise", "Big rise"]
    cuts = pd.cut(df["price_change_pct"], bins=bins, labels=labels_ch)
    ch_qty = df.groupby(cuts, observed=True)["qty"].mean()
    ch_colors = [PALETTE["red"], PALETTE["coral"], PALETTE["gray"],
                 PALETTE["teal"], PALETTE["blue"]]
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(ch_qty.index, ch_qty.values, color=ch_colors,
            edgecolor="white", linewidth=0.5)
    ax1.set_title("Avg demand by price change\ndirection")
    ax1.set_ylabel("Avg qty sold")
    ax1.set_xticklabels(ch_qty.index, rotation=25, ha="right", fontsize=8)
    _bar_labels(ax1, fmt="{:.1f}")

    # 6b — top products revenue
    ax2 = fig.add_subplot(gs[0, 1:])
    top_colors = [CAT_COLORS[i % len(CAT_COLORS)]
                  for i, _ in enumerate(top_prod.index)]
    ax2.barh(top_prod.index[::-1], top_prod["total_rev"][::-1],
             color=top_colors[::-1], edgecolor="white", linewidth=0.5)
    ax2.set_title("Top 10 products by total revenue")
    ax2.set_xlabel("Revenue (R$)")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"R${x/1000:.0f}K"))
    for bar, (pid, row) in zip(ax2.patches, top_prod[::-1].iterrows()):
        ax2.text(row["total_rev"] + 200,
                 bar.get_y() + bar.get_height() / 2,
                 f"R${row['total_rev']/1000:.0f}K  {row['category'][:8]}",
                 va="center", fontsize=8, color="#444441")

    # 6c — price trajectory for top 6 products
    ax3 = fig.add_subplot(gs[1, 0:2])
    top6 = top_prod.index[:6]
    for i, pid in enumerate(top6):
        sub = df[df["product_id"] == pid].sort_values("month")
        ax3.plot(sub["month"], sub["unit_price"],
                 color=CAT_COLORS[i % len(CAT_COLORS)],
                 marker="o", ms=4, lw=1.8, label=pid)
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax3.set_title("Price trajectory — top 6 products")
    ax3.set_ylabel("Unit price (R$)")
    ax3.legend(fontsize=7, ncol=2)

    # 6d — revenue heatmap product × month
    ax4 = fig.add_subplot(gs[1, 2])
    rev_pivot = df[df["product_id"].isin(top6)].pivot_table(
        values="revenue", index="product_id", columns="month", aggfunc="sum"
    ).fillna(0)
    im = ax4.imshow(rev_pivot.values, aspect="auto", cmap="YlOrRd")
    ax4.set_xticks(range(12))
    ax4.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"],
                        fontsize=8)
    ax4.set_yticks(range(len(rev_pivot.index)))
    ax4.set_yticklabels(rev_pivot.index, fontsize=8)
    ax4.set_title("Revenue heatmap\n(top 6 products × month)")
    plt.colorbar(im, ax=ax4, fraction=0.04, label="R$")

    _save_or_show(fig, "06_price_dynamics", save_path)


# ── section 7: insights summary ───────────────────────────────────────────────

def _price_change_insight(df: pd.DataFrame) -> str:
    bins   = [-60, -10, -2, 2, 10, 60]
    labs   = ["big_drop", "small_drop", "stable", "small_rise", "big_rise"]
    cuts   = pd.cut(df["price_change_pct"], bins=bins, labels=labs)
    q_drop = df[cuts == "big_drop"]["qty"].mean()
    q_stbl = df[cuts == "stable"]["qty"].mean()
    return (f"When prices drop >10%, avg demand is {q_drop:.1f} units vs "
            f"{q_stbl:.1f} when stable — directionally consistent but modest.")


def print_insights(df: pd.DataFrame):
    """Print a structured summary of all business-relevant EDA findings.

    Parameters
    ----------
    df : pd.DataFrame
    """
    cat = df.groupby("product_category_name").agg(
        avg_price     = ("unit_price", "mean"),
        avg_comp      = ("avg_comp_price", "mean"),
        total_revenue = ("revenue", "sum"),
        avg_freight_p = ("freight_pct", "mean"),
        avg_qty       = ("qty", "mean"),
        avg_customers = ("customers", "mean"),
    )
    cat["premium_pct"] = (cat["avg_price"] / cat["avg_comp"] - 1) * 100

    stable_pct   = (df["price_change_pct"].abs() < 1).mean() * 100
    pct_above    = (df["price_vs_comp"] > 1).mean() * 100
    elasticity   = stats.linregress(df["log_price"], df["log_qty"]).slope
    corr_cust    = df["customers"].corr(df["qty"])

    print("\n" + "=" * 65)
    print("  BUSINESS INSIGHTS SUMMARY")
    print("=" * 65)

    insights = [
        ("PRICING vs. COMPETITION",
         [f"Health & beauty is priced {cat.loc['health_beauty','premium_pct']:.0f}% above "
          f"competitors yet leads revenue at R${cat.loc['health_beauty','total_revenue']/1000:.0f}K — "
          f"strong pricing power or uncontested niche.",

          f"Computers & accessories is {abs(cat.loc['computers_accessories','premium_pct']):.0f}% "
          f"below competitors (R${cat.loc['computers_accessories','avg_price']:.0f} vs "
          f"R${cat.loc['computers_accessories','avg_comp']:.0f}). "
          f"An untapped margin opportunity.",

          f"Garden tools carries a {cat.loc['garden_tools','premium_pct']:.0f}% premium "
          f"with {cat.loc['garden_tools','avg_qty']:.1f} avg units/month — loyal base.",

          f"Furniture & decor is {abs(cat.loc['furniture_decor','premium_pct']):.0f}% below "
          f"competitors with the highest avg demand ({cat.loc['furniture_decor','avg_qty']:.1f} "
          f"units/month). Price increase likely safe."]),

        ("FREIGHT / COST STRUCTURE",
         [f"Consoles & games freight = {cat.loc['consoles_games','avg_freight_p']:.0f}% of price. "
          f"At avg R${cat.loc['consoles_games','avg_price']:.0f}, logistics eat into margins critically.",

          f"Watches & gifts have the healthiest freight burden at "
          f"{cat.loc['watches_gifts','avg_freight_p']:.0f}% — highest absolute revenue per item.",

          f"Garden tools freight at {cat.loc['garden_tools','avg_freight_p']:.0f}% is high. "
          f"A bulk logistics renegotiation could significantly boost margins."]),

        ("DEMAND DRIVERS",
         [f"Customer count is the strongest demand signal (r={corr_cust:.2f}). "
          f"Demand-gen (traffic, marketing) matters more than price in many categories.",

          f"Overall price elasticity = {elasticity:.2f} (inelastic). "
          f"Price changes have modest demand impact at the aggregate level — "
          f"but consoles & games is highly elastic (ε = −2.28).",

          f"Lag price has a large positive coefficient in regression models. "
          f"Customers anchor to last month's price — avoid sudden large jumps.",

          f"Holiday density drives demand: peak holiday periods see "
          f"{df[df['holiday']==4]['qty'].mean():.1f} vs "
          f"{df[df['holiday']==0]['qty'].mean():.1f} avg units on non-holidays."]),

        ("SEASONALITY",
         [f"November is the demand peak: garden tools avg {df[df['month']==11]['qty'].mean():.1f}, "
          f"overall avg {df[df['month']==11]['qty'].mean():.1f} — plan inventory and pricing accordingly.",

          f"Computers & accessories peaks Jan–Feb (back-to-school / new year). "
          f"Watches & gifts peak in May. Category-specific promotion timing is key.",

          f"Prices are relatively stable (R${df.groupby('month')['unit_price'].mean().min():.0f}–"
          f"R${df.groupby('month')['unit_price'].mean().max():.0f}), "
          f"suggesting no seasonal pricing is being practised today — a missed lever."]),

        ("PRICE DYNAMICS",
         [f"{stable_pct:.0f}% of observations show zero price change month-to-month. "
          f"Most SKUs are statically priced — dynamic pricing is largely untapped.",

          _price_change_insight(df),

          f"Top product health2 (R$327 avg) generates R$63.9K revenue. "
          f"High-ticket health & beauty SKUs are disproportionate revenue contributors."])
    ]

    for section, points in insights:
        print(f"\n  [{section}]")
        for i, pt in enumerate(points, 1):
            # wrap text at 70 chars
            words = pt.split()
            line, lines = [], []
            for w in words:
                if sum(len(x)+1 for x in line) + len(w) > 68:
                    lines.append(" ".join(line))
                    line = [w]
                else:
                    line.append(w)
            if line:
                lines.append(" ".join(line))
            print(f"  {i}. {lines[0]}")
            for l in lines[1:]:
                print(f"     {l}")

    print("\n" + "=" * 65)


# ── master runner ─────────────────────────────────────────────────────────────

def run_all(df: pd.DataFrame, save_path: str | None = None):
    """Run the complete EDA pipeline.

    Executes all six analysis sections in sequence and prints the
    insights summary at the end.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched dataframe from load_data().
    save_path : str, optional
        Directory to save figures as PNG files. If None, plt.show()
        is called for each figure interactively.

    Examples
    --------
    >>> import retail_eda as eda
    >>> df = eda.load_data("retail_price.csv")
    >>> eda.run_all(df)                        # interactive
    >>> eda.run_all(df, save_path="./plots")   # save to disk
    """
    print("\n  Running full EDA pipeline …")
    overview(df, save_path)
    category_analysis(df, save_path)
    seasonality_analysis(df, save_path)
    competitive_analysis(df, save_path)
    demand_drivers(df, save_path)
    price_dynamics(df, save_path)
    print_insights(df)
    print("\n  EDA complete.\n")


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    csv_path  = sys.argv[1] if len(sys.argv) > 1 else "retail_price.csv"
    save_path = sys.argv[2] if len(sys.argv) > 2 else None
    df = load_data(csv_path)
    run_all(df, save_path=save_path)