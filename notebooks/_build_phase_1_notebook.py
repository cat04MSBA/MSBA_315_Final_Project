"""Build ``notebooks/phase_1.ipynb`` from a structured cell list.

The notebook content lives here so it's easy to maintain, diff, and
re-emit. Run this script whenever Phase 1 cell content needs to change:

    python -m notebooks._build_phase_1_notebook

Outputs are cleared on every build; the user runs the notebook to
fill them in.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

OUTPUT = Path(__file__).resolve().parent / "phase_1.ipynb"


def md(text: str) -> dict:
    return nbf.v4.new_markdown_cell(dedent(text).strip("\n"))


def code(text: str) -> dict:
    return nbf.v4.new_code_cell(dedent(text).strip("\n"))


# ---------------------------------------------------------------------------
# Cell content
# ---------------------------------------------------------------------------

CELLS = [
    # ============================================================
    # Header
    # ============================================================
    md("""
        # Phase 1: Data Feasibility Verification

        The goal of this phase is to answer one question: how many films do we
        end up with that have all four required pieces of data? That means a
        screenplay, an IMDb-style rating, a budget, and a revenue figure. The
        answer tells us whether the project can move forward at the planned
        scale, or whether we need to expand or shrink the scope before
        building the rest of the pipeline.

        We're working with two datasets:

        * **MovieSum** (Saxena & Keller, ACL 2024). 2,200 movie screenplays
          in structured XML format, each tagged with an IMDb ID. They're
          stored as JSONL files split into train, val, and test in
          `data/raw/script_data/`.
        * **IMDb-TMDB Movie Metadata Big Dataset (1M)**. A Kaggle dataset
          with about 1.07 million films, carrying IMDb IDs and TMDB IDs
          natively, plus budget, revenue, runtime, multiple rating fields,
          director, and other metadata. The file is at
          `data/raw/ratings_data/IMDB TMDB Movie Metadata Big Dataset (1M).csv`.

        The plan: load both datasets, join them on IMDb ID directly (both
        have it natively, so no fuzzy matching needed), profile the joined
        corpus, and surface anything that needs your attention before Phase 2
        starts.

        Run the cells top to bottom. The first cell (Cell 0) makes everything
        else work regardless of where this notebook lives in the project tree.
    """),

    # ============================================================
    # Bootstrap
    # ============================================================
    md("## 0. Setup"),
    md("""
        This cell finds the project root by walking up from wherever the
        notebook lives until it finds `docs/PROJECT_CONTEXT.txt`. It then
        adds the project root to `sys.path` so `from src... import ...`
        works, and turns on inline plots and module auto-reloading. Safe to
        re-run.

        Note on `%autoreload 2`: this means that whenever you (or anyone
        else) edit a `.py` file inside `src/`, the changes flow into this
        notebook automatically without needing a kernel restart. Very useful
        during development.
    """),
    code("""
        import sys
        from pathlib import Path


        def _find_project_root(start: Path) -> Path:
            \"\"\"Walk up the tree until we find docs/PROJECT_CONTEXT.{txt,md}.\"\"\"
            markers = ("docs/PROJECT_CONTEXT.txt", "docs/PROJECT_CONTEXT.md")
            for candidate in (start.resolve(), *start.resolve().parents):
                if any((candidate / m).is_file() for m in markers):
                    return candidate
            raise RuntimeError(f"Could not find project root from {start!s}.")


        PROJECT_ROOT = _find_project_root(Path.cwd())
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        print("Project root:", PROJECT_ROOT)

        # Auto-reload modules when their source files change.
        # This is what lets edits to src/ flow into the notebook without
        # restarting the kernel.
        get_ipython().run_line_magic("load_ext", "autoreload")
        get_ipython().run_line_magic("autoreload", "2")

        get_ipython().run_line_magic("matplotlib", "inline")

        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
    """),

    md("Common imports the rest of the notebook will use."),
    code("""
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        from src.utils import paths

        paths.ensure_dirs()  # creates data/interim, reports/figures, etc. if missing

        print("Data raw dir:    ", paths.DATA_RAW_DIR)
        print("Data interim:    ", paths.DATA_INTERIM_DIR)
        print("Reports figures: ", paths.REPORTS_FIGURES_DIR)
    """),

    # ============================================================
    # Task 2: Ratings dataset
    # ============================================================
    md("---\n\n## Task 2: Profile the ratings dataset"),
    md("""
        Before we trust this dataset for the join, let's get a feel for what
        it looks like. Three things to check:

        1. How many films total, and how many have the financial fields we
           need?
        2. What's the year coverage? Specifically, does it extend up to recent
           years?
        3. How are the genre, budget, and revenue distributions shaped?

        The loader is column-selective on purpose. Of the 42 columns in the
        raw file, we only read the 20 we'll touch downstream. That keeps
        memory and load time reasonable. The first load takes about 20 to 30
        seconds.
    """),

    md("### 2.1 Load the dataset and look at the headline counts"),
    code("""
        from src.data.load_ratings import load_ratings, summarize_ratings

        ratings = load_ratings()
        counts = summarize_ratings(ratings)

        print("Ratings dataset headline counts:")
        for k, v in counts.items():
            print(f"  {k:>14}: {v:,}")
        ratings.head(3)
    """),
    md("""
        About 1.07 million films total, which is a comprehensive ratings
        catalogue. Of those, around 590 thousand have a valid IMDb ID. That
        sounds like a lot of missing IDs, but most of them are obscure
        foreign or unlinked records, which we don't care about. About 13.8
        thousand have both budget and revenue, which is the realistic upper
        bound on any analysis that needs financial outcomes.

        The size in absolute terms doesn't actually matter for our project.
        What matters is whether the films *that MovieSum has scripts for*
        are present and complete in this dataset. We'll check that in
        Task 4.
    """),

    md("### 2.2 Missingness across the columns we care about"),
    code("""
        cols = [
            "imdb_id", "id", "title", "release_date", "release_year_parsed",
            "budget", "revenue", "runtime",
            "vote_average", "vote_count", "IMDB_Rating", "AverageRating",
            "Meta_score", "popularity", "genres_parsed",
            "Director", "production_companies",
        ]
        miss = ratings[cols].isna().sum().sort_values(ascending=False)
        miss_pct = (miss / len(ratings) * 100).round(1)
        miss_report = pd.DataFrame({"missing": miss, "pct_missing": miss_pct})
        miss_report
    """),
    md("""
        Look at the `pct_missing` column. Some columns are sparsely populated
        across the full dataset (`IMDB_Rating`, `Meta_score`, `Director`,
        `budget`, `revenue` are the worst offenders). Most films just
        don't have these signals filled in.

        That's actually fine for our purposes. We're only going to keep the
        roughly 2,000 films that match MovieSum, and those tend to be
        mainstream titles where these fields are populated. The Task 4 join
        filters us down to that subset.
    """),

    md("### 2.3 Year coverage"),
    code("""
        years = ratings["release_year_parsed"].dropna().astype(int)
        # Clip nonsense future years (a handful say 2099, 2055, etc., which are
        # scheduled releases or noise; we don't want them in the histogram).
        sane = years[(years >= 1900) & (years <= 2025)]

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.hist(sane, bins=range(int(sane.min()), int(sane.max()) + 2),
                edgecolor="black", alpha=0.85)
        ax.set(xlabel="Release year", ylabel="Films",
               title=f"Ratings dataset year distribution: {len(sane):,} films "
                     f"(1900 to 2025; {len(years) - len(sane)} out-of-range dropped)")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        plt.show()

        print(f"Year range (after clipping): {sane.min()} to {sane.max()}")
        print(f"Median year: {int(sane.median())}")
    """),
    md("""
        Year coverage is comprehensive: dense from around 2000 to 2023, with a
        long thin tail going back to 1900. This is the structural property
        that makes this dataset suitable for our project. About 21 percent of
        MovieSum is post-2016, and we need a ratings source that contains
        those films.
    """),

    md("### 2.4 Genre, budget, and revenue summary among films with both"),
    code("""
        mask_both = (ratings["budget"] > 0) & (ratings["revenue"] > 0)
        ratings_both = ratings.loc[mask_both].copy()
        print(f"Films with both budget and revenue > 0: {len(ratings_both):,}")

        exploded = ratings_both.explode("genres_parsed")
        genre_counts = exploded["genres_parsed"].dropna().value_counts()

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.barh(genre_counts.index[::-1], genre_counts.values[::-1],
                color="steelblue", edgecolor="black")
        ax.set(xlabel="Films (multi-genre films counted in each bar)",
               title=f"Genre distribution among {len(ratings_both):,} films "
                     f"with budget and revenue")
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        plt.show()

        summary = pd.DataFrame({
            "budget":  ratings_both["budget"].describe(),
            "revenue": ratings_both["revenue"].describe(),
            "rev_to_budget_ratio": (
                ratings_both["revenue"] / ratings_both["budget"]
            ).describe(),
        }).round(2)
        summary
    """),
    md("""
        Drama, Comedy, Action, and Thriller dominate by a wide margin.
        Documentaries, foreign films, and TV movies are sparse. This shape
        will carry through to the joined corpus.

        Both budget and revenue are heavily right-skewed, meaning the mean
        is much larger than the median and the maximum is much larger than
        the 75th percentile. That's standard for film financials (a few
        blockbusters with $300M-plus budgets stretch the tail). When we get
        to Phase 3 we'll log-transform these (`log1p(budget)` and
        `log1p(revenue)`) so a handful of outliers don't dominate any model
        we train.
    """),

    # ============================================================
    # Task 3: MovieSum
    # ============================================================
    md("---\n\n## Task 3: Load and verify MovieSum"),
    md("""
        MovieSum is the screenplay corpus, 2,200 manually-formatted XML
        scripts, each tagged with an IMDb ID and a short Wikipedia plot
        summary. It's distributed as JSONL across train, val, and test
        splits in `data/raw/script_data/`. Note that those splits are for
        the upstream summarization task that the dataset's authors built
        the corpus for. We'll redo our own splits in Phase 3 once we
        decide on the project-specific train/calibration/test partition.
    """),

    md("### 3.1 Load all three splits and check coverage"),
    code("""
        from src.data.load_moviesum import load_moviesum, imdb_id_validity

        moviesum = load_moviesum()  # ~5 to 10 seconds; reads about 450 MB of JSONL

        print(f"Total screenplays:        {len(moviesum):,}")
        print(f"Origin-split breakdown:   {moviesum['origin_split'].value_counts().to_dict()}")
        print(f"IMDb ID validity:         {imdb_id_validity(moviesum)}")
        moviesum.head(2)
    """),
    md("""
        2,200 screenplays exactly as the README claims (1,800 train, 200
        val, 200 test). All have well-formed `tt`-prefixed IMDb IDs.

        But only 2,188 of those IDs are unique. Twelve IDs appear twice
        each. Those are alternate-title or alternate-draft pairs of the
        same film (M*A*S*H/MASH, Star Wars/Episode IV, Mulholland
        Dr./Mulholland Drive, and similar). Cell 6.1 below prints them
        side-by-side for review.
    """),

    md("### 3.2 Year and length distributions"),
    code("""
        print("Year-in-title summary:")
        print(moviesum["year_in_title"].dropna().describe())

        print("\\nScreenplay length (chars) summary:")
        print(moviesum["script_char_len"].describe())

        fig, (ax_raw, ax_log) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax_raw.hist(moviesum["script_char_len"] / 1000.0, bins=40,
                    edgecolor="black", alpha=0.85)
        ax_raw.set(xlabel="Length (thousand chars)", ylabel="Screenplays",
                   title="Raw scale")
        ax_raw.grid(axis="y", linestyle=":", alpha=0.6)

        ax_log.hist(moviesum["script_char_len"] / 1000.0, bins=40, log=True,
                    edgecolor="black", alpha=0.85)
        ax_log.set(xlabel="Length (thousand chars)",
                   ylabel="Screenplays (log scale)",
                   title="Log y-axis")
        ax_log.grid(axis="y", linestyle=":", alpha=0.6)
        fig.tight_layout()
        plt.show()
    """),
    md("""
        Year range goes from 1931 to 2023, with a median around 2007. About
        21 percent of films are from 2017 onwards. Length is approximately
        bell-shaped around 207,000 characters (which works out to about
        34,000 tokens at roughly 6 characters per token, consistent with
        what the MovieSum paper claims), with a light heavy tail.

        That length number matters for Phase 3. Most pre-trained text models
        we might use for feature extraction (BERT, DistilBERT, RoBERTa, and
        similar) can only read about 512 tokens at a time. Our screenplays
        are roughly 65 times longer than that. So when we get to Phase 3
        we'll need to break each screenplay into smaller chunks (by scene,
        by paragraph, or by fixed-size windows) and either pool the chunk
        features together or do something more clever like hierarchical
        attention. We'll decide the exact strategy then; for now we just
        note that "feed the whole script in" is not an option.
    """),

    md("### 3.3 Confirm the documented XML structure on a sample"),
    code("""
        import xml.etree.ElementTree as ET
        from collections import Counter

        sample = moviesum.iloc[0]
        root = ET.fromstring(sample["script"])

        counts = Counter()
        counts["scene"] = len(root.findall("scene"))
        for tag in ("stage_direction", "scene_description", "character", "dialogue"):
            counts[tag] = sum(len(s.findall(tag)) for s in root.findall("scene"))
        counts["unique_characters"] = len({
            c.text for s in root.findall("scene")
            for c in s.findall("character") if c.text
        })

        print(f"{sample['movie_name']} ({sample['imdb_id']})")
        print("Structural counts:", dict(counts))
    """),
    md("""
        The XML parses cleanly and has positive counts for every documented
        tag (`scene`, `stage_direction`, `scene_description`, `character`,
        `dialogue`). So Phase 3 can rely on a standard XML parse rather than
        ad-hoc regex, which is good news for the parser code.
    """),

    md("### 3.4 Spot-check 5 random screenplays"),
    code("""
        import random

        rng = random.Random(42)
        for i in rng.sample(range(len(moviesum)), 5):
            row = moviesum.iloc[i]
            head = (row["script"] or "")[:400].replace("\\n", " ⏎ ")
            print(f"[{i}] {row['movie_name']} | {row['imdb_id']} | "
                  f"len={row['script_char_len']:,} | split={row['origin_split']}")
            print(f"   {head} ...\\n")
    """),
    md("""
        Five random screenplays, all real films with recognizable scene
        boundaries, character names, and dialogue. A few have cover-page
        metadata (writer, draft date, copyright notice) before the first
        actual scene; that's noise we'll filter in Phase 2's parser. Not
        something to worry about right now.
    """),

    # ============================================================
    # Task 4: Direct IMDb-ID join
    # ============================================================
    md("---\n\n## Task 4: Join MovieSum with the ratings dataset on IMDb ID"),
    md("""
        Both sources carry IMDb IDs natively, so the join is a single line
        of pandas. No normalization, no fuzzy matching, no external bridge.
        The script does this:

        1. Deduplicates MovieSum on IMDb ID (the 12 same-IMDb-ID pairs
           collapse to 1 row each, keeping the longest script per ID).
        2. Deduplicates the ratings dataset on `imdb_id` (a small fraction
           of films appear twice under different TMDB IDs; we keep the row
           with higher `vote_count`).
        3. Left-merges MovieSum onto the deduped ratings rows.
        4. Saves to `data/interim/phase1_joined_corpus.parquet`.
    """),

    md("### 4.1 Run the join"),
    code("""
        from src.data.join_corpus import main as run_join

        joined = run_join()
        matched = joined[joined["id"].notna()]
        unmatched = joined[joined["id"].isna()]
        print(f"\\nMatched: {len(matched):,}  /  Unmatched: {len(unmatched):,}")
        unmatched[["imdb_id", "movie_name", "year_in_title"]] if len(unmatched) else "(no unmatched rows)"
    """),
    md("""
        2,186 of 2,188 MovieSum films matched (99.9 percent). Only 2 films
        are missing from the ratings dataset entirely. The working corpus
        (films with all four signals: matched plus budget plus revenue
        plus rating) comes out to **1,713 films**, which is comfortably
        above the 1,500-film threshold in the corpus-size decision
        criteria.
    """),

    md("### 4.2 Sample matches for verification"),
    code("""
        sample = matched.sample(15, random_state=42)
        sample[[
            "imdb_id", "movie_name", "title_rt", "release_year_parsed",
            "vote_average", "IMDB_Rating", "budget", "revenue",
        ]]
    """),
    md("""
        All 15 should look obviously correct: titles match between MovieSum
        and the ratings side, years match, ratings and budgets and revenues
        are present and sensible. If anything looks off, flag it. Since
        this is a pure-ID join we don't expect mismatches, but it's worth
        glancing at.
    """),

    # ============================================================
    # Task 5: Profile the working corpus
    # ============================================================
    md("---\n\n## Task 5: Profile the 1,713-film working corpus"),
    md("""
        Now the substantive part. For the films that have all four signals,
        we look at how the corpus is shaped. These plots and numbers feed
        directly into the methodology section of the final report.
    """),

    md("### 5.1 Load the working corpus and the summary table"),
    code("""
        from src.data.profile_corpus import load_working_corpus, summary_table

        working = load_working_corpus()
        summary = summary_table(working)
        print(f"Working corpus: {len(working):,} films")
        summary
    """),
    md("""
        The working corpus is 1,713 films. Year range is 1932 to 2023 with a
        median of 2005. Budget median is around $25M, revenue median around
        $64M, rating median around 7.0. ROI median is roughly 2.9x and
        about 80 percent of films are gross-profitable.

        That 80 percent number is striking. The industry's actual
        net-profitable rate is closer to 50 percent. So our corpus is
        survivor-biased: it contains films that got produced in the first
        place, *and* were recognizable enough to land in a major metadata
        aggregator. We need to keep this in mind when we build the
        cost-decision layer in Phase 6, and we may want to reweight or
        otherwise adjust during training. We'll discuss before deciding.
    """),

    md("### 5.2 Year distribution"),
    code("""
        years = working["release_year_parsed"].dropna().astype(int)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.hist(years, bins=range(int(years.min()), int(years.max()) + 2),
                edgecolor="black", alpha=0.85)
        ax.set(xlabel="Release year", ylabel="Films",
               title=f"Working corpus year distribution: N={len(working):,}")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        plt.show()
    """),
    md("""
        The corpus is effectively dense from 1995 to 2022, with a thin tail
        of about 50 films before 1995 going back to 1932. Phase 2 will drop
        the pre-1995 tail (this was already decided and logged 2026-05-02).
        The aim is to keep the corpus in a single coherent era for cleaner
        generalization claims in later phases.
    """),

    md("### 5.3 Genre distribution"),
    code("""
        exploded = working.explode("genres_parsed")
        genre_counts = exploded["genres_parsed"].dropna().value_counts()
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.barh(genre_counts.index[::-1], genre_counts.values[::-1],
                color="steelblue", edgecolor="black")
        ax.set(xlabel="Films (multi-genre films counted in each bar)",
               title=f"Working corpus genre distribution: N={len(working):,}")
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        plt.show()
    """),
    md("""
        Same Drama/Comedy/Thriller/Action lead as in the ratings-only
        profile. The major genres are well-represented enough for
        per-genre cross-validation slices in Phase 4. The thin cells
        (Documentary, TV Movie, Foreign, Western) we'll bucket together
        into an "other" group when stratifying, otherwise we get noisy
        estimates from cells with under 30 films.
    """),

    md("### 5.4 Budget and revenue, raw and log scales"),
    code("""
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        for col, color, ax_raw, ax_log in (
            ("budget",  "steelblue", axes[0, 0], axes[1, 0]),
            ("revenue", "indianred", axes[0, 1], axes[1, 1]),
        ):
            vals = working[col].astype(float)
            ax_raw.hist(vals / 1e6, bins=50, color=color,
                        edgecolor="black", alpha=0.85)
            ax_raw.set(xlabel=f"{col.capitalize()} ($M)", ylabel="Films",
                       title=f"{col.capitalize()}: raw")
            ax_raw.grid(axis="y", linestyle=":", alpha=0.6)

            log_vals = np.log10(vals.clip(lower=1))
            ax_log.hist(log_vals, bins=50, color=color,
                        edgecolor="black", alpha=0.85)
            ax_log.set(xlabel=f"log10({col.capitalize()})", ylabel="Films",
                       title=f"{col.capitalize()}: log10")
            ax_log.grid(axis="y", linestyle=":", alpha=0.6)
        fig.suptitle(f"Budget and revenue: N={len(working):,}")
        fig.tight_layout()
        plt.show()
    """),
    md("""
        The raw-scale histograms are basically useless because of the heavy
        right tail. The log10 versions show clean bell shapes (about $10M
        to $100M for budget centered around $25M; about $1M to $1B for
        revenue centered around $60M). Phase 3 features will be
        log-transformed before any modeling.
    """),

    md("### 5.5 Rating, ROI, and screenplay length"),
    code("""
        roi = working["revenue"] / working["budget"]
        roi_log = np.log10(
            roi.replace([np.inf, -np.inf], np.nan).dropna().clip(lower=1e-3)
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        axes[0].hist(working["effective_rating"], bins=np.arange(0, 10.5, 0.25),
                     edgecolor="black", alpha=0.85)
        axes[0].set(xlabel="Rating (IMDB_Rating preferred)", ylabel="Films",
                    title="Rating")
        axes[0].grid(axis="y", linestyle=":", alpha=0.6)

        axes[1].hist(roi_log, bins=60, color="darkgreen",
                     edgecolor="black", alpha=0.8)
        axes[1].axvline(0, color="black", linestyle="--", alpha=0.7,
                        label="break-even (ROI=1)")
        axes[1].set(xlabel="log10(ROI)", ylabel="Films",
                    title="ROI (revenue / budget)")
        axes[1].legend()
        axes[1].grid(axis="y", linestyle=":", alpha=0.6)

        axes[2].hist(working["script_char_len"] / 1000.0, bins=40,
                     color="orange", edgecolor="black", alpha=0.85)
        axes[2].set(xlabel="Screenplay length (k chars)", ylabel="Films",
                    title="Screenplay length")
        axes[2].grid(axis="y", linestyle=":", alpha=0.6)

        fig.tight_layout()
        plt.show()
    """),
    md("""
        Three quick reads:

        * **Rating**: narrow Gaussian-ish around 7.0. The dynamic range is
          limited (most films cluster between 5.5 and 8.5), so we should
          not expect a huge predictive R-squared in Phase 4. The
          asymmetric-cost decision layer in Phase 6 is what makes the
          system useful, not the headline regression accuracy.
        * **ROI on log scale**: heavy right tail of hits at 10x and beyond,
          and a thin left tail of flops. The break-even line at 0 sits to
          the left of the bulk of the distribution, which is what
          produces the 80 percent gross-profitable number we saw in 5.1.
        * **Screenplay length**: same shape as the full MovieSum length
          plot, similar mean. So the four-signal filter doesn't
          preferentially keep short or long screenplays.
    """),

    md("### 5.6 Bivariate sanity: budget vs revenue, rating vs ROI"),
    code("""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Budget vs revenue (log-log)
        axes[0].scatter(np.log10(working["budget"].clip(lower=1)),
                        np.log10(working["revenue"].clip(lower=1)),
                        s=8, alpha=0.4, color="steelblue")
        axes[0].plot([4, 9], [4, 9], "k--", alpha=0.5,
                     label="break-even (revenue = budget)")
        axes[0].set(xlabel="log10(budget)", ylabel="log10(revenue)",
                    title=f"Budget vs revenue (log-log): N={len(working):,}")
        axes[0].legend()
        axes[0].grid(linestyle=":", alpha=0.5)

        # Rating vs log-ROI
        roi_for_plot = (working["revenue"] / working["budget"]).clip(lower=1e-3)
        axes[1].scatter(working["effective_rating"], np.log10(roi_for_plot),
                        s=8, alpha=0.4, color="indianred")
        axes[1].axhline(0, color="black", linestyle="--", alpha=0.5,
                        label="break-even")
        axes[1].set(xlabel="Rating", ylabel="log10(ROI)",
                    title="Rating vs ROI")
        axes[1].legend()
        axes[1].grid(linestyle=":", alpha=0.5)

        fig.tight_layout()
        plt.show()

        log_budget = np.log10(working["budget"].clip(lower=1))
        log_revenue = np.log10(working["revenue"].clip(lower=1))
        corr_table = pd.DataFrame({
            "log_budget":  log_budget,
            "log_revenue": log_revenue,
            "rating":      working["effective_rating"],
            "log_roi":     np.log10(roi_for_plot),
            "log_length":  np.log10(working["script_char_len"]),
        }).corr(method="pearson").round(3)
        corr_table
    """),
    md("""
        Budget and revenue are strongly positively correlated (around 0.6
        to 0.7), which is what we'd expect: bigger films make more money in
        absolute terms even though many flop. Rating is only weakly
        correlated with ROI and weakly negative with budget (small indie
        films tend to score higher on rating than mid-budget studio
        releases). Screenplay length has weak relationships with
        everything.

        The takeaway: there's no easy "if budget is high, the model wins"
        shortcut. The dialogue features we extract in Phase 3 will need
        to carry most of the predictive signal.
    """),

    md("### 5.7 Per-decade breakdown"),
    code("""
        decade = (working["release_year_parsed"] // 10 * 10).astype("Int64")
        per_decade = working.assign(decade=decade).groupby("decade").agg(
            n_films=("imdb_id", "count"),
            budget_median_M=("budget", lambda s: s.median() / 1e6),
            revenue_median_M=("revenue", lambda s: s.median() / 1e6),
            rating_mean=("effective_rating", "mean"),
        )
        per_decade["roi_median"] = working.assign(decade=decade).groupby("decade").apply(
            lambda d: float((d["revenue"] / d["budget"]).median())
        )
        per_decade.round(2)
    """),
    md("""
        Most of the corpus sits in the 2000s and 2010s (each around 600-plus
        films). Earlier decades are very thin (1930s, 1940s, 1950s have
        fewer than 30 films each). Budget and revenue medians rise over
        time as expected (modern films cost more to make). Rating drifts
        down slightly in recent decades as mainstream releases attract a
        wider rating spread.

        For Phase 4, this means era-stratified cross-validation needs to
        bucket the pre-1980s decades together (or just exclude them, which
        is what the pre-1995 cutoff in Phase 2 will already do).
    """),

    # ============================================================
    # Task 6: Duplicate review
    # ============================================================
    md("---\n\n## Task 6: Review the 12 same-IMDb-ID duplicate pairs"),
    md("""
        MovieSum has 12 IMDb IDs that appear on two rows each. These are
        alternate-title or alternate-draft pairs of the same film. The cell
        below prints them side-by-side and saves a fillable CSV at
        `reports/tables/phase1_moviesum_duplicates_review.csv`.

        After running this cell, open the CSV and mark each row as `keep`
        or `drop` (one keep and one drop per pair). Phase 2 will encode the
        policy you decide on.
    """),

    md("### 6.1 Print the pairs and save the review template"),
    code("""
        from src.data.review_duplicates import main as run_dup_review

        dup_rows = run_dup_review()
        print(f"\\nReturned DataFrame: {len(dup_rows)} rows across "
              f"{dup_rows['imdb_id'].nunique()} duplicate IMDb IDs")
    """),
    md("""
        All 12 pairs are real same-film duplicates. The decision per pair
        is binary: which row stays?

        * The longer script usually represents the more-complete draft, so
          Phase 1's join already preferred the longer one in each pair.
        * For pairs where one row is in `train` and the other in `test`
          (Cherry Falls is one), prefer dropping the `test` copy so the
          upstream MovieSum split doesn't bleed into our future split.
        * "Merge" doesn't make sense for screenplays; just pick one.
    """),

    # ============================================================
    # Task 7: Inventory
    # ============================================================
    md("---\n\n## Task 7: Phase 1 artifact inventory"),
    md("""
        Confirm everything the phase produced is in place. The files below
        feed Phase 2 (the joined parquet) and the final report (figures,
        tables, summary).
    """),
    code("""
        from pathlib import Path

        for label, dirpath in (
            ("Figures",  paths.REPORTS_FIGURES_DIR),
            ("Tables",   paths.REPORTS_TABLES_DIR),
            ("Interim",  paths.DATA_INTERIM_DIR),
            ("Summary",  paths.DOCS_SUMMARIES_DIR),
        ):
            print(f"\\n{label}: {dirpath}")
            for p in sorted(Path(dirpath).glob("*")):
                if p.is_file():
                    size_kb = p.stat().st_size / 1024
                    print(f"  {p.name}  ({size_kb:,.1f} KB)")
    """),

    # ============================================================
    # Conclusion
    # ============================================================
    md("---\n\n## Phase 1 Conclusion"),
    md("""
        ### Headline result

        We have **1,713 films with all four signals** (screenplay, budget,
        revenue, rating). That's comfortably above the 1,500-film threshold
        in the corpus-size decision criteria. Corpus size is no longer a
        constraint on Phase 2.

        ### What we noticed in the EDA, and how Phase 2 will handle it

        These are the concrete things to act on. They translate directly
        into Phase 2 implementation choices. Each item is paired with what
        we plan to do about it. Phase 2 will not just go and apply these
        without checking with you first; some are open and need your call.

        **Already decided (will be encoded in Phase 2 without further discussion):**

        * **Pre-1995 tail of about 50 films.** Drop them. The corpus is
          dense 1995 to 2022, and the long thin tail before that gives us
          fewer than 5 films per year, which is too thin to support
          era-stratified analyses. Decision logged 2026-05-02.
        * **Two MovieSum films missing from the ratings dataset.** Drop
          them. We can't use a film without rating, budget, or revenue.
        * **Ratings dataset has duplicates on `imdb_id`.** Some films
          appear under multiple TMDB IDs (alternate cuts, regional
          releases). We dedupe by `imdb_id` keeping the row with the
          higher `vote_count`, which picks the better-known variant.
          Already implemented in `src/data/join_corpus.py` `dedupe_ratings()`.
        * **Future-year noise (2055, 2099, etc.) in the ratings
          dataset.** Clip `release_year_parsed` to 1900 to 2025 when
          plotting or filtering. The affected rows are obscure and rarely
          intersect with MovieSum, so this is mostly cosmetic.
        * **Effective rating column.** Use `IMDB_Rating` when present and
          fall back to `vote_average` otherwise. IMDB_Rating is
          better-populated on this matched subset and is the metric the
          report ultimately discusses. Phase 2 standardizes on a single
          `effective_rating` column for downstream use.

        **Decided in principle, but Phase 2 will surface options before
        implementing:**

        * **Budget and revenue are heavily right-skewed.** Use
          `log1p(budget)` and `log1p(revenue)` so a few outliers don't
          dominate any model. Phase 2 (or Phase 3, depending on where we
          put the transform) should ask whether you want plain `log1p`,
          a different transform like `log10`, or a quantile-bucketed
          version.
        * **MovieSum dedup policy for the 12 pairs.** Awaiting your row-by-
          row review of `reports/tables/phase1_moviesum_duplicates_review.csv`.
          Phase 2 will encode whatever policy you choose.
        * **Genre long tail.** Documentaries, TV Movies, Foreign,
          Western, etc. have very thin counts. For per-genre
          cross-validation in Phase 4 we'll need to bucket them. Options
          to discuss when we get there: aggregate them all into "other",
          drop them, or keep them and accept noisy per-genre estimates.

        **Observations to carry into later phases (not Phase 2 actions):**

        * **Survivorship bias.** About 80 percent of the working corpus is
          gross-profitable, well above industry's roughly 50 percent net-
          profitable rate. The cost matrix in Phase 6 must use industry
          base rates, not corpus rates. We may also want to reweight at
          training time (Phase 4); we'll discuss before deciding.
        * **Narrow rating range.** Mean around 7.0, standard deviation
          around 0.7. Expect a modest predictive R-squared in Phase 4.
          The asymmetric-cost decision layer in Phase 6 is where the
          system earns its value.
        * **No easy linear shortcut.** Rating is only weakly correlated
          with budget, revenue, ROI, or screenplay length. The dialogue
          features we extract in Phase 3 will have to carry most of the
          predictive weight; budget alone won't tell us whether a film
          is good.
        * **Screenplay length vs. transformer windows.** Screenplays are
          about 34,000 tokens long on average, which is much longer than
          most pre-trained text models can read in a single pass (BERT
          and similar models typically cap at 512 tokens). Phase 3
          feature extraction will need chunking (by scene, paragraph, or
          fixed window) and feature pooling, or a hierarchical approach.
          We'll decide the strategy when we get to Phase 3.
        * **Per-decade thin cells.** Pre-1980s decades each have under 30
          films. The pre-1995 cutoff above already handles most of this;
          for what remains, era-stratified CV in Phase 4 should bucket
          decades.

        ### What to bring to the planning conversation before Phase 2

        Two open items:

        1. The dedup decisions for the 12 MovieSum pairs (review the CSV
           Cell 6.1 emits).
        2. Whether you want the survivorship bias addressed via
           reweighting during training, or just acknowledged in the
           report's limitations section.

        See `docs/summaries/phase_1_summary.txt` for the full narrative
        and per-figure interpretations.
    """),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUTPUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
