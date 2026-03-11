from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm


ISO3_FIX = {
    "FR": "FRA",
    "UK": "GBR",
    "EL": "GRC",
    "SP": "ESP",
    "SW": "SWE",
    "NO": "NOR",
    "JA": "JPN",
    "KO": "KOR",
    "CH": "CHN",
    "FR1": "FRA",
}

USSR_ALIASES = {"USSR", "SUN", "USS"}
USSR_NAME_ALIASES = {
    "USSR",
    "SOVIET UNION",
    "UNION OF SOVIET SOCIALIST REPUBLICS",
}
INVALID_ISO3 = {"-99", "NAN", "NONE", ""}


def _normalize_zip_shp_path(path: str) -> str:
    path = str(path)
    if path.startswith("zip://"):
        return path
    if ".zip!" in path.lower():
        return "zip://" + path
    return path


def _clean_iso3(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    return s.replace({"": np.nan, "NAN": np.nan, "NONE": np.nan, "-99": np.nan})


def _find_column(columns, *, exact=None, contains=None) -> Optional[str]:
    exact = exact or []
    contains = contains or []

    lowered = {str(c).strip().lower(): c for c in columns}
    for key in exact:
        if key.lower() in lowered:
            return lowered[key.lower()]

    for c in columns:
        low = str(c).strip().lower()
        if any(token.lower() in low for token in contains):
            return c
    return None


def _make_world_iso3_key(world: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "ISO_A3" not in world.columns:
        raise ValueError(f"world に ISO_A3 列が無い。columns={list(world.columns)}")

    world = world.copy()
    world["_iso3"] = _clean_iso3(world["ISO_A3"])
    invalid = world["_iso3"].isna()

    if "ADM0_A3" in world.columns:
        world.loc[invalid, "_iso3"] = _clean_iso3(world.loc[invalid, "ADM0_A3"])
        invalid = world["_iso3"].isna()

    if "SOV_A3" in world.columns:
        world.loc[invalid, "_iso3"] = _clean_iso3(world.loc[invalid, "SOV_A3"])
        invalid = world["_iso3"].isna()

    world["_iso3"] = world["_iso3"].replace(ISO3_FIX)
    world.loc[invalid, "_iso3"] = np.nan
    return world


def extract_population_by_iso3_for_year(
    population_df: pd.DataFrame,
    year: int,
    *,
    iso3_col: Optional[str] = None,
    year_col: Optional[str] = None,
    pop_col: Optional[str] = None,
    ussr_until_year: int = 1991,
) -> pd.DataFrame:
    if population_df is None or len(population_df) == 0:
        raise ValueError("population_df が空です。")

    pop = population_df.copy()
    y = int(year)

    if iso3_col is None:
        iso3_col = _find_column(
            pop.columns,
            exact=["iso3", "iso_a3", "iso-3", "iso_3", "country code", "country_code", "code"],
            contains=["iso3"],
        )
    if iso3_col is None:
        raise ValueError(
            f"人口データの ISO3 列を特定できません。iso3_col を指定してください。columns={list(pop.columns)}"
        )

    if year_col is not None or pop_col is not None:
        if year_col is None or pop_col is None:
            raise ValueError("year_col と pop_col はセットで指定してください。")
        long = pop[[iso3_col, year_col, pop_col]].copy()
        long["_year"] = pd.to_numeric(long[year_col], errors="coerce")
        long = long[long["_year"] == y]
        out = long[[iso3_col, pop_col]].rename(columns={iso3_col: "iso3", pop_col: "population"})
    else:
        year_col_wide = None
        for c in pop.columns:
            if str(c).strip() == str(y):
                year_col_wide = c
                break

        if year_col_wide is not None:
            out = pop[[iso3_col, year_col_wide]].rename(columns={iso3_col: "iso3", year_col_wide: "population"})
        else:
            auto_year_col = _find_column(pop.columns, exact=["year", "yr"], contains=["year"])
            auto_pop_col = _find_column(pop.columns, exact=["population", "pop", "value"], contains=["pop"])
            if auto_year_col is None or auto_pop_col is None:
                raise ValueError(
                    "人口データ形式を自動判定できません。"
                    " long形式なら year_col/pop_col、wide形式なら年列（例: '2015'）を確認してください。"
                )
            long = pop[[iso3_col, auto_year_col, auto_pop_col]].copy()
            long["_year"] = pd.to_numeric(long[auto_year_col], errors="coerce")
            long = long[long["_year"] == y]
            out = long[[iso3_col, auto_pop_col]].rename(columns={iso3_col: "iso3", auto_pop_col: "population"})

    out = out.copy()
    out["iso3"] = _clean_iso3(out["iso3"]).replace(ISO3_FIX)
    if y <= int(ussr_until_year):
        out["iso3"] = out["iso3"].replace({k: "RUS" for k in USSR_ALIASES})

    out["population"] = pd.to_numeric(out["population"], errors="coerce")
    out = out[out["iso3"].notna() & np.isfinite(out["population"]) & (out["population"] > 0)].copy()
    out = out.groupby("iso3", as_index=False)["population"].sum()
    return out


def plot_vwt_net_import_export_map_debug_percapita(
    year: int,
    crop: int,
    *,
    country_list_xlsx: str,
    vwt_npy_template: str,
    ne_countries_shp: str,
    population_df: pd.DataFrame,
    pop_iso3_col: Optional[str] = None,
    pop_year_col: Optional[str] = None,
    pop_value_col: Optional[str] = None,
    population_year: Optional[int] = None,
    xlim=(-180, 180),
    ylim=(-60, 85),
    top_n_print: int = 20,
    base_unit: str = "m^3/year",
    percapita_unit: str = "m^3/person/year",
    cmap: str = "seismic",
    clip_quantile: Optional[float] = 0.995,
    missing_color: str = "#BEBEBE",
    border_color: str = "#444444",
    border_lw: float = 0.35,
    show_missing_legend: bool = True,
    debug_iso3: str = "FRA",
    print_no_match_samples: int = 30,
    print_df_only_samples: int = 30,
    ussr_until_year: int = 1991,
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    debug_iso3 = str(debug_iso3).strip().upper()
    y = int(year)
    pop_y = int(population_year) if population_year is not None else y
    pre_ussr = y <= int(ussr_until_year)

    if pre_ussr and debug_iso3 in USSR_ALIASES:
        debug_iso3 = "RUS"

    vwt_path = vwt_npy_template.format(crop=crop, year=year)
    vwt_mat = np.load(vwt_path).astype(float)
    if vwt_mat.ndim != 2:
        raise ValueError(f"VWTmat must be 2D. got shape={vwt_mat.shape}")

    nrow, ncol = vwt_mat.shape
    if nrow != ncol:
        print(f"[WARN] VWTmat is not square: {vwt_mat.shape}. row/colの向きを要確認")

    n = min(nrow, ncol)
    vwt_mat = vwt_mat[:n, :n]
    print("VWTmat shape:", vwt_mat.shape)
    print("VWT file:", vwt_path)

    cl = pd.read_excel(country_list_xlsx)
    iso3_col = _find_column(cl.columns, exact=["iso3", "iso_a3", "iso-3", "iso_3"], contains=["iso3"])
    if iso3_col is None:
        raise ValueError(f"country_list に ISO3 列が見つからない。columns={list(cl.columns)}")
    if len(cl) < n:
        raise ValueError(f"country_list rows ({len(cl)}) < VWT size ({n}). country_listの行数が足りない。")

    cl_n = cl.iloc[:n].copy()
    cl_n["_iso3_raw"] = cl_n[iso3_col].astype(str)
    cl_n["_iso3"] = _clean_iso3(cl_n["_iso3_raw"]).replace(ISO3_FIX)

    country_name_col = _find_column(cl_n.columns, exact=["country name", "country", "name"], contains=["country"])
    if country_name_col is None:
        country_name_col = cl_n.columns[0]

    cl_n["_country_upper"] = cl_n[country_name_col].astype(str).str.strip().str.upper()
    missing_iso = cl_n["_iso3"].isna()
    cl_n.loc[missing_iso & cl_n["_country_upper"].isin(USSR_NAME_ALIASES), "_iso3"] = "USSR"

    import_by_country = np.nansum(vwt_mat, axis=0)
    export_by_country = np.nansum(vwt_mat, axis=1)

    df_net = pd.DataFrame(
        {
            "iso3_raw": cl_n["_iso3"].values,
            "import": import_by_country,
            "export": export_by_country,
        }
    )
    if pre_ussr:
        df_net["iso3"] = df_net["iso3_raw"].replace({k: "RUS" for k in USSR_ALIASES})
    else:
        df_net["iso3"] = df_net["iso3_raw"]

    df_net = df_net[df_net["iso3"].notna()].copy()
    df_net = df_net.groupby("iso3", as_index=False)[["import", "export"]].sum()
    df_net["net"] = df_net["import"] - df_net["export"]

    pop_by_iso3 = extract_population_by_iso3_for_year(
        population_df=population_df,
        year=pop_y,
        iso3_col=pop_iso3_col,
        year_col=pop_year_col,
        pop_col=pop_value_col,
        ussr_until_year=ussr_until_year,
    )
    print(f"population year: {pop_y}, rows={len(pop_by_iso3)}")

    df_net = df_net.merge(pop_by_iso3, on="iso3", how="left")
    valid_pop = np.isfinite(df_net["population"]) & (df_net["population"] > 0)
    for col in ["import", "export", "net"]:
        df_net[f"{col}_pc"] = np.where(valid_pop, df_net[col] / df_net["population"], np.nan)

    print(f"population valid ratio: {int(valid_pop.sum())}/{len(df_net)}")

    df_pos = df_net[df_net["net_pc"] > 0].sort_values("net_pc", ascending=False)
    df_neg = df_net[df_net["net_pc"] < 0].sort_values("net_pc", ascending=True)
    print(f"\nTop {top_n_print} net per-capita positive (net_pc>0):")
    print(df_pos.head(top_n_print)[["iso3", "population", "import_pc", "export_pc", "net_pc"]])
    print(f"\nTop {top_n_print} net per-capita negative (net_pc<0):")
    print(df_neg.head(top_n_print)[["iso3", "population", "import_pc", "export_pc", "net_pc"]])

    ne_countries_shp = _normalize_zip_shp_path(ne_countries_shp)
    world = gpd.read_file(ne_countries_shp)
    world = _make_world_iso3_key(world)
    m = world.merge(df_net, left_on="_iso3", right_on="iso3", how="left")

    print("\n" + "=" * 70)
    print(f"[DEBUG] Investigate ISO3 = {debug_iso3}")
    print("=" * 70)
    print(f"[DEBUG] world has {debug_iso3}? ->", (world["_iso3"] == debug_iso3).any())
    print(f"[DEBUG] df_net has {debug_iso3}? ->", (df_net["iso3"] == debug_iso3).any())
    if (df_net["iso3"] == debug_iso3).any():
        print(df_net[df_net["iso3"] == debug_iso3][["iso3", "population", "import_pc", "export_pc", "net_pc"]].head(5))

    no_match = m[m["_iso3"].notna() & m["net_pc"].isna()][["_iso3"]].drop_duplicates()
    print(f"\n[DEBUG] sample world-iso3 but missing in df_net/pop (no match): n={len(no_match)}")
    print(no_match.head(print_no_match_samples).to_string(index=False))

    net_only = df_net[~df_net["iso3"].isin(world["_iso3"].dropna())][["iso3"]].drop_duplicates()
    print(f"\n[DEBUG] sample df_net iso3 but missing in world: n={len(net_only)}")
    print(net_only.head(print_df_only_samples).to_string(index=False))
    print("=" * 70 + "\n")

    vals = m["net_pc"].astype(float).to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("net_pc の有効値が無い（人口マージ失敗 or population<=0）。")

    absvals = np.abs(vals)
    if clip_quantile is not None:
        q = float(clip_quantile)
        if not (0 < q <= 1):
            raise ValueError("clip_quantile must be in (0, 1].")
        vmax = float(np.nanquantile(absvals, q))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.nanmax(absvals))
    else:
        vmax = float(np.nanmax(absvals))

    if vmax == 0:
        vmax = 1.0

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_axis_off()
    ax.set_title(f"Virtual-water net per capita (import - export) [{percapita_unit}] | {year} (crop={crop})", fontsize=14)

    m.plot(ax=ax, color=missing_color, linewidth=0.0, edgecolor="none", zorder=1)
    has = np.isfinite(m["net_pc"].astype(float))
    m.loc[has].plot(
        ax=ax,
        column="net_pc",
        cmap=cmap,
        norm=norm,
        linewidth=0.0,
        edgecolor="none",
        legend=True,
        zorder=2,
    )
    m.boundary.plot(ax=ax, color=border_color, linewidth=border_lw, zorder=3)

    if show_missing_legend:
        from matplotlib.patches import Patch

        ax.legend(
            handles=[Patch(facecolor=missing_color, edgecolor="none", label="No data / no match / no population")],
            loc="lower left",
            frameon=True,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    plt.show()

    return df_net, m


def animate_vwt_net_import_export_percapita(
    crop: int,
    year_start: int,
    year_end: int,
    *,
    country_list_xlsx: str,
    vwt_npy_template: str,
    ne_countries_shp: str,
    population_df: pd.DataFrame,
    pop_iso3_col: Optional[str] = None,
    pop_year_col: Optional[str] = None,
    pop_value_col: Optional[str] = None,
    population_year: Optional[int] = None,
    xlim=(-180, 180),
    ylim=(-60, 85),
    percapita_unit: str = "m^3/person/year",
    fps: int = 6,
    interval_ms: Optional[int] = None,
    out_mp4: Optional[str] = None,
    out_gif: Optional[str] = None,
    dpi: int = 150,
    cmap: str = "seismic",
    clip_quantile: Optional[float] = 0.995,
    missing_color: str = "#BEBEBE",
    border_color: str = "#444444",
    border_lw: float = 0.35,
    show_missing_legend: bool = True,
    ussr_until_year: int = 1991,
) -> FuncAnimation:
    years = list(range(int(year_start), int(year_end) + 1))
    if len(years) == 0:
        raise ValueError("year_start <= year_end になるように指定してください。")
    if interval_ms is None:
        interval_ms = int(1000 / max(1, int(fps)))

    cl = pd.read_excel(country_list_xlsx)
    iso3_col = _find_column(cl.columns, exact=["iso3", "iso_a3", "iso-3", "iso_3"], contains=["iso3"])
    if iso3_col is None:
        raise ValueError(f"country_list に ISO3 列が見つからない。columns={list(cl.columns)}")

    vwt0 = np.load(vwt_npy_template.format(crop=crop, year=years[0])).astype(float)
    if vwt0.ndim != 2:
        raise ValueError(f"VWTmat must be 2D. got shape={vwt0.shape}")

    n = min(vwt0.shape[0], vwt0.shape[1])
    if vwt0.shape[0] != vwt0.shape[1]:
        print(f"[WARN] VWTmat is not square: {vwt0.shape}. row/colの向きを要確認")
    if len(cl) < n:
        raise ValueError(f"country_list rows ({len(cl)}) < VWT size ({n}). country_listの行数が足りない。")

    cl_n = cl.iloc[:n].copy()
    cl_n["_iso3_raw"] = cl_n[iso3_col].astype(str)
    cl_n["_iso3"] = _clean_iso3(cl_n["_iso3_raw"]).replace(ISO3_FIX)

    country_name_col = _find_column(cl_n.columns, exact=["country name", "country", "name"], contains=["country"])
    if country_name_col is None:
        country_name_col = cl_n.columns[0]
    cl_n["_country_upper"] = cl_n[country_name_col].astype(str).str.strip().str.upper()
    missing_iso = cl_n["_iso3"].isna()
    cl_n.loc[missing_iso & cl_n["_country_upper"].isin(USSR_NAME_ALIASES), "_iso3"] = "USSR"

    ne_countries_shp = _normalize_zip_shp_path(ne_countries_shp)
    world = gpd.read_file(ne_countries_shp)
    world = _make_world_iso3_key(world)

    pop_cache: dict[int, pd.DataFrame] = {}
    netpc_by_year: dict[int, pd.DataFrame] = {}
    global_abs_vals = []

    for y in years:
        vwt_y = np.load(vwt_npy_template.format(crop=crop, year=y)).astype(float)
        if vwt_y.ndim != 2:
            raise ValueError(f"年{y}のVWTmatが2次元でない: shape={vwt_y.shape}")
        if min(vwt_y.shape[0], vwt_y.shape[1]) < n:
            raise ValueError(f"年{y}のVWTサイズが小さい: {vwt_y.shape}, expected at least {n}x{n}")
        if vwt_y.shape[0] != vwt_y.shape[1]:
            print(f"[WARN] year={y}: VWTmat is not square: {vwt_y.shape}. row/colの向きを要確認")
        vwt_y = vwt_y[:n, :n]

        import_by_country = np.nansum(vwt_y, axis=0)
        export_by_country = np.nansum(vwt_y, axis=1)
        df_y = pd.DataFrame(
            {
                "iso3_raw": cl_n["_iso3"].values,
                "import": import_by_country,
                "export": export_by_country,
            }
        )
        if y <= int(ussr_until_year):
            df_y["iso3"] = df_y["iso3_raw"].replace({k: "RUS" for k in USSR_ALIASES})
        else:
            df_y["iso3"] = df_y["iso3_raw"]

        df_y = df_y[df_y["iso3"].notna()].copy()
        df_y = df_y.groupby("iso3", as_index=False)[["import", "export"]].sum()
        df_y["net"] = df_y["import"] - df_y["export"]

        pop_y = int(population_year) if population_year is not None else int(y)
        if pop_y not in pop_cache:
            pop_cache[pop_y] = extract_population_by_iso3_for_year(
                population_df=population_df,
                year=pop_y,
                iso3_col=pop_iso3_col,
                year_col=pop_year_col,
                pop_col=pop_value_col,
                ussr_until_year=ussr_until_year,
            )

        df_y = df_y.merge(pop_cache[pop_y], on="iso3", how="left")
        valid_pop = np.isfinite(df_y["population"]) & (df_y["population"] > 0)
        df_y["net_pc"] = np.where(valid_pop, df_y["net"] / df_y["population"], np.nan)
        netpc_by_year[y] = df_y[["iso3", "net_pc"]].copy()

        vals = df_y["net_pc"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            global_abs_vals.append(np.abs(vals))

    if len(global_abs_vals) == 0:
        raise ValueError("全年で net_pc の有効値が無い（人口マージ失敗 or population<=0）。")
    global_abs_vals = np.concatenate(global_abs_vals)

    if clip_quantile is not None:
        q = float(clip_quantile)
        if not (0 < q <= 1):
            raise ValueError("clip_quantile must be in (0, 1].")
        vmax = float(np.nanquantile(global_abs_vals, q))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.nanmax(global_abs_vals))
    else:
        vmax = float(np.nanmax(global_abs_vals))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    print(f"Global per-capita scale: vmax={vmax:g} (clip_quantile={clip_quantile})")
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_axis_off()
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(f"net per capita (import - export) [{percapita_unit}]")

    def update(frame_i: int):
        y = years[frame_i]
        ax.cla()
        ax.set_axis_off()
        pop_label = int(population_year) if population_year is not None else y
        ax.set_title(
            f"Virtual-water net per capita (import - export) [{percapita_unit}] | {y} (pop={pop_label}, crop={crop})",
            fontsize=14,
        )

        m = world.merge(netpc_by_year[y], left_on="_iso3", right_on="iso3", how="left")
        m.plot(ax=ax, color=missing_color, linewidth=0.0, edgecolor="none", zorder=1)

        has = np.isfinite(m["net_pc"].astype(float))
        if has.any():
            m.loc[has].plot(
                ax=ax,
                column="net_pc",
                cmap=cmap,
                norm=norm,
                linewidth=0.0,
                edgecolor="none",
                legend=False,
                zorder=2,
            )

        m.boundary.plot(ax=ax, color=border_color, linewidth=border_lw, zorder=3)

        if show_missing_legend:
            from matplotlib.patches import Patch

            ax.legend(
                handles=[Patch(facecolor=missing_color, edgecolor="none", label="No data / no match / no population")],
                loc="lower left",
                frameon=True,
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        return []

    anim = FuncAnimation(fig, update, frames=len(years), interval=interval_ms, blit=False)

    if out_mp4:
        anim.save(out_mp4, writer="ffmpeg", fps=fps, dpi=dpi)
        print("Saved MP4:", out_mp4)

    if out_gif:
        anim.save(out_gif, writer="pillow", fps=fps, dpi=dpi)
        print("Saved GIF:", out_gif)

    return anim


def save_vwt_net_and_percapita_npy_by_year(
    crop: int,
    year_start: int,
    year_end: int,
    *,
    country_list_xlsx: str,
    vwt_npy_template: str,
    population_df: pd.DataFrame,
    out_dir: str,
    pop_iso3_col: Optional[str] = None,
    pop_year_col: Optional[str] = None,
    pop_value_col: Optional[str] = None,
    population_year: Optional[int] = None,
    net_npy_template: str = "net_{crop}_{year}.npy",
    net_pc_npy_template: str = "net_pc_{crop}_{year}.npy",
    iso3_order_npy: str = "iso3_order.npy",
    save_iso3_each_year: bool = False,
    iso3_yearly_npy_template: str = "iso3_{crop}_{year}.npy",
    ussr_until_year: int = 1991,
) -> pd.DataFrame:
    years = list(range(int(year_start), int(year_end) + 1))
    if len(years) == 0:
        raise ValueError("year_start <= year_end になるように指定してください。")

    cl = pd.read_excel(country_list_xlsx)
    iso3_col = _find_column(cl.columns, exact=["iso3", "iso_a3", "iso-3", "iso_3"], contains=["iso3"])
    if iso3_col is None:
        raise ValueError(f"country_list に ISO3 列が見つからない。columns={list(cl.columns)}")

    vwt0 = np.load(vwt_npy_template.format(crop=crop, year=years[0])).astype(float)
    if vwt0.ndim != 2:
        raise ValueError(f"VWTmat must be 2D. got shape={vwt0.shape}")

    n = min(vwt0.shape[0], vwt0.shape[1])
    if vwt0.shape[0] != vwt0.shape[1]:
        print(f"[WARN] VWTmat is not square: {vwt0.shape}. row/colの向きを要確認")
    if len(cl) < n:
        raise ValueError(f"country_list rows ({len(cl)}) < VWT size ({n}). country_listの行数が足りない。")

    cl_n = cl.iloc[:n].copy()
    cl_n["_iso3_raw"] = cl_n[iso3_col].astype(str)
    cl_n["_iso3"] = _clean_iso3(cl_n["_iso3_raw"]).replace(ISO3_FIX)

    country_name_col = _find_column(cl_n.columns, exact=["country name", "country", "name"], contains=["country"])
    if country_name_col is None:
        country_name_col = cl_n.columns[0]
    cl_n["_country_upper"] = cl_n[country_name_col].astype(str).str.strip().str.upper()
    missing_iso = cl_n["_iso3"].isna()
    cl_n.loc[missing_iso & cl_n["_country_upper"].isin(USSR_NAME_ALIASES), "_iso3"] = "USSR"

    # USSR系をRUSへ寄せた共通順序で保存する
    iso3_order = (
        cl_n["_iso3"]
        .replace({k: "RUS" for k in USSR_ALIASES})
        .dropna()
        .astype(str)
        .sort_values()
        .drop_duplicates()
        .to_numpy()
    )
    if iso3_order.size == 0:
        raise ValueError("iso3_order が空です。country_list のISO3を確認してください。")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / iso3_order_npy, iso3_order)

    pop_cache: dict[int, pd.DataFrame] = {}
    summary_rows = []

    for y in years:
        vwt_y = np.load(vwt_npy_template.format(crop=crop, year=y)).astype(float)
        if vwt_y.ndim != 2:
            raise ValueError(f"年{y}のVWTmatが2次元でない: shape={vwt_y.shape}")
        if min(vwt_y.shape[0], vwt_y.shape[1]) < n:
            raise ValueError(f"年{y}のVWTサイズが小さい: {vwt_y.shape}, expected at least {n}x{n}")
        if vwt_y.shape[0] != vwt_y.shape[1]:
            print(f"[WARN] year={y}: VWTmat is not square: {vwt_y.shape}. row/colの向きを要確認")
        vwt_y = vwt_y[:n, :n]

        import_by_country = np.nansum(vwt_y, axis=0)
        export_by_country = np.nansum(vwt_y, axis=1)
        df_y = pd.DataFrame(
            {
                "iso3_raw": cl_n["_iso3"].values,
                "import": import_by_country,
                "export": export_by_country,
            }
        )

        if int(y) <= int(ussr_until_year):
            df_y["iso3"] = df_y["iso3_raw"].replace({k: "RUS" for k in USSR_ALIASES})
        else:
            df_y["iso3"] = df_y["iso3_raw"]

        df_y = df_y[df_y["iso3"].notna()].copy()
        df_y = df_y.groupby("iso3", as_index=False)[["import", "export"]].sum()
        df_y["net"] = df_y["import"] - df_y["export"]

        pop_y = int(population_year) if population_year is not None else int(y)
        if pop_y not in pop_cache:
            pop_cache[pop_y] = extract_population_by_iso3_for_year(
                population_df=population_df,
                year=pop_y,
                iso3_col=pop_iso3_col,
                year_col=pop_year_col,
                pop_col=pop_value_col,
                ussr_until_year=ussr_until_year,
            )
        df_y = df_y.merge(pop_cache[pop_y], on="iso3", how="left")

        valid_pop = np.isfinite(df_y["population"]) & (df_y["population"] > 0)
        df_y["net_pc"] = np.where(valid_pop, df_y["net"] / df_y["population"], np.nan)

        aligned = df_y.set_index("iso3").reindex(iso3_order)
        net_arr = aligned["net"].to_numpy(dtype=float)
        net_pc_arr = aligned["net_pc"].to_numpy(dtype=float)

        net_name = net_npy_template.format(crop=crop, year=y)
        net_pc_name = net_pc_npy_template.format(crop=crop, year=y)
        np.save(out_path / net_name, net_arr)
        np.save(out_path / net_pc_name, net_pc_arr)

        if save_iso3_each_year:
            iso_name = iso3_yearly_npy_template.format(crop=crop, year=y)
            np.save(out_path / iso_name, iso3_order)

        summary_rows.append(
            {
                "year": int(y),
                "pop_year": int(pop_y),
                "n_iso3": int(iso3_order.size),
                "n_valid_population": int(np.isfinite(net_pc_arr).sum()),
                "net_npy": str(out_path / net_name),
                "net_pc_npy": str(out_path / net_pc_name),
            }
        )

    summary = pd.DataFrame(summary_rows)
    print(f"Saved yearly npy: years={years[0]}-{years[-1]}, files={len(years) * 2}")
    print(f"Saved iso3 order: {out_path / iso3_order_npy} (n={iso3_order.size})")
    return summary


def _normalize_iso3_key(iso3: str) -> str:
    key = str(iso3).strip().upper()
    key = ISO3_FIX.get(key, key)
    if key in USSR_ALIASES:
        key = "RUS"
    return key


def build_vwt_trade_npy_store_all_crops(
    crops: Iterable[int],
    year_start: int,
    year_end: int,
    *,
    country_list_xlsx: str,
    vwt_npy_template: str,
    population_df: pd.DataFrame,
    out_dir: str,
    pop_iso3_col: Optional[str] = None,
    pop_year_col: Optional[str] = None,
    pop_value_col: Optional[str] = None,
    population_year: Optional[int] = None,
    ussr_until_year: int = 1991,
) -> pd.DataFrame:
    years = list(range(int(year_start), int(year_end) + 1))
    crop_ids = [int(c) for c in crops]
    if len(years) == 0:
        raise ValueError("year_start <= year_end になるように指定してください。")
    if len(crop_ids) == 0:
        raise ValueError("crops が空です。")

    cl = pd.read_excel(country_list_xlsx)
    iso3_col = _find_column(cl.columns, exact=["iso3", "iso_a3", "iso-3", "iso_3"], contains=["iso3"])
    if iso3_col is None:
        raise ValueError(f"country_list に ISO3 列が見つからない。columns={list(cl.columns)}")

    vwt0 = np.load(vwt_npy_template.format(crop=crop_ids[0], year=years[0])).astype(float)
    if vwt0.ndim != 2:
        raise ValueError(f"VWTmat must be 2D. got shape={vwt0.shape}")
    n = min(vwt0.shape[0], vwt0.shape[1])
    if len(cl) < n:
        raise ValueError(f"country_list rows ({len(cl)}) < VWT size ({n}). country_listの行数が足りない。")

    cl_n = cl.iloc[:n].copy()
    cl_n["_iso3_raw"] = cl_n[iso3_col].astype(str)
    cl_n["_iso3"] = _clean_iso3(cl_n["_iso3_raw"]).replace(ISO3_FIX)
    country_name_col = _find_column(cl_n.columns, exact=["country name", "country", "name"], contains=["country"])
    if country_name_col is None:
        country_name_col = cl_n.columns[0]
    cl_n["_country_upper"] = cl_n[country_name_col].astype(str).str.strip().str.upper()
    missing_iso = cl_n["_iso3"].isna()
    cl_n.loc[missing_iso & cl_n["_country_upper"].isin(USSR_NAME_ALIASES), "_iso3"] = "USSR"

    iso3_order = (
        cl_n["_iso3"]
        .replace({k: "RUS" for k in USSR_ALIASES})
        .dropna()
        .astype(str)
        .sort_values()
        .drop_duplicates()
        .to_numpy()
    )
    if iso3_order.size == 0:
        raise ValueError("iso3_order が空です。country_list のISO3を確認してください。")

    n_crops = len(crop_ids)
    n_years = len(years)
    n_iso3 = int(iso3_order.size)

    export_total = np.full((n_crops, n_years, n_iso3), np.nan, dtype=float)
    import_total = np.full((n_crops, n_years, n_iso3), np.nan, dtype=float)
    net_total = np.full((n_crops, n_years, n_iso3), np.nan, dtype=float)
    export_pc = np.full((n_crops, n_years, n_iso3), np.nan, dtype=float)
    import_pc = np.full((n_crops, n_years, n_iso3), np.nan, dtype=float)
    net_pc = np.full((n_crops, n_years, n_iso3), np.nan, dtype=float)
    population_used = np.full((n_crops, n_years, n_iso3), np.nan, dtype=float)

    pop_cache: dict[int, pd.DataFrame] = {}
    summary_rows = []

    for ci, crop in enumerate(crop_ids):
        for yi, y in enumerate(years):
            vwt_y = np.load(vwt_npy_template.format(crop=crop, year=y)).astype(float)
            if vwt_y.ndim != 2:
                raise ValueError(f"crop={crop}, year={y} のVWTmatが2次元でない: shape={vwt_y.shape}")
            if min(vwt_y.shape[0], vwt_y.shape[1]) < n:
                raise ValueError(f"crop={crop}, year={y} のVWTサイズが小さい: {vwt_y.shape}, expected at least {n}x{n}")
            vwt_y = vwt_y[:n, :n]

            import_by_country = np.nansum(vwt_y, axis=0)
            export_by_country = np.nansum(vwt_y, axis=1)
            df_y = pd.DataFrame(
                {
                    "iso3_raw": cl_n["_iso3"].values,
                    "import": import_by_country,
                    "export": export_by_country,
                }
            )
            if int(y) <= int(ussr_until_year):
                df_y["iso3"] = df_y["iso3_raw"].replace({k: "RUS" for k in USSR_ALIASES})
            else:
                df_y["iso3"] = df_y["iso3_raw"]

            df_y = df_y[df_y["iso3"].notna()].copy()
            df_y = df_y.groupby("iso3", as_index=False)[["import", "export"]].sum()
            df_y["net"] = df_y["import"] - df_y["export"]

            pop_y = int(population_year) if population_year is not None else int(y)
            if pop_y not in pop_cache:
                pop_cache[pop_y] = extract_population_by_iso3_for_year(
                    population_df=population_df,
                    year=pop_y,
                    iso3_col=pop_iso3_col,
                    year_col=pop_year_col,
                    pop_col=pop_value_col,
                    ussr_until_year=ussr_until_year,
                )
            df_y = df_y.merge(pop_cache[pop_y], on="iso3", how="left")

            valid_pop = np.isfinite(df_y["population"]) & (df_y["population"] > 0)
            df_y["export_pc"] = np.where(valid_pop, df_y["export"] / df_y["population"], np.nan)
            df_y["import_pc"] = np.where(valid_pop, df_y["import"] / df_y["population"], np.nan)
            df_y["net_pc"] = np.where(valid_pop, df_y["net"] / df_y["population"], np.nan)

            aligned = df_y.set_index("iso3").reindex(iso3_order)

            export_total[ci, yi, :] = aligned["export"].to_numpy(dtype=float)
            import_total[ci, yi, :] = aligned["import"].to_numpy(dtype=float)
            net_total[ci, yi, :] = aligned["net"].to_numpy(dtype=float)
            export_pc[ci, yi, :] = aligned["export_pc"].to_numpy(dtype=float)
            import_pc[ci, yi, :] = aligned["import_pc"].to_numpy(dtype=float)
            net_pc[ci, yi, :] = aligned["net_pc"].to_numpy(dtype=float)
            population_used[ci, yi, :] = aligned["population"].to_numpy(dtype=float)

            summary_rows.append(
                {
                    "crop": int(crop),
                    "year": int(y),
                    "pop_year": int(pop_y),
                    "n_iso3": int(n_iso3),
                    "n_valid_population": int(np.isfinite(net_pc[ci, yi, :]).sum()),
                }
            )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / "iso3_order.npy", iso3_order.astype(str))
    np.save(out_path / "crop_ids.npy", np.array(crop_ids, dtype=int))
    np.save(out_path / "years.npy", np.array(years, dtype=int))
    np.save(out_path / "export_total.npy", export_total)
    np.save(out_path / "import_total.npy", import_total)
    np.save(out_path / "net_total.npy", net_total)
    np.save(out_path / "export_pc.npy", export_pc)
    np.save(out_path / "import_pc.npy", import_pc)
    np.save(out_path / "net_pc.npy", net_pc)
    np.save(out_path / "population_used.npy", population_used)

    summary = pd.DataFrame(summary_rows)
    print(
        "Saved trade store:",
        f"crops={n_crops}, years={n_years}, iso3={n_iso3},",
        f"arrays=7 + index(3) at {out_path}",
    )
    return summary


def load_vwt_trade_npy_store(store_dir: str) -> dict[str, Any]:
    path = Path(store_dir)
    store: dict[str, Any] = {
        "iso3_order": np.load(path / "iso3_order.npy", allow_pickle=False),
        "crop_ids": np.load(path / "crop_ids.npy", allow_pickle=False),
        "years": np.load(path / "years.npy", allow_pickle=False),
        "export_total": np.load(path / "export_total.npy", allow_pickle=False),
        "import_total": np.load(path / "import_total.npy", allow_pickle=False),
        "net_total": np.load(path / "net_total.npy", allow_pickle=False),
        "export_pc": np.load(path / "export_pc.npy", allow_pickle=False),
        "import_pc": np.load(path / "import_pc.npy", allow_pickle=False),
        "net_pc": np.load(path / "net_pc.npy", allow_pickle=False),
        "population_used": np.load(path / "population_used.npy", allow_pickle=False),
    }
    store["iso3_to_idx"] = {str(v): i for i, v in enumerate(store["iso3_order"].tolist())}
    store["crop_to_idx"] = {int(v): i for i, v in enumerate(store["crop_ids"].tolist())}
    store["year_to_idx"] = {int(v): i for i, v in enumerate(store["years"].tolist())}
    return store


def get_vwt_trade_record(
    store: dict[str, Any],
    *,
    crop: int,
    year: int,
    iso3: str,
) -> dict[str, float]:
    c = int(crop)
    y = int(year)
    key = _normalize_iso3_key(iso3)

    if c not in store["crop_to_idx"]:
        raise KeyError(f"crop={c} が store にありません。")
    if y not in store["year_to_idx"]:
        raise KeyError(f"year={y} が store にありません。")
    if key not in store["iso3_to_idx"]:
        raise KeyError(f"iso3={key} が store にありません。")

    ci = store["crop_to_idx"][c]
    yi = store["year_to_idx"][y]
    ii = store["iso3_to_idx"][key]

    return {
        "export_total": float(store["export_total"][ci, yi, ii]),
        "import_total": float(store["import_total"][ci, yi, ii]),
        "net_total": float(store["net_total"][ci, yi, ii]),
        "export_pc": float(store["export_pc"][ci, yi, ii]),
        "import_pc": float(store["import_pc"][ci, yi, ii]),
        "net_pc": float(store["net_pc"][ci, yi, ii]),
        "population_used": float(store["population_used"][ci, yi, ii]),
    }
