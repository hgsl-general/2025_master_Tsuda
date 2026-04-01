#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from netCDF4 import Dataset


COMBOS: List[Tuple[str, str]] = [
    ("ssp126", "ssp1"),
    ("ssp370", "ssp3"),
    ("ssp585", "ssp3"),
]
SYSTEMS = ["irri", "rain"]
CROPS = ["rice", "wheat"]


def parse_args() -> argparse.Namespace:
    default_root = Path.cwd().parent / "Iizumi"
    parser = argparse.ArgumentParser(
        description=(
            "Create time-series GIF animations from ensemble mean NetCDF files "
            "for selected scenarios and systems."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Root directory that contains rice/ and wheat/ (default: ../Iizumi).",
    )
    parser.add_argument(
        "--ensemble-subdir",
        type=str,
        default="ensemble_mean_5gcm",
        help="Subdirectory name where ensemble files are stored.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="animations_ensemble_mean_5gcm",
        help="Subdirectory name created under each crop directory for GIF outputs.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Frames per second for GIF.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="DPI for saved GIF frames.",
    )
    return parser.parse_args()


def _copy_to_temp(src: Path, temp_dir: Path, name: str) -> Path:
    dst = temp_dir / name
    shutil.copyfile(src, dst)
    return dst


def _to_data_and_mask(var_t, var_obj) -> Tuple[np.ndarray, np.ndarray]:
    if np.ma.isMaskedArray(var_t):
        data = np.asarray(var_t.data, dtype=np.float64)
        missing = np.ma.getmaskarray(var_t).copy()
    else:
        data = np.asarray(var_t, dtype=np.float64)
        missing = np.zeros(data.shape, dtype=bool)

    for attr in ("_FillValue", "missing_value"):
        if hasattr(var_obj, attr):
            fv = float(np.array(getattr(var_obj, attr)).item())
            missing |= np.isclose(data, fv, rtol=0.0, atol=0.0)
    return data, missing


def _years_from_time(time_var) -> np.ndarray:
    units = str(getattr(time_var, "units", ""))
    if "since" in units:
        # Expected pattern: "months since YYYY-MM-DD ..."
        after = units.split("since", 1)[1].strip()
        base_year = int(after.split("-", 1)[0])
    else:
        base_year = 1981
    tvals = np.asarray(time_var[:], dtype=np.float64)
    years = base_year + np.rint(tvals / 12.0).astype(int)
    return years


def build_animation(input_nc: Path, output_gif: Path, fps: int, dpi: int) -> None:
    temp_root = Path(tempfile.gettempdir()) / f"anim_{uuid.uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        temp_nc = _copy_to_temp(input_nc, temp_root, "in.nc")
        with Dataset(str(temp_nc), "r") as nc:
            lat = np.asarray(nc.variables["lat"][:], dtype=np.float64)
            lon = np.asarray(nc.variables["lon"][:], dtype=np.float64)
            var = nc.variables["var"]
            years = _years_from_time(nc.variables["time"])
            n_time = len(nc.dimensions["time"])

            # 0..360 -> -180..180 ordering for map-like display.
            lon_wrap = ((lon + 180.0) % 360.0) - 180.0
            order = np.argsort(lon_wrap)
            lon_plot = lon_wrap[order]

            # Get robust color limits from all valid cells in this file.
            all_vals = []
            for t in range(n_time):
                d, m = _to_data_and_mask(var[t, :, :], var)
                d = d.copy()
                d[m] = np.nan
                valid = d[np.isfinite(d)]
                if valid.size:
                    all_vals.append(valid)
            if not all_vals:
                raise RuntimeError(f"No valid values found in {input_nc.name}")
            cat = np.concatenate(all_vals)
            vmin = float(np.nanpercentile(cat, 2))
            vmax = float(np.nanpercentile(cat, 98))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                vmin = float(np.nanmin(cat))
                vmax = float(np.nanmax(cat))

            fig, ax = plt.subplots(figsize=(12, 4.8))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("#f0f0f0")

            d0, m0 = _to_data_and_mask(var[0, :, :], var)
            d0 = d0[:, order].copy()
            m0 = m0[:, order]
            d0[m0] = np.nan
            mesh = ax.pcolormesh(
                lon_plot,
                lat,
                d0,
                shading="auto",
                cmap="YlGn",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlim(-180, 180)
            ax.set_ylim(-60, 80)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            title = ax.set_title(f"{input_nc.stem} (year={years[0]}, time index=0)")
            cbar = fig.colorbar(mesh, ax=ax, label="Yield")
            cbar.ax.tick_params(labelsize=9)
            fig.tight_layout()

            # pcolormesh update expects flattened array.
            ny, nx = d0.shape

            def update(frame: int):
                d, m = _to_data_and_mask(var[frame, :, :], var)
                d = d[:, order].copy()
                m = m[:, order]
                d[m] = np.nan
                mesh.set_array(d.ravel())
                title.set_text(f"{input_nc.stem} (year={years[frame]}, time index={frame})")
                return (mesh, title)

            anim = FuncAnimation(
                fig,
                update,
                frames=n_time,
                interval=max(1, int(1000 / fps)),
                blit=False,
            )

            output_gif.parent.mkdir(parents=True, exist_ok=True)
            writer = PillowWriter(fps=fps)
            anim.save(str(output_gif), writer=writer, dpi=dpi)
            plt.close(fig)
            print(f"[OK] {output_gif}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def iter_targets(root: Path, ensemble_subdir: str) -> Iterable[Tuple[Path, Path]]:
    for crop in CROPS:
        src_dir = root / crop / ensemble_subdir
        for rcp, ssp in COMBOS:
            for sys in SYSTEMS:
                in_name = f"cygma_ensemble-5gcm_{rcp}_{ssp}_{crop}_{sys}_yld_1981_2100.nc"
                yield src_dir / in_name, root / crop


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    targets = list(iter_targets(root, args.ensemble_subdir))
    missing = [p for p, _ in targets if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input files:\n" + "\n".join(str(p) for p in missing)
        )

    print(f"root: {root}")
    print(f"target_count: {len(targets)}")
    for input_nc, crop_dir in targets:
        out_dir = crop_dir / args.output_subdir
        out_name = input_nc.stem + ".gif"
        out_gif = out_dir / out_name
        build_animation(input_nc=input_nc, output_gif=out_gif, fps=args.fps, dpi=args.dpi)

    print("Completed all animations.")


if __name__ == "__main__":
    main()
