#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import uuid
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from netCDF4 import Dataset


FILENAME_RE = re.compile(
    r"^cygma_"
    r"(?P<model>[^_]+)_"
    r"(?P<rcp>[^_]+)_"
    r"(?P<ssp>[^_]+)_"
    r"(?P<crop>[^_]+)_"
    r"(?P<system>[^_]+)_"
    r"(?P<variable>[^_]+)_"
    r"(?P<period_start>\d{4})_"
    r"(?P<period_end>\d{4})"
    r"\.nc$"
)


EXPECTED_MODELS = [
    "gfdl-esm4",
    "ipsl-cm6a-lr",
    "mri-esm2-0",
    "mpi-esm1-2-hr",
    "ukesm1-0-ll",
]


KeyType = Tuple[str, str, str, str, str, str, str]


def parse_args() -> argparse.Namespace:
    default_input = Path.cwd().parent / "Iizumi" / "wheat"
    parser = argparse.ArgumentParser(
        description=(
            "Build multi-GCM ensemble mean NetCDF files for wheat data. "
            "If any model is missing at a grid/time point, output is set to fill value."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help="Directory that contains cygma_*.nc files (default: ../Iizumi/wheat).",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="ensemble_mean_5gcm",
        help="Subdirectory (created under input-dir) for generated files.",
    )
    parser.add_argument(
        "--ensemble-model-name",
        type=str,
        default="ensemble-5gcm",
        help="Token used in output filename at climate-model position.",
    )
    return parser.parse_args()


def group_files(input_dir: Path) -> Dict[KeyType, Dict[str, Path]]:
    groups: Dict[KeyType, Dict[str, Path]] = defaultdict(dict)
    files = sorted(input_dir.glob("cygma_*.nc"))
    if not files:
        raise FileNotFoundError(f"No cygma_*.nc files found in: {input_dir}")

    for f in files:
        m = FILENAME_RE.match(f.name)
        if not m:
            continue
        model = m.group("model")
        key: KeyType = (
            m.group("rcp"),
            m.group("ssp"),
            m.group("crop"),
            m.group("system"),
            m.group("variable"),
            m.group("period_start"),
            m.group("period_end"),
        )
        groups[key][model] = f

    return groups


def assert_group_integrity(groups: Dict[KeyType, Dict[str, Path]]) -> None:
    if not groups:
        raise RuntimeError("No valid groups parsed from file names.")

    expected = set(EXPECTED_MODELS)
    errors: List[str] = []
    for key, model_map in sorted(groups.items()):
        models = set(model_map.keys())
        missing = sorted(expected - models)
        extra = sorted(models - expected)
        if missing or extra:
            errors.append(
                f"key={key} missing={missing if missing else '-'} extra={extra if extra else '-'}"
            )
    if errors:
        preview = "\n".join(errors[:10])
        raise RuntimeError(
            "Model set mismatch found in grouped files.\n"
            f"{preview}\n(total mismatch groups: {len(errors)})"
        )


def _sentinel_values(var) -> List[float]:
    vals: List[float] = []
    for attr in ("_FillValue", "missing_value"):
        if hasattr(var, attr):
            val = getattr(var, attr)
            try:
                vals.append(float(np.array(val).item()))
            except Exception:
                pass
    return vals


def _any_missing(arr, sentinels: List[float]) -> np.ndarray:
    if np.ma.isMaskedArray(arr):
        mask = np.ma.getmaskarray(arr).copy()
        values = np.asarray(arr, dtype=np.float64)
    else:
        values = np.asarray(arr, dtype=np.float64)
        mask = np.zeros(values.shape, dtype=bool)
    for s in sentinels:
        mask |= np.isclose(values, s, rtol=0.0, atol=0.0)
    return mask


def _extract_values(arr) -> np.ndarray:
    if np.ma.isMaskedArray(arr):
        return np.asarray(arr.data, dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def _copy_coord_var(src_ds: Dataset, dst_ds: Dataset, name: str) -> None:
    src_var = src_ds.variables[name]
    dst_var = dst_ds.createVariable(name, src_var.dtype, src_var.dimensions)
    dst_var[:] = src_var[:]
    for attr in src_var.ncattrs():
        dst_var.setncattr(attr, src_var.getncattr(attr))


def _validate_alignment(template_ds: Dataset, candidate_ds: Dataset, path: Path) -> None:
    for dim_name in ("time", "lat", "lon"):
        if dim_name not in candidate_ds.dimensions:
            raise RuntimeError(f"{path.name}: missing dimension '{dim_name}'")
        if len(template_ds.dimensions[dim_name]) != len(candidate_ds.dimensions[dim_name]):
            raise RuntimeError(f"{path.name}: dimension length mismatch in '{dim_name}'")
    for var_name in ("time", "lat", "lon", "var"):
        if var_name not in candidate_ds.variables:
            raise RuntimeError(f"{path.name}: missing variable '{var_name}'")
    for coord in ("time", "lat", "lon"):
        a = np.asarray(template_ds.variables[coord][:], dtype=np.float64)
        b = np.asarray(candidate_ds.variables[coord][:], dtype=np.float64)
        if not np.allclose(a, b, rtol=0.0, atol=0.0):
            raise RuntimeError(f"{path.name}: coordinate mismatch in '{coord}'")
    if template_ds.variables["var"].dimensions != candidate_ds.variables["var"].dimensions:
        raise RuntimeError(f"{path.name}: var dimension order mismatch")


def write_ensemble_file(
    model_paths: Dict[str, Path],
    output_path: Path,
    ensemble_model_name: str,
    key: KeyType,
) -> None:
    rcp, ssp, crop, system, variable, period_start, period_end = key
    ordered_paths = [model_paths[m] for m in EXPECTED_MODELS]

    # netCDF4 in this environment cannot directly open Japanese path strings.
    # Workaround: copy input files and output file through an ASCII temp directory.
    temp_root = Path(tempfile.gettempdir()) / f"wheat_ens_{uuid.uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_in_paths: List[Path] = []
    temp_out_path = temp_root / "out.nc"

    try:
        for i, p in enumerate(ordered_paths):
            tmp_p = temp_root / f"in_{i}.nc"
            shutil.copyfile(p, tmp_p)
            temp_in_paths.append(tmp_p)

        with ExitStack() as stack:
            src_datasets = [stack.enter_context(Dataset(str(p), mode="r")) for p in temp_in_paths]
            template_ds = src_datasets[0]
            for ds, p in zip(src_datasets[1:], ordered_paths[1:]):
                _validate_alignment(template_ds, ds, p)

            with Dataset(str(temp_out_path), mode="w", format="NETCDF4") as out_ds:
                for dim_name in ("lon", "lat", "time"):
                    dim = template_ds.dimensions[dim_name]
                    out_ds.createDimension(dim_name, None if dim.isunlimited() else len(dim))

                _copy_coord_var(template_ds, out_ds, "lon")
                _copy_coord_var(template_ds, out_ds, "lat")
                _copy_coord_var(template_ds, out_ds, "time")

                src_var0 = template_ds.variables["var"]
                out_fill = float(getattr(src_var0, "_FillValue", -999000000.0))
                out_var = out_ds.createVariable(
                    "var",
                    src_var0.dtype,
                    src_var0.dimensions,
                    fill_value=out_fill,
                    zlib=True,
                    complevel=1,
                )
                for attr in src_var0.ncattrs():
                    if attr == "_FillValue":
                        continue
                    out_var.setncattr(attr, src_var0.getncattr(attr))

                for attr in template_ds.ncattrs():
                    out_ds.setncattr(attr, template_ds.getncattr(attr))
                created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                history_msg = (
                    f"{created_at} created by ensemble averaging models={','.join(EXPECTED_MODELS)}; "
                    f"missing rule=any-missing->fill; ensemble_model_name={ensemble_model_name}"
                )
                if "history" in out_ds.ncattrs():
                    prev = str(out_ds.getncattr("history"))
                    out_ds.setncattr("history", f"{history_msg}\n{prev}")
                else:
                    out_ds.setncattr("history", history_msg)

                time_len = len(template_ds.dimensions["time"])
                sentinels_each = [_sentinel_values(ds.variables["var"]) for ds in src_datasets]
                for t in range(time_len):
                    arrs = [ds.variables["var"][t, :, :] for ds in src_datasets]
                    missing_mask = np.zeros(arrs[0].shape, dtype=bool)
                    vals: List[np.ndarray] = []
                    for arr, sentinels in zip(arrs, sentinels_each):
                        missing_mask |= _any_missing(arr, sentinels)
                        vals.append(_extract_values(arr))
                    stacked = np.stack(vals, axis=0)
                    mean_vals = np.mean(stacked, axis=0, dtype=np.float64)
                    mean_vals[missing_mask] = out_fill
                    out_var[t, :, :] = mean_vals.astype(np.float32)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(temp_out_path, output_path)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    print(f"[OK] {output_path.name}")
    print(
        "     key="
        f"{rcp}_{ssp}_{crop}_{system}_{variable}_{period_start}_{period_end}"
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (input_dir / args.output_subdir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    groups = group_files(input_dir)
    assert_group_integrity(groups)

    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"group_count: {len(groups)}")
    print(f"models: {EXPECTED_MODELS}")

    for key in sorted(groups.keys()):
        rcp, ssp, crop, system, variable, period_start, period_end = key
        out_name = (
            f"cygma_{args.ensemble_model_name}_{rcp}_{ssp}_{crop}_{system}_"
            f"{variable}_{period_start}_{period_end}.nc"
        )
        out_path = output_dir / out_name
        write_ensemble_file(
            model_paths=groups[key],
            output_path=out_path,
            ensemble_model_name=args.ensemble_model_name,
            key=key,
        )

    print(f"Completed. Created {len(groups)} ensemble files in: {output_dir}")


if __name__ == "__main__":
    main()
