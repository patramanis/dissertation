from __future__ import annotations

from pathlib import Path

import pandas as pd

from ModularMonolith.data.train_window import TRAIN_DATE_END, TRAIN_DATE_START


BASE = Path("ModularMonolith") / "data"
RAW2 = BASE / "raw_data_2"
RAW3 = BASE / "raw_data_3"
PROCESSED = BASE / "processed_data_1"


def _read_dates_parquet(path: Path) -> pd.DatetimeIndex:
    df = pd.read_parquet(path, engine="pyarrow", columns=["Date"])
    d = pd.to_datetime(df["Date"], errors="raise").dt.normalize().dt.tz_localize(None)
    return pd.DatetimeIndex(d)


def _check_folder(folder: Path, anchor: pd.DatetimeIndex) -> list[str]:
    problems: list[str] = []
    for p in sorted(folder.glob("*.parquet")):
        d = _read_dates_parquet(p)
        if not d.equals(anchor):
            problems.append(f"{folder.name}/{p.name}")
    return problems


def main() -> None:
    spdr2 = RAW2 / "SPDR.parquet"
    if not spdr2.exists():
        raise FileNotFoundError(spdr2)

    anchor = _read_dates_parquet(spdr2)
    anchor_train = anchor[(anchor >= TRAIN_DATE_START) & (anchor <= TRAIN_DATE_END)]

    required = [RAW2 / "SPY.parquet", RAW3 / "SPY.parquet"]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    bad_raw2 = _check_folder(RAW2, anchor)
    bad_raw3 = _check_folder(RAW3, anchor)

    bad_proc: list[str] = []
    for p in sorted(PROCESSED.glob("features_h*.parquet")):
        d = _read_dates_parquet(p)
        if len(d) == 0:
            bad_proc.append(f"processed_data_1/{p.name}")
            continue

        d_unique = pd.DatetimeIndex(pd.to_datetime(pd.Series(d).unique(), errors="raise")).sort_values()
        if len(d_unique) != len(anchor_train):
            bad_proc.append(f"processed_data_1/{p.name}")
            continue
        if set(d_unique.tolist()) != set(anchor_train.tolist()):
            bad_proc.append(f"processed_data_1/{p.name}")
            continue

    if bad_raw2 or bad_raw3 or bad_proc:
        msg = []
        if bad_raw2:
            msg.append("raw_data_2 mismatched: " + ", ".join(bad_raw2))
        if bad_raw3:
            msg.append("raw_data_3 mismatched: " + ", ".join(bad_raw3))
        if bad_proc:
            msg.append("processed_data_1 date issues: " + ", ".join(bad_proc))
        raise SystemExit("\n".join(msg))

    print(f"Aligned OK: trading days={len(anchor)} raw_data_2 files={len(list(RAW2.glob('*.parquet')))} raw_data_3 files={len(list(RAW3.glob('*.parquet')))}")


if __name__ == "__main__":
    main()
