from __future__ import annotations
import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = (THIS_DIR.parent / "src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
import numpy as np
import pandas as pd
from cv.weights import compute_linear_decay_weights, validate_weights

log = logging.getLogger(__name__)

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLY", "XLV", "XLU"]
FORBIDDEN_PATTERNS = ["fwd_", "forward", "future", "next_", "label_", "target_"]

def _workspace_root() -> Path:
    return THIS_DIR.parent


class ReadinessChecker:    
    def __init__(self, workspace_root: Path | None = None):
        self.ws = workspace_root or _workspace_root()
        self.results: dict[int, dict[str, Any]] = {}
    
    def check_horizon(self, h: int) -> dict[str, Any]:        
        log.info("=" * 60)
        log.info("Checking horizon h=%d", h)
        log.info("=" * 60)
        
        result = {
            "horizon": h,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "checks": {},
            "overall_pass": True,
            "critical_failures": [],
            "warnings": [],
        }
        
        result["checks"]["inventory"] = self._check_inventory(h)
        
        panel_path = self.ws / "data" / "processed" / "panel" / f"panel_h{h}.csv"
        panel: pd.DataFrame | None = None
        if panel_path.exists():
            panel = pd.read_csv(panel_path, parse_dates=["Date"])
            result["checks"]["panel_integrity"] = self._check_panel_integrity(panel, h)
            result["checks"]["label_recompute"] = self._check_label_recompute(panel, h)
            result["checks"]["forbidden_features"] = self._check_forbidden_features(panel)
            result["checks"]["nan_profile"] = self._check_nan_profile(panel)
        else:
            result["checks"]["panel_integrity"] = {"passed": False, "error": "Panel file not found"}
            result["checks"]["label_recompute"] = {"passed": False, "error": "Panel file not found"}
            result["checks"]["forbidden_features"] = {"passed": False, "error": "Panel file not found"}
            result["checks"]["nan_profile"] = {"passed": False, "error": "Panel file not found"}
        
        result["checks"]["cv_specs"] = self._check_cv_specs(h)
        
        if panel is not None:
            result["checks"]["decay_weights"] = self._check_decay_weights(panel, h)
        else:
            result["checks"]["decay_weights"] = {"passed": False, "error": "Panel file not found"}
        
        for check_name, check_result in result["checks"].items():
            if not check_result.get("passed", False):
                if check_result.get("critical", False):
                    result["critical_failures"].append(check_name)
                    result["overall_pass"] = False
                else:
                    result["warnings"].append(check_name)
        
        self.results[h] = result
        return result
    
    def _check_inventory(self, h: int) -> dict[str, Any]:
        
        log.info("[1/7] Inventory check...")
        
        required_files = {
            "panel": self.ws / "data" / "processed" / "panel" / f"panel_h{h}.csv",
            "panel_manifest": self.ws / "data" / "processed" / "panel" / f"panel_h{h}_manifest.json",
            "labels": self.ws / "data" / "labels" / "excess" / f"labels_h{h}.csv",
            "prices_close": self.ws / "data" / "interim" / "label_inputs" / "prices_close.csv",
            "cv_rolling": self.ws / "configs" / "cv" / f"cv_rolling_h{h}.json",
            "cv_expanding_equal": self.ws / "configs" / "cv" / f"cv_expanding_equal_h{h}.json",
            "cv_expanding_linear_decay": self.ws / "configs" / "cv" / f"cv_expanding_linear_decay_h{h}.json",
        }
        
        found = {}
        missing = []
        
        for name, path in required_files.items():
            exists = path.exists()
            found[name] = {"path": str(path), "exists": exists}
            if not exists:
                missing.append(name)
        
        passed = len(missing) == 0
        log.info("Inventory: %s (missing: %s)", "PASS" if passed else "FAIL", missing or "none")
        
        return {
            "passed": passed,
            "critical": True,
            "files": found,
            "missing": missing,
        }
    
    def _check_panel_integrity(self, panel: pd.DataFrame, h: int) -> dict[str, Any]:
        
        log.info("[2/7] Panel integrity check...")
        
        issues = []
        
        required_cols = ["Date", "Sector", "label_excess", "y_gate"]
        missing_cols = [c for c in required_cols if c not in panel.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        n_dupes = panel.duplicated(subset=["Date", "Sector"]).sum()
        if n_dupes > 0:
            issues.append(f"Duplicate (Date, Sector) keys: {n_dupes}")
        
        group_sizes = panel.groupby("Date")["Sector"].nunique()
        min_size = group_sizes.min()
        max_size = group_sizes.max()
        if min_size != 9 or max_size != 9:
            issues.append(f"Sectors per date not exactly 9: min={min_size}, max={max_size}")
        
        actual_sectors = sorted(panel["Sector"].unique().tolist())
        if actual_sectors != sorted(SECTORS):
            issues.append(f"Unexpected sectors: {actual_sectors}")
        
        try:
            pd.to_datetime(panel["Date"])
        except Exception as e:
            issues.append(f"Date column not parseable: {e}")
        
        nan_label = panel["label_excess"].isna().sum()
        nan_ygate = panel["y_gate"].isna().sum()
        if nan_label > 0 or nan_ygate > 0:
            issues.append(f"NaN in targets: label_excess={nan_label}, y_gate={nan_ygate}")
        
        passed = len(issues) == 0
        log.info("    Panel integrity: %s (issues: %d)", "PASS" if passed else "FAIL", len(issues))
        
        return {
            "passed": passed,
            "critical": True,
            "n_rows": int(len(panel)),
            "n_cols": int(len(panel.columns)),
            "n_dates": int(panel["Date"].nunique()),
            "date_range": {
                "start": str(panel["Date"].min().date()),
                "end": str(panel["Date"].max().date()),
            },
            "sectors_per_date": {"min": int(min_size), "max": int(max_size)},
            "issues": issues,
        }
    
    def _check_label_recompute(self, panel: pd.DataFrame, h: int) -> dict[str, Any]:
        
        log.info("[3/7] Label recompute check...")
        
        prices_path = self.ws / "data" / "interim" / "label_inputs" / "prices_close.csv"
        if not prices_path.exists():
            return {"passed": False, "critical": True, "error": "prices_close.csv not found"}
        
        prices = pd.read_csv(prices_path, parse_dates=["Date"])
        px = prices.sort_values("Date").set_index("Date")
        
        denom = px.shift(1)
        numer = px.shift(-(h - 1))
        ret = numer / denom - 1.0
        
        spy_ret = ret["SPY"]
        sec_ret = ret[SECTORS]
        label_excess_recompute = sec_ret.sub(spy_ret, axis=0)
        
        mismatches = 0
        max_diff = 0.0
        sample_errors = []
        
        for _, row in panel.iterrows():
            d = pd.Timestamp(row["Date"]).normalize()
            s = str(row["Sector"])
            panel_le = float(row["label_excess"]) if not pd.isna(row["label_excess"]) else np.nan
            
            if pd.isna(panel_le):
                continue
            
            if d in label_excess_recompute.index:
                recompute_le = float(label_excess_recompute.at[d, s])  # type: ignore[arg-type]
                diff = abs(panel_le - recompute_le)
                max_diff = max(max_diff, diff)
                
                if diff > 1e-8:
                    mismatches += 1
                    if len(sample_errors) < 5:
                        sample_errors.append({
                            "date": str(d.date()),
                            "sector": s,
                            "panel": float(panel_le),
                            "recomputed": float(recompute_le),
                            "diff": float(diff),
                        })
        
        cost_bps = 10.0
        cost = cost_bps / 10_000.0
        expected_gate = (panel["label_excess"] > cost).astype(int)
        got_gate = panel["y_gate"].astype(int)
        gate_mismatches = (expected_gate != got_gate).sum()
        
        passed = mismatches == 0 and gate_mismatches == 0
        log.info("Label recompute: %s (label_excess_mismatches=%d, y_gate_mismatches=%d, max_diff=%.2e)",
                 "PASS" if passed else "FAIL", mismatches, gate_mismatches, max_diff)
        
        return {
            "passed": passed,
            "critical": True,
            "label_excess_mismatches": int(mismatches),
            "y_gate_mismatches": int(gate_mismatches),
            "max_abs_diff": float(max_diff),
            "sample_errors": sample_errors,
            "label_formula": "label_excess = (px[t+h-1]/px[t-1] - 1) sector - SPY",
            "y_gate_formula": "y_gate = 1 if label_excess > cost (10bps) else 0",
        }
    
    def _check_forbidden_features(self, panel: pd.DataFrame) -> dict[str, Any]:
        log.info("[4/7] Forbidden features check...")
        
        non_features = {
            "Date", "Sector", "label_excess", "y_gate", "rel_rank", "rank_target",
            "cost_bps", "horizon", "label_contract", "sample_weight"
        }
        feature_cols = [c for c in panel.columns if c not in non_features]
        
        forbidden_found = []
        for col in feature_cols:
            col_lower = col.lower()
            for pattern in FORBIDDEN_PATTERNS:
                if pattern in col_lower:
                    forbidden_found.append({"column": col, "matched_pattern": pattern})
                    break
        
        if "Date" in feature_cols:
            forbidden_found.append({"column": "Date", "matched_pattern": "date_as_feature"})
        
        passed = len(forbidden_found) == 0
        log.info("Forbidden features: %s (found: %d)", "PASS" if passed else "FAIL", len(forbidden_found))
        
        return {
            "passed": passed,
            "critical": True,
            "n_features": len(feature_cols),
            "forbidden_found": forbidden_found,
            "patterns_checked": FORBIDDEN_PATTERNS,
        }
    
    def _check_nan_profile(self, panel: pd.DataFrame) -> dict[str, Any]:
        log.info("[5/7] NaN profile check...")
        
        non_features = {
            "Date", "Sector", "label_excess", "y_gate", "rel_rank", "rank_target",
            "cost_bps", "horizon", "label_contract", "sample_weight"
        }
        feature_cols = [c for c in panel.columns if c not in non_features]
        
        nan_rates = panel[feature_cols].isna().mean().sort_values(ascending=False)
        high_nan_threshold = 0.02
        high_nan_cols = nan_rates[nan_rates > high_nan_threshold]
        
        stds = panel[feature_cols].std()
        constant_cols = stds[stds == 0].index.tolist()
        
        passed = len(constant_cols) == 0
        log.info("NaN profile: %s (high_nan_features=%d, constant_features=%d)",
                 "PASS" if passed else "WARN", len(high_nan_cols), len(constant_cols))
        
        return {
            "passed": passed,
            "critical": False,
            "total_features": len(feature_cols),
            "high_nan_features": len(high_nan_cols),
            "high_nan_threshold": high_nan_threshold,
            "high_nan_cols": {str(k): float(v) for k, v in high_nan_cols.head(10).items()},
            "constant_features": constant_cols,
            "overall_nan_rate": float(panel[feature_cols].isna().mean().mean()),
        }
    
    def _check_cv_specs(self, h: int) -> dict[str, Any]:
        log.info("[6/7] CV specs check...")
        
        cv_types = ["rolling", "expanding_equal", "expanding_linear_decay"]
        results = {}
        all_passed = True
        
        for cv_type in cv_types:
            cv_path = self.ws / "configs" / "cv" / f"cv_{cv_type}_h{h}.json"
            
            if not cv_path.exists():
                results[cv_type] = {"passed": False, "error": "File not found"}
                all_passed = False
                continue
            
            try:
                data = json.loads(cv_path.read_text(encoding="utf-8"))
                folds = data.get("folds", [])
                
                issues = []
                
                for fold in folds:
                    train_start = pd.Timestamp(fold["train_start"])
                    train_end = pd.Timestamp(fold["train_end"])
                    test_start = pd.Timestamp(fold["test_start"])
                    test_end = pd.Timestamp(fold["test_end"])
                    
                    if test_start <= train_end:
                        issues.append(f"Fold {fold['fold_id']}: test_start ({test_start.date()}) <= train_end ({train_end.date()})")
                    
                    purge = fold.get("purge_gap", 0)
                    embargo = fold.get("embargo", 0)
                    if purge < h:
                        issues.append(f"Fold {fold['fold_id']}: purge_gap ({purge}) < horizon ({h})")
                    if embargo < h:
                        issues.append(f"Fold {fold['fold_id']}: embargo ({embargo}) < horizon ({h})")
                
                if cv_type == "rolling":
                    for fold in folds:
                        train_start = pd.Timestamp(fold["train_start"])
                        train_end = pd.Timestamp(fold["train_end"])
                        train_days = (train_end - train_start).days
                        if train_days < 1000 or train_days > 2000:
                            issues.append(f"Fold {fold['fold_id']}: unusual train length ({train_days} days)")
                
                if cv_type.startswith("expanding"):
                    prev_train_end = None
                    for fold in folds:
                        train_end = pd.Timestamp(fold["train_end"])
                        if prev_train_end is not None and train_end <= prev_train_end:
                            issues.append(f"Fold {fold['fold_id']}: train_end not growing")
                        prev_train_end = train_end
                
                passed = len(issues) == 0
                results[cv_type] = {
                    "passed": passed,
                    "n_folds": len(folds),
                    "issues": issues[:5],
                }
                
                if not passed:
                    all_passed = False
                    
            except Exception as e:
                results[cv_type] = {"passed": False, "error": str(e)}
                all_passed = False
        
        log.info("CV specs: %s", "PASS" if all_passed else "FAIL")
        
        return {
            "passed": all_passed,
            "critical": True,
            "cv_types": results,
        }
    
    def _check_decay_weights(self, panel: pd.DataFrame, h: int) -> dict[str, Any]:        
        log.info("[7/7] Decay weights check...")
        
        cv_path = self.ws / "configs" / "cv" / f"cv_expanding_linear_decay_h{h}.json"
        
        if not cv_path.exists():
            return {"passed": False, "critical": False, "error": "CV config not found"}
        
        try:
            data = json.loads(cv_path.read_text(encoding="utf-8"))
            folds = data.get("folds", [])
            
            if not folds:
                return {"passed": False, "critical": False, "error": "No folds in config"}
            
            last_fold = folds[-1]
            
            weights = compute_linear_decay_weights(
                panel,
                train_start=last_fold["train_start"],
                train_end=last_fold["train_end"],
                min_weight=0.1,
            )
            
            validation = validate_weights(
                weights, panel,
                train_start=last_fold["train_start"],
                train_end=last_fold["train_end"],
            )
            
            passed = validation["valid"]
            log.info("Decay weights: %s (monotonic=%s, min=%.3f, max=%.3f)",
                     "PASS" if passed else "FAIL",
                     validation["monotonic_increasing"],
                     validation["min_weight"] or 0,
                     validation["max_weight"] or 0)
            
            return {
                "passed": passed,
                "critical": False,
                "test_fold_id": last_fold["fold_id"],
                "validation": validation,
            }
            
        except Exception as e:
            return {"passed": False, "critical": False, "error": str(e)}
    
    def generate_reports(self, output_dir: Path | None = None) -> None:
        out_dir = output_dir or (self.ws / "diagnostics" / "readiness")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for h, result in self.results.items():
            json_path = out_dir / f"readiness_h{h}.json"
            json_path.write_text(
                json.dumps(result, indent=2, sort_keys=True, default=str),
                encoding="utf-8"
            )
            log.info("JSON report: %s", json_path)
            
            md_path = out_dir / f"readiness_h{h}.md"
            md_content = self._generate_markdown(result)
            md_path.write_text(md_content, encoding="utf-8")
            log.info("Markdown report: %s", md_path)
    
    def _generate_markdown(self, result: dict) -> str:
     
        h = result["horizon"]
        lines = [
            f"AutoGluon Readiness Report: Horizon h={h}",
            "",
            f"Generated: {result['checked_at_utc']}",
            "",
            f"Overall Status: {'PASS' if result['overall_pass'] else 'FAIL'}",
            "",
        ]
        
        if result["critical_failures"]:
            lines.append("Critical Failures")
            for f in result["critical_failures"]:
                lines.append(f"FAIL {f}")
            lines.append("")
        
        if result["warnings"]:
            lines.append("Warnings")
            for w in result["warnings"]:
                lines.append(f"WARNING {w}")
            lines.append("")
        
        lines.append("Check Details")
        lines.append("")
        
        for check_name, check_result in result["checks"].items():
            status = "PASS" if check_result.get("passed", False) else "FAIL"
            lines.append(f"{status} {check_name.replace('_', ' ').title()}")
            lines.append("")
            

            for key, value in check_result.items():
                if key in ("passed", "critical"):
                    continue
                if isinstance(value, dict) and len(str(value)) > 200:
                    lines.append(f"- **{key}**: (see JSON)")
                elif isinstance(value, list) and len(value) > 5:
                    lines.append(f"- **{key}**: {value[:5]} ...")
                else:
                    lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        return "\n".join(lines)

def main(
    horizons: list[int] | None = None,
    output_dir: Path | None = None,
) -> dict[int, dict]:
    checker = ReadinessChecker()
    hs = horizons or [5, 21, 63]
    
    for h in hs:
        checker.check_horizon(h)
    
    checker.generate_reports(output_dir)
    
    print("READINESS SUMMARY")
    
    for h, result in checker.results.items():
        status = "PASS" if result["overall_pass"] else "FAIL"
        print(f"  h={h:2d}: {status}")
        if result["critical_failures"]:
            for f in result["critical_failures"]:
                print(f"X {f}")
    
    return checker.results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoGluon readiness check")
    p.add_argument("horizons", default="5,21,63", help="Comma-separated horizons")
    p.add_argument("output-dir", default=None, help="Output directory for reports")
    p.add_argument("log-level", default="INFO", help="Logging level")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    out_dir = Path(args.output_dir) if args.output_dir else None
    
    main(horizons=horizons, output_dir=out_dir)