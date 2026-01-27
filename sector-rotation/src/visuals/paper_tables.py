from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

def format_percent(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}\\%"


def format_float(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def format_bold_best(
    df: pd.DataFrame, 
    column: str, 
    higher_better: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    
    if higher_better:
        best_idx = df[column].idxmax()
    else:
        best_idx = df[column].idxmin()
    
    df.loc[best_idx, column] = f"\\textbf{{{df.loc[best_idx, column]}}}"
    
    return df


def add_significance_stars(
    value: float,
    pvalue: float,
) -> str:
    formatted = f"{value:.2f}"
    
    if pvalue < 0.001:
        return f"{formatted}***"
    elif pvalue < 0.01:
        return f"{formatted}**"
    elif pvalue < 0.05:
        return f"{formatted}*"
    else:
        return formatted

def generate_strategy_comparison_table(
    metrics: pd.DataFrame,
    caption: str = "Strategy Performance Comparison",
    label: str = "tab:strategy_comparison",
    output_path: Path | str | None = None,
) -> str:

    df = metrics.copy()
    
    format_map = {
        "return": ("Return (\\%)", lambda x: format_float(x * 100)),
        "cagr": ("CAGR (\\%)", lambda x: format_float(x * 100)),
        "sharpe": ("Sharpe", lambda x: format_float(x)),
        "sortino": ("Sortino", lambda x: format_float(x)),
        "calmar": ("Calmar", lambda x: format_float(x)),
        "max_drawdown": ("Max DD (\\%)", lambda x: format_float(x * 100)),
        "volatility": ("Vol (\\%)", lambda x: format_float(x * 100)),
        "win_rate": ("Win Rate (\\%)", lambda x: format_float(x * 100)),
    }
    
    new_cols = {}
    for col in df.columns:
        col_lower = col.lower().replace(" ", "_")
        for key, (name, fmt) in format_map.items():
            if key in col_lower:
                new_cols[col] = name
                df[col] = df[col].apply(fmt)
                break
    
    df = df.rename(columns=new_cols)
    
    latex = df.to_latex(
        escape=False,
        index=True,
        column_format="l" + "r" * len(df.columns),
        caption=caption,
        label=label,
    )
    
    latex = latex.replace("\\toprule", "\\toprule")
    latex = latex.replace("\\midrule", "\\midrule")
    latex = latex.replace("\\bottomrule", "\\bottomrule")
    
    latex = latex.replace(
        "\\end{tabular}",
        "\\end{tabular}\n\\\\[0.5em]\n"
        "\\footnotesize Note: Transaction costs of 10 bps applied."
    )
    
    if output_path:
        output_path = Path(output_path)
        output_path.write_text(latex, encoding="utf-8")
        log.info("Saved table to %s", output_path)
    
    return latex


def generate_horizon_comparison_table(
    metrics_by_horizon: dict[int, pd.DataFrame],
    mode: str,
    caption: str | None = None,
    label: str | None = None,
    output_path: Path | str | None = None,
) -> str:
    if caption is None:
        caption = f"{mode.replace('_', ' ').title()} Performance by Horizon"
    if label is None:
        label = f"tab:{mode}_horizon"
    
    rows = []
    for horizon, metrics in metrics_by_horizon.items():
        row = {"Horizon": f"H={horizon}"}
        if isinstance(metrics, pd.DataFrame):
            row.update(metrics.iloc[0].to_dict())
        else:
            row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows).set_index("Horizon")
    
    return generate_strategy_comparison_table(
        df, 
        caption=caption,
        label=label,
        output_path=output_path,
    )

def generate_ic_summary_table(
    ic_stats: dict[str, dict[str, float]],
    caption: str = "Information Coefficient Summary",
    label: str = "tab:ic_summary",
    output_path: Path | str | None = None,
) -> str:
    rows = []
    
    for strategy, stats in ic_stats.items():
        row = {
            "Strategy": strategy,
            "Mean IC": format_float(stats.get("mean", 0), 4),
            "Std IC": format_float(stats.get("std", 0), 4),
            "IC IR": format_float(stats.get("ir", 0)),
        }
        
        if "t_stat" in stats and "pvalue" in stats:
            row["t-stat"] = add_significance_stars(
                stats["t_stat"],
                stats["pvalue"],
            )
        
        rows.append(row)
    
    df = pd.DataFrame(rows).set_index("Strategy")
    
    latex = df.to_latex(
        escape=False,
        column_format="lrrrr",
        caption=caption,
        label=label,
    )
    
    latex = latex.replace(
        "\\end{tabular}",
        "\\end{tabular}\n\\\\[0.5em]\n"
        "\\footnotesize *$p<0.05$, **$p<0.01$, ***$p<0.001$"
    )
    
    if output_path:
        Path(output_path).write_text(latex, encoding="utf-8")
    
    return latex

def generate_regime_performance_table(
    regime_metrics: pd.DataFrame,
    caption: str = "Performance by Market Regime",
    label: str = "tab:regime_performance",
    output_path: Path | str | None = None,
) -> str:
    df = regime_metrics.copy()
    
    df = df.map(lambda x: format_float(x) if isinstance(x, (int, float)) else x)
    
    latex = df.to_latex(
        escape=False,
        column_format="l" + "r" * len(df.columns),
        caption=caption,
        label=label,
    )
    
    if output_path:
        Path(output_path).write_text(latex, encoding="utf-8")
    
    return latex

def generate_model_summary_table(
    fold_metrics: list[dict[str, float]],
    caption: str = "Cross-Validation Results Summary",
    label: str = "tab:cv_summary",
    output_path: Path | str | None = None,
) -> str:
    df = pd.DataFrame(fold_metrics)
    df.index = [f"Fold {i+1}" for i in range(len(df))]
    
    mean_row = df.mean()
    mean_row.name = "Mean"
    df = pd.concat([df, mean_row.to_frame().T])
    
    std_row = df.iloc[:-1].std()
    std_row.name = "Std"
    df = pd.concat([df, std_row.to_frame().T])
    
    df = df.map(lambda x: format_float(x, 4) if isinstance(x, (int, float)) else x)
    
    latex = df.to_latex(
        escape=False,
        column_format="l" + "r" * len(df.columns),
        caption=caption,
        label=label,
    )
    
    latex = latex.replace("Mean &", "\\textbf{Mean} &")
    
    if output_path:
        Path(output_path).write_text(latex, encoding="utf-8")
    
    return latex

def generate_feature_importance_table(
    importance: pd.Series | dict[str, float],
    top_n: int = 15,
    caption: str = "Top Feature Importances",
    label: str = "tab:feature_importance",
    output_path: Path | str | None = None,
) -> str:
    if isinstance(importance, dict):
        importance = pd.Series(importance)
    
    top = importance.nlargest(top_n)
    
    df = pd.DataFrame({
        "Feature": top.index,
        "Importance": top.values,
        "Rank": range(1, len(top) + 1),
    }).set_index("Rank")
    
    df["Importance"] = df["Importance"].apply(lambda x: format_float(x, 4))
    
    latex = df.to_latex(
        escape=False,
        column_format="rlr",
        caption=caption,
        label=label,
    )
    
    if output_path:
        Path(output_path).write_text(latex, encoding="utf-8")
    
    return latex

def generate_paired_comparison_table(
    comparisons: list[dict[str, Any]],
    caption: str = "Paired Strategy Comparisons",
    label: str = "tab:paired_tests",
    output_path: Path | str | None = None,
) -> str:
    rows = []
    
    for comp in comparisons:
        row = {
            "Strategy A": comp["strategy_a"],
            "Strategy B": comp["strategy_b"],
            "$\\Delta$ Sharpe": format_float(comp.get("diff", 0)),
            "t-stat": add_significance_stars(
                comp.get("t_stat", 0),
                comp.get("pvalue", 1),
            ),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    latex = df.to_latex(
        escape=False,
        index=False,
        column_format="llrr",
        caption=caption,
        label=label,
    )
    
    latex = latex.replace(
        "\\end{tabular}",
        "\\end{tabular}\n\\\\[0.5em]\n"
        "\\footnotesize *$p<0.05$, **$p<0.01$, ***$p<0.001$. "
        "Paired t-test on rolling 252-day Sharpe ratios."
    )
    
    if output_path:
        Path(output_path).write_text(latex, encoding="utf-8")
    
    return latex

def generate_turnover_table(
    turnover_stats: dict[str, dict[str, float]],
    caption: str = "Portfolio Turnover and Transaction Costs",
    label: str = "tab:turnover",
    output_path: Path | str | None = None,
) -> str:
    rows = []
    
    for strategy, stats in turnover_stats.items():
        row = {
            "Strategy": strategy,
            "Avg Daily Turnover (\\%)": format_float(
                stats.get("avg_turnover", 0) * 100, 2
            ),
            "Total Turnover": format_float(
                stats.get("total_turnover", 0), 1
            ),
            "Cost Drag (bps/yr)": format_float(
                stats.get("cost_drag", 0) * 10000, 1
            ),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows).set_index("Strategy")
    
    latex = df.to_latex(
        escape=False,
        column_format="lrrr",
        caption=caption,
        label=label,
    )
    
    if output_path:
        Path(output_path).write_text(latex, encoding="utf-8")
    
    return latex

def generate_full_results_table(
    results: dict[str, dict[str, Any]],
    output_dir: Path | str,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tables = {}
    
    metrics_rows = []
    for key, result in results.items():
        if "vectorbt" in result:
            vbt = result["vectorbt"]
            row = {
                "Strategy": key,
                "return": vbt.get("total_return", 0),
                "cagr": vbt.get("cagr", 0),
                "sharpe": vbt.get("sharpe", 0),
                "sortino": vbt.get("sortino", 0),
                "max_drawdown": vbt.get("max_drawdown", 0),
                "calmar": vbt.get("calmar", 0),
            }
            metrics_rows.append(row)
    
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows).set_index("Strategy")
        
        tables["strategy_comparison"] = generate_strategy_comparison_table(
            metrics_df,
            output_path=output_dir / "strategy_comparison.tex",
        )
    
    log.info("Generated %d LaTeX tables in %s", len(tables), output_dir)
    
    return tables