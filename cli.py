# cli.py
import typer
from price_simulator import PowerPrices
from enum import StrEnum
from typing import Optional

app = typer.Typer(help="Generate synthetic wholesale power prices")

class OutputFormat(StrEnum):
    CSV    = "csv"
    MEMORY = "memory"
    BOTH   = "both"

@app.command()
def generate(
    start_date:  str           = typer.Argument(...,                    help="Start date e.g. 2026-01-01"),
    end_date:    str           = typer.Argument(...,                    help="End date e.g. 2026-12-31"),
    output:      OutputFormat  = typer.Option(OutputFormat.MEMORY,      help="Output format"),
    filepath:    Optional[str] = typer.Option(None,                     help="Folder to save CSV"),
    filename:    Optional[str] = typer.Option(None,                     help="Custom filename"),
) -> None:
    """Generate a synthetic power price curve."""
    pp = PowerPrices()
    save = output in (OutputFormat.CSV, OutputFormat.BOTH)
    pp.generate_price_curve(start_date, end_date, save_to_csv=save, filepath=filepath, filename=filename)
    typer.echo(typer.style("✓ Curve generated successfully", fg=typer.colors.GREEN))

@app.command()
def plot(
    start_date: str            = typer.Argument(..., help="Start date"),
    end_date:   str            = typer.Argument(..., help="End date"),
    curve:      Optional[str]  = typer.Option(None,  help="Specific curve to plot"),
) -> None:
    """Generate and plot a price curve."""
    pp = PowerPrices()
    pp.generate_price_curve(start_date, end_date)
    pp.plot_curves(curve_to_plot=curve)

@app.command()
def analyse(
    start_date: str = typer.Argument(..., help="Start date"),
    end_date:   str = typer.Argument(..., help="End date"),
) -> None:
    """Analyse statistics for generated curves."""
    pp = PowerPrices()
    pp.generate_price_curve(start_date, end_date)
    stats = pp.analyse_curves()
    for curve_name, curve_stats in stats.items():
        typer.echo(f"\n{curve_name}")
        for stat, value in curve_stats.items():
            typer.echo(f"  {stat}: {value:.2f}")

if __name__ == "__main__":
    app()