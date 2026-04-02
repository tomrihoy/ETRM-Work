import pandas as pd
import typer

from price_simulator import PowerPrices, compute_stats, plot_curves
from dataclass_definitions import OutputFormat
app = typer.Typer(help="Generate synthetic wholesale power prices")


@app.command()
def generate(
    start_date:  str           = typer.Argument(..., help="Start date e.g. 2026-01-01"),
    end_date:    str           = typer.Argument(..., help="End date e.g. 2026-12-31"),
    output_format: OutputFormat  = typer.Option(OutputFormat.STDOUT, help="Output format"),
    filepath:    str | None = typer.Option(None, help="Folder to save CSV"),
    filename:    str | None = typer.Option(None, help="Custom filename"),
) -> None:
    """Generate a synthetic power price curve."""
    pp = PowerPrices()
    pp.generate_price_curve(start_date, end_date, output_format=output_format, filepath=filepath, filename=filename)
    typer.echo(typer.style("✓ Curve generated successfully", fg=typer.colors.GREEN))

@app.command()
def plot(
    filepath: str = typer.Argument(..., help="Path to a saved price curve CSV"),
) -> None:
    """Generate and plot a price curve."""
    plot_curves(filepath)

@app.command()
def analyse(
    filepath: str = typer.Argument(..., help="Path to a saved price curve CSV"),
) -> None:
    """Analyse statistics from a saved price curve CSV."""
    df = pd.read_csv(filepath)
    stats = compute_stats(df['power_prices'])
    for stat, value in stats.items():
        typer.echo(f"  {stat}: {value:.2f}")


if __name__ == "__main__":
    app()
