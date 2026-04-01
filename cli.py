import typer

from price_simulator import PowerPrices

app = typer.Typer(help="Generate synthetic wholesale power prices")


@app.command()
def generate(
    start_date:  str           = typer.Argument(..., help="Start date e.g. 2026-01-01"),
    end_date:    str           = typer.Argument(..., help="End date e.g. 2026-12-31"),
    save_to_csv:      bool  = typer.Option(None, help="Output format"),
    filepath:    str | None = typer.Option(None, help="Folder to save CSV"),
    filename:    str | None = typer.Option(None, help="Custom filename"),
) -> None:
    """Generate a synthetic power price curve."""
    pp = PowerPrices()
    pp.generate_price_curve(start_date, end_date, save_to_csv=save_to_csv, filepath=filepath, filename=filename)
    typer.echo(typer.style("✓ Curve generated successfully", fg=typer.colors.GREEN))

@app.command()
def plot(
    filepath: str = typer.Argument(..., help="Path to a saved price curve CSV"),
) -> None:
    """Generate and plot a price curve."""
    PowerPrices.cli_plot_curve(filepath)

@app.command()
def analyse(
    filepath: str = typer.Argument(..., help="Path to a saved price curve CSV"),
) -> None:
    """Analyse statistics from a saved price curve CSV."""
    import pandas as pd
    df = pd.read_csv(filepath)
    stats = PowerPrices._compute_stats(df['power_prices'])
    for stat, value in stats.items():
        typer.echo(f"  {stat}: {value:.2f}")


if __name__ == "__main__":
    app()
