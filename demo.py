# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair==6.0.0",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "ucimlrepo==0.0.7",
# ]
# ///

import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import altair as alt

    from ucimlrepo import fetch_ucirepo
    return fetch_ucirepo, mo


@app.cell
def _(mo):
    mo.md(text="## Data Fetching")
    return


@app.cell
def _(fetch_ucirepo, mo):
    @mo.cache()
    def fetch_bike_data():
        bike_data_source = fetch_ucirepo(id=560) 
        bike_df = bike_data_source.data.original
        return bike_df

    bike_df = fetch_bike_data()
    bike_df
    return


if __name__ == "__main__":
    app.run()
