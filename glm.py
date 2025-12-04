# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair==6.0.0",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "scikit-learn==1.7.2",
#     "statsmodels==0.14.5",
# ]
# ///

import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium", app_title="Introduction to GLMS")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    import numpy as np

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from sklearn.model_selection import train_test_split
    return alt, mo, np, pd, sm, train_test_split


@app.cell
def _(mo):
    mo.md(text= "# Generalized Linear Models - Seoul Rented Bikes")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(text="""
    The objective of this notebook application is to introduce students to generalized linear models in python

    Data source: <code>https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand</code>
    """).callout(kind="info")
    return


@app.cell
def _(mo):
    label1 = mo.md(text="**1. Click the select to see the variables descriptions**")
    return (label1,)


@app.cell
def _(column_description):
    column_description
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **2. Inspect the dataframe interacting with the ui elements**
    """)
    return


@app.cell(hide_code=True)
def datafetch(mo, pd):
    ## fetching data
    @mo.cache()
    def fetch_bike_data():
        url = "https://raw.githubusercontent.com/datagus/glm_exercise_marimo/main/data/SeoulBikeData.csv"
        bike_df = pd.read_csv(url, encoding='latin-1')
    
        # Remove units from all columns
        bike_df.columns = bike_df.columns.str.replace(r'\(.*?\)', '', regex=True).str.strip()
        return bike_df

    bike_df = fetch_bike_data()
    bike_df
    return (bike_df,)


@app.cell(hide_code=True)
def add_description(bike_df, mo):
    #creating dictionary for column descriptions
    bike_dict = {
        "Date": "Date the bike was rented",
        "Rented Bike Count": "Count of bikes rented at each hour",
        "Hour": "Hour of the day",
        "Temperature": "Temperature in Celsius",
        "Humidity": "Percentage humidity",
        "Wind speed": "Wind speed in m/s",
        "Visibility": "Visibility in units of 10 meters",
        "Dew point temperature": "Dew point temperature in Celsius",
        "Solar Radiation": "Solar radiation in MJ/m²",
        "Rainfall": "Rainfall in mm",
        "Snowfall": "Snowfall in cm",
        "Seasons": "Season (Winter, Spring, Summer, Autumn)",
        "Holiday": "Holiday vs. no holiday",
        "Functioning Day": "NoFunc (Non-functional hours) or Fun (Functional hours)"
    }

    # Create dropdown with all columns from bike_df
    column_selector = mo.ui.dropdown(
        options=bike_df.columns.tolist(),
        value=bike_df.columns[0],
        full_width=True
    )
    return bike_dict, column_selector


@app.cell(hide_code=True)
def selector_description(bike_dict, column_selector, label1, mo):
    # creating ui to check the column descriptions
    selected_column = column_selector.value

    des = mo.md(bike_dict[selected_column])

    column_description = mo.hstack([label1, mo.vstack([column_selector, des], gap=0.05)])
    return (column_description,)


@app.cell(hide_code=True)
def _(bike_df, pd):
    #changing the date format and splitting them

    bike_df["Date"] = pd.to_datetime(bike_df["Date"], format="%d/%m/%Y")
    bike_df['year'] = bike_df["Date"].dt.year
    bike_df['month'] = bike_df["Date"].dt.month
    bike_df['month_name'] = bike_df["Date"].dt.month_name()
    bike_df['weekday'] = bike_df["Date"].dt.dayofweek + 1
    bike_df['weekday_name'] = bike_df["Date"].dt.day_name()
    bike_df['day'] = bike_df["Date"].dt.day
    return


@app.cell(hide_code=True)
def _(bike_df, np):
    #converting circular data

    # Hour (0–23)
    bike_df["hour_sin"] = np.sin(2 * np.pi * bike_df["Hour"] / 24)
    bike_df["hour_cos"] = np.cos(2 * np.pi * bike_df["Hour"] / 24)

    # Day of week (0=Mon ... 6=Sun)
    bike_df["dow_sin"] = np.sin(2 * np.pi * bike_df["weekday"] / 7)
    bike_df["dow_cos"] = np.cos(2 * np.pi * bike_df["weekday"] / 7)

    # Month (1–12)
    bike_df["month_sin"] = np.sin(2 * np.pi * bike_df["month"] / 12)
    bike_df["month_cos"] = np.cos(2 * np.pi * bike_df["month"] / 12)
    return


@app.cell(hide_code=True)
def _(bike_df, mo):
    #creating year and month selectors
    year_selector = mo.ui.dropdown(
        options=bike_df["year"].unique().tolist(),
        value=2017,
        full_width=True
    )

    month_selector = mo.ui.dropdown(
        options=bike_df["month_name"].unique().tolist(),
        value="December"
    )

    day_selector = mo.ui.dropdown(
        options=bike_df["day"].unique().tolist(),
        value="1",
        full_width=True
    )
    return day_selector, month_selector, year_selector


@app.cell(hide_code=True)
def _(day_selector, mo, month_selector, year_selector):
    #stacking the selector

    selectors = mo.hstack([year_selector, month_selector,day_selector], justify="center")
    return (selectors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(text="## **Exploration charts**")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **3. Select the date you want to visualize**
    On the left plot you see the count of bikes rented per hour in a single day. On the right you see the average count of bikes rented in a specific month
    """)
    return


@app.cell(hide_code=True)
def _(bike_df, day_selector, month_selector, year_selector):
    #creating subselected dataframe based on the selectors
    single_df = (
        bike_df.loc[
            (bike_df["year"]==year_selector.value) & 
            (bike_df["month_name"]==month_selector.value) &
            (bike_df["day"]==day_selector.value),
            ]

    )
    return (single_df,)


@app.cell(hide_code=True)
def _(alt, single_df):
    #creating time series chart based on the selectors

    chart = alt.Chart(single_df).mark_line(point=True).encode(
        x=alt.X("Hour", 
                axis=alt.Axis(
                    labelAngle=-45,  # Rotate labels 
                    title='Time'
                )),# :T tells Altair this is a temporal (time) field
        y="Rented Bike Count:Q",  # :Q tells Altair this is a quantitative field
        tooltip=["Temperature","Solar Radiation", "Rainfall","Snowfall"]
    ).properties(
        width=350,
        height=300,
        title="Count of Rented Bikes for the specified Date"
    ).configure_axis(
        grid=False
    )
    return (chart,)


@app.cell(hide_code=True)
def _(bike_df):
    #grouping dataframe to get average counts per day

    grouped_df = (
        bike_df.copy()
          .groupby(["Date","year", "Seasons","month_name", "day","weekday", "weekday_name","Holiday"])
          .mean(numeric_only=True)       # mean for all numeric columns
          .round(2)
          .reset_index()
    )


    drop_columns = ["Hour", "month"]
    grouped_df = grouped_df.drop(drop_columns, axis=1)

    discrete_columns = ["Rented Bike Count","Humidity", "Visibility"]
    grouped_df[discrete_columns] = grouped_df[discrete_columns].round(0).astype(int)
    return (grouped_df,)


@app.cell(hide_code=True)
def _(grouped_df, month_selector):
    #subselecting grouped_df based on the month selector

    grouped_month = (
        grouped_df.loc[
            (grouped_df["month_name"]==month_selector.value)
            ]
    )
    return (grouped_month,)


@app.cell(hide_code=True)
def _(alt, grouped_month, month_selector):
    #getting the timeseries average counts plot 

    chart_month = alt.Chart(grouped_month).mark_line(point=True).encode(
        x=alt.X("Date:T", 
                axis=alt.Axis(
                    labelAngle=-45,  # Rotate labels 
                    title='Time'
                )),# :T tells Altair this is a temporal (time) field
        y="Rented Bike Count:Q",  # :Q tells Altair this is a quantitative field
        color=alt.value("purple"),
        tooltip=["Rented Bike Count","Temperature","Solar Radiation", "Rainfall","Snowfall"]
    ).properties(
        width=350,
        height=300,
        title=f"Average count of rented bikes for {month_selector.value}"
    ).configure_axis(
        grid=False
    )
    return (chart_month,)


@app.cell(hide_code=True)
def _(chart, chart_month, mo, selectors):
    mo.hstack([
        mo.vstack(
            [selectors, chart], 
            heights="equal"
                 ),
        mo.vstack(
            ["\n",chart_month],
            align="center"
        )
    ],widths="equal"
             )
    return


@app.cell(hide_code=True)
def _():
    #mo.md(text="## Splitting the data")
    return


@app.cell(hide_code=True)
def _(bike_df, pd, sm, train_test_split):
    df = bike_df.copy()

    df = df.rename(columns={"Rented Bike Count":"bike_count","Wind speed":"Wind_speed","Dew point temperature":"Dew_point_temperature","Solar Radiation":"Solar_Radiation"})

    y = df[["bike_count"]]
    X = df[["Date","Hour", 
            "Temperature", 
            "Humidity", 
            "Wind_speed", 
            "Visibility", 
            "Dew_point_temperature", 
            "Solar_Radiation", 
            "Rainfall", 
            "Snowfall", 
            "year", 
            "month",
            "month_name",
            "weekday", 
            "day", 
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin", 
            "month_cos"
           ]]

    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=42) 

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    return test, train


@app.cell
def _():
    #mo.md(text="## Putting interactive predictors")
    return


@app.cell
def _(mo):
    # Create checkboxes for variable selection
    predictor_names = ['Hour', 'Temperature', 'Humidity', 'Wind_speed', 'Visibility', 'Dew_point_temperature', 'Solar_Radiation', 'Rainfall', 'Snowfall', 'year', 'month', 'weekday', 'day', 'hour_sin','hour_cos','dow_sin','dow_cos','month_sin', 'month_cos']

    predictor_selector = mo.ui.array([
        mo.ui.checkbox(label=var, value=False) for var in predictor_names
    ])
    return predictor_names, predictor_selector


@app.cell
def _(mo, predictor_selector):
    predictors = mo.vstack([
        mo.md("**4.Select Variables for Model**"),
        mo.hstack(
            [predictor_selector[i] for i in range(0, len(predictor_selector))],
            widths="equal",
            wrap=True
        )
    ])
    return (predictors,)


@app.cell
def _(predictor_names, predictor_selector):
    # Now this will react to changes
    def put_name(number):
       if predictor_selector[number].value:
           a = predictor_names[number] + " + "
       else:
           a = ""
       return a
    return (put_name,)


@app.cell
def _(predictor_selector, put_name):
    for i in range(0,len(predictor_selector)):
       if i == 0:
           formula_pred = put_name(i)
       else:
           formula_pred = formula_pred + put_name(i)

    formula_pred = formula_pred[:-3]
    return


@app.cell
def _(predictor_names, predictor_selector):
    coli = [predictor_names[j] for j in range(0,len(predictor_selector)) if predictor_selector[j].value]
    return (coli,)


@app.cell
def _():
    #run_model_btn = mo.ui.run_button(label="Fit Model")
    #run_model_btn
    return


@app.cell
def _():
    #run_model_btn.value
    return


@app.cell
def _(sm, train):
    def glm_model(coli):

        y = train["bike_count"]
        X = train[coli]
        X = sm.add_constant(X)
        #formula = (f"""bike_count ~ {formula_pred}""")
        family = sm.families.Poisson()
        #model = smf.GLM(formula, train, family=family)
        model =sm.GLM(y,X,family=family)
        result = model.fit()
        return result
    return (glm_model,)


@app.cell
def _(coli, glm_model, mo, pd):
    result = glm_model(coli)

    # Create tabs for different model outputs
    model_tabs = mo.ui.tabs({
            "Model Statistics":
            pd.DataFrame({
                'Metric': ['Deviance', 'Null Deviance', 'Deviance Explained', 'Log-Likelihood', 'AIC'],
                'Value': [
                    round(result.deviance,2),
                    round(result.null_deviance,2),
                    1 - (result.deviance / result.null_deviance),
                    round(result.llf,2),
                    round(result.aic,2)
                ]
            }),
        "Coefficients":
            pd.DataFrame({
                'Coefficient': result.params,
                'Std Error': result.bse,
                'z': result.tvalues,
                'P>|z|': result.pvalues,
                '[0.025': result.conf_int()[0],
                '0.975]': result.conf_int()[1]
            })
    })
    return model_tabs, result


@app.cell
def _(mo):
    mo.md(text="## **Add variables to the model**")
    return


@app.cell
def _(mo, model_tabs, predictors, residuals_histogram, time_series_chart):
    final_ui = mo.hstack(
        [
    
        mo.vstack([predictors, model_tabs]),
        mo.vstack([time_series_chart,residuals_histogram])
    ],     widths=[1, 1]
                        )
    final_ui
    return


@app.cell
def _():
    ##mo.md(text="## Predicting")
    return


@app.cell
def _(coli, result, sm, test):
    X_test_with_const = sm.add_constant(test[coli])
    y_predict = result.predict(X_test_with_const)
    return (y_predict,)


@app.cell
def _(pd, test, y_predict):
    predictions_comparison = pd.DataFrame({
        'Date': test["Date"],
        'Hour': test["Hour"],
        'year': test["year"],
        'month': test["month_name"],
        'day': test["day"],
        'Actual': test["bike_count"],
        'Predicted': y_predict,
        'Residual': test["bike_count"] - y_predict
    })
    return (predictions_comparison,)


@app.cell(hide_code=True)
def _(alt, predictions_comparison):
    # Reshape data for Altair (melt to long format)
    plot_data = predictions_comparison.melt(
        id_vars=['Date'], 
        value_vars=['Actual', 'Predicted'],
        var_name='Type',
        value_name='Bike Count'
    )

    # Create the chart with monthly aggregation
    time_series_chart = alt.Chart(plot_data).mark_point().encode(
        x=alt.X('yearmonth(Date):T', 
                axis=alt.Axis(
                    labelAngle=-45,
                    title='Month'
                )),
        y=alt.Y('mean(Bike Count):Q', 
                title='Average Bike Count',
                axis=alt.Axis(format='.0f')),
        color=alt.Color('Type:N', 
                        scale=alt.Scale(domain=['Actual', 'Predicted'], 
                                       range=['#1f77b4', '#ff7f0e']),
                        legend=alt.Legend(title='Value Type')),
        tooltip=[
            alt.Tooltip('yearmonth(Date):T', title='Month'),
            alt.Tooltip('Type:N', title='Type'),
            alt.Tooltip('mean(Bike Count):Q', title='Average Count', format='.0f')
        ]
    ).properties(
        width=350,
        height=200,
        title='Monthly Average: Actual vs Predicted Bike Rentals'
    ).configure_axis(
        grid=False
    )
    return (time_series_chart,)


@app.cell(hide_code=True)
def _(alt, predictions_comparison):
    residuals_histogram = alt.Chart(predictions_comparison).mark_bar(color='red').encode(
        x=alt.X('Residual:Q', bin=alt.Bin(maxbins=50), title='Residual'),
        y=alt.Y('count()', title='Frequency'),
        tooltip=[
            alt.Tooltip('Residual:Q', bin=alt.Bin(maxbins=50), title='Residual Range'),
            alt.Tooltip('count()', title='Count')
        ]
    ).properties(
        width=350,
        height=200,
        title='Distribution of Residuals (Actual - Predicted)'
    )
    return (residuals_histogram,)


if __name__ == "__main__":
    app.run()
