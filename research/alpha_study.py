import pandas as pd

from dataload import AlphaLoader, InvestableUniverseLoader, ReturnsLoader, RiskFactorDefLoader, RiskFactorExposuresLoader
from datetime import timedelta
import tqdm
import numpy as np
from typing import List

def qcutStats(df:pd.DataFrame, pivotCol: str, cuts:int, valueCols: List[str]) -> pd.DataFrame:
    # Pivot and calculate mean and standard error in one expression
    def stde(x):
        return np.std(x, ddof=1) / np.sqrt(len(x))
    def avg(x):
        return np.mean(x)
    pivot = df.pivot_table(
        index=pd.qcut(df[pivotCol],cuts),   # Columns to group by
        values=valueCols,         # Columns to aggregate
        aggfunc={ C : [ avg, stde ] for C in valueCols},
        observed=False
    )

    # Renaming columns for clarity (optional)
    columnOrder = []
    for c in valueCols:
        columnOrder.append((c,'avg'))
        columnOrder.append((c,'stde'))
    pivot = pivot[columnOrder]
    #pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    #pivot.reset_index(inplace=True)
    return pivot

def csv_download_link(df, csv_file_name, delete_prompt=True):
    """Display a download link to load a data frame as csv from within a Jupyter notebook"""
    df.to_csv(csv_file_name, index=True)
    from IPython.display import FileLink,display
    display(FileLink(csv_file_name))
    if delete_prompt:
        a = input('Press enter to delete the file after you have downloaded it.')
        import os
        os.remove(csv_file_name)

def get_alpha_stats(meanAdjust = False ) -> pd.DataFrame:
    AL = AlphaLoader()
    UL = InvestableUniverseLoader()
    RL = ReturnsLoader()
    returnDates = RL.getDates() #will serve as a poor-mens business calendar
    alphaDates = AL.getDates()
    dfs = []
    for index,dt in enumerate(tqdm.tqdm(alphaDates)):
        if index != len(alphaDates)-1: #Not the last date
            universe = UL.getDataAsOf(dt)

            # derive primary fundamental factor for all assets in universe
            riskFactorDefs = RiskFactorDefLoader().getDataAsOf(dt)
            riskFactorExposures = RiskFactorExposuresLoader().getDataAsOf(dt)
            sectorFactors = {f'beta_{i+1}':i for i in range(8,28)} #beta column name -> factor number for fundamental factors
            primFactor = abs(riskFactorExposures.set_index('cusip')[list(sectorFactors.keys())]).idxmax(axis=1).map(sectorFactors)
            primFactor.name = 'Factor Number'
            primFactor = primFactor.to_frame().reset_index().merge(riskFactorDefs, on='Factor Number')[
                ['cusip', 'Factor Code', 'Factor Name']].set_index('cusip')
            primFactor.rename(columns = {'Factor Code':'PrimFactorCode','Factor Name':'PrimFactorName'},inplace=True)

            alphaEstimates = AL.getDataAsOf(dt)[['cusip','Name','Alpha']]
            alphaEstimates = alphaEstimates[alphaEstimates.cusip.isin(set(universe.cusip.to_list()))].dropna().set_index('cusip')
            if meanAdjust:
                alphaEstimates['Alpha'] = alphaEstimates['Alpha'] - alphaEstimates['Alpha'].mean()
            periodStartDate = dt+timedelta(days=1) # Assume alphas come out after EOD. So we will get into it next day
            periodEndDates = [r for r in returnDates if r >= periodStartDate][0:5] + [dt+timedelta(weeks=horizon) for horizon in [1,2,3,4]]
            periodEndLabels = [f'{d}bd' for d in range(1,6)] + [f'{w}wk' for w in range(1,5)]
            for index,periodEndDt in enumerate(periodEndDates):
                #get returns for this period - assume alphas come out after close, so assume we will get into it next day ( and ignore open->close )
                returns = RL.getPeriodReturns(periodStartDate,periodEndDt,alphaEstimates.index.to_list())
                returns.index.name='cusip'
                returns.name=f'Return_{periodEndLabels[index]}'
                alphaEstimates[returns.name]=returns
            alphaEstimates['ForecastDate'] = dt
            alphaEstimates = alphaEstimates.merge(primFactor, on='cusip', how='left')
            dfs.append(alphaEstimates)
    return pd.concat(dfs)


def remove_outliers(df, colsToConsider, upper_quantile=0.995, lower_quantile=0.005):

    # Calculate the 1st and 99th percentiles for each numeric column
    lower_bound = df[colsToConsider].quantile(lower_quantile)
    upper_bound = df[colsToConsider].quantile(upper_quantile)

    # Create a boolean mask for rows that fall within the 1st to 99th percentile range for all numeric columns
    mask = (df[colsToConsider] >= lower_bound) & (df[colsToConsider] <= upper_bound)

    # Keep rows where all numeric values in a row fall within the bounds
    mask = mask.all(axis=1)

    # Filter the original DataFrame (including non-numeric columns) based on the mask
    return df[mask], df[~mask],lower_bound,upper_bound

if __name__ == '__main__':
    alpha_stats = get_alpha_stats()

