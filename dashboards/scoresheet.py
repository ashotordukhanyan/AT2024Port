import pickle
import logging
import glob
import numpy as np
import pandas as pd
from typing import List
from datetime import date,timedelta

def getPortfolioBetas(trade_name:str,stats_directory:str, needed_dates:List[date]):
    df = pd.DataFrame()
    for stats in portfolioBetaStats(trade_name,stats_directory,needed_dates):
        dt = stats[0]
        betas = stats[1]
        betaSeries = pd.Series(betas)
        betaSeries.name = dt.isoformat()
        df[betaSeries.name]= betaSeries
    return df


def portfolioBetaStats(trade_name:str,stats_directory:str, needed_dates:List[date]):
    for stats in getStats(trade_name,stats_directory):
        rebalDate = stats['date']
        if rebalDate in needed_dates:
            yield rebalDate, stats['portfolioBetas']

def getStats(trader_name:str, stats_directory:str):
    for statsFile in glob.glob(f'{stats_directory}/pstats_{trader_name}_*.pkl'):
        with open(statsFile,'rb') as f:
            stats = pickle.load(f)
            yield stats

def _clean(x):
    return float(x) if isinstance(x,np.float64) else x

def getPNLStats(trader_name:str,stats_directory:str,checkpoints):
    rebalCosts = {}; shortingCosts={};dailyReturns={}
    for stats in getStats(trader_name,stats_directory):
        dt = stats['date']
        rebalCosts[dt] = float(stats['rebalCost'])
        shortingCosts[dt] = float(stats['shortingCost'])
        for d,r in stats['dailyReturns'].items():
            dailyReturns[d] = r
    dailyReturns = pd.Series(dailyReturns)
    rebalCosts = pd.Series(rebalCosts)
    shortingCosts = pd.Series(shortingCosts)
    #return rebalCosts, shortingCosts, dailyReturns
    periodGrossRtns = [0]; periodRebalCosts = [0]; periodShoringCosts = [0]

    for i in range(1,len(checkpoints)):
        cp = checkpoints[i]
        previous_cp = checkpoints[i-1]
        (1+dailyReturns[(dailyReturns.index > previous_cp) & (dailyReturns.index <= cp)]).prod()
        periodGrossRtn = float(100*((1+.01*(dailyReturns[(dailyReturns.index > previous_cp) & (dailyReturns.index <= cp)])).prod()-1))
        periodRebalCost = float(rebalCosts[(rebalCosts.index > previous_cp) & (rebalCosts.index <= cp)].sum())
        periodShoringCost = float(shortingCosts[(shortingCosts.index > previous_cp) & (shortingCosts.index <= cp)].mean()) * 4/52 ##annualized
        periodGrossRtns.append(periodGrossRtn)
        periodShoringCosts.append(periodShoringCost)
        periodRebalCosts.append(periodRebalCost)
    pnls = pd.DataFrame(index = checkpoints)
    pnls['gross'] = periodGrossRtns
    pnls['rebalCosts'] = periodRebalCosts
    pnls['shortingCosts'] = periodShoringCosts
    return pnls

def getReturnStats(trader_name:str,stats_directory:str,checkpointDates:List[date]):
    acc=[]
    for stats in getStats(trader_name,stats_directory):
        acc.append(stats)
        rebalDate = stats['date']
        if rebalDate in checkpointDates:
            acc2 = acc.copy()
            acc = []
            yield rebalDate, acc2

def buildStatSheet(trader_name:str, stats_directory:str):
    portLevelMeasures = ['totalRisk','factorRisk','residRisk','portAlpha','rebalCost','shortingCost','date']
    portLevelStats = {x:[] for x in portLevelMeasures}
    portLevelStats['longHoldings'], portLevelStats['shortHoldings'] = [],[]
    portLevelStats['longWeights'], portLevelStats['shortWeights'] = [],[]
    dailyReturns = dict(Date=[],Return=[])
    allHoldings = {}
    cusip2name = {}
    for stats in getStats(trader_name,stats_directory):
        rebalDate = stats['date']
        for s in portLevelMeasures:
            portLevelStats[s].append(_clean(stats[s]))
        portLevelStats['longHoldings'].append(len( [x for x in stats['holdings'].values() if x > 0]))
        portLevelStats['shortHoldings'].append(len([x for x in stats['holdings'].values() if x < 0]))
        portLevelStats['longWeights'].append(sum([x for x in stats['holdings'].values() if x > 0]))
        portLevelStats['shortWeights'].append(sum([x for x in stats['holdings'].values() if x < 0]))
        for dt,ret in stats['dailyReturns'].items():
            dailyReturns['Date'].append(dt)
            dailyReturns['Return'].append(ret)
        allHoldings[rebalDate] = stats['holdings']
        for cusip,name in stats['secData'].items():
            if cusip in cusip2name and cusip2name[cusip] != name:
                logging.warning(f'cusip {cusip} changed names from {cusip2name[cusip]} to {name} on {rebalDate.isoformat()}')
            cusip2name[cusip] = name

    portfolioDF = pd.DataFrame.from_dict(portLevelStats)
    dailyReturnsDF = pd.DataFrame.from_dict(dailyReturns)

    #Build holdings dataframe - cusips and names, dates as columnns, weights as values
    holdingsDF= pd.DataFrame()
    allCusips = sorted(cusip2name.keys())
    holdingsDF['cusip'] = allCusips
    holdingsDF['name'] = [cusip2name[c] for c in allCusips]
    allRebalDates = sorted(allHoldings.keys())
    for dt in allRebalDates:
        holdingsDF[dt] = [allHoldings[dt].get(c,0) for c in allCusips]


    return portfolioDF, dailyReturnsDF, holdingsDF

if __name__ == '__main__':
    portfDF, dailyReturnsDF, holdingsDF = buildStatSheet('Luddite','../trading/stats')