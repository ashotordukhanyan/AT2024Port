import os.path
import pickle
import logging
import glob
import numpy as np
import pandas as pd
from typing import List
from datetime import date,timedelta
import os
import bisect
def getStatData(trader_name:str,stats_directory:str):
    holdingsRows = []
    portBetaRows = []
    portfolioRebalRows = []; PORT_MEASURES = ['date','totalRisk','factorRisk','residRisk','portAlpha','rebalCost','shortingCost']
    dailyReturns = {}

    for s in getStats(trader_name, stats_directory):
        dt = s['date']
        cusip2Name = s['secData']
        for cusip,weight in s['holdings'].items():
            holdingsRows.append([dt, cusip, weight, cusip2Name[cusip]])
        for factor,beta in s['portfolioBetas'].items():
            portBetaRows.append([dt,factor,beta])
        for d,r in s['dailyReturns'].items():
            dailyReturns[d] = r
        portfolioRebalRows.append([s[x] for x in PORT_MEASURES])


    #Prepare portfolio level frame
    portRebalDf = pd.DataFrame(portfolioRebalRows,columns=PORT_MEASURES).set_index('date')

    #Prepare a frame of holdings across rebalance dates
    holdingsDF = pd.DataFrame(holdingsRows,columns=['date','cusip','weight','name'])
    cusip2lastName = holdingsDF.groupby(['cusip'])['name'].last()
    holdingsDF = holdingsDF.pivot(index='cusip',columns='date',values='weight').fillna(0).join(cusip2lastName)
    #make name first column
    holdingsDF = holdingsDF[[holdingsDF.columns[-1]] + list(holdingsDF.columns[:-1])]
    #append turnover to portRebalDl
    rebalDates = sorted(portRebalDf.index.to_list())
    turnovers = {}
    for i in range(0,len(rebalDates)):
        turnover = holdingsDF[rebalDates[i]].abs().sum() if i==0 else (holdingsDF[rebalDates[i]] - holdingsDF[rebalDates[i-1]]).abs().sum()
        turnovers[rebalDates[i]] = float(turnover)
    portRebalDf['turnover'] = pd.Series(turnovers)

    #prepare a frame of portfolio betas across rebalance dates
    portBetaDF = pd.DataFrame(portBetaRows,columns=['date','factor','beta'])
    portBetaDF = portBetaDF.pivot(index='factor',columns='date',values='beta').fillna(0)
    factorsInOrder = list(s['portfolioBetas'].keys())
    #reorder portBetaDF by index to match factorsInOrder
    portBetaDF = portBetaDF.loc[factorsInOrder]

    #series of daily returns
    dailyReturns = pd.Series(dailyReturns)
    dailyReturns.index.name='date'
    dailyReturns.name='return'

    return portRebalDf, holdingsDF,portBetaDF,dailyReturns

def generateStatsSheet(trader_name:str,stats_directory:str,checkpoints:List[date]) -> str :
    portRebalDf, holdingsDF, portBetaDF, dailyReturns = getStatData(trader_name,stats_directory)
    checkpointReturns = dailyReturns.groupby([ checkpoints[bisect.bisect_left(checkpoints,x)] for x in dailyReturns.index.to_list()]).\
        apply(lambda g: 100 * ((1 + 0.01 * g).prod() - 1))

    gb=portRebalDf.groupby([checkpoints[bisect.bisect_left(checkpoints,x)] for x in sorted(portRebalDf.index.to_list())])
    agg_rules = {x:'mean' for x in 'totalRisk factorRisk residRisk portAlpha turnover shortingCost'.split()}
    agg_rules['rebalCost'] = 'sum'
    portCheckPointStats = gb.agg(agg_rules)
    portCheckPointStats['shortingCost'] = portCheckPointStats['shortingCost'] * 4/52 ##annualize
    portCheckPointStats['grossReturn'] = checkpointReturns
    portRebalDf['grossReturn'] = dailyReturns
    portCheckPointBetas = portBetaDF[checkpoints]

    fileName = f'{trader_name}_stats.xlsx'
    with pd.ExcelWriter(f'{trader_name}_stats.xlsx',date_format="YYYY-MM-DD") as writer:
        holdingsDF.to_excel(writer, sheet_name='holdings', float_format='%.4f')
        portCheckPointBetas.to_excel(writer,sheet_name='betas',float_format='%.4f')
        #checkpointReturns.to_excel(writer, sheet_name='returns', float_format='%.4f')
        portCheckPointStats.T.to_excel(writer,sheet_name='portfolio',float_format='%.4f')
        portRebalDf.T.to_excel(writer,sheet_name='full_portfolio',float_format='%.4f')
        portBetaDF.to_excel(writer,sheet_name='full_betas',float_format='%.4f')
        #dailyReturns.to_excel(writer,sheet_name='full_returns',float_format='%.4f')
    return os.path.abspath(fileName)

# def getPortfolioBetas(trade_name:str,stats_directory:str, needed_dates:List[date]):
#     df = pd.DataFrame()
#     for stats in portfolioBetaStats(trade_name,stats_directory,needed_dates):
#         dt = stats[0]
#         betas = stats[1]
#         betaSeries = pd.Series(betas)
#         betaSeries.name = dt.isoformat()
#         df[betaSeries.name]= betaSeries
#     return df
#
#
# def portfolioBetaStats(trade_name:str,stats_directory:str, needed_dates:List[date]):
#     for stats in getStats(trade_name,stats_directory):
#         rebalDate = stats['date']
#         if rebalDate in needed_dates:
#             yield rebalDate, stats['portfolioBetas']

def getStats(trader_name:str, stats_directory:str):
    for statsFile in glob.glob(f'{stats_directory}/pstats_{trader_name}_*.pkl'):
        with open(statsFile,'rb') as f:
            stats = pickle.load(f)
            yield stats

def _clean(x):
    return float(x) if isinstance(x,np.float64) else x

# def getPNLStats(trader_name:str,stats_directory:str,checkpoints):
#     rebalCosts = {}; shortingCosts={};dailyReturns={}
#     for stats in getStats(trader_name,stats_directory):
#         dt = stats['date']
#         rebalCosts[dt] = float(stats['rebalCost'])
#         shortingCosts[dt] = float(stats['shortingCost'])
#         for d,r in stats['dailyReturns'].items():
#             dailyReturns[d] = r
#     dailyReturns = pd.Series(dailyReturns)
#     rebalCosts = pd.Series(rebalCosts)
#     shortingCosts = pd.Series(shortingCosts)
#     #return rebalCosts, shortingCosts, dailyReturns
#     periodGrossRtns = [0]; periodRebalCosts = [0]; periodShoringCosts = [0]
#
#     for i in range(1,len(checkpoints)):
#         cp = checkpoints[i]
#         previous_cp = checkpoints[i-1]
#         (1+dailyReturns[(dailyReturns.index > previous_cp) & (dailyReturns.index <= cp)]).prod()
#         periodGrossRtn = float(100*((1+.01*(dailyReturns[(dailyReturns.index > previous_cp) & (dailyReturns.index <= cp)])).prod()-1))
#         periodRebalCost = float(rebalCosts[(rebalCosts.index > previous_cp) & (rebalCosts.index <= cp)].sum())
#         periodShoringCost = float(shortingCosts[(shortingCosts.index > previous_cp) & (shortingCosts.index <= cp)].mean()) * 4/52 ##annualized
#         periodGrossRtns.append(periodGrossRtn)
#         periodShoringCosts.append(periodShoringCost)
#         periodRebalCosts.append(periodRebalCost)
#     pnls = pd.DataFrame(index = checkpoints)
#     pnls['gross'] = periodGrossRtns
#     pnls['rebalCosts'] = periodRebalCosts
#     pnls['shortingCosts'] = periodShoringCosts
#     return pnls
#
# def getReturnStats(trader_name:str,stats_directory:str,checkpointDates:List[date]):
#     acc=[]
#     for stats in getStats(trader_name,stats_directory):
#         acc.append(stats)
#         rebalDate = stats['date']
#         if rebalDate in checkpointDates:
#             acc2 = acc.copy()
#             acc = []
#             yield rebalDate, acc2
#
# def buildStatSheet(trader_name:str, stats_directory:str):
#     portLevelMeasures = ['totalRisk','factorRisk','residRisk','portAlpha','rebalCost','shortingCost','date']
#     portLevelStats = {x:[] for x in portLevelMeasures}
#     portLevelStats['longHoldings'], portLevelStats['shortHoldings'] = [],[]
#     portLevelStats['longWeights'], portLevelStats['shortWeights'] = [],[]
#     dailyReturns = dict(Date=[],Return=[])
#     allHoldings = {}
#     cusip2name = {}
#     for stats in getStats(trader_name,stats_directory):
#         rebalDate = stats['date']
#         for s in portLevelMeasures:
#             portLevelStats[s].append(_clean(stats[s]))
#         portLevelStats['longHoldings'].append(len( [x for x in stats['holdings'].values() if x > 0]))
#         portLevelStats['shortHoldings'].append(len([x for x in stats['holdings'].values() if x < 0]))
#         portLevelStats['longWeights'].append(sum([x for x in stats['holdings'].values() if x > 0]))
#         portLevelStats['shortWeights'].append(sum([x for x in stats['holdings'].values() if x < 0]))
#         for dt,ret in stats['dailyReturns'].items():
#             dailyReturns['Date'].append(dt)
#             dailyReturns['Return'].append(ret)
#         allHoldings[rebalDate] = stats['holdings']
#         for cusip,name in stats['secData'].items():
#             if cusip in cusip2name and cusip2name[cusip] != name:
#                 logging.warning(f'cusip {cusip} changed names from {cusip2name[cusip]} to {name} on {rebalDate.isoformat()}')
#             cusip2name[cusip] = name
#
#     portfolioDF = pd.DataFrame.from_dict(portLevelStats)
#     dailyReturnsDF = pd.DataFrame.from_dict(dailyReturns)
#
#     #Build holdings dataframe - cusips and names, dates as columnns, weights as values
#     holdingsDF= pd.DataFrame()
#     allCusips = sorted(cusip2name.keys())
#     holdingsDF['cusip'] = allCusips
#     holdingsDF['name'] = [cusip2name[c] for c in allCusips]
#     allRebalDates = sorted(allHoldings.keys())
#     for dt in allRebalDates:
#         holdingsDF[dt] = [allHoldings[dt].get(c,0) for c in allCusips]
#
#
#     return portfolioDF, dailyReturnsDF, holdingsDF

if __name__ == '__main__':
    checkpoints = [date(2017,12,27) + timedelta(weeks=4*i) for i in range(87)]
    trader_name = 'Luddite2'
    stats_directory = '../trading/stats'
    filename = generateStatsSheet(trader_name,stats_directory,checkpoints)
    print(filename)
