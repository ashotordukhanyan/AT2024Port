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
def getStatData(trader_name:str,stats_directory:str,max_date:date):
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
    portRebalDf = portRebalDf[portRebalDf.index <= max_date]

    #Prepare a frame of holdings across rebalance dates
    holdingsDF = pd.DataFrame(holdingsRows,columns=['date','cusip','weight','name'])
    holdingsDF = holdingsDF[holdingsDF['date'] <= max_date]

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
    portBetaDF = portBetaDF[portBetaDF['date'] <= max_date]

    portBetaDF = portBetaDF.pivot(index='factor',columns='date',values='beta').fillna(0)
    factorsInOrder = list(s['portfolioBetas'].keys())
    #reorder portBetaDF by index to match factorsInOrder
    portBetaDF = portBetaDF.loc[factorsInOrder]

    #series of daily returns
    dailyReturns = pd.Series(dailyReturns)
    dailyReturns = dailyReturns[dailyReturns.index <= max_date]
    dailyReturns.index.name='date'
    dailyReturns.name='return'

    return portRebalDf, holdingsDF,portBetaDF,dailyReturns

def generateStatsSheet(trader_name:str,stats_directory:str,checkpoints:List[date]) -> str :
    portRebalDf, holdingsDF, portBetaDF, dailyReturns = getStatData(trader_name,stats_directory,checkpoints[-1])
    checkpointReturns = dailyReturns.groupby([ checkpoints[bisect.bisect_left(checkpoints,x)] for x in dailyReturns.index.to_list()]).\
        apply(lambda g: 100 * ((1 + 0.01 * g).prod() - 1))

    gb=portRebalDf.groupby([checkpoints[bisect.bisect_left(checkpoints,x)] for x in sorted(portRebalDf.index.to_list())])
    agg_rules = {x:'mean' for x in 'totalRisk factorRisk residRisk portAlpha turnover shortingCost'.split()}
    agg_rules['rebalCost'] = 'sum'
    portCheckPointStats = gb.agg(agg_rules)
    portCheckPointStats['shortingCost'] = portCheckPointStats['shortingCost'] * 4/52 ##annualize
    portCheckPointStats['grossReturn'] = checkpointReturns
    portCheckPointStats['Portfolio Returns % (Gross)'] = portCheckPointStats['grossReturn']/100.0
    portCheckPointStats['Transaction & Shorting Costs %'] = (portCheckPointStats['rebalCost'] + portCheckPointStats['shortingCost'])/100.0
    portRebalDf['grossReturn'] = dailyReturns
    portCheckPointBetas = portBetaDF[checkpoints]

    fileName = f'{trader_name}_statsos.xlsx'
    with pd.ExcelWriter(f'{fileName}',date_format="YYYY-MM-DD") as writer:
        holdingsDF.to_excel(writer, sheet_name='holdings', float_format='%.4f')
        portCheckPointBetas.to_excel(writer,sheet_name='betas',float_format='%.4f')
        #checkpointReturns.to_excel(writer, sheet_name='returns', float_format='%.4f')
        portCheckPointStats.T.to_excel(writer,sheet_name='portfolio',float_format='%.4f')
        portRebalDf.T.to_excel(writer,sheet_name='full_portfolio',float_format='%.4f')
        portBetaDF.to_excel(writer,sheet_name='full_betas',float_format='%.4f')
        #dailyReturns.to_excel(writer,sheet_name='full_returns',float_format='%.4f')
    return os.path.abspath(fileName)


def getStats(trader_name:str, stats_directory:str):
    for statsFile in glob.glob(f'{stats_directory}/pstats_{trader_name}_*.pkl'):
        with open(statsFile,'rb') as f:
            stats = pickle.load(f)
            yield stats

def _clean(x):
    return float(x) if isinstance(x,np.float64) else x

if __name__ == '__main__':
    #checkpoints = [date(2017,12,27) + timedelta(weeks=4*i) for i in range(87)]
    checkpoints = [date(2014, 8, 13) + timedelta(weeks=4 * i) for i in range(90)]
    trader_names = ['Luddite1_1','Luddite3_1','Luddite5_1','Luddite10_1']
    stats_directory = '../trading/statsos'
    for trader_name in trader_names:
        filename = generateStatsSheet(trader_name,stats_directory,checkpoints)
        print('Generated ',filename)
