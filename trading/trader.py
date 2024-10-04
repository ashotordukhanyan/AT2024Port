import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
from datetime import date, timedelta
from dataload import InvestableUniverseLoader, AlphaLoader, ReturnsLoader
from risk.riskcalc import RiskCalculator
from portopt import PortOptimizer
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

class Trader(ABC):
    def __init__(self,initial_portfolio: Dict[str,float] = None, name = None):
        ''' Create a new trader with an initial portfolio
            :param initial_portfolio: A dictionary of cusip to weights
        '''
        self.portfolio = initial_portfolio or {}
        #self.universe_loader = InvestableUniverseLoader()
        #self.alpha_loader = AlphaLoader()
        self.name = name or self.__class__.__name__
        #self.lastRebalanceDate = None

    def saveStats(self,stats:Dict,dt:date):
        fname = f'./statsos/pstats_{self.name}_{dt.strftime("%Y%m%d")}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(stats,f)
    def loadState(self,dt:date):
        fname = f'./statsos/pstats_{self.name}_{dt.strftime("%Y%m%d")}.pkl'
        with open(fname,'rb') as f:
            stats = pickle.load(f)
        self.portfolio = stats['holdings']

    def rescale(self,weights:np.ndarray,max_concentration:float, max_iterations = 10):
        ''' Rescale weights to ensure that long weights and short weights sum to 1 and -1 respectively without breaking max_concentration
            :param weights: The weights to rescale
        '''
        longWeights = np.where(weights>0,weights,0)
        iterations = 0
        while( abs(sum(longWeights) - 1) > .05): ##more than 5% un or over allocated
            longWeights = np.clip(longWeights/sum(longWeights),0,max_concentration)
            iterations += 1
            if iterations > max_iterations:
                print(longWeights)
                raise ValueError('Max iterations reached attempting to rescale long weights')

        iterations = 0
        shortWeights = np.where(weights<0,weights,0)
        while ( abs(-sum(shortWeights) - 1) > .05 ): ##more than 5% un or over allocated
            shortWeights = np.clip(shortWeights/-sum(shortWeights),-max_concentration,0)
            if iterations > max_iterations:
                print(shortWeights)
                raise ValueError('Max iterations reached attempting to rescale short weights')
        return longWeights + shortWeights


    def rebalance(self,dt:date, tradable_universe: np.ndarray, riskCal: RiskCalculator, alphas: np.ndarray, txnCosts: np.ndarray,
                  shortingCosts: np.ndarray, secData: pd.DataFrame):
        ''' Rebalance the portfolio for a given date
            :param dt: The date for which the portfolio is rebalanced
            :param tradable_universe: A list of cusips in the universe
            :param riskCal: RiskCalculator instance
            :param alphas: alphas (e.g. 0.0007 for 7bps)
            :param txn_costs: transaction costs
            :param shorting_costs: shorting costs (cost of holding a short position)
        '''

        initial_weights = np.array([self.portfolio.get(cusip, 0) for cusip in tradable_universe])
        riskData = riskCal.calculateRisk(tradable_universe,initial_weights)
        newWeights = self.calculateNewWeights(dt, tradable_universe, initial_weights,alphas,txnCosts,shortingCosts,riskData)
        newWeights = self.rescale(newWeights,0.25)

        trades = self.updatePositions(tradable_universe,newWeights)
        newRiskData = riskCal.calculateRisk(tradable_universe,np.array([self.portfolio.get(cusip,0) for cusip in tradable_universe]))
        logging.warning('%s Doing %d trades. Have positions in %d assets', self.name, len(trades), len(self.portfolio))

        #calculate stats to save for further analysis
        portStats = newRiskData.copy()
        t_positions = np.array([self.portfolio.get(cusip,0) for cusip in tradable_universe])
        t_trades = np.array([trades.get(cusip,0) for cusip in tradable_universe])
        portStats['portAlpha'] = t_positions@alphas
        portStats['rebalCost'] = np.abs(t_trades)@txnCosts
        portStats['shortingCost'] = -1*np.where(t_positions<0,t_positions,0)@shortingCosts
        portStats['holdings'] = self.portfolio
        portStats['date'] = dt
        portStats['secData'] =  secData.loc[self.portfolio.keys()]['Name'].to_dict() #asset cusip -> name mapping

        #self.lastRebalanceDate = dt
        return portStats

    def updatePositions(self,tradable_universe:List[str],newWeights:np.ndarray) -> Dict[str,float]:
        ''' Update the portfolio with new positions
            :param positions: A dictionary of cusip to weights
            :return: A dictionary of trades indexed by cusip
        '''

        trades = {}
        min_trade = 1e-6
        for index,asset in enumerate(tradable_universe):
            newWeight = newWeights[index]
            oldWeight = self.portfolio.get(asset,0)
            tradeWeight = float(newWeight - oldWeight)

            if abs(tradeWeight) > 1e-6:
                self.portfolio[asset] = newWeight
                trades[asset] = tradeWeight

        # assets that we had a position in but are no longer in investable or risk universes
        assetsToUnload = [c for c in self.portfolio.keys() if c not in tradable_universe]

        for asset in assetsToUnload:
            trades[asset] = -self.portfolio[asset]
            del self.portfolio[asset]

        return trades

    @abstractmethod
    def calculateNewWeights(self,dt:date,instruments: List[str], initialWeights:np.ndarray,
                            alphas : np.ndarray, txn_costs: np.ndarray, shorting_costs: np.ndarray, risk_data: Dict) -> np.ndarray:
        ''' Calculate new weights based on the current portfolio, universe and alpha
            :param dt: The date for which the weights are calculated
            :param instruments: A list of cusips in the universe
            :param initialWeights: The current portfolio weights
            :param alphas: alphas (e.g. 0.0007 for 7bps)
            :param txn_costs: transaction costs
            :param shorting_costs: shorting costs (cost of holding a short position)
            :param risk_data: risk data from RiskCalculator.calculateRisk
            :return: new weights
        '''
        pass


class OptimizingTrader(Trader):
    ''' Trader that uses classical Markowitz optimization to rebalance the portfolio'''
    def __init__(self,initial_portfolio: Dict[str,float] = None, risk_multiplier = 10., alpha_multiplier = 1., \
                 txn_cost_multiplier = 1., shorting_cost_multiplier = 1., max_contentration:float = 0.25, name = None,
                 max_iterations = 100,
                 avoidSectors = [] ):
        super().__init__(initial_portfolio,name)
        self.risk_multiplier = risk_multiplier
        self.alpha_multiplier = alpha_multiplier
        self.txn_cost_multiplier = txn_cost_multiplier
        self.shorting_cost_multiplier = shorting_cost_multiplier
        self.max_concentration = max_contentration
        self.avoidSectors = avoidSectors
        self.max_iterations = max_iterations
        self.optimizer = None

    def calculateNewWeights(self, dt: date, instruments: List[str], initialWeights: np.ndarray,
                                alphas: np.ndarray, txn_costs: np.ndarray, shorting_costs: np.ndarray,
                                risk_data: Dict) -> np.ndarray:

        sectorBounds = None
        if self.avoidSectors is not None:
            ##If we want to avoid sectors - import sell only constraints on positions in those sectors
            sectorBounds = []
            for index,instrument in enumerate(instruments):
                sector = risk_data['primarySectors'][instrument]
                bound = None
                if sector in self.avoidSectors:
                    currentPosition = initialWeights[index]
                    bound = (0,currentPosition) if currentPosition>=0 else (currentPosition,0)
                sectorBounds.append(bound)

        VCV = risk_data['assetVCV'] + np.diag(risk_data['assetResidualVar'])
        attempt = 0
        converged = False
        prev_weights = initialWeights
        while not converged and attempt < 10:
            self.optimizer = PortOptimizer(VCV,initialWeights,alphas,txn_costs,shorting_costs,self.risk_multiplier,self.alpha_multiplier,
                                   self.txn_cost_multiplier,self.shorting_cost_multiplier,self.max_concentration,weight_bounds=sectorBounds)
            solution = self.optimizer.solve()
            weights = solution['optPositions']
            if sum(abs(weights - prev_weights)) < .01:
                logging.debug('Converged after %d iterations',attempt)
                converged = True
            prev_weights = weights
        if not converged:
            logging.warning('Did not converge after %d iterations',attempt)
        return weights

class HighConviction(Trader):
    ''' Trader that trades based on alpha only - top x% long, botton x% short - equally weighted'''
    def __init__(self,initial_portfolio: Dict[str,float] = None, name = None, long_pct = 0.1, short_pct = 0.1):
        super().__init__(initial_portfolio,name)
        self.long_pct = long_pct
        self.short_pct = short_pct

    def calculateNewWeights(self, dt: date, instruments: List[str], initialWeights: np.ndarray,
                                alphas: np.ndarray, txn_costs: np.ndarray, shorting_costs: np.ndarray,
                                risk_data: Dict) -> np.ndarray:
        longs = np.where(alphas >= np.percentile(alphas,100*(1-self.long_pct)))[0]
        shorts = np.where(alphas <= np.percentile(alphas,100*self.short_pct))[0]
        newWeights = np.zeros(len(alphas))
        newWeights[longs] = 1/len(longs)
        newWeights[shorts] = -1/len(shorts)
        return newWeights

class Indiscriminate(Trader):
    ''' Trader that trades based on alpha only  - all stocks long or short weighted by alpha '''
    def __init__(self,initial_portfolio: Dict[str,float] = None, name = None):
        super().__init__(initial_portfolio,name)

    def calculateNewWeights(self, dt: date, instruments: List[str], initialWeights: np.ndarray,
                                alphas: np.ndarray, txn_costs: np.ndarray, shorting_costs: np.ndarray,
                                risk_data: Dict) -> np.ndarray:
        longs = np.where(alphas >= 0)[0]
        shorts = np.where(alphas <= 0)[0]
        newWeights = np.zeros(len(alphas))
        newWeights[longs] = alphas[longs]/np.sum(alphas[longs])
        newWeights[shorts] = -1*alphas[shorts]/np.sum(alphas[shorts])
        return newWeights

class Guild:
    ''' A collection of PMs that each have access to the same alpha etc. data but trade independently'''
    def __init__(self, traders:List[Trader], MULTITHREAD = False):
        self.traders = traders
        ##assert that the list of trader names is unique
        assert len(set([t.name for t in self.traders])) == len(self.traders)

        self.universe_loader = InvestableUniverseLoader()
        self.alpha_loader = AlphaLoader()
        self.returns_loader = ReturnsLoader()
        self.lastRebalanceDate = None
        if MULTITHREAD:
            self.executor = ThreadPoolExecutor(max_workers=len(self.traders))
            self.executor.__enter__()

    def rebalance(self,dt:date, dryMode:bool = False):
        ''' Rebalance all PM portfolio for a given date
            :param dt: The date for which the portfolio is rebalanced
        '''
        firstRebalance = self.lastRebalanceDate == None
        if not firstRebalance:
            assert dt > self.lastRebalanceDate
            #Calculate daily returns for the period that just ended
            # combine universe across all traders
            all_cusips = sorted(list(set((np.concatenate([ list(t.portfolio.keys()) for t in self.traders ]).tolist()))))
            dailyReturns = self.returns_loader.getDailyReturns(self.lastRebalanceDate+timedelta(days=1),dt,all_cusips)
            past_dates = sorted(dailyReturns.Date.unique().tolist())

            trader_returns = {}
            for trader in self.traders:
                portfolio_cusips = list(trader.portfolio.keys())
                portfolio_weights = np.array(list(trader.portfolio.values()))
                portfolio_daily_returns = {}
                for past_date in past_dates :
                    portfolio_daily_returns[past_date] = float(dailyReturns[dailyReturns.Date == past_date].set_index('CUSIP').\
                        reindex(portfolio_cusips).fillna(0)['Return'].to_numpy().dot(portfolio_weights))
                trader_returns[trader.name] = portfolio_daily_returns

        logging.warning('Rebalancing portfolios for %s', dt)

        secData = self.universe_loader.getDataAsOf(dt).set_index('cusip')
        riskCal = RiskCalculator(dt)
        #lets check if we have market data - sometimes we have alphas for delisted assets
        validMD = set(riskCal.getCoveredAssets()) if firstRebalance else set(self.returns_loader.getDataAsOf(dt).dropna()['CUSIP'].to_list())
        tradable_universe = [c for c in secData.index.to_list() if
                             c in riskCal.getCoveredAssets() and c in validMD]  # assets we can trade
        alphaData = self.alpha_loader.getDataAsOf(dt).set_index('cusip')
        #sort alpha dataframe in the order of cusips in universe df
        alphas = alphaData.reindex(tradable_universe)['Alpha'].fillna(0).to_numpy()/100.0 #translate from pct
        mktCaps = secData.loc[tradable_universe]['Base.Currency.Mkt.Cap'].to_numpy() * 1e6 #translate from millions
        txnCosts = (20.0 -15.0 * (np.log(mktCaps)-np.log(min(mktCaps)))/ (np.log(max(mktCaps))-np.log(min(mktCaps))))/10000.0
        shortingCosts = (100.0 -80.0 * (np.log(mktCaps)-np.log(min(mktCaps)))/ (np.log(max(mktCaps))-np.log(min(mktCaps))))/10000.0
        assert len(alphas) == len(tradable_universe) and not any(np.isnan(alphas))
        assert len(txnCosts) == len(tradable_universe) and not any(np.isnan(txnCosts))
        assert len(shortingCosts) == len(tradable_universe ) and not any(np.isnan(shortingCosts))
        #TODO - add ThreadPoolExecutor parralelization here to execute each trader on a separate thread
        def _rebalTrader(trader):
            logging.warning('Rebalancing portfolio for %s', trader.name)
            portStats = trader.rebalance(dt, tradable_universe, riskCal, alphas, txnCosts, shortingCosts, secData)
            portStats['dailyReturns'] = trader_returns[trader.name] if not firstRebalance else {}
            if not dryMode:
                trader.saveStats(portStats, dt)
            return True
        if self.executor is not None:
           waitForIt = list(self.executor.map(_rebalTrader,self.traders)) #map to list needed because returned gen is lazy
        else:
            for trader in self.traders:
                _rebalTrader(trader)
        self.lastRebalanceDate = dt
    def finish(self):
        if self.executor is not None:
            self.executor.__exit__(None,None,None)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING,format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
    #trader = OptimizingTrader(alpha_multiplier=3.0, name='OptimizingTraderHA')
    traders = [ OptimizingTrader(name='OldSchool'),
                OptimizingTrader(alpha_multiplier=3.0, name='Gambler'),
                OptimizingTrader(alpha_multiplier=3.0, risk_multiplier = 0.1, name='Cowboy'),
                HighConviction(name='HiCon'),
                Indiscriminate(name='Indiscriminate')]
    traders = [ OptimizingTrader(name='Luddite',avoidSectors='USIT USHT USHE'.split(),risk_multiplier = 3.0 )]
    traders = [
        OptimizingTrader(name='Luddite1_1', avoidSectors='USIT USHT USHE'.split(), risk_multiplier=1.0),
        OptimizingTrader(name='Luddite3_1', avoidSectors='USIT USHT USHE'.split(), risk_multiplier=3.0, txn_cost_multiplier=3.0),
        OptimizingTrader(name='Luddite5_1', avoidSectors='USIT USHT USHE'.split(), risk_multiplier=3.0, txn_cost_multiplier=5.0),
        OptimizingTrader(name='Luddite10_1', avoidSectors='USIT USHT USHE'.split(), risk_multiplier=3.0, txn_cost_multiplier=10.0)
    ]
    ##traders = [OptimizingTrader(name='OldSchool')]

    guild = Guild(traders,MULTITHREAD=True)
    #for the sake of this exercise assume that we rebalance at the same frequency as alphas are available
    LAST_CHECKPOINT_DATE = date(2020,11,25)
    rebalance_dates = guild.alpha_loader.getDates()
    #earliest date for risk model
    rebalance_dates = [rd for rd in rebalance_dates if rd >= date(2014, 8, 13) ]

    if LAST_CHECKPOINT_DATE in rebalance_dates:
        guild.lastRebalanceDate = LAST_CHECKPOINT_DATE
        for t in guild.traders:
            t.loadState(LAST_CHECKPOINT_DATE)
        rebalance_dates = [d for d in rebalance_dates if d > LAST_CHECKPOINT_DATE]

    for d in tqdm(rebalance_dates):
        guild.rebalance(d, dryMode=False)
    guild.finish()