import logging
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
from trading.trader import Trader,OptimizingTrader,Guild

traders = [OptimizingTrader(name='Luddite', avoidSectors='USIT USHT USHE'.split(),txn_cost_multiplier=1.0,
                            shorting_cost_multiplier=1.0,risk_multiplier=5.0,max_iterations=1000,alpha_multiplier=1.0 )]
trader = traders[0]
#guild = Guild(traders)
alpha_loader = AlphaLoader()
rebalance_dates = alpha_loader.getDates()[0:1]
universe_loader = InvestableUniverseLoader()
returns_loader = ReturnsLoader()

for dt in rebalance_dates:
    for _ in range(5):
        secData = universe_loader.getDataAsOf(dt).set_index('cusip')
        riskCal = RiskCalculator(dt)
        # lets check if we have market data - sometimes we have alphas for delisted assets
        tradable_universe = [c for c in secData.index.to_list() if
                             c in riskCal.getCoveredAssets() ]  # assets we can trade
        #alphaData = alpha_loader.getDataAsOf(dt).set_index('cusip')
        #TEMP OVERRITE TO KEEP ALPHA CONSTANT !
        alphaData = alpha_loader.getDataAsOf(rebalance_dates[0]).set_index('cusip')
        # sort alpha dataframe in the order of cusips in universe df
        alphas = alphaData.reindex(tradable_universe)['Alpha'].fillna(0).to_numpy() / 100.0  # translate from pct
        mktCaps = secData.loc[tradable_universe]['Base.Currency.Mkt.Cap'].to_numpy() * 1e6  # translate from millions
        txnCosts = (20.0 - 15.0 * (np.log(mktCaps) - np.log(min(mktCaps))) / (
                    np.log(max(mktCaps)) - np.log(min(mktCaps)))) / 10000.0
        shortingCosts = (100.0 - 80.0 * (np.log(mktCaps) - np.log(min(mktCaps))) / (
                    np.log(max(mktCaps)) - np.log(min(mktCaps)))) / 10000.0
        assert len(alphas) == len(tradable_universe) and not any(np.isnan(alphas))
        assert len(txnCosts) == len(tradable_universe) and not any(np.isnan(txnCosts))
        assert len(shortingCosts) == len(tradable_universe) and not any(np.isnan(shortingCosts))
        asset = 'G2519Y10'
        logging.warning('Rebalancing portfolio for %s', trader.name)
        logging.warning('Alpha for %s is %.4f', asset, alphas[tradable_universe.index(asset)])
        portStats = trader.rebalance(dt, tradable_universe, riskCal, alphas, txnCosts, shortingCosts, secData)
        logging.warning("POSITION FOR %s NOW %.4f", asset, trader.portfolio.get('G2519Y10',0) )
