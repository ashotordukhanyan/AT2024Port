import logging
from datetime import date
from typing import List, Optional, Set

import numpy as np

from dataload import RiskFactorExposuresLoader, RiskCorrLoader, RiskFactorDefLoader

class RiskCalculator:
    def __init__(self, asofdate: date ):
        self.asofdate = asofdate
        rfeLoader, rcLoader, rfdLoader = RiskFactorExposuresLoader(), RiskCorrLoader(), RiskFactorDefLoader()
        assert rfeLoader.getLatestAvailableDate(asofdate) == rcLoader.getLatestAvailableDate(asofdate) == rfdLoader.getLatestAvailableDate(asofdate)
        self.riskFactorExposures = rfeLoader.getDataAsOf(asofdate).copy()
        self.riskFactorExposures['resid_SD_YR'] = self.riskFactorExposures['resid_SD'] * np.sqrt(12) #for some reason resid_SD is monthly
        self.riskCorr = rcLoader.getDataAsOf(asofdate)
        self.riskFactorDef = rfdLoader.getDataAsOf(asofdate).sort_values('Factor Number')
        assert self.riskFactorDef['Factor Name'].to_list() == self.riskCorr.columns.to_list() ##factors are ordered

        self.riskUniverse = set(self.riskFactorExposures.cusip.dropna().to_list()) # assets covered by risk model

        #Calculate cusip->primary (non PCA) sector mapping for securities in the risk model
        sectorFactors = {f'beta_{i + 1}': i for i in range(8, 28)}  # beta column name -> factor number for fundamental factors
        primFactor = abs(self.riskFactorExposures.set_index('cusip')[list(sectorFactors.keys())]).idxmax(axis=1).map(
            sectorFactors)
        primFactor.name = 'Factor Number'
        primFactor = primFactor.to_frame().reset_index().merge(self.riskFactorDef, on='Factor Number')[
            ['cusip', 'Factor Code', 'Factor Name']].set_index('cusip')
        primFactor.rename(columns={'Factor Code': 'PrimFactorCode', 'Factor Name': 'PrimFactorName'}, inplace=True)
        self.primaryFactorMapping = primFactor.to_dict()

    def getCoveredAssets(self) -> Set[str]:
        return self.riskUniverse

    def getPrimaryFactorCodeMapping(self) -> dict:
        return self.primaryFactorMapping['PrimFactorCode']

    def getPrimaryFactorNameMapping(self) -> dict:
        return self.primaryFactorMapping['PrimFactorName']

    def calculateRisk(self, assets:List[str], weights:np.ndarray):
        assert len(weights) == len(assets)
        notCoveredAssets = [ x for x in assets if x not in self.riskUniverse]
        if notCoveredAssets:
            logging.warning('Assets %s not covered by risk model', ','.join(notCoveredAssets))

        universeFactorExposures = self.riskFactorExposures.set_index('cusip').reindex(assets).reset_index().fillna(0)
        stdDevMatrix = np.diag(np.sqrt(self.riskFactorDef['Factor Variance'].to_numpy()))
        factorVCV = stdDevMatrix @ self.riskCorr.to_numpy() @ stdDevMatrix
        numModelFactors = self.riskFactorDef.shape[0]
        betas = universeFactorExposures[[f'beta_{i}' for i in range(1, numModelFactors + 1)]].to_numpy()
        assetVCV = (betas @ factorVCV @ betas.T)
        residualSD = universeFactorExposures['resid_SD_YR'].to_numpy() / 100.0  # idiosyncratic risk

        factorVariance = weights.T@assetVCV@weights
        residVariance = (residualSD**2)@(weights**2)

        if sum(abs(weights)) > 0:
            assert factorVariance > 0 and residVariance > 0

        totalVariance = factorVariance + residVariance
        factorRisk = np.sqrt(factorVariance); residRisk = np.sqrt(residVariance); totalRisk = np.sqrt(totalVariance)
        portfolioBetas = dict(zip(self.riskFactorDef['Factor Name'].to_list(),(weights@betas).tolist()))

        primarySectors = { a: self.primaryFactorMapping['PrimFactorCode'].get(a,'UNKNOWN') for a in assets }
        return dict(factorRisk=factorRisk.tolist(), residRisk=residRisk.tolist(), totalRisk=totalRisk.tolist(),
                    factorVariance=factorVariance.tolist(), residVariance=residVariance.tolist(), totalVariance=totalVariance.tolist(),
                    portfolioBetas=portfolioBetas, assetVCV=assetVCV, assetResidualVar=(residualSD**2).tolist(), primarySectors=primarySectors)


#TEMP DEVELOPMENT CODE
# if __name__ == '__main__':
#     rc = RiskCalculator(date(2023,1,1))
#     universe = ['55607P20', '15376610', '20171230', '20171220', '31188H10', '69766020', 'G0681212', '18682W20', '69807K10']
#     weights = np.ones(len(universe))
#     risk = rc.calculateRisk(universe,weights)
#     print(risk)
