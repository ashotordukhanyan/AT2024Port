import unittest
from datetime import date
import numpy as np
from risk.riskcalc import RiskCalculator
from dataload import InvestableUniverseLoader

class TestRiskCalculator(unittest.TestCase):
    def setUp(self) -> None:
        self.test_date = date(2023,1,1)
        self.universe = InvestableUniverseLoader().getDataAsOf(self.test_date)['cusip'].to_list()[0:100]
        self.rc = RiskCalculator(self.test_date)
        #self.rc.loadRiskData(self.test_date, self.universe)

    def test_calculateRisk(self):
        weights = np.ones(len(self.universe))
        risk = self.rc.calculateRisk(self.universe,weights)
        self.assertTrue(all([ x in risk for x in ['factorRisk', 'residRisk','totalRisk','factorVariance',\
                                                       'residVariance','totalVariance','portfolioBetas']]))
        self.assertAlmostEqual(risk['factorRisk'], 24.87,2)
        self.assertAlmostEqual(risk['residRisk'], 2.78,2)
        self.assertAlmostEqual(risk['totalRisk'], 25.02,2)
        self.assertAlmostEqual(risk['factorVariance'], 618.43,2)
        self.assertAlmostEqual(risk['residVariance'], 7.73,2)
        self.assertAlmostEqual(risk['totalVariance'], 626.17,2)
        self.assertEqual(len(risk['portfolioBetas']), 41)

if __name__ == '__main__':
    unittest.main()