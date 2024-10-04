import unittest
from dataload import RiskFactorExposuresLoader, RiskCorrLoader, RiskFactorDefLoader,AlphaLoader,\
    ReturnsLoader,GeneralUniverseLoader,InvestableUniverseLoader
import logging

class TestRiskCalculator(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        self.loaderClasses = [RiskFactorExposuresLoader, RiskCorrLoader, RiskFactorDefLoader, AlphaLoader, ReturnsLoader, GeneralUniverseLoader, InvestableUniverseLoader]

    def test_riskConsistency(self):
        self.assertEqual(RiskFactorExposuresLoader().getDates(),RiskCorrLoader().getDates())
        self.assertEqual(RiskCorrLoader().getDates(), RiskFactorDefLoader().getDates())

    def test_all_dates(self):
        availableDates = [ c().getDates() for c in self.loaderClasses ]
        alldates = set(availableDates[0])
        for dates in availableDates[1:]:
            alldates = alldates.union(dates)
        alldates = sorted(list(alldates))
        with open('calendar.csv', 'w') as f:
            print('Date,weekday,' ,  ','.join([c.__name__ for c in self.loaderClasses]),file=f)
            for d in alldates:
                print(d.strftime('%Y-%m-%d'),',',d.weekday(),end=',',file=f)
                for dates in availableDates:
                    print('Y' if d in dates else 'N',end=',',file=f)
                print(file=f)

    def test_schemaConsistency(self):
        for clazz in self.loaderClasses:
            logging.warning('Testing schema consistency for %s', clazz.__name__)
            loader = clazz()
            dates = loader.getDates()
            df0 = loader.getDataAsOf(dates[0]);
            schema = df0.dtypes
            totalRows = 0
            for index, d in enumerate(dates):
                logging.debug('Validating schema for %s date %s', clazz, d)
                df = loader.getDataAsOf(d)
                assert df is not None and len(df) > 0
                self.assertTrue(len(df.dtypes) == len(schema),f'Schema length mismatch for {clazz.__name__} date {d} vs {loader.getDates()[index - 1]} ')
                self.assertTrue((df.dtypes==schema).all(),f'Schema mismatch for {clazz.__name__} date {d} vs {loader.getDates()[index - 1]} ')
                totalRows += len(df)
            self.assertTrue(totalRows>0)