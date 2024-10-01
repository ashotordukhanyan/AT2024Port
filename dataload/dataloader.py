import functools
import glob
from typing import List,Optional
import re
import os
from datetime import datetime, date
import pandas as pd
import bisect

#temp
os.environ['DATA_DIR'] = 'C:/Users/orduk/PycharmProjects/AT2024Port/data' #TEMP
_DATA_DIR = os.environ['DATA_DIR']
class CSVDataLoader:
    ''' Loads tabular data from a directory of CSV or txt files '''
    def __init__(self, data_dir:str, glob_file_pattern:str, sep:str=',', dtype:dict=None, skiprows=None, columns=None):
        self.data_dir = data_dir
        self.file_pattern = glob_file_pattern
        self.sep = sep
        self.dtype = dtype
        self.skiprows = skiprows
        self.columns = columns

    @functools.lru_cache(maxsize=5)
    def getDates(self) -> List[date]:
        ''' Sorted list of dates for which this dataloader has data'''
        dates = []
        for f in glob.glob(f'{self.data_dir}/{self.file_pattern}'):
            fname = os.path.basename(f)
            match = re.search(r'(\d{8})', fname)
            if match:
                date_str = match.group(1)
                try:
                    date_obj = datetime.strptime(date_str, '%Y%m%d').date()
                    dates.append(date_obj)
                except:
                    pass
        return sorted(dates)

    @functools.lru_cache(maxsize=100)
    def getLatestAvailableDate(self,asof: date) -> date:
        dates = self.getDates()
        if len(dates) == 0:
            raise ValueError('No data found')
        if asof is None:
            return dates[-1]
        else:
            idx = bisect.bisect_right(dates, asof)
            if idx == 0:
                raise ValueError(f'No data as of {asof}')
            return dates[idx - 1]

    def getFileNameForDate(self,dt:date) -> str:
        return f'{self.data_dir}/{self.file_pattern.replace("*", dt.strftime("%Y%m%d"))}'

    def getDataAsOf(self, asof:date) -> pd.DataFrame :
        dt = self.getLatestAvailableDate(asof)
        fname = self.getFileNameForDate(dt)
        return self.readFile(fname)

    @functools.lru_cache(maxsize=4)
    def readFile(self,fname:str) -> pd.DataFrame:
        df = pd.read_csv(fname,sep=self.sep,dtype=self.dtype, skiprows = self.skiprows, names=self.columns,\
                           encoding_errors='replace')
        #drop columns that begin with 'unused'
        df = df[[c for c in df.columns if not c.startswith('unused')]]
        return df

class GeneralUniverseLoader(CSVDataLoader):
    def __init__(self):
        super().__init__(_DATA_DIR+'/GeneralUniverse', 'Universe *.txt', sep='|', \
                         dtype={'cusip':str,'Name':str,'Base.Currency.Mkt.Cap':float})

class InvestableUniverseLoader(CSVDataLoader):
    def __init__(self):
        super().__init__(_DATA_DIR+'/InvestableUniverse', 'InvestableUniverse *.txt', sep='|',\
                         dtype={'cusip':str,'Name':str,'Base.Currency.Mkt.Cap':float})

class ReturnsLoader(CSVDataLoader):
    def __init__(self):
        super().__init__(_DATA_DIR+'/Returns', 'Returns*c.csv', sep=',', dtype={'CUSIP':str,'Return':float} )

    def getDailyReturns(self,startDate:date, endDate:date,cusips:Optional[List[str]] = None) -> pd.DataFrame:
        ''' Get daily returns for a list of cusips between two dates (including start and end dates)'''
        dates = [dt for dt in self.getDates() if startDate <= dt <= endDate]
        dfs = []
        for dt in dates:
            df = self.readFile(self.getFileNameForDate(dt))
            if cusips is not None:
                df = df[df['CUSIP'].isin(cusips)].copy()
            df['Date'] = dt
            dfs.append(df)
        return pd.concat(dfs)

    def getPeriodReturns(self, startDate: date, endDate: date, cusips: Optional[List[str]] = None) -> pd.DataFrame:
        ''' Get start->end cumulative returns for a list of cusips between two dates (including start and end dates)'''
        dailyReturns = self.getDailyReturns(startDate,endDate,cusips).sort_values('Date')
        dailyReturns['Return'] = 1 + dailyReturns['Return']/100.0
        df = dailyReturns.groupby('CUSIP')['Return'].prod()
        #translate back to pct returns
        df=100.*df-100.
        return df

class AlphaLoader(CSVDataLoader):
    def __init__(self):
        super().__init__(_DATA_DIR+'/Alphas', 'Alphas *.txt', sep='|', \
                         dtype={'cusip':str,'Name':str, 'Alpha':float, 'Trade.Cost.Mkt.Cap':float})
    def getAlphas(self, startDate:Optional[date] = date(1900,1,1), endDate:Optional[date] = date(200,1,1), cusips:Optional[List[str]] = None) -> pd.DataFrame:
        ''' Get alphas for a list of cusips for a range of dates'''
        dates = [dt for dt in self.getDates() if startDate <= dt <= endDate]
        dfs = []
        for dt in dates:
            df = self.readFile(self.getFileNameForDate(dt))
            if cusips is not None:
                df = df[df['cusip'].isin(cusips)].copy()
            df['Date'] = dt
            dfs.append(df)
        return pd.concat(dfs)
    @staticmethod
    def GetAlphas(startDate:Optional[date] = date(1900,1,1), endDate:Optional[date] = date(2090,1,1),
                  cusips:Optional[List[str]] = None) -> pd.DataFrame:
        return AlphaLoader().getAlphas(startDate,endDate,cusips)

class RiskCorrLoader(CSVDataLoader):
    def __init__(self):
        super().__init__(_DATA_DIR+'/US 2_19_9g', 'FF_RSQ_RSQRM_US_v2_19_9g_USD_*_Correl.txt', sep='|', skiprows=2)

class RiskFactorDefLoader(CSVDataLoader):
    def __init__(self):
        super().__init__(_DATA_DIR+'/US 2_19_9g', 'FF_RSQ_RSQRM_US_v2_19_9g_USD_*_FactorDef.txt', sep='|', skiprows=2,
                         dtype={'Factor Number':int,'Factor Name':str,'Factor Code':str,'Factor Variance':float})

class RiskFactorExposuresLoader(CSVDataLoader):
    @staticmethod
    def ColumnNames():
        C = ['cusip','name','unused1','price','resid_SD','unused2']

        for fn in range(1,42):
            C.append(f'beta_{fn}')
        C = C + ['unused3','currency', 'total_risk','unused4','unused5','unused6','unused7','unused8','unused9']
        return C
    @staticmethod
    def ColumnTypes():
        r = {}
        for c in RiskFactorExposuresLoader.ColumnNames():
            if c in ['cusip','name','currency']:
                r[c] = str
            elif c.startswith('unused'):
                r[c] = object
            elif c in ['price','total_risk','resid_SD']:
                r[c] = float
            elif c.startswith('beta_'):
                r[c] = float
            else:
                r[c] = object
        return r
    def __init__(self):
        super().__init__(_DATA_DIR +'/US 2_19_9g', 'RSQRM_US_v2_19_9g*_USDCusip.csv', sep=',', \
                         columns=RiskFactorExposuresLoader.ColumnNames(), dtype=RiskFactorExposuresLoader.ColumnTypes())

