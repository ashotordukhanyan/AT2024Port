from dataload import ReturnsLoader, GeneralUniverseLoader, InvestableUniverseLoader, AlphaLoader,\
    RiskCorrLoader,RiskFactorDefLoader,RiskFactorExposuresLoader

from pandas.api.types import is_numeric_dtype

import logging
#for loaderC in [GeneralUniverseLoader, InvestableUniverseLoader, AlphaLoader,ReturnsLoader,RiskCorrLoader,RiskFactorDefLoader]:
for loaderC in [RiskFactorExposuresLoader, RiskCorrLoader, RiskFactorDefLoader]:
    logging.warning('Validating schema for %s', loaderC.__name__)
    loader = loaderC()
    dates = loader.getDates()
    df0 = loader.getDataAsOf(dates[0]); schema = df0.dtypes
    numCols = [c for c in df0.columns if is_numeric_dtype(df0[c])]
    ranges = {c:(df0[c].min(),df0[c].max()) for c in numCols}
    totalRows = 0
    for index,d in enumerate(dates):
        logging.debug('Validating schema for %s date %s', loaderC, d)
        df = loader.getDataAsOf(d)
        assert df is not None and len(df)>0
        for c in numCols:
            ranges[c] = (min(ranges[c][0],df[c].min()),max(ranges[c][1],df[c].max()))

        if not (df.dtypes == schema).all():
            logging.error('Previous schema: %s', schema)
            logging.error('This schema: %s', df.dtypes)
            raise ValueError(f'Schema mismatch for {loaderC.__name__} date {d} vs {loader.getDates()[index-1]} ')
        totalRows += len(df)
    logging.warning('Schema validated for %s', loaderC.__name__)
    logging.warning('%d rows across %d dates', totalRows, len(dates))
    logging.warning('Schema: %s', schema)
    logging.warning('Ranges: %s', ranges)