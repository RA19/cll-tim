
import numpy as np
import pandas as pd
from scipy import stats


def serialize_dates_long(df, start_col, end_col, normalise = True):
    """
    Function to expand vertically a date range given a start and end date
    """
    if normalise:
        dates = [pd.date_range(r[0], r[1], normalize=True).to_series() 
        for r in df[[start_col, end_col]].values]
    
    lens = [len(x) for x in dates]

    df_long = pd.DataFrame( {col:np.repeat(df[col].values, lens) for col in df.columns})
    df_long.insert(1, 'date', np.concatenate(dates))

    return df_long


def mem_usage(pandas_obj):
    
    """
    To check memory usage
    From https://www.dataquest.io/blog/pandas-big-data/
    """
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


def combine_col_names(df):
    return [ "_".join(list(filter(None, col))) for col in df.columns.values]


def reduce_mem_usage(props):
    """
    From Kaggle https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
    """
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in.
    excluded_types = [object, np.datetime64]
    for col in props.columns:
        if props[col].dtype not in excluded_types:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist



def shannonFunc(y):
    unu, inverse = np.unique(y, return_inverse=True)
    y = np.bincount(inverse)

    notabs = ~np.isnan(y)
    t = y[notabs] / np.sum(y[notabs])
    t = t[t != 0]
    H = -np.sum(t*np.log(t))
    return H


def richness(y):
    unu, inverse = np.unique(y, return_inverse=True)
    y = np.bincount(inverse)

    notabs = ~np.isnan(y)
    t = y[notabs]
    D = np.sum(t != 0)
    return float(D)


def giniFunc(y):
    unu, inverse = np.unique(y, return_inverse=True)
    y = np.bincount(inverse)
    notabs = ~np.isnan(y)
    t = y[notabs] / np.sum(y[notabs])
    D = 1 - np.sum(t**2)
    return D


def simpson(y):
    unu, inverse = np.unique(y, return_inverse=True)
    y = np.bincount(inverse)

    notabs = ~np.isnan(y)
    t = y[notabs] / np.sum(y[notabs])
    D = np.sum(t**2)
    return D


def dom(y):
    unu, inverse = np.unique(y, return_inverse=True)
    y = np.bincount(inverse)

    notabs = ~np.isnan(y)
    t = y[notabs] / np.sum(y[notabs])
    if len(t) == 0:
        D = -9999
    else:
        D = np.max(t)
    return D


def evenFunc(y):

    unu, inverse = np.unique(y, return_inverse=True)
    y = np.bincount(inverse)

    notabs = ~np.isnan(y)
    t = y[notabs] / np.sum(y[notabs])
    n = float(np.sum(t != 0))
    t = t[t != 0]
    H = -np.sum(t*np.log(t))
    den = np.log(n)
    if den == 0:
        ans = -9999
    else:
        ans = H/np.log(n)
    return ans


def slopes(x, y):

    slp, intercept, r, p, stder = scipy.stats.linregress(x, y)
    return slp


def slopesIntercept(x, y):

    slp, intercept, r, p, stder = scipy.stats.linregress(x, y)
    preddy = slp*(np.max(x)+30)+intercept
    return preddy
