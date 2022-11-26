import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# create a function to calculate the simple moving average
def SMA(data, period=30, column='close'):
    return data[column].rolling(window=period).mean()

# create a function to calculate the exponential moving average
def EMA(data, period=20, column='close'):
    return data[column].ewm(span=period, adjust=False).mean()

# create a function to calculate the MACD and signal line
def MACD(data, fast_period=12, slow_period=26, signal_period=9, column='close'):
    data['fast_ema'] = EMA(data, period=fast_period, column=column)
    data['slow_ema'] = EMA(data, period=slow_period, column=column)
    data['macd'] = data['fast_ema'] - data['slow_ema']
    data['signal'] = EMA(data, period=signal_period, column='macd')
    return data

# create a function to calculate the Bollinger Bands
def BBANDS(data, period=20, column='close'):
    data['ma'] = SMA(data, period, column=column)
    data['bb_up'] = data['ma'] + 2 * data[column].rolling(window=period).std()
    data['bb_dn'] = data['ma'] - 2 * data[column].rolling(window=period).std()
    data['bb_width'] = data['bb_up'] - data['bb_dn']
    return data

# create a function to calculate the Relative Strength Index
def RSI(data, period=14, column='close'):
    delta = data[column].diff(1)
    delta = delta.dropna()
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    data['up'] = up
    data['down'] = down
    AVG_Gain = SMA(data, period, column='up')
    AVG_Loss = abs(SMA(data, period, column='down'))
    RS = AVG_Gain / AVG_Loss
    data['RSI'] = 100.0 - (100.0 / (1.0 + RS))
    return data

# create a function to calculate the Stochastic Oscillator
def STOCH(data, period=14, column='close'):
    data['highest_high'] = data[column].rolling(window=period).max()
    data['lowest_low'] = data[column].rolling(window=period).min()
    data['fast_k'] = 100 * (data[column] - data['lowest_low']) / (data['highest_high'] - data['lowest_low'])
    data['slow_d'] = data['fast_k'].rolling(window=3).mean()
    return data

# create a function to calculate the Commodity Channel Index
def CCI(data, period=20, column='close'):
    data['PP'] = (data['high'] + data['low'] + data[column]) / 3
    # print(PP)
    data['ma'] = SMA(data, period, column='PP')
    data['md'] = data['PP'] - data['ma']
    data['cci'] = data['md'] / (0.015 * data['md'].rolling(window=period).std())
    return data

# create a function to calculate the Average Directional Index
def ADX(data, period=14, column='close'):
    i = 0
    UpI = [0]
    DoI = [0]
    data.index = range(0, len(data))
    while (i + 1 <= data.index[-1]):
        UpMove = data.loc[i + 1, 'high'] - data.loc[i, 'high']
        DoMove = data.loc[i, 'low'] - data.loc[i + 1, 'low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    data['UpI'] = pd.Series(UpI)
    data['DoI'] = pd.Series(DoI)
    PosDI = SMA(data, period, column='UpI')
    NegDI = SMA(data, period, column='DoI')
    data['ADX'] = abs((PosDI - NegDI) / (PosDI + NegDI))
    return data

# create a function to calculate the On Balance Volume
def OBV(data, column='close'):
    i = 0
    OBV = [0]
    while i < data.index[-1]:
        if data.loc[i + 1, column] - data.loc[i, column] > 0:
            OBV.append(data.loc[i + 1, 'volume'])
        if data.loc[i + 1, column] - data.loc[i, column] == 0:
            OBV.append(0)
        if data.loc[i + 1, column] - data.loc[i, column] < 0:
            OBV.append(-data.loc[i + 1, 'volume'])
        i = i + 1
    data['obv'] = pd.Series(OBV).cumsum()
    return data

# create a function to calculate the Chaikin Oscillator
def Chaikin(data):
    AD = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
    AD = AD.fillna(0)
    AD = AD * data['volume']
    data['chaikin'] = AD.ewm(span=3, adjust=False).mean() - AD.ewm(span=10, adjust=False).mean()
    return data

# create a function to calculate the Money Flow Index
def MFI(data, period=14, column='close'):
    PP = (data['high'] + data['low'] + data[column]) / 3
    data['TP'] = PP
    data['PMF'] = 0
    data['NMF'] = 0
    i = 0
    while i < data.index[-1]:
        if data.loc[i + 1, 'TP'] > data.loc[i, 'TP']:
            data.loc[i + 1, 'PMF'] = data.loc[i + 1, 'TP'] * data.loc[i + 1, 'volume']
            data.loc[i + 1, 'NMF'] = 0
        if data.loc[i + 1, 'TP'] < data.loc[i, 'TP']:
            data.loc[i + 1, 'PMF'] = 0
            data.loc[i + 1, 'NMF'] = data.loc[i + 1, 'TP'] * data.loc[i + 1, 'volume']
        if data.loc[i + 1, 'TP'] == data.loc[i, 'TP']:
            data.loc[i + 1, 'PMF'] = 0
            data.loc[i + 1, 'NMF'] = 0
        i = i + 1
    data['MFR'] = data['PMF'].rolling(window=period).sum() / data['NMF'].rolling(window=period).sum()
    data['MFI'] = 100 - 100 / (1 + data['MFR'])
    return data

# create a function to calculate the Ease of Movement
def EVM(data, period=14, column='close'):
    dm = ((data['high'] + data['low']) / 2) - ((data['high'].shift(1) + data['low'].shift(1)) / 2)
    br = (data['volume'] / 100000000) / ((data['high'] - data['low']))
    data['EVM'] = dm / br
    data['EVM_MA'] = data['EVM'].rolling(window=period).mean()
    return data

# create a function to calculate the Volume-price Trend
def VPT(data, column='close'):
    i = 0
    VPT = [0]
    while i < data.index[-1]:
        VPT.append(VPT[i] + ((data.loc[i + 1, column] - data.loc[i, column]) / data.loc[i, column]) * data.loc[i + 1, 'volume'])
        i = i + 1
    data['VPT'] = pd.Series(VPT)
    return data

# create a function to calculate the Negative Volume Index
def NVI(data, column='close'):
    i = 0
    NVI = [1000]
    while i < data.index[-1]:
        if data.loc[i + 1, column] - data.loc[i, column] > 0:
            NVI.append(NVI[i] + data.loc[i + 1, 'volume'])
        if data.loc[i + 1, column] - data.loc[i, column] == 0:
            NVI.append(NVI[i])
        if data.loc[i + 1, column] - data.loc[i, column] < 0:
            NVI.append(NVI[i] - data.loc[i + 1, 'volume'])
        i = i + 1
    data['NVI'] = pd.Series(NVI)
    return data

# create a function to calculate the Accumulation/Distribution
def ACCDIST(data, column='close'):
    CLV = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
    CLV = CLV.fillna(0)
    data['AD'] = CLV * data['volume']
    data['AD'] = data['AD'].cumsum()
    return data

# create a function to calculate the Keltner Channel
def KELCH(data, period=10):
    KelChM = (data['high'].rolling(window=period).max() + data['low'].rolling(window=period).min() + data['close']) / 3
    data['KelChM'] = KelChM
    data['KelChU'] = KelChM + (data['high'].rolling(window=period).max() - data['low'].rolling(window=period).min()) * 0.5
    data['KelChD'] = KelChM - (data['high'].rolling(window=period).max() - data['low'].rolling(window=period).min()) * 0.5
    return data

# create a function to calculate the Donchian Channel
def DONCH(data, period=20):
    data['DonchianU'] = data['high'].rolling(window=period).max()
    data['DonchianD'] = data['low'].rolling(window=period).min()
    return data

# create a function to calculate the Standard Deviation
def STDDEV(data, period=20, column='close'):
    data['STDDEV'] = data[column].rolling(window=period).std()
    return data

# create a function to calculate the Average True Range
def ATR(data, period=14):
    data['H-L'] = abs(data['high'] - data['low'])
    data['H-PC'] = abs(data['high'] - data['close'].shift(1))
    data['L-PC'] = abs(data['low'] - data['close'].shift(1))
    data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    data['ATR'] = data['TR'].rolling(window=period).mean()
    return data

# create a function to calculate the True Strength Index
def TSI(data, r=25, s=13, column='close'):
    M = pd.Series(data[column].diff(1))
    aM = abs(M)
    EMA1 = M.ewm(span=r, min_periods=r - 1).mean()
    aEMA1 = aM.ewm(span=r, min_periods=r - 1).mean()
    EMA2 = EMA1.ewm(span=s, min_periods=s - 1).mean()
    aEMA2 = aEMA1.ewm(span=s, min_periods=s - 1).mean()
    data['TSI'] = EMA2 / aEMA2
    return data

# create a function to calculate the Ultimate Oscillator
def ULTOSC(data):
    i = 0
    TR_l = [0]
    BP_l = [0]
    while i < data.index[-1]:
        TR = max(data.loc[i + 1, 'high'], data.loc[i, 'close']) - min(data.loc[i + 1, 'low'], data.loc[i, 'close'])
        TR_l.append(TR)
        BP = data.loc[i + 1, 'close'] - min(data.loc[i + 1, 'low'], data.loc[i, 'close'])
        BP_l.append(BP)
        i = i + 1
    data['TR'] = pd.Series(TR_l)
    data['BP'] = pd.Series(BP_l)
    data['UO'] = 100 * (4 * data['BP'].rolling(window=7).sum() / data['TR'].rolling(window=7).sum() + 2 * data['BP'].rolling(window=14).sum() / data['TR'].rolling(window=14).sum() + data['BP'].rolling(window=28).sum() / data['TR'].rolling(window=28).sum()) / 7
    return data

# create a function to calculate the Coppock Curve
def COPP(data, period=14):
    M = data['close'].diff(period * 11 - 1)
    N = data['close'].shift(period * 11 - 1)
    ROC1 = M / N
    M = data['close'].diff(period - 1)
    N = data['close'].shift(period - 1)
    ROC2 = M / N
    data['COPP'] = pd.Series(ROC1 + ROC2, name='COPP').ewm(span=10, min_periods=9).mean()
    return data

def call_data(URL): ## 
    df = pd.read_csv(URL, names=['time', 'open', 'high', 'low', 'close', 'volume'], delimiter = ";",)
    df.time = pd.to_datetime(df.time, format = '%Y.%m.%d %H:%M:%S.%f')
    df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time\ 
    df = df.drop_duplicates()
    return df


def define_indicators_features(MES):
    # series
    MD = MACD( MES, 12, 26, 9)
    for i in MD.iloc[:,-4:].columns:
        MES[i] = MD[i]
    
    sma21 = SMA(MES, period=21, column='close').to_frame()
    sma50 = SMA(MES, period=50, column='close').to_frame()

    MES['sma21'] = sma21
    MES['sma50'] = sma50

    MES['EMA21'] = EMA(MES, period=21, column='close')
    MES['EMA50'] = EMA(MES, period=50, column='close')

    # BBANDS RSI STOCH CCI ADX OBV Chaikin MFI EVM VPT NVI ACCDIST KELCH DONCH STDDEV ATR TSI ULTOSC COPP
    MES = BBANDS(MES, period=20, column='close')
    MES = RSI(MES, period=14, column='close')
    MES = STOCH(MES, period=14, column='close')
    MES = CCI(MES, period=20, column='close')
    MES = ADX(MES, period=14, column='close')
    MES = OBV(MES, column='close')
    MES = Chaikin(MES)
    MES = MFI(MES, period=14, column='close')
    MES = EVM(MES, period=14, column='close')
    MES = VPT(MES)
    MES = NVI(MES)
    MES = ACCDIST(MES)
    MES = KELCH(MES, period=14)
    MES = DONCH(MES, period=14)
    MES = STDDEV(MES, period=14, column='close')
    MES = ATR(MES, period=14)
    MES = TSI(MES, r=25, s=13, column='close')
    MES = ULTOSC(MES)
    MES = COPP(MES, period=14)
    return  MES
    
# get sample data range

data = pd.read_csv("MES 06-21.Last.txt", names=['time', 'open', 'high', 'low', 'close', 'volume'], 
                   delimiter = ";", index_col='time')[-3000:]

define_indicators_features(data)