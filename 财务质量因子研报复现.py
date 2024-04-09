import pandas as pd
import pymongo
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm

# 中性化处理，OLS回归
def neutralization_industry(X, Y):
    result = sm.OLS(Y.astype(float), X.astype(float)).fit()
    return result.resid

# -------------------
# 因子计算
# -------------------

data =pd.read_pickle(r'D:\Desktop\营业利润财务质量\data\ttl_data')

# step1:rolling 3 年 12 期数据回归。
# train_data = data.loc['2012-03-31':'2014-12-31'] #三年训练集时间
backtest_data = data.loc['2013-03-31':]  # 数据起点2012，一年时间滞后，2013-2015是3年时间窗口
ttl_time = backtest_data.index.unique()

# 双重索引
backtest_data.reset_index(inplace=True)
backtest_data = backtest_data.groupby(['REPORT_PERIOD', 'S_INFO_WINDCODE']).sum()

factor = pd.DataFrame()  # 原始因子
for t in range(len(ttl_time)-12):
    rolling_data = backtest_data.loc[ttl_time[t]:ttl_time[t+11], :]
    factor_indu = pd.DataFrame()  # 同一时间不同行业的，横向粘贴

    # 以行业为单位的面板回归，个体固定、时点固定，被解释变量为单季度营业利润率，解释变量除以上四个指标外添加 t-4 期营业利润率，
    for indu in list(rolling_data['indu_code'].unique()):
        indu_data = rolling_data[rolling_data['indu_code'] == indu]
        indu_data.reset_index(inplace=True)
        indu_data.set_index(['S_INFO_WINDCODE', 'REPORT_PERIOD'], inplace=True)
        exog = indu_data[['accruals_ind', 'inventory_ind', 'fix_assets_ind', 'cash_flows_ind', 'oper_profit_ratio_t4']]
        res_fe = PanelOLS(indu_data['oper_profit_ratio'], exog, entity_effects=True, time_effects=True)  # 个体固定+时间固定
        results_fe = res_fe.fit()

        # Step 2: 原始因子值为残差波动率，并进行行业内标准化
        residuals = pd.DataFrame(results_fe.resids)    # 残差
        residuals = residuals.groupby('S_INFO_WINDCODE').std()   # 残差波动率
        residuals = (residuals['residual'] - residuals['residual'].mean()) / residuals['residual'].std()  # 标准化
        residuals = pd.DataFrame(residuals)

        # 加时间索引
        # residuals['date'] = ttl_time[t + 11]
        # residuals.reset_index(drop=True)
        # residuals.set_index('date', inplace=True)

        factor_indu = pd.concat([factor_indu, residuals])  # 同一天，不同行业的因子横向合并

    # Step 3: 对原始因子进行市值中性处理
    mkt_val =rolling_data.loc[ttl_time[t+11], ['S_DQ_MV', 'indu_code']].reset_index()

    # 1: 对市值进行行业中性化处理
    X = pd.get_dummies(mkt_val.indu_code)
    Y = mkt_val.S_DQ_MV
    mkt_val['mkt_val_neu'] = neutralization_industry(X, Y)

    # 2: 对市值进行全市场标准化处理
    mkt_val['mkt_val_std'] = (mkt_val['mkt_val_neu'] - mkt_val['mkt_val_neu'].mean()) / mkt_val['mkt_val_neu'].std()

    # 3：对因子进行市值中性化处理
    factor_indu = factor_indu.merge(mkt_val[['S_INFO_WINDCODE', 'mkt_val_std']], how='left', on='S_INFO_WINDCODE')
    X_1 = factor_indu.mkt_val_std
    Y_1 = factor_indu.residual
    factor_indu['residual_mkt'] = neutralization_industry(X, Y)

    # 4：对因子进行全市场标准化处理
    factor_indu[ttl_time[t+11]] = (factor_indu['residual_mkt'] - factor_indu['residual_mkt'].mean()) / factor_indu['residual_mkt'].std()

    # 合并所有行业所有日期的因子值
    factor_indu.set_index('S_INFO_WINDCODE', inplace=True)
    factor = pd.concat([factor, factor_indu[ttl_time[t+11]]], axis=1)  # 不同时间，因子竖向合并

# 输出行为股票代码，列位时间的因子
factor_t = factor.T
factor_t.index = pd.to_datetime(factor_t.index, format="%Y%m%d")
factor_t.index.name = 'REPORT_PERIOD'

# 两个财报期权重月度再平衡
# 将季度factor处理成月度factor
factor_month = factor_t.resample('M').last().copy()
for month in range(len(factor_month.index)-1):
    if month % 3 == 1:
        factor_month.iloc[month,:] = (2/3) * factor_month.iloc[month-1,:] + (1/3) * factor_month.iloc[month+2,:]
    elif month % 3 == 2:
        factor_month.iloc[month,:] = (1/3) * factor_month.iloc[month-2,:] + (2/3) * factor_month.iloc[month+1,:]
    else:
        factor_month.iloc[month,:] = factor_month.iloc[month,:]
factor_month
factor_t.to_pickle(r'D:\Desktop\营业利润财务质量\data\backtest_factor_2')
