"""
Functions to implement the SARIMA(X) model for load forecasting.
Data grouping is weekly, no shaping performed.
"""

# # ERM Quantitative Risk

import os
import datetime
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as rmse


class SarimaModel:
    # # instance counter
    instance = 0

    def __init__(self):
        # data root
        self.default_filepath = r'C:\Users\ee080463\EWEC\EWEC - GF - ERM - Documents\Quant Risk Models\_data\_Forecasting Models'
        self.default_filename = 'weeklydata.xlsx'
        self.default_data_start = '2016-01-01'

        # model objects
        self.input_df = None
        self.output_df = None
        self.model = None
        self.prediction = None

        # model information
        self.model_version = 1.0
        self.version_date = datetime.datetime(year=2023, month=3, day=9)

        # class info
        SarimaModel.instance += 1

    # # import time series data which is required to fit the model. return import data.
    def import_data(self):

        self.input_df = pd.read_excel(fr'{self.default_filepath}\{self.default_filename}')
        self.input_df.columns = ['volume']  # rename column to standardise model variables

        df_length = len(self.input_df['volume'])

        # # set datetime-index using default start date, frequency is weekly
        self.input_df.index = pd.date_range(start=self.default_data_start, periods=df_length,
                                            freq='W')

        return self.input_df

    # # fit Sarima(x), return model and output_df.
    def fit_model(self, save_model=False):

        if self.input_df is None:
            self.import_data()

        # # trend_options = ['n', 'c', 't', 'ct']

        self.model = sm.tsa.statespace.SARIMAX(self.input_df['volume'], order=(1, 0, 0), seasonal_order=(1, 0, 1, 52),
                                               # TODO - Optimise p,d,q P,D,Q,(λ) with grid search.
                                               trend='n').fit()  # TODO - Fit exogenous data to improve accuracy.
        fit_values = self.model.fittedvalues

        self.output_df = pd.concat([self.input_df, fit_values], axis=1)
        self.output_df.columns = ['actual', 'forecast']
        self.output_df['residual'] = self.output_df['forecast'] - self.output_df['actual']
        self.output_df['residual_pct'] = self.output_df['residual'] / self.output_df['actual']

        self.output_df['month'] = self.output_df.index.month

        # # is save_model arg True, save model to pkl file
        if save_model:
            self.__save_model(model=self.model, filename='holt_winters.pkl')

        return self.model, self.output_df

    # # plot auto-correlation and partial auto-correlation plots.
    def plot_auto_correlation(self, lags=None):

        if self.input_df is None:
            self.import_data()

        layout = (1, 3)

        raw = plt.subplot2grid(layout, (0, 0))
        acf = plt.subplot2grid(layout, (0, 1))
        pacf = plt.subplot2grid(layout, (0, 2))

        raw.plot(self.input_df)
        sm.tsa.graphics.plot_acf(self.input_df, lags=lags, ax=acf, zero=False)
        sm.tsa.graphics.plot_pacf(self.input_df, lags=lags, ax=pacf, zero=False)
        sns.despine()
        plt.tight_layout()
        plt.show()

    # # plot the fit model with the observations.
    def plot_model(self):

        if self.output_df is None:
            self.fit_model()

        fig, axs = plt.subplots()
        axs.plot(self.output_df.index, self.output_df['actual'], label='Actual')
        axs.plot(self.output_df.index, self.output_df['forecast'], label='Forecast')

        axs.set_facecolor('gainsboro')
        axs.set_ylabel('Weekly Load (MWh)')
        axs.grid(axis='y')
        axs.legend()
        plt.suptitle('SARIMA Model (DEVELOPMENT ONLY, Weekly granularity w/o shaping)')
        plt.show()

    # # analyse how the model is performing vs observations on a monthly basis.
    def plot_residuals_by_month(self):

        if self.output_df is None:
            self.fit_model()

        # analyse residuals by month
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(nrows=3, ncols=4)
        plots = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]

        for i in plots:
            index = plots.index(i)

            df_temp = self.output_df.copy()
            df_temp = df_temp.loc[df_temp['month'] == index + 1]
            i.hist(df_temp['residual_pct'], density=False, bins=10)
            i.set_title(
                f'{index + 1} (µ: {df_temp["residual_pct"].mean():.2f}%,σ: {df_temp["residual_pct"].std():.2f}%)',
                fontdict={'fontsize': 10})
            i.set_facecolor('gainsboro')
            i.grid(axis='y')
            i.set_xlim(left=self.output_df['residual_pct'].min(), right=self.output_df['residual_pct'].max())
            i.set_ylim(top=10, bottom=0)

        plt.suptitle('Historical monthly error distribution\n(the seasonal effects of forecast accuracy)')
        plt.show()

    # Dickey–Fuller tests the null hypothesis that a unit root is present in an autoregressive time series model.
    def plot_dickey_fuller(self):

        if self.input_df is None:
            self.import_data()

        monthly_temp = self.input_df.copy().resample('M').mean()
        monthly_temp = monthly_temp.dropna(axis=0)

        dftest = ts.adfuller(monthly_temp, )

        df_output = pd.Series(dftest[0:4], index=['TEST STAT', 'P-VAL', 'LAGS', 'OBSERVATIONS'])

        for key, value in dftest[4].items():
            df_output[f'Critical Val: {key}'] = value
        print(df_output)

        rol_mean = monthly_temp.rolling(window=12).mean()
        rol_std = monthly_temp.rolling(window=12).std()

        monthly_temp['lag_12'] = monthly_temp.shift(12)
        monthly_temp['seasonal_diff'] = monthly_temp['volume'] - monthly_temp['lag_12']

        fig, ax = plt.subplots(nrows=2)
        x = monthly_temp.index

        ax[0].plot(x, monthly_temp['volume'], label='Original')
        ax[0].plot(rol_mean, label='Mean')
        ax[0].plot(rol_std, label='STD')

        ax[1].plot(x, monthly_temp['volume'], label='Original')
        ax[1].plot(x, monthly_temp['lag_12'], label='lag_12')
        ax[1].plot(x, monthly_temp['seasonal_diff'], label='seasonal_diff')

        for i in range(0, 2):
            ax[i].set_facecolor('gainsboro')
            ax[i].legend()
            ax[i].grid(axis='y')

        plt.suptitle("Dickey Fuller test for unit root")
        plt.show()

    # # predict future values.
    def predict(self, start="2022-12-25", end="2023-03-31"):

        # # if the model has not been fit, fit the model.
        if self.model is None:
            self.fit_model()

        # # make prediction
        self.prediction = self.model.predict(start=start, end=end)

        return self.prediction

    # # plot model prediction.
    def plot_prediction(self):

        if self.prediction is None:
            self.predict()

        fig, axs = plt.subplots()
        axs.plot(self.output_df.index, self.output_df['actual'], label='Actual')
        axs.plot(self.output_df.index, self.output_df['forecast'], label='Model')
        axs.plot(self.prediction, label='Prediction')

        axs.legend()
        axs.grid(axis='y')
        axs.set_facecolor('gainsboro')

        plt.suptitle('Sarima Model')
        plt.show()

    @staticmethod  # save model to pickle
    def __save_model(model, filename):

        if filename in os.listdir():
            print('Model file already exists, would you like to override? (y / n) ')
            command = input()
            if 'y' not in command:
                print('Model not saved.')
                exit()

        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        file.close()
        print('Model saved to file.')

    @staticmethod  # load pickle file
    def __load_model(filename):

        if filename not in os.listdir():
            print('Sorry, file not found.')
            exit()

        with open(filename, 'rb') as file:
            object_file = pickle.load(file)
            file.close()

        return object_file

    # # score the model using mape, mae and rmse
    def analyse_model_performance(self):

        if self.output_df is None:
            self.fit_model()

        score_dict = {'mape': None, 'mae': None, 'rmse': None}

        mape_score = mape(self.output_df['actual'], self.output_df['forecast']) * 100
        score_dict['mape'] = mape_score

        mae_score = mae(self.output_df['actual'], self.output_df['forecast'])
        score_dict['mae'] = mae_score

        rmse_score = rmse(self.output_df['actual'], self.output_df['forecast'], squared=False)
        score_dict['rmse'] = rmse_score

        text = f'\n--------- MODEL SCORING ---------\n' \
               f'MAPE:   {mape_score:.2f}%\n' \
               f'MAE:    {mae_score:,.2f}\n' \
               f'RMSE:   {rmse_score:,.2f}\n' \
               '---------------------------------'
        print(text)
        score_dict['text'] = text

        # # MINCER-ZARNOWITZ test (OLS ON THE MODEL OUTPUT VS ACTUAL) - THIS EVALUATES BIAS AND ACCURACY

        X = sm.add_constant(self.input_df['volume'])
        mincer_zarnowitz = sm.OLS(self.model.fittedvalues, X).fit()

        # # print MZ info to terminal
        print(
            f'-- MINCER-ZARNOWITZ PARAMETERS --\nBias:                  {mincer_zarnowitz.params.const:,.2f} MWh\n'
            f'Relative Efficiency:   {mincer_zarnowitz.params.volume * 100:.2f}%\n'
            f'---------------------------------'
        )

        return score_dict


if __name__ == '__main__':
    sar = SarimaModel()
    sar.plot_residuals_by_month()
