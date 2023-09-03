
"""
Medium term Holt Winters Exponential Smoothing forecast.
Data grouping is weekly, no shaping performed.
"""

# # ERM Quantitative Risk

import datetime
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as rmse
from statsmodels.tsa.filters.hp_filter import hpfilter  # Hodrick Prescott filter for cyclic & trend separation
# could also use seasonal_decompose or STL decomposition ^
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import os


class HoltWinters:
    # # instance counter
    instance = 0

    def __init__(self):

        # data root
        self.default_filepath = r'FILEPATH'
        self.default_filename = 'FILENAME.xlsx'
        self.default_data_start = '%y-%m-%d'

        # model objects
        self.input_df = None
        self.output_df = None
        self.model = None

        # model information
        self.model_version = 1.0
        self.version_date = datetime.datetime(year=2023, month=3, day=9)

        # class info
        HoltWinters.instance += 1

    # # import time series data which is required to fit the model. return import data.
    def import_data(self):

        self.input_df = pd.read_excel(fr'{self.default_filepath}\{self.default_filename}')
        self.input_df.columns = ['volume']  # rename column to standardise model variables

        df_length = len(self.input_df['volume'])

        # # set datetime-index using default start date, frequency is weekly
        self.input_df.index = pd.date_range(start=self.default_data_start, periods=df_length,
                                            freq='W')

        return self.input_df

    # # extract trend from import_data and fit the HW model. return model and output_df.
    def fit_model(self, save_model=False):

        self.import_data()  # run import data function

        # Hodrick Prescott filter for cyclic & trend separation, other filters are available but haven't been tested.
        cycles, trend = hpfilter(self.input_df['volume'], lamb=1600 * 124)

        # # assign trend and cycle variables from the filter to input dataframe.
        self.input_df['trend'] = trend
        self.input_df['cycles'] = cycles

        # # declare and fit Holt Winters Exponential Smoothing model.
        self.model = ExponentialSmoothing(self.input_df['volume'], trend='mul', seasonal='add', seasonal_periods=52,
                                          use_boxcox=True, initialization_method='estimated').fit()

        # # score the model using mape, mae and rmse
        mape_score = mape(self.input_df['volume'], self.model.fittedvalues) * 100
        mae_score = mae(self.input_df['volume'], self.model.fittedvalues)
        rmse_score = rmse(self.input_df['volume'], self.model.fittedvalues, squared=False)

        print(f'--------- MODEL SCORING ---------\n'
              f'MAPE:   {mape_score:.2f}%\n'
              f'MAE:    {mae_score:,.2f}\n'
              f'RMSE:   {rmse_score:,.2f}\n'
              f'---------------------------------')

        # # is save_model arg True, save model to pkl file
        if save_model:
            self.__save_model(model=self.model, filename='holt_winters.pkl')

        # # generate the output of the model
        self.output_df = pd.DataFrame(
            np.c_[
                self.input_df['volume'], self.model.level, self.model.trend, self.model.season, self.model.fittedvalues,
                self.input_df[
                    'volume'] - self.model.fittedvalues],
            columns=[r"actual", r"level", r"trend", r"season", r"forecast", 'errors'],
            index=self.input_df.index)

        return self.model, self.output_df

    # # provides an analysis of forecast performance 
    def analyse_forecast(self, start="2022-12-25", end="2023-12-31"):

        # # if the model has not been fit, fit the model.
        if self.model is None:
            self.fit_model()

        # # declare plot
        fig, axs = plt.subplots(3)

        # # set x as time series, stdev for error bars and generate future forecast
        x = self.output_df.index
        stdev = self.output_df['errors'].std() * 1.96
        prediction = self.model.predict(start=start, end=end)

        # # plot one: actual vs forecast performance
        axs[0].plot(x, self.output_df['actual'], label='Actual Load', color='cadetblue')
        axs[0].plot(x, self.output_df['forecast'], label='HW Model Fit', color='peru')
        axs[0].plot(x, self.output_df['errors'], label='Model Residuals', color='navy', lw=1)
        axs[0].hlines(0, x.min(), x.max(), linestyles='dashed', color='white', lw=1)

        axs[0].plot(prediction.index, prediction, ls='--', label='Load Forecast', color='black', lw=1)
        axs[0].plot(prediction.index, prediction + stdev, ls='--', color='black', lw=1)
        axs[0].plot(prediction.index, prediction - stdev, ls='--', color='black', lw=1)
        axs[0].fill_between(prediction.index, prediction + stdev, prediction - stdev, color='red', alpha=0.3,
                            label='95% Confidence')

        # # plot formatting
        axs[0].set_ylabel('Weekly Load (MWh)')

        # # plot two: residual histogram (note, no testing for normality has been performed).

        mu, sigma = self.output_df['errors'].mean(), self.output_df['errors'].std()

        count, bins, ignored = axs[1].hist(self.output_df['errors'], bins=75, density=True, alpha=0.8,
                                           label='residual histogram')

        axs[1].plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
                    np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), label='normal distribution')

        axs[1].set_ylabel('Residuals Histogram')

        # # plot three: MINCER-ZARNOWITZ test (OLS ON THE MODEL OUTPUT VS ACTUAL) - THIS EVALUATES BIAS AND ACCURACY

        lin_reg = LinearRegression().fit(self.input_df['volume'].to_numpy().reshape(-1, 1), self.model.fittedvalues)

        X = sm.add_constant(self.input_df['volume'])
        mincer_zarnowitz = sm.OLS(self.model.fittedvalues, X).fit()

        # # print MZ info to terminal
        print(
            f'Mincer-Zarnowitz Parameters\nBias: {mincer_zarnowitz.params.const:,.2f} MWh\n'
            f'Relative Efficiency: {mincer_zarnowitz.params.volume * 100:.2f}%\n'
            f'---------------------------------'
        )

        self.input_df['volume'] = self.input_df['volume'].sort_values(ascending=True)
        axs[2].scatter(self.input_df['volume'], self.model.fittedvalues, lw=0.5, marker='x', label='residuals')
        axs[2].plot(self.input_df['volume'], lin_reg.predict(self.input_df['volume'].to_numpy().reshape(-1, 1)),
                    ls='--', label='Mincer-Zarnowitz regression')
        axs[2].plot(self.input_df['volume'], self.input_df['volume'], ls='--', color='red', label='1-to-1')

        axs[2].set_ylabel('Mincer-Zarnowitz Regression')
        axs[2].set_xlabel(
            f'Bias: {mincer_zarnowitz.params.const:,.2f} MWh, Relative Efficiency: {mincer_zarnowitz.params.volume * 100:.2f}%')

        for i in range(3):
            axs[i].set_facecolor('gainsboro')
            axs[i].grid(axis='y')
            axs[i].legend()

        plt.suptitle('\nHolt Winters Exponential Smoothing Model\n(Weekly, pre-shaping)')
        plt.show()

    # make prediction and return.
    def predict(self, start="2022-12-25", end="2023-12-31"):

        # # if the model has not been fit, fit the model.
        if self.model is None:
            self.fit_model()

        # # make prediction
        prediction = self.model.predict(start=start, end=end)

        return prediction

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

        text = f'--------- MODEL SCORING ---------\n' \
               f'MAPE:   {mape_score:.2f}%\n' \
               f'MAE:    {mae_score:,.2f}\n' \
               f'RMSE:   {rmse_score:,.2f}\n' \
               '---------------------------------'
        print(text)
        score_dict['text'] = text

        return score_dict


if __name__ == '__main__':
    hw = HoltWinters()
    hw.fit_model(save_model=False)
    hw.analyse_forecast()
