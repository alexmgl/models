import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import sklearn.preprocessing as sk_pre
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance


# TODO - MAYBE SPLIT THESE INTO TIME SERIES TOOLS AND STATIC DATA TOOLS.

class EWEC_ML:

    def __init__(self, filepath, model, y_col):

        # read in data
        self.data_filepath = filepath  # filepath to the master data
        self.master_df = None
        self.__import_data()

        # model
        self.model = model

        # target variable information
        self.y_col = y_col
        self.y = self.df_master[self.y_col]

        # independent variable
        self.X = self.master_df.copy()
        self.X = self.X.drop(columns=[self.y_col])
        self.X_cols = list(self.X.columns)

    def __import_data(self):
        self.df_master = pd.read_excel(self.data_filepath)  # read in the data
        # TODO - df = df.astype(np.uint8, errors='ignore') # reduce memory footprint
        # LOOK INTO THIS ^. INTERESTING WAY TO REDUCE MEMORY FOOTPRINT OF THE PROBLEM

    def __remove_outliers(self):
        # # signal analysis, test for outliers and remove if required
        pass  # custom function to clean data

    def __categorical_label_encoding(self, discrete_cols):

        # Label encoding for categoricals
        for col_name in self.df_master.select_dtypes("object"):
            self.df_master[col_name], _ = self.df_master[col_name].factorize()

        # All discrete features should now have integer dtypes (double-check this before using MI!)
        discrete_features = self.df_master.dtypes == int

    def __target_encoding(self):
        """
         Target encoding is any kind of encoding that replaces a feature's categories with some number derived from the target.
         """

        # e.g. df["new_column"] = df.groupby("categorical")["continuous"].transform("mean")
        # ^ This kind of target encoding is sometimes called a mean encoding. Applied to a binary target, it's also
        # called bin counting. (Other names you might come across include: likelihood encoding, impact encoding, and leave-one-out encoding.)

    # a function measuring associations between a feature and the target
    def __feature_utility(self, X, y, show=True):

        """ Mutual information is like correlation in that it measures a relationship between two quantities.
        The least possible mutual information between quantities is 0. When MI is 0, the quantities are independent.
        In theory there's no upper bound to what MI can be. In practice values above 2.0 or so are uncommon.
        Mutual information is a logarithmic quantity, so it increases very slowly """

        mi_scores = mutual_info_regression(X, y)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)

        def plot_mi_scores(scores=mi_scores):
            scores = scores.sort_values(ascending=True)
            width = np.arange(len(scores))
            ticks = list(scores.index)
            plt.figure(dpi=100, figsize=(8, 5))
            plt.barh(width, scores)
            plt.yticks(width, ticks)
            plt.title("Mutual Information Scores")

        if show:
            plot_mi_scores(mi_scores)

        return mi_scores

    # to interpolate functions and to smooth signals
    def fourier_transform(self):
        pass

    def correlation_matrix(self, df):
        # # correlation_matrix = df.corr()
        # # correlation_matrix['target variable'].sort_values(ascending=False)

        plt.matshow(df.corr())  # # Calculate standard correlation coefficient (Pearson's r) between every pair of attributes

        plt.colorbar()
        plt.show()

        # # also see Pandas scatter_matrix() method from Pandas.plotting

    # #  Variance inflation factor (VIF)
    def multicollinearity_vif(self):
        # Variance inflation factor (VIF)
        # from statsmodels.stats.outliers_influence import variance_inflation_factor
        # ridge regression or principal component regression are better at handling multicollinearity
        pass

    # Clustering With K-Means
    def feature_engineering(self, model):

        """ For a feature to be useful, it must have a relationship to the target that our model is able to learn."""
        # determine which features are the most important with mutual information
        # create new features by combining existing ones
        # encode high-cardinality categoricals with a target encoding
        # create segmentation features with k-means clustering
        # decompose a dataset's variation into features with principal component analysis (PCA) - To maximise variance explained.

        """ Clustering simply means the assigning of data points to groups based upon how similar the points are to each other.
         Applied to a single real-valued feature, clustering acts like a traditional "binning" or "discretisation" transform.
         On multiple features, it's like "multi-dimensional binning" (sometimes called vector quantization)."""

        # Create cluster feature
        """ Since k-means clustering is sensitive to scale, it can be a good idea rescale or normalize data with extreme values."""
        kmeans = KMeans(n_clusters=6)
        self.X["Cluster"] = kmeans.fit_predict(self.X)
        self.X["Cluster"] = self.X["Cluster"].astype("category")

        self.X.head()

        # TODO - PRINCIPAL COMPONENT ANALYSIS (PCA)
        """ 1. Standardize the datapoints
            2. Find the covariance matrix from the given datapoints
            3. Carry out eigen-value decomposition of the covariance matrix
            4. Sort the eigenvalues and eigenvectors"""

    # prepare and clean the dataset
    def pre_processing(self):
        # dealing with missing data
        # - we can remove the full row
        # - Get rid of a data class
        # - Set the missing value to some other, this is called imputation.
        # - - imputation can be done manually, via sklearn's SimpleImputer or via more advanced methods (KNNImputer, IterativeImputer) in sklearn

        # handling categorical data
        # - we need to encode these values (can use sklearn OrdinalEncoder)
        # one hot encoding using pd.get_dummies
        # be careful with one hot encoding that no new categorical features will be included in the model - raise expection if this is the case. OneHotEncoder function will not do this.

        # if the data has fat tails or extreme values, it's sensible to first transform the dataset (i.e. log or some other transformer) before scaling.
        # or bucket the data

        # feature scaling via normalization or standardization
        scaler = sk_pre.StandardScaler().set_output(transform="pandas")

        # # - also see min-max scaling (also called normalization)
        min_max_scaler = sk_pre.MinMaxScaler()

    def feature_selection(self):
        # Principal Component Analysis (PCA)
        pass

    def explore_data(self):

        # e.g. Check cross-correlation with Pearson correlation coefficient

        pass

    # Split the prepared dataset, return train test split
    def split(self, X, y, test_size=0.3, random_state=0):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size,
                                                                                random_state=random_state)

        return self.X_train, self.X_test, self.y_train, self.y_test

    """ Sampling both training observations and predictors is known as random patches method, whereas sampling 
    only the predictors and keeping the whole training dataset is called random subspaces method."""

    # validate model effectiveness against unseen (test) data
    def cross_validation(self, model, X, y):

        score = -cross_val_score(model, X, y, cv=3)  # # FYI - utility func to need to reverse the sign (i.e. +ve to -ve)

        return score

    # validate model effectiveness using statistical tests
    def information_criteria(self):
        # aic - most useful when working with small data set or time series analysis. The lower the AIC score the better.
        # bic
        # mdl

        pass

    def fit_model(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.prediction = self.model.predict_y(self.X_test)

        return self.prediction

    def evaluate(self, show=True):

        mape = mean_absolute_percentage_error(self.y_test, self.prediction)
        mse = mean_squared_error(self.y_test, self.prediction)
        rmse = mean_squared_error(self.y_test, self.prediction, squared=False)
        mae = mean_absolute_error(self.y_test, self.prediction)

        # MINCER - ZARNOWITZ

        X = sm.add_constant(self.prediction)
        mincer_zarnowitz = sm.OLS(self.y_test, X).fit()
        print(mincer_zarnowitz.summary())

        reg = LinearRegression().fit(self.prediction, self.y_test)
        print(reg.coef_, reg.intercept)

        # PERMUTATION IMPORTANCE
        # TODO
        r = permutation_importance(self.model, self.X_test, self.y_test,n_repeats = 30,random_state = 0)

        for i in r.importances_mean.argsort()[::-1]:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                print(f"{self.X_test.feature_names[i]:<8}"
                f"{r.importances_mean[i]:.3f}"
                f" +/- {r.importances_std[i]:.3f}")

        # TODO - ANOTHER WAY TO LOOK AT FEATURE IMPORTANCE
        # https://shap.readthedocs.io/en/latest/
        # SHAPELY VALUES - GAME THEORETIC APPROACH TO FEATURE IMPORTANCE
        # The Shapley value is the average contribution of a feature value to the prediction in different coalitions.

        # --------------------------------------

        if show:
            print(f'-- MODEL SCORE --\n\n'
                  f'MAPE: {mape}%\n'
                  f'MSE: {mse}\n'
                  f'RMSE: {rmse}\n'
                  f'MAE: {mae}\n\n')

        return {'mape': mape, 'mse': mse, 'rmse': rmse, 'mae': mae}

    def hyper_parameter_optimisation(self):
        # sklearn.model_selection.GridSearchCV()
        # sklearn.model_selection.RandomizedSearchCV()
        # sklearn.model_selection.HalvingGridSearchCV()
        # sklearn.model_selection.HalvingRandomSearchCV()
        pass

    def declare_model(self, model):
        self.model = model

    def predict_y(self):
        return self.model.predict(self.X_test)

    def run_all(self):

        self.pre_processing()
        self.cross_validation()
        self.feature_selection()
        self.split()
        self.fit_model()
        self.predict_y()
        self.evaluate()


if __name__ == '__main__':

    EWEC_ML()
