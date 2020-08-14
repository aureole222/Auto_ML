# Auto_ML
This project provides an integrated toolkits for data auto-preprocessing and a framework for hyperparamter tuning.
## Required Packages:
•	tensorflow >=2.0.0 \
•	sklearn\
•	mxnet\
•	lightGBM\
•	seaborn\
•	tqdm


## Data preprocessing toolkit:
This contains two constructor: \
•	TimeSeries_DataFrame  \
•	Non_TimeSeries_DataFrame

To initiate the function, an example is :     NYC_cab_object = Non_TimeSeries_DataFrame(df = NYC_data,label_col='if_tip_paid') \
where we need to state the data we are working with and the column that we need to classify on.

And it contains following functions:

### Class function and methods:
@property \
def data_type(self):\
This function shows the type of data in each column

@property\
def na_report(self):\
This function reports the number and the percentage of missing data in each column

@property\
def data_quality_score (self):\
This function performs a simple logistic regression on the dataset to get a baseline precision and accuracy score
  
  
def Convert_numerical(self, col_names, noise = ','):\
This function convert string to float if the data type was incorrect because of the number is stored as string or there is comma or space in between. For example ‘ 64,300 ’ are converted to int 64300.\
**col_names**: columns that need to be converted to numerical value\
**noise**: the wrong string that need to be removed.

def update (self):\
This function updates the numerical and categorical columns’ names and store into the dataset



### def auto_feature_selection(self, n_features = 'auto'):

This function aims to selected the best N features from all available features using a combination of different algorithms.\
**n_features**: The number of feature we want to keep. If it equals ‘auto’, half of the features are selected

Feature selection are based on following algorithms simultaneously and then decided by majority vote\
•	Chi-2\
•	Recursive feature elimination (RFE) http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html \
•	Embedded LASSO regression or L1 penalized Logistic Regression\
•	Random Forest Importance\
•	LightGBM: Light GBM is a gradient boosting framework that uses tree based learning algorithm. https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc



### def imputate(self, method={} , hyperparas = [128,0.5,100,10000])
This function aims to deal with missing data in the pandas dataframe.\
**Hyperparas**: only for GAIN method, useless for all other methods\
**Method**: is a dictionary in the format of, for example:   method = {'drop': [col1 , col2]    , 'mean': [col]}\
‘drop’ | ‘mean’
‘median’ | 
‘mode’ | 
‘max_occurance’ | 
‘KNN’ | 
‘GAIN’ | 
‘Simple_Datawig’ | 
‘Simple_MICE’


### def Noralization(self, columns=[], method='Max_Min'):
This function aims to normalize the value of numerical features. The scale is important for algorithms such as PCA and neural networks.\
**Columns**: This is a list containing list of numerical features that need to be normalized.\
**Method**: It is a string choose from the following\
'MaxMin' | 
‘Std’ | 
‘Robust’ | 
‘MaxAbs’


### def scale_transformation(self, cols = [], method = 'Yeo-Johnson'):       

This function aims to scale the values of numerical feature to achieve a better distribution\
**Cols**: This contains columns that need to be transformed. If no column is selected, it will automatically transform feature with skewness >10\
**Method**: This is a string choose from following;\
‘Box-Cox’ | ‘Yeo-Johnson’




### def outlier_removal(self,columns=[],limits=[0.05,0.95],method='Trim'):

This function aims to remove the outlier with the give percentage in order to achieve a better ML fitting result\
**columns**: This is a list containing list of numerical features that need to check the outliers.\
**limits**: The upper and lower percentage of word you want to remove. Not applicable on Gaussian method.\
**method**: It is a string choose from the following:	\
‘Trim’ | ‘Inter-quantal’ | ‘Gaussian’ | ‘Quantiles’

### def hampel_filter(self,columns, window_size=10, n_sigmas=3):   

This function removes outliers only on time-series object! The Hampel filter is a member of the class of decsion filters that replaces the central value in the data window with the median if it lies far enough from the median to be deemed an outlier. \
Link:  https://link.springer.com/article/10.1186/s13634-016-0383-6#:~:text=The%20Hampel%20filter%20is%20a,to%20be%20deemed%20an%20outlier. \
**columns**: This is a list containing list of numerical features that need to check the outliers.\
**window_size**: This is the number of historical values to look at\
**n_sigmas**: hyperparameter for sensitivity

### def discretization(self, columns, method='EqualFrequency', n_bin=5, force_multi_iter = False):

This function aims to discretize the numerical values that is sickly distributed or the small change in value makes no difference.\
**columns**: This is a list containing list of numerical features that need to discretize.\
**n_bins**:  This is the number of unique values we want the final discretization to achieve. This will not be strictly followed by method of ‘DecesionTree’ and ‘MDLP_Entropy’, as the numbers are machine selected.\
**force_multi_iter**: This ask us whether to force another round of discretization based on information gain algorithm (IG) on top of the ‘MDLP_Entropy’ method if the automatic discretization achieved a number of unique values that is less than n_bins\
**method**:  It is a string choose from the following:\
‘EqualWidth’ | ‘EqualFrequency’ | ‘Kmeans’ | ‘DecisionTree’ | ‘MDLP_Entropy’


### def linear_interpolation(self, columns, method='fill_between'):

This function interpolate missing values only on time-series object!\
**columns**: This is a list containing list of numerical features that need to be interpolated.\
**method**: It is a string choose from the following\
‘fill_between’ | ‘EMW’


### def get_top_abs_correlations(self, n=5,only_numeric=True):

The function will visualize the correlation if the data dimension<=10. It will return pairs of features that has highest correlation\
**n**: Top n highest correlated features are shown\
**only_numeric**: this asks whether we only look numerical columns’ correlations


### def feature_encoder(self, columns, method='OneHot'):

The function will give numerical multi-dimensional embedding for the categorical features \
**columns**: This is a list containing list of numerical features that need to be embedding.\
**method**: It is a string choose from the following


### def merge_column(self, columns, target_col=['untitle']):

This function aims to reduce the dimension of data using PCA method. \
**columns**: This is a list ask for numerical columns that need to be merged\
**target_col**: This is a list that ask for the names of columns that save the dimension-reduced data. The length of the list represent the target dimension\
The function select one of method below for dimension reduction automatically based on data type:\
**Method 1**: Principle component analysis (PCA) if entry are all numerical data\
**Method 2**: Factor analysis for mixture data (FAMD) if columns are a mixture of numerical and categorical variable. http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/115-famd-factor-analysis-of-mixed-data-in-r-essentials/ \
**Method 3**: Multiple correspondence analysis (MFA) if columns are all categorical data

### def auto_merge_high_correlation(self):

This function aims to provide automatic dimension reduction by performing PCA/MCA/PAMD on each group of high correlation feature.\
The algorithm detects the set features that is highly correlated to each other in form of list of list. Then the features in each list are highly correlated to each other and hence merged into one feature using dimension reduction algorithms
 





