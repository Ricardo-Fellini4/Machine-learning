import numpy
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
url = 'D:\Machine learning\SPECTF heart prediction\SPECTF.train.csv'
dataset = read_csv(url, header=None)
print(dataset)
# D:\Machine learning\SPECTF heart prediction\project.py
#$D:\Machine learning\SPECTF heart prediction\SPECTF.train.csv
# types
set_option('display.max_rows', 500)
print(dataset.dtypes)
set_option('display.width', 50)
print(dataset.head(20))
print(dataset.groupby(44).size()) #Do 44 là thứ tự của cột chứa các giá trị phân loại.
set_option('precision', 3)
print(dataset.describe())
dataset.hist()
pyplot.show()