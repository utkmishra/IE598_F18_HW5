import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
import seaborn as sns
from sklearn import metrics
from sklearn.svm import SVC

# Extracting the principal components step-by-step

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()

cols = ['Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline', 'Class label']

# Feature correlation analysis

def feature_scatter(df_wine, cols):
    sns.set(style='whitegrid',context='notebook')
    sns.pairplot(df_wine[cols])
    plt.show()
feature_scatter(df_wine,cols)

# draw heatmap

def draw_heatmap(df_wine,cols):
    corr = df_wine[cols].corr()
    sns.set(font_scale=1.5)
    sns.heatmap(corr,cbar=True,annot=False,square=True,fmt='.2f',
                annot_kws={'size':10},yticklabels=cols,
                xticklabels=cols)
    plt.show()
    
draw_heatmap(df_wine,cols)

# draw boxplot
def draw_boxplot(df_wine, cols):
    # standardize data
    temp_data = df_wine.copy()
    for c in cols:
        col = temp_data[c]
        temp_data[c] = (col - col.mean()) / col.std()
    sns.boxplot(data=temp_data, orient='h')
    plt.show()
draw_boxplot(df_wine,cols)

# Splitting the data into 80% training and 30% test subsets.

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,random_state=42)

#LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)

lr_y_train_pred = lr.predict(X_train)
print( "Logistic Regression train accuracy score: ",metrics.accuracy_score(y_train, lr_y_train_pred) )
lr_y_pred = lr.predict(X_test)
print( "Logistic Regression test accuracy score: ",metrics.accuracy_score(y_test, lr_y_pred) )

#SVM

svm = SVC(kernel = 'linear', C= 1.0, random_state= 1)
svm.fit(X_train, y_train)
svm_y_train_pred = svm.predict(X_train)
print( "SVM Regression train accuracy score: ",metrics.accuracy_score(y_train, svm_y_train_pred) )
svm_y_pred = svm.predict(X_test)
print("SVM Regression test accuracy score: ",metrics.accuracy_score(y_test, svm_y_pred) )

# Standardizing the data

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

#Logistic Regression fitted on PCA transformed data

lr = LogisticRegression()
lr.fit(X_train_pca, y_train)
pca_lr_y_train_pred = lr.predict(X_train_pca)
pca_lr_y_pred = lr.predict(X_test_pca)
print( "Logistic Regression, on PCA transformed Data, train accuracy score: ",metrics.accuracy_score(y_train, pca_lr_y_train_pred) )
print("Logistic Regression, on PCA transformed Data, test accuracy score: ",metrics.accuracy_score(y_test, pca_lr_y_pred) )

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],y=X[y == cl, 1],alpha=0.6,c=cmap(idx),edgecolor='black',marker=markers[idx],label=cl)


plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PCA_LR_PC 1')
plt.ylabel('PCA_LR_PC 2')
plt.legend(loc='lower left')
plt.show()

#SVM Regression fitted on PCA transformed data

pca = PCA(n_components=2)
svm = SVC(kernel = 'linear', C= 1.0, random_state= 1)
svm.fit(X_train_pca, y_train)
pca_svm_y_train_pred = svm.predict(X_train_pca)
pca_svm_y_pred = svm.predict(X_test_pca)
print( "SVM Regression, on PCA transformed Data, train accuracy score: ",metrics.accuracy_score(y_train, pca_svm_y_train_pred) )
print("SVM Regression, on PCA transformed Data, test accuracy score: ",metrics.accuracy_score(y_test, pca_svm_y_pred) )

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],y=X[y == cl, 1],alpha=0.6,c=cmap(idx),edgecolor='black',marker=markers[idx],label=cl)


plot_decision_regions(X_train_pca, y_train, classifier=svm)
plt.xlabel('PCA_SVM_PC 1')
plt.ylabel('PCA_SVM_PC 2')
plt.legend(loc='lower left')
plt.show()

#LDA

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

#Logistic Regression fitted on LDA transformed data

lr = LogisticRegression()
lr.fit(X_train_lda, y_train)
lda_lr_y_train_pred = lr.predict(X_train_lda)
lda_lr_y_pred = lr.predict(X_test_lda)
print( "Logistic Regression, on LDA transformed Data, train accuracy score: ",metrics.accuracy_score(y_train, lda_lr_y_train_pred) )
print("Logistic Regression, on LDA transformed Data, test accuracy score: ",metrics.accuracy_score(y_test, lda_lr_y_pred) )

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],y=X[y == cl, 1],alpha=0.6,c=cmap(idx),edgecolor='black',marker=markers[idx],label=cl)


plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('LDA_LR_PC 1')
plt.ylabel('LDA_LR_PC 2')
plt.legend(loc='lower left')
plt.show()

#SVM Regression fitted on LDA transformed data

svm = SVC(kernel = 'linear', C= 1.0, random_state= 1)
svm.fit(X_train_lda, y_train)
lda_svm_y_train_pred = svm.predict(X_train_lda)
lda_svm_y_pred = svm.predict(X_test_lda)
print( "SVM Regression, on LDA transformed Data, train accuracy score: ",metrics.accuracy_score(y_train, lda_svm_y_train_pred) )
print("SVM Regression, on LDA transformed Data, test accuracy score: ",metrics.accuracy_score(y_test, lda_svm_y_pred) )

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],y=X[y == cl, 1],alpha=0.6,c=cmap(idx),edgecolor='black',marker=markers[idx],label=cl)


plot_decision_regions(X_train_pca, y_train, classifier=svm)
plt.xlabel('LDA_SVM_PC 1')
plt.ylabel('LDA_SVM_PC 2')
plt.legend(loc='lower left')
plt.show()

#kPCA

scikit_kpca = KernelPCA(n_components=2,kernel='rbf', gamma=1.0) #Gamma values can be updated here
X_train_skernpca = scikit_kpca.fit_transform(X_train_std, y_train)
X_test_skernpca= scikit_kpca.transform(X_test_std)

#Logistic Regression fitted on KPCA transformed data

lr = LogisticRegression()
lr.fit(X_train_skernpca, y_train)
kpca_lr_y_train_pred = lr.predict(X_train_skernpca)
kpca_lr_y_pred = lr.predict(X_test_skernpca)
print( "KPCA Logistic Regression train accuracy score (gamma=1.0): ",metrics.accuracy_score(y_train, kpca_lr_y_train_pred) )
print("KPCA Logistic Regression test accuracy score (gamma=1.0): ",metrics.accuracy_score(y_test, kpca_lr_y_pred) )

#SVM Regression fitted on KPCA transformed dataset

svm = SVC(kernel = 'linear', C= 1.0, random_state= 1)
svm.fit(X_train_skernpca, y_train)
kpca_svm_y_train_pred = svm.predict(X_train_skernpca)
kpca_svm_y_pred = svm.predict(X_test_skernpca)
print( "KPCA SVM Regression train accuracy score (gamma=1.0): ",metrics.accuracy_score(y_train, kpca_svm_y_train_pred) )
print("KPCA SVM Regression test accuracy score (gamma=1.0): ",metrics.accuracy_score(y_test, kpca_svm_y_pred) )

print("My name is Utkarsh Mishra")
print("My NetID is: umishra3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")