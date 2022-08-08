# -*- coding: utf-8 -*-
"""
Author: Caner Canlıer - 21702121
"""
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.manifold import TSNE
from sammon import sammon as SammonMapping # I used a GitHub repo (https://github.com/tompollard/sammon) to use sammon mapping.


### Load the Dataset and Split the Data Into Half to create Train and Test Data

# Load dataset
data = loadmat("C:/Users/ccane/Desktop/ge461/PROJECT 2/digits.mat")
features = data["digits"]
labels = data["labels"]
# Split data into half -> train and test 
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, shuffle=True)


# Question 1.1:
pca = PCA(n_components=30)
Xt = pca.fit_transform(x_train)

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=list(labels))
plt.ylabel('percentange of explained variance')
plt.xlabel('principal component')
plt.title('scree plot')
plt.show()

pca = PCA(n_components=400).fit(x_train)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("Principal Component vs. Variance")
plt.xlabel("Principal Component ")
plt.ylabel("Variance")
plt.show()

ev=sorted(pca.explained_variance_,reverse=True)
plt.plot(np.cumsum(ev))
plt.title("Principal Component vs. Eigenvalue")
plt.xlabel("Principal Component ")
plt.ylabel("Eigenvalue")
plt.show()


# Question 1.2:
x_train_mean = (pca.mean_.reshape(20,20)).T
plt.imshow(x_train_mean)
plt.title("Mean for the Training Data Set")

print(pca.components_.shape)

for i in range(10):
   EV=(pca.components_[i].reshape(20,20)).T
   plt.imshow(EV)
   plt.show()


# Question 1.3:
train_errors = []
test_errors = []
components = []
for i in range (2,100):
    components.append(i)
for i in components:
    pca = PCA(n_components=i)
    pca.fit(x_train)

    x_train_projected = pca.transform(x_train)
    x_test_projected = pca.transform(x_test)

    clf = GaussianNB()
    clf.fit(x_train_projected, (y_train.T)[0])

    # predicting train data
    predicted_train = clf.predict(x_train_projected)
    expected_train = (y_train.T)[0]
    matches_train = (predicted_train == expected_train)
    train_errors.append(1-(matches_train.sum() / float(len(matches_train))))
    # predicting test data
    predicted_test = clf.predict(x_test_projected)
    expected_test = (y_test.T)[0]
    matches_test = (predicted_test == expected_test)
    test_errors.append(1-(matches_test.sum() / float(len(matches_test))))

plt.plot(components,train_errors,color='g',label='Train Errors')
plt.ylabel("Error rate of Train set")
plt.xlabel("Number of Components")
plt.title("Number of Components Vs. Train Error rate")

plt.show()
plt.plot(components,test_errors,color='r',label='Test Errors')
plt.ylabel("Error rate of Test Set")
plt.xlabel("Number of Components")
plt.title("Number of Components Vs. Test Error rate")
plt.show()

plt.plot(components,train_errors,color='g',label='Train Errors')
plt.plot(components,test_errors,color='r',label='Test Errors')
plt.ylabel("Error rate")
plt.xlabel("Number of Components")
plt.title("Number of Components Vs. Error rate")
plt.legend()
plt.show()


"""## Apply PCA"""

#Question 2
## Apply LDA
model = LinearDiscriminantAnalysis()
X_r2 = model.fit(x_train, (y_train.T)[0]).transform(x_train)

# Question 2.1: Displaying new bases 
for index in range(9):
    plt.axis("off")
    plt.subplot(3, 3, index+1)
    plt.imshow((model.scalings_[:,index]).reshape(20,20))
    
plt.suptitle("LDA Bases")
plt.show()

# Question 2.2-3: Create Different Subspaces

train_errors = []
test_errors = []
components = []
for i in range (1,10):
    components.append(i)

for i in components:
    lda = LinearDiscriminantAnalysis(n_components=i)
    lda.fit(x_train, (y_train.T)[0])

    x_train_projected = lda.transform(x_train)
    x_test_projected = lda.transform(x_test)

    clf = GaussianNB()
    clf.fit(x_train_projected, (y_train.T)[0])

    # predicting train data
    predicted_train = clf.predict(x_train_projected)
    expected_train = (y_train.T)[0]
    matches_train = (predicted_train == expected_train)
    train_errors.append(1-(matches_train.sum() / float(len(matches_train))))
    # predicting test data
    predicted_test = clf.predict(x_test_projected)
    expected_test = (y_test.T)[0]
    matches_test = (predicted_test == expected_test)
    test_errors.append(1-(matches_test.sum() / float(len(matches_test))))

plt.plot(components,train_errors,color='g',label='Train Errors')
plt.ylabel("Error rate of Train set")
plt.xlabel("Dimension of Each Subspace")
plt.title("Dimension of Each Subspace Vs. Train Error rate")

plt.show()
plt.plot(components,test_errors,color='r',label='Test Errors')
plt.ylabel("Error rate of Test Set")
plt.xlabel("Dimension of Each Subspace")
plt.title("Dimension of Each Subspace Vs. Test Error rate")
plt.show()

plt.plot(components,train_errors,color='g',label='Train Errors')
plt.plot(components,test_errors,color='r',label='Test Errors')
plt.ylabel("Error rate")
plt.xlabel("Dimension of Each Subspace")
plt.title("Dimension of Each Subspace Vs. Error rate")
plt.legend()
plt.show()


#Question 3.1:
# Sammon's Mapping

#Changing maximater value will give different results. I tried 100,200 and 500. For each result I change the value in the code.
[y_200,E] = SammonMapping(features, n=2, maxiter=100, maxhalves=20) 


# Plot
plt.scatter(y_200[labels[:,0] == 0, 0], y_200[labels[:,0] == 0, 1], marker='s',label="0")
plt.scatter(y_200[labels[:,0] == 1, 0], y_200[labels[:,0] == 1, 1], marker='s',label="1")
plt.scatter(y_200[labels[:,0] == 2, 0], y_200[labels[:,0] == 2, 1], marker='s',label="2")
plt.scatter(y_200[labels[:,0] == 3, 0], y_200[labels[:,0] == 3, 1], marker='s',label="3")
plt.scatter(y_200[labels[:,0] == 4, 0], y_200[labels[:,0] == 4, 1], marker='s',label="4")
plt.scatter(y_200[labels[:,0] == 5, 0], y_200[labels[:,0] == 5, 1], marker='s',label="5")
plt.scatter(y_200[labels[:,0] == 6, 0], y_200[labels[:,0] == 6, 1], marker='s',label="6")
plt.scatter(y_200[labels[:,0] == 7, 0], y_200[labels[:,0] == 7, 1], marker='s',label="7")
plt.scatter(y_200[labels[:,0] == 8, 0], y_200[labels[:,0] == 8, 1], marker='s',label="8")
plt.scatter(y_200[labels[:,0] == 9, 0], y_200[labels[:,0] == 9, 1], marker='s',label="9")
plt.title("Sammon's Mapping for 200 Iterations")
plt.legend(loc=0, bbox_to_anchor=(1,1), title="Digits")
plt.show()

#Question 3.2:

# Applying t-SNE

tSNE = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=2000, n_iter_without_progress=500, random_state=10)
result = tSNE.fit_transform(features)

plt.scatter(result[labels[:,0] == 0, 0], result[labels[:,0] == 0, 1], marker='.', label="0")
plt.scatter(result[labels[:,0] == 1, 0], result[labels[:,0] == 1, 1], marker='.', label="1")
plt.scatter(result[labels[:,0] == 2, 0], result[labels[:,0] == 2, 1], marker='.', label="2")
plt.scatter(result[labels[:,0] == 3, 0], result[labels[:,0] == 3, 1], marker='.', label="3")
plt.scatter(result[labels[:,0] == 4, 0], result[labels[:,0] == 4, 1], marker='.', label="4")
plt.scatter(result[labels[:,0] == 5, 0], result[labels[:,0] == 5, 1], marker='.', label="5")
plt.scatter(result[labels[:,0] == 6, 0], result[labels[:,0] == 6, 1], marker='.', label="6")
plt.scatter(result[labels[:,0] == 7, 0], result[labels[:,0] == 7, 1], marker='.', label="7")
plt.scatter(result[labels[:,0] == 8, 0], result[labels[:,0] == 8, 1], marker='.', label="8")
plt.scatter(result[labels[:,0] == 9, 0], result[labels[:,0] == 9, 1], marker='.', label="9")
plt.title('t-SNE Mapping for Perplexity = 10 and 2000 iterations')
plt.legend(loc=0, bbox_to_anchor=(1,1), title="Digits")
plt.show()

tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=2000, random_state=10)
result = tsne.fit_transform(features)

plt.scatter(result[labels[:,0] == 0, 0], result[labels[:,0] == 0, 1], marker='.', label="0")
plt.scatter(result[labels[:,0] == 1, 0], result[labels[:,0] == 1, 1], marker='.', label="1")
plt.scatter(result[labels[:,0] == 2, 0], result[labels[:,0] == 2, 1], marker='.', label="2")
plt.scatter(result[labels[:,0] == 3, 0], result[labels[:,0] == 3, 1], marker='.', label="3")
plt.scatter(result[labels[:,0] == 4, 0], result[labels[:,0] == 4, 1], marker='.', label="4")
plt.scatter(result[labels[:,0] == 5, 0], result[labels[:,0] == 5, 1], marker='.', label="5")
plt.scatter(result[labels[:,0] == 6, 0], result[labels[:,0] == 6, 1], marker='.', label="6")
plt.scatter(result[labels[:,0] == 7, 0], result[labels[:,0] == 7, 1], marker='.', label="7")
plt.scatter(result[labels[:,0] == 8, 0], result[labels[:,0] == 8, 1], marker='.', label="8")
plt.scatter(result[labels[:,0] == 9, 0], result[labels[:,0] == 9, 1], marker='.', label="9")
plt.title('t-SNE Mapping for Perplexity = 20 and 2000 iterations')
plt.legend(loc=0, bbox_to_anchor=(1,1), title="Digits")
plt.show()

tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1000, random_state=10)
result = tsne.fit_transform(features)

plt.scatter(result[labels[:,0] == 0, 0], result[labels[:,0] == 0, 1], marker='.', label="0")
plt.scatter(result[labels[:,0] == 1, 0], result[labels[:,0] == 1, 1], marker='.', label="1")
plt.scatter(result[labels[:,0] == 2, 0], result[labels[:,0] == 2, 1], marker='.', label="2")
plt.scatter(result[labels[:,0] == 3, 0], result[labels[:,0] == 3, 1], marker='.', label="3")
plt.scatter(result[labels[:,0] == 4, 0], result[labels[:,0] == 4, 1], marker='.', label="4")
plt.scatter(result[labels[:,0] == 5, 0], result[labels[:,0] == 5, 1], marker='.', label="5")
plt.scatter(result[labels[:,0] == 6, 0], result[labels[:,0] == 6, 1], marker='.', label="6")
plt.scatter(result[labels[:,0] == 7, 0], result[labels[:,0] == 7, 1], marker='.', label="7")
plt.scatter(result[labels[:,0] == 8, 0], result[labels[:,0] == 8, 1], marker='.', label="8")
plt.scatter(result[labels[:,0] == 9, 0], result[labels[:,0] == 9, 1], marker='.', label="9")
plt.title('t-SNE Mapping for Perplexity = 10 and 1000 iterations')
plt.legend(loc=0, bbox_to_anchor=(1,1), title="Digits")
plt.show()


tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=1000, random_state=10)
result = tsne.fit_transform(features)

plt.scatter(result[labels[:,0] == 0, 0], result[labels[:,0] == 0, 1], marker='.', label="0")
plt.scatter(result[labels[:,0] == 1, 0], result[labels[:,0] == 1, 1], marker='.', label="1")
plt.scatter(result[labels[:,0] == 2, 0], result[labels[:,0] == 2, 1], marker='.', label="2")
plt.scatter(result[labels[:,0] == 3, 0], result[labels[:,0] == 3, 1], marker='.', label="3")
plt.scatter(result[labels[:,0] == 4, 0], result[labels[:,0] == 4, 1], marker='.', label="4")
plt.scatter(result[labels[:,0] == 5, 0], result[labels[:,0] == 5, 1], marker='.', label="5")
plt.scatter(result[labels[:,0] == 6, 0], result[labels[:,0] == 6, 1], marker='.', label="6")
plt.scatter(result[labels[:,0] == 7, 0], result[labels[:,0] == 7, 1], marker='.', label="7")
plt.scatter(result[labels[:,0] == 8, 0], result[labels[:,0] == 8, 1], marker='.', label="8")
plt.scatter(result[labels[:,0] == 9, 0], result[labels[:,0] == 9, 1], marker='.', label="9")
plt.title('t-SNE Mapping for Perplexity = 20 and 1000 iterations')
plt.legend(loc=0, bbox_to_anchor=(1,1), title="Digits")
plt.show()