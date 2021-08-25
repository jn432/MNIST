#Project in degree
#classification problem using the MNIST numbers dataset
#tasks were to do an end-to-end project, including exploratory analysis, selecting and tuning models and evaluation

#to load/manipulate the dataset
from sklearn.datasets import fetch_openml
import numpy as np
#for exploratory analysis
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stat
#3 models chosen
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#for parameter tuning
from sklearn.model_selection import GridSearchCV
#for evaluation
from sklearn.metrics import classification_report, confusion_matrix

#download the dataset

your_data_home = 'C:/Users/FFMaster/Desktop/CSCI316/Assignments/Ass1'
mnist = fetch_openml('mnist_784', version=1, data_home=your_data_home)

#set features as X, class as Y
X = mnist['data']
Y = mnist['target']

#PART A - discover and visualise the data

#print the first 10 digits that are labeled the same
#used to visualise and compare same classes as each other
printed = 0
for j in range(10):
    label = str(j)
    for i in range(len(Y)):
        if Y[i] == label:
            printed += 1
            print("Index:", i, "\tTrue value: ", Y[i])
            some_digit = X[i]
            some_digit_image = some_digit.reshape(28,28)
            plt.imshow(some_digit_image,cmap = mpl.cm.binary,interpolation='nearest')
            plt.axis('off')
            plt.show()
        if printed >= 10:
            printed = 0
            break

#comments on viewing the data
#check for missing values or other weird values
#0
# index 63 - the 0 is a little weird, unsure if it is noise or just poor handwriting
#1
# index 24 - is indeed a 1, but most of the others are single vertical line 1's
# index 67 - noisy data, is a 1, but has noise around it for some reason
# index 70 - similar to 24, but not as pronounced
#2
# some 2's have the loop at the bottom, some do not
#3
# index 10 - noise? hard to tell
# index 27 - a little bit of noise on the top tip of 3
# index 49 - tiny bit of noise to the side
#4
# index 53 - a bit of noise on the right side
# index 89 - a bit of noise spot on the left side
#5
# index 47 - a bit of noise on the right side
# index 132 - looks more like 1 to me
# some tips of the 5 are disconnected
#6
# nothing interesting
#7
# index 38 - is a 7, but some people add the extra slash, outlier that is important to note
#8
# nothing interesting
#9
# some have a more pronounced hook at the bottom
#overall thoughts
# the numbers have different thickness, probably due to different pens
# darker pixels are where the pen went, so the data with a little bit of noise but no nearby dark pixels
# could be cleaned, but could also be ignored, most of the data is clean

#print the "average" image
some_digit = X.mean(0)
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image,cmap = mpl.cm.binary,interpolation='nearest')
plt.axis()
plt.show()
#shows where the most likely pixels to have darker areas, confirms that the majority
#of numbers are written at the middle of the image, and is where the most information
#can be found

#explore 7 some more, since it has an interesting image(index 38)
#print the mean of all 7's
X7 = X[Y == '7']
some_digit = X7.mean(0)
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image,cmap = mpl.cm.binary,interpolation='nearest')
plt.axis()
plt.show()
#confirmation that the dash is just an outlier, most interestingly the average 7 is
#clearly a 7 to the eye

#try mean for each number after seeing good results from 7
for i in range(10):
    Xtemp = X[Y == str(i)]
    some_digit = Xtemp.mean(0)
    some_digit_image = some_digit.reshape(28,28)
    plt.imshow(some_digit_image,cmap = mpl.cm.binary,interpolation='nearest')
    plt.axis()
    plt.show()

#try min/max for all images??
some_digit = X.max(0)
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image,cmap = mpl.cm.binary,interpolation='nearest')
plt.axis()
plt.show()
#white spots are unused pixels, can be removed to generalize model
#confirms that corners are not being written on

#min is worthless
some_digit = X.min(0)
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image,cmap = mpl.cm.binary,interpolation='nearest')
plt.axis()
plt.show()


#check if distribution is skewed for either train or test set
plt.hist(Y, color='#0504aa', alpha=0.7, rwidth=0.85)
plt.title('Distribution of each class')
plt.xlabel('Digit')
plt.ylabel('Frequency')
plt.show()
#looks to be uniformly distributed, 1's is slightly higher

#look at values within some rows
print(X[0])
print(X[4502])
print(X[60233])

#stats and 5 number summary
mode = stat.mode(X, axis=0)
median = np.median(X, axis=0)
mean = np.mean(X, axis=0)
var = np.var(X, axis=0)
min = np.amin(X, axis=0)
max = np.amax(X, axis=0)
q1 = np.quantile(X, 0.25, axis=0)
q3 = np.quantile(X, 0.75, axis=0)
for i in range(len(mode[0][0])):
    #cartesian coordinates of the attribute
    #1st number is pixels across, 2nd number is pixels down
    print(i%28, int(i/28))
    #verifying sparsity with mode, then checking skewness of data using mean/median
    print("Mode:", mode[0][0][i], "\tMedian:", median[i], '\tMean:', mean[i])
    #variance
    print("Variance:", var[i])
    #5 number summary
    print("Min:", min[i], "\tQ1:", q1[i],"\tMedian:", median[i], "\tQ3:", q3[i], '\tMax:', max[i], '\n')
    #some attributes are all 0's, I suspect they are the corners of each image

#sparse
#looks to be sparse data, high numbers are where the pen went(dark pixels), around it will be other dark
#pixels or lighter pixels
#the most common value in each attribute is 0, confirming sparse data
#skewness
#due to sparseness of the data, most attributes also right skewed. Attributes that are not skewed are the
#corners of the image, where nothing is being written, and towards the middle of the image, where numbers
#are more likely to be written
#variance
#suspicion that the attributes with high variance are where the numbers are written, the middle of the image
#and will be important in distinguishing in classifying the number

#PART B - prepare the data for machine learning algorithms

#normalize data from 0 to 1(min-max normalization)
#min value is 0, max is 255 for all columns since this is image data
#(value - min)/(max - min) = value/255
X = X/255

#split into training and test sets
X_train = X[:60000]
X_test = X[60000:]
Y_train = Y[:60000]
Y_test = Y[60000:]

#PART C, D, E

#model 1 - Decision trees

#default params
dt = tree.DecisionTreeClassifier()
dt.fit(X_train, Y_train)

#initial evaluation
Y_pred = dt.predict(X_train)
print(confusion_matrix(Y_train, Y_pred))
print(classification_report(Y_train, Y_pred))
Y_pred = dt.predict(X_test)
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))


#parameter tuning

#choose some parameters
parameter_space = {
    'criterion': ['gini', 'entropy'],
    'max_features':[None,'auto','log2']
}

dt_gs = tree.DecisionTreeClassifier()
clf = GridSearchCV(dt_gs, parameter_space, n_jobs=4)
clf.fit(X_train,Y_train)
print('Best parameters found:\n', clf.best_params_)

#results found that using entropy as splitting method is better, None(default) for max_features
#retrain with new params and see the results
dt = tree.DecisionTreeClassifier(criterion='entropy',max_features=None)
dt.fit(X_train, Y_train)
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
#does not seem to have improved at all...


#model 2 - MLP

#setting learning_rate_init to 0.1 for faster training
mlp = MLPClassifier(learning_rate_init=0.1)
mlp.fit(X_train, Y_train)

#initial evaluation
Y_pred = mlp.predict(X_train)
print(confusion_matrix(Y_train, Y_pred))
print(classification_report(Y_train, Y_pred))
Y_pred = mlp.predict(X_test)
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
#already good results, 90% accuracy

#parameter tuning
parameter_space = {
    'hidden_layer_sizes': [(50),(100),(50,20),(100,20)],
    'solver': ['sgd', 'adam']
}

mlp_gs = MLPClassifier(learning_rate_init=0.1)
clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=4)
clf.fit(X_train,Y_train)
print('Best parameters found:\n', clf.best_params_)
#{'hidden_layer_sizes': 100, 'solver': 'sgd'}

mlp = MLPClassifier(hidden_layer_sizes=(100), learning_rate_init=0.1, solver='sgd')
mlp.fit(X_train, Y_train)
Y_pred = mlp.predict(X_test)
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
#change the solver to 'sgd'(stochastic gradient descent) - results improved dramatically
# this one gets to 98%


#model 3 - Random Forest

#create and train random forest
#default params are pretty good, 97% on test set, but predicting takes some time(n=100)
#n=50 also gives 97%, n=10 gives 95%, n=20 gave 96%, n=300 still gave 97% accuracy(improved in the digits)
#n_jobs sets the number of processes, -1 means use all available(default is 1)
rf = RandomForestClassifier(n_jobs=4)
rf.fit(X_train, Y_train)

#initial evaluation
Y_pred = rf.predict(X_train)
print(confusion_matrix(Y_train, Y_pred))
print(classification_report(Y_train, Y_pred))
Y_pred = rf.predict(X_test)
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

#parameter tuning
parameter_space = {
    'n_estimators': [100, 150],
    'criterion': ['gini', 'entropy'],
}

rf_gs = RandomForestClassifier(n_jobs=4)
clf = GridSearchCV(rf_gs, parameter_space, n_jobs=4)
clf.fit(X_train,Y_train)
print('Best parameters found:\n', clf.best_params_)
#{'criterion': 'gini', 'n_estimators': 150}

rf = RandomForestClassifier(n_jobs=4, n_estimators=150)
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
Y_pred = rf.predict(X_test)
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))