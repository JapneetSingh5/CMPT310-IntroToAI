from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

# Do not change anything in this code
X, y = make_classification(
    n_samples=1000,
    n_features=12,
    n_informative=4,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=5,
    shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=43)

RFClassifier = RandomForestClassifier()
RFClassifier.fit(X_train, y_train)
y_test_predicted = RFClassifier.predict(X_test)

feature_importances = RFClassifier.feature_importances_
print('Feature Importances: \n')
for i in range(0, 12):
    print('Feature '+str(i+1), feature_importances[i])

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), feature_importances, align='center')
plt.xticks(range(X_train.shape[1]), ['Feature ' + str(i) for i in range(1,X_train.shape[1]+1)], rotation=90)
plt.tight_layout()
plt.show()

accuracy = RFClassifier.score(X_test, y_test)
print('Accuracy using score function: ', accuracy)
accuracy2 = metrics.accuracy_score(y_test, y_test_predicted)
print('Accuracy using accuracy_score function: ', accuracy2)
