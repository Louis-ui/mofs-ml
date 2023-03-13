from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from dataClean import Xtrain, Ytrain, Xtest, Ytest


# Classification

## 定义一个保存模型的字典，根据 key 来选择加载哪个模型
models = {
    "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
    "svm": SVC(kernel="rbf", gamma="auto"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "mlp": MLPClassifier()
}

# 训练的数据
# print("训练的数据")

# print("XTrain")
# print(Xtrain)
# print(Xtrain.astype("float"))

# print("YTrain")
# print(Ytrain)
# print(Ytrain.astype("float"))

# print("YTrain")
# print(Ytrain)

## 训练模型
# model = models["random_forest"]
# model.fit(Xtrain.astype("float"), Ytrain.astype("float"))

## 预测并输出分类结果报告
# print("模型评估")
# predictions = model.predict(Xtest.astype("float"))

# print("predictions")
# print(predictions)

# target_names = ['Heat_furfural']
# # target_names = ['Henry_furfural','Henry_Tip5p','Heat_furfural','Heat_Tip5p']
# print(classification_report(Ytest, predictions, target_names=target_names))