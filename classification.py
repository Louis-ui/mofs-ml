from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from resultAnalysis import *
import shap


# Classification

# 定义一个保存模型的字典，根据 key 来选择加载哪个模型
models = {
    "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
    "svm": SVC(kernel="rbf", gamma="auto"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=20),
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


def classificate(Xtrain, Ytrain, Xtest, Ytest, data):

    for function in models:
        model = models[function]
                # 训练模型
        # model = models["decision_tree"]
        model.fit(Xtrain, Ytrain)

        # 预测并输出分类结果报告
        # print("模型评估")
        predictions = model.predict(Xtest)

        # print("predictions")
        # print(predictions)

        regressionAnalysis(function, ['type'], Ytest, predictions)

        # calc_feature_importance_shap(model, data)


def calc_feature_importance_shap(tree_model, data):
    """
    计算特征重要性
    :param tree_model: 树模型
    :param data: 特征数据
    :returns: 特征重要度
    """
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(data)
    # shap_values_df = pd.DataFrame(shap_values)
    shap.summary_plot(shap_values, data)
    # return shap_values_df.abs().mean(axis=0)


