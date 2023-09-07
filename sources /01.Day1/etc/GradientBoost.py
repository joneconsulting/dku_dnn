from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, 
                                                    stratify=cancer.target, random_state=0)

gbc = GradientBoostingClassifier(random_state=0) # 기본값: max_depth=3, learning_rate=0.1
gbc.fit(x_train, y_train)
score_train = gbc.score(x_train, y_train) # train set 정확도
print('{:.3f}'.format(score_train))
# 1.000 -> 과적합
score_test = gbc.score(x_test, y_test) # 일반화 정확도
print('{:.3f}'.format(score_test))
# 0.958

## overfitting 방지 -> 트리의 깊이를 줄여 pre-pruning을 강하게
print("## pre-pruning")
gbc = GradientBoostingClassifier(random_state=0, max_depth=1)
gbc.fit(x_train, y_train)
score_train_pre = gbc.score(x_train, y_train) # train set 정확도
print('{:.3f}'.format(score_train_pre))
# 0.995 (낮아짐)
score_test_pre = gbc.score(x_test, y_test) # 일반화 정확도
print('{:.3f}'.format(score_test_pre))
# 0.965 (높아짐)

## learning_rate
print("## learning_rate")
gbc = GradientBoostingClassifier(random_state=0, max_depth=3, learning_rate=0.01) # 기본값 0.1
gbc.fit(x_train, y_train)
score_train_lr = gbc.score(x_train, y_train)
print('{:.3f}'.format(score_train_lr))
# 0.995
score_test_lr = gbc.score(x_test, y_test) 
print('{:.3f}'.format(score_test_lr))
# 0.944 