# 1. 사용할 것 들만 미리 import 해온다!
import numpy as np
from sklearn.linear_model import LinearRegression

x = list(range(0,21))
y = [2*num + 3 for num in x]

x = np.array(x).reshape([-1,1])
y = np.array(y).reshape([-1,1])


# 2. 나 이 모델 쓸꺼야 선언한다!
lm = LinearRegression()

# 3. 모델.fit(X,y)로 학습시킨다.
# 15개 학습한다. 0~14
lm.fit(x[:15], y[:15])

# y = 2x + 3 이었다.
print(lm.coef_)  #기울기
print(lm.intercept_) #절편


# 4. 모델.predict(X)로 예측값을 만든다!
# 15부터 예측한다. 15~20
pred = lm.predict(x[15:])

#예측값
print(pred)

# 실제값
print(y[15:])