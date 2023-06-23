import tensorflow as tf
import numpy

# 주어진 좌표
# ex)  (1,2), (2,6), (3,10), (4,17)
x_data = [1,2,3,4]
y_data = [2,6,10,17]

# 변수 초기 지정값
a_init = 2
b_init = 3

# TensorFlow 변수
a = tf.Variable(a_init)
b = tf.Variable(b_init) 

# 모델 정의 (함수식)
def model(x):
    return a * tf.pow(b, x)

# 손실 함수 정의
def loss_function(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# 옵티마이저 정의
optimizer = tf.optimizers.Adam(learning_rate=0.1)

# 최적화
for epoch in range(1000):
    # 그래디언트 계산을 위한 GradientTape
    with tf.GradientTape() as tape:
        # 모델 예측
        y_pred = model(x_data)
        # 손실 값 계산
        loss = loss_function(y_pred, y_data)
    
    # 그래디언트 계산
    gradients = tape.gradient(loss, [a, b])
    
    # 변수 업데이트
    optimizer.apply_gradients(zip(gradients, [a, b]))

# 최종 결과 출력
print("a =", a.numpy())
print("b =", b.numpy())
