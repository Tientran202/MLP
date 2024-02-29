# load thư viện và dữ liệu
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv

# Load file X_train, Y_train, X_test
X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")
X_test = np.load("X_test.npy")

# Chuyển đổi về hình dạng 28x28
X_train_reshaped = X_train.reshape(-1, 28, 28)
X_test_reshaped = X_test.reshape(-1, 28, 28)

# # vẽ một số dữ liệu ở tập train
# for i in range(100):
#     plt.subplot(10, 10, 1 + i)
#     plt.title(str(Y_train[i]))
#     image = X_train_reshaped[i]
#     plt.imshow(image, cmap="gray")
# plt.show()

# chuyển dữ liệu x_train về khoảng 0 và 1
from tensorflow.keras.utils import to_categorical


X_train_reshaped, X_test = X_train_reshaped / 255.0, X_test / 255.0
# chuyển dữ liệu y_train từ label sang encode
Y_train = to_categorical(Y_train)


# định nghĩa hàm loss
loss_fn = tf.keras.losses.CategoricalCrossentropy()
# định nghĩa thuật toán tối ưu
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01)


# xây dựng mô hình
def create_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28,28)),

            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    return model


# tạo mô hình và in ra bảng tổng kết
model = create_model()

# xây dựng hàm lưu lại mô hình dựa theo loss của tập validation
weights_filepath = "./weights/"
callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=weights_filepath,
    monitor="val_loss",
    verbose=1,
    save_best_only=False,
    save_weights_only=False,
)

# bắt đầu training
his = model.fit(
    X_train_reshaped,
    Y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    callbacks=callback,
)

# # vẽ đường loss trên tập train và tập validation
# plt.plot(his.history["val_loss"], c="coral", label="validation loss line")
# plt.plot(his.history["loss"], c="blue", label="train loss line")
# legend = plt.legend(loc="upper center")
# plt.show()

# # vẽ đường accuracy trên tập train và tập validation
# plt.plot(his.history["val_accuracy"], c="coral", label="validation accuracy line")
# plt.plot(his.history["accuracy"], c="blue", label="train accuracy line")
# legend = plt.legend(loc="lower center")
# plt.show()


# Load file mô hình đã huấn luyện
model = create_model()
model = tf.keras.models.load_model("./weights/")

# Đánh giá mô hình trên tập test
loss, acc = model.evaluate(X_test_reshaped, verbose=0)
print("loss tập test = ", loss, "| accuracy tập test = ", acc)

data = [["index", "label"]]


def append_dataPT(x, y):
    data.append([x, y])


# lấy 1 hình ảnh bất kỳ ở tập test và dự đoán
for i in range(len(X_test)):
    print(f" \n {i} ")
    input_image = X_test_reshaped[i]
    # plt.imshow(input_image, cmap=plt.get_cmap("gray"))
    # print("shape của 1 bức ảnh", input_image.shape)
    input_image = np.expand_dims(input_image, axis=0)
    # print("shape phù hợp với mô hình là 3 chiều", input_image.shape)
    output = model.predict(input_image)
    # print("số dự đoán là :", output.argmax())
    # plt.show()
    append_dataPT(i, output.argmax())

with open("123TGMT2002_9H53_6.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)
