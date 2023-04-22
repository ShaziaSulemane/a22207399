import keras
import image_network_library
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

d = image_network_library.read_json(
    path="/home/shazia/PycharmProjects/a22207399/dataset/via_project_6Nov2022_10h31m_json.json")
print(d)

tensors = image_network_library.make_tensors(d)
print(tensors)

print("-------------------------------------------------------------------")

x_train, y_train = image_network_library.preprocessing.create_dataset(
    images_folder="/home/shazia/PycharmProjects/a22207399/dataset",
    y_train_dict=tensors)
print(len(x_train))
print(len(y_train))

print("-------------------------------------------------------------------")

print(x_train[0])
print(y_train[0])

print("--------------------------------------------------------------------")
model = image_network_library.create_model(model_url="https://tfhub.dev/google/efficientnet/b7/feature-vector/1",
                                           target_size=(256, 256), num_coordinates=3)
print(model.layers)

print("--------------------------------------------------------------------")
model.compile(optimizer="adam",
              loss="mse",
              metrics=["acc"])
y_train = tf.reshape(y_train, (y_train.shape[0], -1))
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_train, y_train))

print("--------------------------------------------------------------------")
model1 = image_network_library.ViT(
    image_size=256,
    patch_size=8,
    num_coordinates=3,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

model1.build(input_shape=x_train.shape)

model1.compile(optimizer="adam",
               loss="mse",
               metrics=["acc"])

print(x_train.shape)
print(y_train.shape)

print("----------------------------------------------------------------")
print(model1.layers)
print(model1.summary())

history1 = model.fit(x_train, y_train, epochs=5, validation_data=(x_train, y_train))

img = tf.random.normal(shape=[1, 256, 256, 3])
preds = model1(img)  # (1, 1000)
print(preds)

print("-----------------------------------------------------------------")
