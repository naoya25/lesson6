import os
from gc import callbacks
import numpy as np
import tensorflow
from PIL import Image
from tensorflow.keras import Model, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

ResNet_model = ResNet50(
    weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3))
)

# 既存のResNetモデルの出力部分をinputという変数に代入
inputs = ResNet_model.output

x = Flatten()(inputs)

x = Dense(2048, kernel_regularizer=tensorflow.keras.regularizers.l2(0.001), activation="relu")(x)

x = Dropout(0.25)(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.25)(x)

prediction = Dense(8, activation="softmax")(x)

model = Model(inputs=ResNet_model.input, outputs=prediction)

# 変更箇所
lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.00005, decay_steps=10000, decay_rate=1e-6
)
optimizer = tensorflow.keras.optimizers.RMSprop(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
# ここまで

gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=20,
    horizontal_flip=True,
    fill_mode="constant",
    cval=0,
    validation_split=0.2,
)

train_gen = gen.flow_from_directory(
    "./train", target_size=(224, 224), class_mode="categorical", shuffle=True, subset="training"
)

val_gen = gen.flow_from_directory(
    "./train", target_size=(224, 224), class_mode="categorical", shuffle=True, subset="validation"
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.0,
    patience=3,
)

model.fit_generator(
    train_gen,
    steps_per_epoch=35,
    epochs=20,
    verbose=1,
    validation_data=val_gen,
    callbacks=[early_stopping],
)

model.save("ftmodel.h5")
