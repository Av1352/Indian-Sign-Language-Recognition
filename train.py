from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

train_dir = 'output/train'
val_dir = 'output/test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

sign_model = Sequential()

sign_model.add(Conv2D(32, kernel_size=(
    3, 3), activation='relu', input_shape=(48, 48, 1)))
sign_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
sign_model.add(MaxPooling2D(pool_size=(2, 2)))
sign_model.add(Dropout(0.25))

sign_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
sign_model.add(MaxPooling2D(pool_size=(2, 2)))
sign_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
sign_model.add(MaxPooling2D(pool_size=(2, 2)))
sign_model.add(Dropout(0.25))

sign_model.add(Flatten())
sign_model.add(Dense(1024, activation='relu'))
sign_model.add(Dropout(0.5))
sign_model.add(Dense(7, activation='softmax'))

sign_model.compile(loss='categorical_crossentropy', optimizer=Adam(
    learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])

sign_model_info = sign_model.fit(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=75,
    validation_data=val_generator,
    validation_steps=7178 // 64
)

sign_model.save_weights('model.h5')

score = sign_model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])