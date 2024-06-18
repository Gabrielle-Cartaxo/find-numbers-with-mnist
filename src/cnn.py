import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# Carregar o dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizar os dados
X_train = X_train / 255.0
X_test = X_test / 255.0

# Adicionar uma dimensão para os canais de cor (CNN espera 3D)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Converter os labels para o formato one-hot
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Definir o modelo convolucional
model_cnn = Sequential([
    Input(shape=(28, 28, 1)),  # Input é uma imagem 28x28 com 1 canal
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilar o modelo
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model_cnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)

# Salvar o modelo
model_cnn.save('modelos/cnn.h5')
