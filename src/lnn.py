import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Carregar o dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizar os dados
X_train = X_train / 255.0
X_test = X_test / 255.0

# Remodelar para um vetor linear (flatten)
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

# Converter os labels para o formato one-hot
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Definir o modelo linear
model_linear = Sequential([
    Input(shape=(28 * 28,)),  # Input é uma imagem achatada de 28x28 pixels
    Dense(128, activation='relu'),  # Primeira camada densa com 128 neurônios
    Dense(10, activation='softmax')  # Camada de saída com 10 neurônios para 10 classes
])

# Compilar o modelo
model_linear.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model_linear.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)

# Salvar o modelo
model_linear.save('modelos/linear.h5')
