import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3  # Importar InceptionV3
import pandas as pd


from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout  # Adicionando Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU

import tensorflow as tf



# Diretórios
train_dir = 'C:\\Users\\Julio\\OneDrive\\Área de Trabalho\\Faculdade\\Projetos Integradores\\image captcha\\training'
validation_dir = 'C:\\Users\\Julio\\OneDrive\\Área de Trabalho\\Faculdade\\Projetos Integradores\\image captcha\\validation'
test_dir = 'C:\\Users\\Julio\\OneDrive\\Área de Trabalho\\Faculdade\\Projetos Integradores\\image captcha\\test'


# Atualizar o diretório para cada classe
train_semafaro_dir = os.path.join(train_dir, 'traffic')
train_faixa_dir = os.path.join(train_dir, 'crosswalk')
train_onibus_dir = os.path.join(train_dir, 'onibus')
train_hidrante_dir = os.path.join(train_dir, 'hidrante')
train_bicicleta_dir = os.path.join(train_dir,'bicicleta')


validation_semafaro_dir = os.path.join(validation_dir, 'traffic')
validation_faixa_dir = os.path.join(validation_dir, 'crosswalk')
validation_onibus_dir = os.path.join(validation_dir, 'onibus')
validation_hidrante_dir = os.path.join(validation_dir, 'hidrante')
validation_bicicleta_dir = os.path.join(validation_dir,'bicicleta')


test_semafaro_dir = os.path.join(test_dir, 'traffic')
test_faixa_dir = os.path.join(test_dir, 'crosswalk')
test_onibus_dir = os.path.join(test_dir, 'onibus')
test_hidrante_dir = os.path.join(test_dir, 'hidrante')
test_bicicleta_dir = os.path.join(test_dir,'bicicleta')


# Número de classes
num_classes = 5

# Dimensões das imagens
img_width, img_height = 224, 224

# Geradores de dados
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Atualizar os diretórios para cada classe
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=12,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=12,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=12,
    class_mode='categorical',
    shuffle=False
)


# Carregar a VGG16 com pesos pré-treinados (excluindo as camadas fully-connected)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Adicionar camadas personalizadas para classificação
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.1)(x)  # Adicione dropout se desejar
x = Dense(512, activation='relu')(x)
x = Dropout(0.1)(x)  # Adicione dropout se desejar

predictions = Dense(num_classes, activation='softmax')(x)

# Montar o modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar os pesos da VGG16 para a etapa de treinamento inicial
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])


# Exibir a arquitetura do modelo
model.summary()
# Treinar o modelo
history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=15,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)
    )

model.save('model_inception.keras')
print("Modelo salvo com sucesso.")

# Extrair informações do histórico
train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

# Avaliar o desempenho no conjunto de teste

# Carregar o modelo treinado
loaded_model = load_model('model_vgg16.keras')
print("Modelo carregado com sucesso.")
# Prever as classes para as imagens de teste
predictions = loaded_model.predict(test_generator, steps=len(test_generator))
predicted_classes = np.argmax(predictions, axis=1)


# Obter as classes verdadeiras
true_classes = test_generator.classes

# Calcular o F1 Score
f1 = f1_score(true_classes, predicted_classes, average='weighted')
print("F1 Score:", f1)

# Calcular a matriz de confusão manualmente
confusion_mtx = np.zeros((num_classes, num_classes), dtype=int)  # Alteração aqui

class_accuracy = {}
for i in range(num_classes):
    total_samples = np.sum(true_classes == i)
    correct_predictions = np.sum((true_classes == i) & (predicted_classes == i))
    class_accuracy[i] = correct_predictions / total_samples if total_samples > 0 else 0

print("Precisão de cada classe:")
for i in range(num_classes):
    print(f"Classe {i}: {class_accuracy[i]:.2%}")

# Calcular a precisão global
global_accuracy = accuracy_score(true_classes, predicted_classes)
print("Precisão global:", global_accuracy)

# Restante do código...
for i in range(len(predicted_classes)):
    confusion_mtx[true_classes[i], predicted_classes[i]] += 1

# Definir os nomes das classes
class_names = ['semafaro', 'faixa', 'onibus', 'hidrante', 'bicicleta']

# Criar um DataFrame a partir da matriz de confusão
confusion_df = pd.DataFrame(confusion_mtx, index=class_names, columns=class_names)

# Plotar a matriz de confusão usando seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Verdadeira')
plt.title('Matriz de Confusão')
plt.show()

test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# Plotar as curvas de loss e accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.tight_layout()
plt.show()

