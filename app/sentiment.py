import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

texts = [
    "Me encanta este producto","Excelente servicio","Muy buena experiencia",
    "Todo perfecto","Gran trabajo","Me gustó mucho","Fantástico resultado",
    "Muy recomendable","Estoy satisfecho","Buenísimo",

    "Es normal","Está bien supongo","Nada especial","Aceptable","Más o menos",
    "No está mal","Regular servicio","Puede mejorar","Está decente","No es la gran cosa",

    "Esto es horrible","Eres un idiota","Pésimo servicio","No me gustó para nada",
    "Terrible experiencia","Asco total","Qué basura","Muy malo","Es un desastre","Odio esto"
]

labels_text = ["normal"]*10 + ["regular"]*10 + ["toxico"]*10

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels_text)
labels = np.array(label_tokenizer.texts_to_sequences(labels_text)) - 1
labels = labels.flatten()

tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=12, padding='post')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=2000, output_dim=32),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 🔥 Entrenar UNA sola vez cuando el contenedor inicia
model.fit(padded, labels, epochs=40, verbose=0)

label_map = {0: "normal", 1: "regular", 2: "toxico"}


def predict_batch(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    pads = pad_sequences(sequences, maxlen=12, padding='post')
    predictions = model.predict(pads, verbose=0)
    return [label_map[np.argmax(pred)] for pred in predictions]