from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, TextVectorization
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses

import requests
import json
import numpy as np

target = requests.get("http://localhost:8000/target/")
targetJSON = target.json()

sentences = []
labels = []
names = []
for item in targetJSON:
    if item["predictor"] != "tbc":
        print(item)
        sentences.append(item["title"])
        if item["predictor"] in names:
            pass
        else:
            names.append(item["predictor"])
            
        labels.append(names.index(item["predictor"]))
        
total = len(sentences)
total_80 = round(total * 0.8)
total_20 = total - total_80

sentences_v = sentences[total_80:]
labels_v = labels[total_80:]
sentences = sentences[0:total_80]
labels = labels[0:total_80]

raw_train_ds = tf.data.Dataset.from_tensor_slices((sentences,labels))
raw_val_ds = tf.data.Dataset.from_tensor_slices((sentences_v,labels_v))

raw_train_ds = raw_train_ds.batch(batch_size=2)
raw_val_ds = raw_val_ds.batch(batch_size=2)

raw_train_ds.class_names = names
raw_val_ds.class_names = names

VOCAB_SIZE = 10000

binary_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='binary')

train_text = raw_train_ds.map(lambda text, labels: text)
binary_vectorize_layer.adapt(train_text)

def binary_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return binary_vectorize_layer(text), label

binary_train_ds = raw_train_ds.map(binary_vectorize_text)
binary_val_ds = raw_val_ds.map(binary_vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

def configure_dataset(dataset):
  return dataset.cache().prefetch(buffer_size=AUTOTUNE)

binary_train_ds = configure_dataset(binary_train_ds)
binary_val_ds = configure_dataset(binary_val_ds)

binary_model = tf.keras.Sequential([layers.Dense(len(raw_train_ds.class_names))])

binary_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

history = binary_model.fit(
    binary_train_ds, validation_data=binary_val_ds, epochs=300)

export_model = tf.keras.Sequential(
    [binary_vectorize_layer, binary_model,
     layers.Activation('sigmoid')])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=['accuracy'])

def get_string_labels(predicted_scores_batch):
  predicted_int_labels = tf.math.argmax(predicted_scores_batch, axis=1)
  print(predicted_int_labels)
  print(raw_train_ds.class_names)  
  predicted_labels = tf.gather(raw_train_ds.class_names, predicted_int_labels)
  return predicted_labels

inputs = sentences_v[4:10]
predicted_scores = export_model.predict(inputs)
#print(predicted_scores)
predicted_labels = get_string_labels(predicted_scores)
for input, label in zip(inputs, predicted_labels):
  print("Question: ", input)
  print("Predicted label: ", label.numpy().decode('utf-8'))

  