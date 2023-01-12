import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_history(path):
    history = pd.read_csv(path)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.legend()
    plt.title('Accuracy')
    plt.show()
