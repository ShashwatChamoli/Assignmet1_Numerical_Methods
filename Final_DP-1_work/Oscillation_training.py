import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.layers import Dense
from keras.optimizers.legacy import Adam as LegacyAdam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.callbacks import LearningRateScheduler

######################################
######     Data preparation     ######
######################################

loaded_array = np.loadtxt('data_and_labels.csv', delimiter=',')

no_of_classes = 6

# Separating the loaded array back into data and labels
x = loaded_array[:, :-1] # Features
y = loaded_array[:, -1]  # Labels

# Splitting the data into training, testing, and validation sets with shuffling
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, shuffle=True, random_state=42) 

# Convert labels to one-hot encoding
y_train_encoded = to_categorical(y_train, num_classes=no_of_classes )
y_test_encoded = to_categorical(y_test, num_classes=no_of_classes )
y_val_encoded = to_categorical(y_val, num_classes=no_of_classes )


######################################
######     Machine learning     ######
######################################

# Building the model
model = Sequential([
 # Adding 1D convolutional layers
    Conv1D(filters=8, kernel_size=45, activation='relu', input_shape=(x_train.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.2),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=16, kernel_size=45, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=45, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=45, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    MaxPooling1D(pool_size=2),


    Flatten(),  # Flatten the output from convolutional layers for Dense layers

    Dense(units=128, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=no_of_classes , activation='softmax')  # Using softmax activation for classification
])

# Compiling the model
learning_rate = 0.001
model.compile(optimizer=LegacyAdam(learning_rate = learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])  # Using categorical crossentropy for classification
model.summary()
'''
# Fitting the model
batch_size=2**8
# Define a learning rate scheduler function
factor = 1/10
epoch_change = 4
def lr_scheduler(epoch, learning_rate):
    if epoch % epoch_change == 0 and epoch != 0:
        learning_rate = learning_rate * factor
        return learning_rate # Reduce the learning rate every n no. of epochs
    return learning_rate

# Create a learning rate scheduler callback
#lr_scheduler_callback = LearningRateScheduler(lr_scheduler)
lr_scheduler_callback = [LearningRateScheduler(lr_scheduler), 
                         tf.keras.callbacks.ModelCheckpoint(filepath = 'spectra_generated',
                                                             monitor = 'val_accuracy',
                                                               mode='auto',
                                                                 save_best_only=True, verbose=1,)]

# Fitting the model with the learning rate scheduler callback
history = model.fit(x_train, y_train_encoded, 
                    validation_data=(x_val, y_val_encoded), 
                    epochs=24, 
                    batch_size=batch_size, 
                    verbose=1, 
                    callbacks=[lr_scheduler_callback])

print('Model saved successfully')

#######################################
######          Results          ######
#######################################

# Prediciting
predictions = model.predict(x_test, batch_size = 100, verbose = 0)
predicted_labels = np.argmax(predictions, axis = -1) # Finding best predicted labels

# Classification report
report = classification_report(y_test, predicted_labels)
print(report)
with open('classification_report.txt', 'w') as file:
    file.write(report)

# Calculate and plot confusion matrix
cm = confusion_matrix(y_test, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
#plt.title(f'Confusion Matrix (batch size = {batch_size}, initial learning rate = {learning_rate} multiplied by {factor} after every {epoch_change} epoch)')
plt.title('Confusion Matrix')
plt.savefig('Confusion_Matrix.pdf')
plt.show()

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test_encoded)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
with open('test_results.txt', 'w') as file:
    file.write(f"Loss(on test data): {test_loss}\n")
    file.write(f"Accuracy(on test data): {test_accuracy}\n")

print("Test results saved to 'test_results.txt'")

fontsize = 14

# Plotting the training and validation loss
plt.plot(history.history['loss'], label='Training Loss') 
plt.plot(history.history['val_loss'], label='Validation Loss') 
plt.yscale('log') 
plt.xlabel('Epoch', fontsize = fontsize)
plt.ylabel('Loss', fontsize = fontsize)
#plt.title(f'Training and Validation Loss {batch_size}, {learning_rate} multiplied by {factor} every {epoch_change} epoch') 
plt.title('Training and Validation Loss', fontsize = fontsize)
plt.legend(fontsize = fontsize) 
plt.xticks(fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.grid(True) 
plt.savefig('Training_and_Validation_Loss.png')
plt.show()

# Plotting the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.xlabel('Epoch', fontsize = fontsize) 
plt.ylabel('Accuracy', fontsize = fontsize) 
#plt.title(f'Training and Validation Accuracy {batch_size}, {learning_rate} multiplied by {factor} every {epoch_change} epoch')
plt.title('Training metrics for the first model', fontsize = fontsize)
plt.legend(fontsize = fontsize)  
plt.xticks(fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.grid(True) 
plt.savefig('Training_and_Validation_Accuracy.png') 
plt.show()
'''