# This code is for predicting using the model. 
# You can see the histogram of probabilities and can plot the interpolated values
# stars provided by subrat = [2398, 5307, 5390, 5474, 6344, 6393, 6404]

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from collections import Counter
from collections import defaultdict


##################################
######   Data Preparation   ######
##################################

spacing_names = ['Linear', 'Quadratic', 'Square root', 'Hyperbolic', 'Rational 1', 'Rational 2']

# Loading the star names
file_path = 'Final_Star_Names.txt'
star_names = np.loadtxt(file_path, dtype=int)
print('We have ', len(star_names), ' stars')

data = []

# Open the text file and read the values
with open('interpolated_values.txt', 'r') as file:
    # Read each line from the file
    for line in file:
        # Spliting using whitespace and conerting the values to float
        row_values = [float(value) for value in line.split()]
        data.append(row_values)

amp = np.array(data)
# Normalizing the interpolated data
for i in range(len(amp)):
    max = np.max(amp[i])
    amp[i] = amp[i]/max
    print(np.max(amp[i]))


'''
# Modifying the data to remove very small modes
for i in range(len(amp)):
    max_value = np.max(amp[i])
    for j in range(len(amp[i])):
        if amp[i][j] < 0.1 * max_value:
            amp[i][j] = 0


# Plotting the modified data for the star

index = 257
freq = np.linspace(1, 100, 1000)
plt.plot(freq, amp[index])
plt.title(f'Modified data for star: TIC {star_names[index]}')
plt.show()
'''

###################################
######      Predictions      ######
###################################

no_of_classes = 6

# Load the model
model = tf.keras.models.load_model('spectra_generated')

predictions = model.predict(amp, batch_size = 100, verbose = 0)
print(predictions)
'''
predicted_labels = []
for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        if predictions[i][j] > 0.7:
            predicted_labels.append(j)
        elif j==(len(predictions[i]) - 1):
            predicted_labels.append(2000)

# Counting the occurrences of each number in the array
counts = Counter(predicted_labels)

# Print the counts for each label
for i in range(8):
    print(f"The label {i} occurs {counts[i]} times.")
'''


predicted_labels = np.argmax(predictions, axis = -1) # Finding best predicted labels
# Counting the occurrences of each number in the array
counts = Counter(predicted_labels)

# Create a dictionary to store indices for each label
indices_dict = defaultdict(list)

# Iterate over the predicted labels and store indices
for idx, label in enumerate(predicted_labels):
    indices_dict[label].append(idx)

# Write indices to separate text files for each label
for label, indices in indices_dict.items():
    filename = f"label_{label}_{spacing_names[label]}_star_names.txt"
    with open(filename, "w") as file:
        file.write("\n".join(map(str, star_names[indices])))
    print(f"Indices for label {label} written to {filename}")

# Print the counts for each label
for i in range(no_of_classes):
    print(f"The label {i} occurs {counts[i]} times.")

# Labels (starts from 0)
labels = np.arange(no_of_classes)

#np.savetxt('All_probabilities.txt', predictions)

index = [2398, 5307, 5390, 5474, 6344, 6393, 6404] # Subrat stars
# Plotting the predicted probabilities 

for index in index:
    plt.figure(figsize=(10, 6))
    plt.bar(labels, predictions[index])
    plt.xlabel('Spacing label')
    plt.ylabel(f'Predicted Probability')
    plt.title(f'Star name : TIC {star_names[index]}')
    plt.xticks(labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'TIC{star_names[index]}.png')
    plt.show()
    plt.close()



'''
# Counting the occurrences of each number in the array
counts = Counter(predicted_labels)

# Print the counts for each number
for num in range(9):  # Assuming numbers range from 0 to 8
    print(f"Number {num} occurs {counts[num]} times.")
'''
