# To generate spectra for further training
# Corrected for sine spacing
# Normalized after adding chaotic modes too

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import random
import math

# Generally we have 1 to 10 visible modes
n_vals = np.arange(1, 11)
# Our spectrum will consists of frequency from 1 to 100
freq_samples = 1000
freq = np.linspace(1, 100, freq_samples)

names = ['Linear', 'Quadratic', 'Square root', 'Hyperbolic', 'Rational 1', 'Rational 2']

######################################################
#                                                    #
#           Defining the required functions          #
#                                                    #
######################################################

def parameteres(spacing_name):
    
    # Min and Max values for parameters
    if spacing_name == 'Linear':
        N_samples = 17000 # No. of tuples of parameters you want (Note : Parameters for quadratic are a, b, c)
        parameter_min = [1e-1, 1e-3, 1e-1]
        parameter_max = [1e+1, 1e+1, 1e+1]
    if spacing_name == 'Quadratic':
        N_samples = 120000
        parameter_min = [1e-1, 1e-2, 1e-3]
        parameter_max = [1e+1, 1e+1, 1e+1]
    if spacing_name == 'Square root':
        N_samples = 27000
        parameter_min = [1e-2, 1e-3, 1e-2]
        parameter_max = [1e+3, 1e+1, 1e+2]
    if spacing_name == 'Inverse':
        N_samples = 13500
        parameter_min = [1e-1, 1e-1, 1e-1]
        parameter_max = [1e+3, 1e+1, 1e+1]
    if spacing_name == 'Sinusoidal':
        N_samples = 20000000
        parameter_min = [1e-2, 1e-10, 1e-1]
        parameter_max = [1e+3, 2*np.pi, 1e+3]   
    if spacing_name == 'Hyperbolic':
        N_samples = 5500000
        parameter_min = [1e-2, 1e-2, 1e-2]
        parameter_max = [1e+1, 1e+1, 1e+1]
    if spacing_name == 'Rational 1':
        N_samples = 90000
        parameter_min = [1e-1, 1e-3, 1e-1]
        parameter_max = [1e+1, 1e+1, 1e+3]
    if spacing_name == 'Rational 2':
        N_samples = 14000
        parameter_min = [1e-2, 1e-1, 1e-1]
        parameter_max = [1e+2, 1e+1, 1e+1]
    # This will have rows = N_samples and columns = 3(for a, b, c). 
    # So, this will contain all a values in 1st column, b in 2nd etc.
    parameter_vals = np.random.uniform(parameter_min, parameter_max, [N_samples,3])
    # Lists having values of a, b and c
    a_vals, b_vals, c_vals = parameter_vals.T
    return (a_vals, b_vals, c_vals)

def function(a, b, c, spacing_name, n_vals):
    if spacing_name == 'Linear':
        #if a >= b/2 :
        # Linear
        return (a * n_vals + b)
    if spacing_name == 'Quadratic':
        # Quadratic
        #if a >= b/5 :
        return (a * n_vals**2 + b * n_vals + c)
    if spacing_name == 'Square root':
        # Square root
        return (np.sqrt((a * n_vals + b)))
    '''
    if spacing_name == 'Inverse':
        # Inverse function
        return (a * (b * n_vals + c)**(-1))
    '''
    if spacing_name == 'Sinusoidal':
        # Sinusoidal function
        return (a * np.sin(b * n_vals + c))  
    if spacing_name == 'Hyperbolic':
        # Hyperbolic function
        return (np.sinh(b * n_vals + c))
    if spacing_name == 'Rational 1':
        # Rational 1 function
        return ((a*n_vals + b)**(3/2))
    if spacing_name == 'Rational 2':
        # Rational 2 function
        return (np.sqrt(a*(n_vals**2) + b**2)) 

# Function to add sinc functions on modal frequencies
def insert_sinc(freq_mode, amp_mode, freq, resolution):
    # Amplitude should be zero on non modal frequencies
    amp = np.zeros_like(freq)
    # Inserting sinc functions on all modal frequencies
    for (i,j) in zip(freq_mode, amp_mode):
        if i != 0:
            arg = (freq-i)/resolution
            amp += j * np.sinc(arg).__abs__()
    return amp

# Function to add noise to the spectrum
def insert_noise(without_noise, freq):
    noise = np.random.normal(0,1,len(freq)).__abs__()
    noise /= max(noise) # Normalized the noise so that its maximum = 1.
    noise_factor = np.random.uniform(1,5,1)
    noise *= noise_factor/20 * max(without_noise)
    with_noise = without_noise + noise
    # Normalizing
    with_noise /= max(with_noise)
    return (with_noise)

# With threshold
'''
def to_add_chaotic_modes(freq_samples, freq, mode_freq, amp_modes):
    # Adding some chaotic modes
    chaotic_modes_numbers = np.random.choice(np.arange(50, 100))
    # Selecting random indices in freq domain
    random_indices = np.random.choice(freq_samples, size=chaotic_modes_numbers, replace=False)
    # Get the selected values
    chaotic_modes = freq[random_indices]
    chaotic_modes = chaotic_modes[ (chaotic_modes>=5) & (chaotic_modes<=80) ]
    # Find the common elements
    common_elements = np.intersect1d(chaotic_modes, mode_freq)
    # Filter out the common elements
    if len(common_elements)>0:
        chaotic_modes_new = chaotic_modes[~np.isin(chaotic_modes, common_elements)]
        chaotic_modes = chaotic_modes_new
    else :
        chaotic_modes_new = chaotic_modes
    for index_chaotic in range(len(chaotic_modes_new)):
        # Find the index of the closest element
        closest_index = np.abs(mode_freq - chaotic_modes_new[index_chaotic]).argmin()
        min_diff_chaotic = np.abs(mode_freq[closest_index] - chaotic_modes_new[index_chaotic])
        if min_diff_chaotic < 5:
            index_to_delete = np.where(chaotic_modes == chaotic_modes_new[index_chaotic])[0]
            chaotic_modes = np.delete(chaotic_modes, index_to_delete)
        if len(chaotic_modes) > 0: 
            # Adding sinc functions on chaotic modes
            power_amp = np.random.uniform(-1, 0.6*math.log10(np.max(amp_modes)), len(chaotic_modes))
            # Adding random resolution for chaotic modes
            res = np.random.uniform(5, 10)
            amp_modes = amp_modes + insert_sinc(chaotic_modes, 10**power_amp, freq, 1/res)
        else:
            chaotic_modes = [0]
    return chaotic_modes
'''

# Without threshold
def to_add_chaotic_modes(freq_samples, freq, mode_freq, amp_modes):
    # Adding some chaotic modes
    chaotic_modes_numbers = np.random.choice(np.arange(50, 100))
    # Selecting random indices in freq domain
    random_indices = np.random.choice(freq_samples, size=chaotic_modes_numbers, replace=False)
    # Get the selected values
    chaotic_modes = freq[random_indices]
    chaotic_modes = chaotic_modes[ (chaotic_modes>=5) & (chaotic_modes<=80) ]
    # Find the common elements
    common_elements = np.intersect1d(chaotic_modes, mode_freq)
    # Filter out the common elements
    if len(common_elements)>0:
        chaotic_modes = chaotic_modes[~np.isin(chaotic_modes, common_elements)]
    if len(chaotic_modes) > 0: 
        # Adding sinc functions on chaotic modes
        power_amp = np.random.uniform(-4, 0.1*math.log10(np.max(amp_modes)), len(chaotic_modes))
        # Adding random resolution for chaotic modes
        res = np.random.uniform(20, 25)
        amp_modes = amp_modes + insert_sinc(chaotic_modes, 10**power_amp, freq, 1/res)
    else:
        chaotic_modes = [0]
    return chaotic_modes, amp_modes

######################################################
#                                                    #
#        Algorithm to generate synthetic data        #
#                                                    #
######################################################

def Algo(spacing_name):
    a_vals, b_vals, c_vals = parameteres(spacing_name)
    # Some required empty lists
    mode_amp = []
    spacing_type = []
    no_of_modes = []
    no_of_chaotic_modes = []
    mode_freq_to_draw = []
    chaotic_mode_freq_to_draw = []
    a_opt, b_opt, c_opt = [], [], []
    for a, b, c in zip(a_vals, b_vals, c_vals):
        # Calling the function we want the spacing to follow
        mode_freq = function(a, b, c, spacing_name, n_vals)
        if mode_freq is not None:
            mode_freq = mode_freq[ (mode_freq>=5) & (mode_freq<=80) ]
            # Frequencies of modes should be between 5 and 80 and also we should have at least 6 modes
            if len(mode_freq) > 5:
                min_diff = min(np.diff(mode_freq))
                if min_diff > 4:
                    # We want the amplitude's exponent distribution in log scale
                    power_amp = np.random.uniform(-1, 0,len(mode_freq))
                    # Inserting the sinc function on modes so that it looks like a real spectrum.
                    # Since most Delta-Scuti stars are observed by TESS for ~30 days, resolution = 1/30 d^{-1} but selecting 1/5 will give us better resolution.
                    amp_modes_without_noise = insert_sinc(mode_freq, 10**power_amp, freq, 1/25)
                    # Adding noise to the spectrum
                    amp_modes = insert_noise(amp_modes_without_noise, freq)
                    # Adding chaotic modes
                    chaotic_modes, amp_modes = to_add_chaotic_modes(freq_samples, freq, mode_freq, amp_modes)
                    # Normalizing
                    amp_max = np.max(amp_modes)
                    amp_modes = amp_modes/amp_max

                    # To remove the chaotic modes comment above and uncomment below
                    #chaotic_modes = [0]
                    # Appending these amplitude modes to the lists we initialized
                    mode_amp.append(amp_modes)
                    # Appending no. of modes
                    no_of_modes.append(len(mode_freq))
                    no_of_chaotic_modes.append(len(chaotic_modes))

                    # Appending the freq at modes
                    mode_freq_to_draw.append(mode_freq)
                    chaotic_mode_freq_to_draw.append(chaotic_modes)
                    # Appending the optimal values of parameters
                    a_opt.append(a) 
                    b_opt.append(b) 
                    if spacing_name in ['Linear', 'Square root', 'Rational 1', 'Rational 2']: # Value of c doesn't make sense for these spacings
                        c_opt.append(0)
                    else:
                        c_opt.append(c)


    # Converting all the lists to arrays
    mode_amp = np.array(mode_amp)
    no_of_modes = np.array(no_of_modes)
    no_of_chaotic_modes = np.array(no_of_chaotic_modes)
    a_opt = np.array(a_opt)
    b_opt = np.array(b_opt)
    c_opt = np.array(c_opt)
    print('We have', len(mode_amp), 'spectra for', spacing_name, 'spacing')
    return mode_freq_to_draw, chaotic_mode_freq_to_draw, mode_amp, no_of_modes, no_of_chaotic_modes, a_opt, b_opt, c_opt

# Spacing names
#names = ['Linear', 'Quadratic', 'Square root', 'Inverse', 'Sinusoidal']
#names = ['Quadratic'] 
#names = ['Linear', 'Quadratic', 'Square root', 'Sinusoidal', 'Hyperbolic', 'Rational 1', 'Rational 2']

data = np.zeros((len(names)*freq_samples*10, freq_samples))
labels = np.zeros((len(names)*freq_samples*10, 1))
s = 0
label = 0

'''
for i in range(len(names)):
    print(f'Working with {names[i]} spacing')

    # Retreving the values
    mode_freq_to_draw, chaotic_mode_freq_to_draw, mode_amp, no_of_modes, no_of_chaotic_modes, a_opt, b_opt, c_opt = Algo(names[i])

    # Saving the data in matrix format
    for j in range(10000): # Took the range 10,000 instead of len(mode_amp) cuz we need 10k spectra
        data[s] = mode_amp[j]
        labels[s] = label
        s += 1

    label += 1
'''
    # Uncomment the below line if you want to save the pdf file with plots for every spacing
'''
    # Create a PDF file to save the plots
    #pdf_filename = f"plots_{names[i]}.pdf"
    #pdf_pages = PdfPages(pdf_filename)
    chaotic_mode_freq_to_draw_without_zero = []
    #for m in range(10000):  # To print 1000 pages of every spacing
    for m in range(5):
        #j = m # Uncomment if you want the plots from 0 to 10000
        j = random.randint(0, 9999) # Uncomment if you want plots randomly

        plt.figure(figsize=[14.9, 4])
        plt.plot(freq, mode_amp[j])  # Plot a single spectrum
        plt.title(f"{names[i]} : No. of true and chaotic modes = {no_of_modes[j]} and {no_of_chaotic_modes[j]}, a = {a_opt[j]}, b = {b_opt[j]}, c = {c_opt[j]}")
        for point in mode_freq_to_draw[j]:
            plt.axvline(x=point, color='r', linestyle='--')

        #if not np.any(chaotic_mode_freq_to_draw[j] == 0):
        for point in chaotic_mode_freq_to_draw[j]:
            if point != 0:
                plt.axvline(x=point, color='m', linestyle='--')       
        plt.tight_layout()
        # Save the current plot to the PDF file
        #pdf_pages.savefig()
        print(f'{names[i]} :', j, f'/ {len(mode_amp)} th plot saved in pdf')
        plt.close()  # Close the figure to free up memory
    # Close the PDF file
    #pdf_pages.close()
    print("Plots saved to", pdf_filename)
pdf_pages.close()
'''


pdf_filename = f"plots.pdf"
pdf_pages = PdfPages(pdf_filename)

for i in range(len(names)):
    print(f'Working with {names[i]} spacing')

    # Retreving the values
    mode_freq_to_draw, chaotic_mode_freq_to_draw, mode_amp, no_of_modes, no_of_chaotic_modes, a_opt, b_opt, c_opt = Algo(names[i])

    # Saving the data in matrix format
    for j in range(10000): # Took the range 10,000 instead of len(mode_amp) cuz we need 10k spectra
        data[s] = mode_amp[j]
        labels[s] = label
        s += 1

    label += 1

    # Uncomment the below line if you want to save the pdf file with plots for every spacing

    # Create a PDF file to save the plots
    #pdf_filename = f"plots_{names[i]}.pdf"
    #pdf_pages = PdfPages(pdf_filename)
    chaotic_mode_freq_to_draw_without_zero = []
    #for m in range(10000):  # To print 1000 pages of every spacing
    for m in range(5):
        #j = m # Uncomment if you want the plots from 0 to 10000
        j = random.randint(0, 9999) # Uncomment if you want plots randomly

        plt.figure(figsize=[14.9, 4])
        plt.plot(freq, mode_amp[j], lw = 2, zorder = 2)  # Plot a single spectrum
        #plt.title(f"{names[i]} : No. of true and chaotic modes = {no_of_modes[j]} and {no_of_chaotic_modes[j]}, a = {a_opt[j]}, b = {b_opt[j]}, c = {c_opt[j]}")
        plt.title('Synthetic data')
        sum = 0
        for point in mode_freq_to_draw[j]:
            if sum == 0:
                plt.axvline(x=point, color='r', linestyle='--', zorder = 1, label = 'Non-Chaotic modes')
            else:
                plt.axvline(x=point, color='r', linestyle='--', zorder = 1)
            sum = sum +1
        plt.legend()
        '''
        #if not np.any(chaotic_mode_freq_to_draw[j] == 0):
        for point in chaotic_mode_freq_to_draw[j]:
            if point != 0:
                plt.axvline(x=point, color='m', linestyle='--')    
        '''   
        plt.ylabel('Amplitude(mmag)')
        plt.xlabel('frequency(1/d)')
        plt.tight_layout()
        # Save the current plot to the PDF file
        pdf_pages.savefig()
        print(f'{names[i]} :', j, f'/ {len(mode_amp)} th plot saved in pdf')
        plt.close()  # Close the figure to free up memory

    # Close the PDF file
    #pdf_pages.close()

    print("Plots saved to", pdf_filename)
    # Open a text file in write mode
    with open(f"optimal_params_{names[i]}.txt", "w") as file:
        for ind in range(10000):
            file.write(f"{a_opt[ind]} {b_opt[ind]} {c_opt[ind]}\n")
    print(f"optimal_params_{names[i]}.txt saved now")


pdf_pages.close()

# Concatenate data and labels along axis 1 to form a single array
combined_array = np.column_stack((data, labels.reshape(-1, 1)))
#print(combined_array[50000])

# Save the combined array as a CSV file
file_path = 'data_and_labels.csv'
if os.path.exists(file_path):
    # If the file exists, delete it
    print('CSV File already existed, so just deleted it')
    os.remove(file_path)
np.savetxt(file_path, combined_array, delimiter=',', fmt='%f')
print('csv file with the data and labels saved succesfully:)')

