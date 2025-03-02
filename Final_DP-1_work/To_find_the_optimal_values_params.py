import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Initialize empty lists for storing values
a_opt = []
b_opt = []
c_opt = []

spacing_names = ['Linear', 'Quadratic', 'Square root', 'Hyperbolic', 'Rational 1', 'Rational 2']

for names in spacing_names:
    # Open the text file in read mode
    with open(f"optimal_params_{names}.txt", "r") as file:
        # Iterate over each line in the file
        for line in file:
            # Split the line into values using whitespace as the delimiter
            values = line.split()

            # Extract values for a_opt, b_opt, and c_opt and convert them to floats
            a_opt.append(float(values[0]))
            b_opt.append(float(values[1]))
            c_opt.append(float(values[2]))

    # Create a PDF file to save the plots
    pdf_filename = f"{names}_params_graphs.pdf"
    with PdfPages(pdf_filename) as pdf:
        # Plot histograms
        plt.figure()
        plt.hist(a_opt)
        plt.title('a_optimal values')
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.hist(b_opt)
        plt.title('b_optimal values')
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.hist(c_opt)
        plt.title('c_optimal values')
        pdf.savefig()
        plt.close()

        # Plot scatter plots
        plt.figure()
        plt.scatter(a_opt, b_opt, s=10)
        plt.xlabel('a_opt')
        plt.ylabel('b_opt')
        plt.title('b_opt vs a_opt')
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.scatter(c_opt, b_opt, s=10)
        plt.xlabel('c_opt')
        plt.ylabel('b_opt')
        plt.title('b_opt vs c_opt')
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.scatter(a_opt, c_opt, s=10)
        plt.xlabel('a_opt')
        plt.ylabel('c_opt')
        plt.title('c_opt vs a_opt')
        pdf.savefig()
        plt.close()

    print(f"All plots saved to {pdf_filename}")

