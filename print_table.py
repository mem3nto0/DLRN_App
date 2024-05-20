#%%
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np


def Table_print(Tau_solutions, Tau_true, kind_top):

    Tau_solutions = np.trunc(Tau_solutions*100)/100
    Tau_true = np.trunc(Tau_true*100)/100

    if kind_top == 1:

        # Sample data for the table
        data = [
            ["true value", 0, 0,0,0,0,0,0],
            ["Solution 1", 0, 0,0,0,0,0,0],
        ]

    if kind_top == 3:
        
        data = [
            ["true value", 0, 0,0,0,0,0,0],
            ["Solution 1", 0, 0,0,0,0,0,0],
            ["Solution 2", 0, 0,0,0,0,0,0],
            ["Solution 3", 0, 0,0,0,0,0,0]
        ]

    data[0][1:len(Tau_true)+1] = Tau_true

    for i in range(Tau_solutions.shape[0]):

        data[i+1][1:len(Tau_solutions[i])+1] = Tau_solutions[i]
    
    # Table headers
    headers = ["Solutions", r'$\tau$1', r'$\tau$2', r'$\tau$3', r'$\tau$4', r'$\tau$5', r'$\tau$6', r'$\tau$7']

    # Generate the table as a string
    table_string = tabulate(data, headers, tablefmt="grid")

    # Create a figure and subplot for the table
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis('off')  # Turn off axis for the subplot

    # Create a table with a more fancy frame
    table = ax.table(cellText=data, colLabels=headers, cellLoc="center", loc="center", colColours=["#c4c4c4"] * 8)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Scale the table for a fancier appearance

    # Set a title for the figure
    fig.suptitle("Time constant values")

    # Show the figure with the fancy table
    plt.show()