#%%

import matplotlib.pyplot as plt
import numpy as np

def model_drawing(ax, matrix_data, prob):
    matrix = matrix_data.copy()
    matrix = matrix.T

    for ii in range(5):
        matrix[ii, ii] = 0

    # Define the positions of the states in the plot
    positions = [(0, 1.5), (-1.5, 0), (1.5, 0), (-1.5, -1.5), (1.5, -1.5)]

    arrow_count = 1

    # Iterate through the matrix and draw the arrows
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] != 0:
                arrow_width = 0.02  # Adjust arrow width based on preference
                arrow_head_width = 0.2  # Adjust arrowhead width based on preference
                arrow_head_length = 0.2  # Adjust arrowhead length based on preference
                radius = 0.2  # Radius of the circle
                
                # Calculate the intersection point of the arrow with the circle
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                dx, dy = x2 - x1, y2 - y1
                d = np.sqrt(dx**2 + dy**2)
                arrow_length = d - radius - arrow_head_length / 2  # Adjusted arrow length
                
                # Draw the arrow and arrowhead up to the circle's edge
                ax.arrow(
                    x1,
                    y1,
                    arrow_length * dx / d,
                    arrow_length * dy / d,
                    width=arrow_width,
                    fc='black', ec='black',  # Set face color and edge color
                    head_width=arrow_head_width,
                    head_length=arrow_head_length
                    )

                # Define the fraction to move the circle closer to the end of the arrow
                circle_fraction = 0.65  # Adjust as needed

                # Calculate the new circle position
                mid_x = x1 + circle_fraction * (x2 - x1)
                mid_y = y1 + circle_fraction * (y2 - y1)

                ax.add_patch(plt.Circle((mid_x, mid_y), radius=0.15, color='white', ec='black'))
                ax.text(mid_x, mid_y, str(arrow_count), color='black',
                        ha='center', va='center', fontsize=6)

                arrow_count += 1

    # Add the states as circles
    for pos in positions:
        circle = plt.Circle(pos, radius=0.2, color='black', linewidth=2, fill=None)
        ax.add_patch(circle)

    # Add the state names
    state_names = ["A", "B", "C", "D", "GS"]
    for i in range(len(positions)):
        if i <= 2:
            ax.text(
                positions[i][0],
                positions[i][1] + 0.5,
                state_names[i],
                ha="center",
                va="center",
                size=12
            )
        else:
            ax.text(
                positions[i][0],
                positions[i][1] - 0.5,
                state_names[i],
                ha="center",
                va="center",
                size=12
            )
    # Set the limits of the plot
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.1)
    ax.axis("off")
    title = ("Kinetic Model, \nprob. = {}".format(np.trunc(prob*10**2)/(10**2)))
    ax.set_title(title, fontsize=10)