import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_from_csv(filename, x_label="X-Axis", y_label="Y-Axis", save_path=None, y_decimal_places=1):
    """
    Reads a CSV file and plots the data, then saves the figure.
    
    Parameters:
        filename (str): Path to the CSV file.
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
        save_path (str): Optional path to save the plot image.
        y_decimal_places (int): Number of decimal places for y-axis percentage formatting.
    """
    # Load data
    data = pd.read_csv(filename, header=None, names=['x', 'y'])

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(data['x'], data['y'], marker='o', linestyle='-')
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.grid(True)

    # Format y-axis ticks as percentages with specified decimal places
    y_format = f'{{:.{y_decimal_places}f}}%'  # Creates format string like "{:.1f}%" or "{:.2f}%"
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: y_format.format(y)))

    # Increase tick size
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # Determine save path
    if save_path is None:
        save_path = os.path.splitext(filename)[0] + ".png"  # Save with same name as CSV but .png

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory

    print(f"Plot saved as {save_path}")

def plot_two_curves_from_csv(filename1, filename2, x_label="X-Axis", y_label="Y-Axis", save_path=None, y_decimal_places=1):
    """
    Reads two CSV files and plots both datasets in the same figure with a legend, 
    ensuring the y-axis is displayed as percentages.

    Parameters:
        filename1 (str): Path to the first CSV file.
        filename2 (str): Path to the second CSV file.
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
        save_path (str): Optional path to save the plot image.
        y_decimal_places (int): Number of decimal places for y-axis percentage formatting.
    """
    # Load data
    data1 = pd.read_csv(filename1, header=None, names=['x', 'y'])
    data2 = pd.read_csv(filename2, header=None, names=['x', 'y'])

    # Plot
    plt.figure(figsize=(4, 4))
    plt.plot(data1['x'], data1['y'], marker='o', linestyle='-', label="Small Disruption", linewidth=2)
    plt.plot(data2['x'], data2['y'], marker='s', linestyle='--', label="Large Disruption", linewidth=2)
    
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=12)

    # Format y-axis ticks as percentages with specified decimal places
    y_format = f'{{:.{y_decimal_places}f}}%'  # Creates format string like "{:.1f}%" or "{:.2f}%"
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: y_format.format(y)))

    # Increase tick size
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    # Determine save path
    if save_path is None:
        save_path = "combined_plot.pdf"

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory

    print(f"Plot saved as {save_path}")

if __name__ == "__main__":
    # First dataset: Display y-ticks with 2 decimal places (e.g., 3.45%)
    plot_from_csv('corr_interpoloate_costsavings(small_p_small_yieldloss).csv', 
                  x_label=r'$p_0/p$', 
                  y_label="Cost Savings",
                  save_path='Python_Plots/costsavings(small_p_small_yieldloss).pdf',
                  y_decimal_places=2)

    # Second dataset: Display y-ticks with 1 decimal place (e.g., 3.4%)
    plot_from_csv('corr_interpoloate_costsavings(small_p_large_yieldloss).csv', 
                  x_label=r'$p_0/p$', 
                  y_label="Cost Savings",
                  save_path='Python_Plots/costsavings(small_p_large_yieldloss).pdf',
                  y_decimal_places=1)

    # Plot both curves in the same figure
    plot_two_curves_from_csv('corr_interpoloate_costsavings(small_p_small_yieldloss).csv', 
                             'corr_interpoloate_costsavings(small_p_large_yieldloss).csv', 
                             x_label=r'$p_0/p$', 
                             y_label="Cost Savings",
                             save_path='Python_Plots/combined_costsavings.pdf')
