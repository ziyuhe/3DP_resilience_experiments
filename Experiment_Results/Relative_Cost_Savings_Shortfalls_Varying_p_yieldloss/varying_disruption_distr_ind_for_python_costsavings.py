import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Load data from Excel
filename = 'varying_disruption_distr_ind_for_python_costsavings.xlsx'


# Plot 1: Yield Loss Rate vs. Cost Savings
# Load data
plot1_data = pd.read_excel(filename, sheet_name='Plot 1')

# Create the plot
plt.figure(figsize=(4, 4))  # Set consistent figure size
colors = ['#1f77b4','black', '#ff7f0e']
labels = ['0.2x (Low)', '5x (Mid)', '10x (High)']

for idx, col in enumerate(plot1_data.columns[1:]):
    plt.plot(plot1_data['Yield Loss Rate'], plot1_data[col], '-o', label=labels[idx], color=colors[idx], markersize=8, linewidth=2)

plt.xlabel(r'$\alpha_j$', fontsize=20)
plt.ylabel('Cost Savings (%)', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title=r"$p_j$: multiple of baseline", loc='lower left', ncol=1, fontsize=13, title_fontsize=15)

# Adjust tick parameters for larger font size
plt.tick_params(axis='both', which='major', labelsize=14)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=12)  # Minor ticks (if applicable)

# Format y-axis ticks as percentages
current_ticks = plt.gca().get_yticks()
plt.gca().set_yticklabels([f'{int(tick)}%' for tick in current_ticks])

plt.ylim([-4,6])
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}%'))


plt.savefig('Python_Plots/CostSavings/Varying_DisruptDistr_Fixed_yieldloss_(Fixed_C3DP_Case11)_CostSavings.pdf', bbox_inches='tight')
plt.close()





# Plot 2: Marginal Failure Rate vs. Cost Savings
# Load data
plot2_data = pd.read_excel(filename, sheet_name='Plot 2', header=0)

plt.figure(figsize=(4, 4))  # Set consistent figure size
labels = ['0.2x (Low)', '8x (Mid)', '14x (High)']

for idx, col in enumerate(plot2_data.columns[1:]):
    plt.plot(plot2_data['Marginal Failure Rate'], plot2_data[col], '-o', label=labels[idx], color=colors[idx], markersize=8, linewidth=2)

plt.xlabel(r'$p_j$', fontsize=20)
plt.ylabel('Cost Savings (%)', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)


# Adjust tick parameters for larger font size
plt.tick_params(axis='both', which='major', labelsize=14)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=12)  # Minor ticks (if applicable)

plt.legend(title=r"$\alpha_j$: multiple of baseline", loc='lower left', ncol=1, fontsize=13, title_fontsize=15)

plt.ylim([-4,6])
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}%'))

plt.savefig('Python_Plots/CostSavings/Varying_DisruptDistr_Fixed_p_(Fixed_C3DP_Case11)_CostSavings.pdf', bbox_inches='tight')
plt.close()








# Plot 3: Capacity vs. Cost Savings
# Load the data
plot3_data = pd.read_excel(filename, sheet_name='Plot 3')

# Define custom styles
markers = ['o', 's', '^', 'd', 'v']  # Marker styles
colors = ['#6baed6', '#1f77b4', 'black', '#fdae61', '#ff7f0e']  # Colors for the curves

# Create the plot
plt.figure(figsize=(4, 4))  # Consistent figure size

# Fill area between -3 and 0 with shallow grey
plt.fill_between(
    x=[0, 20],  # x-axis range
    y1=-6,  # Lower y bound
    y2=0,  # Upper y bound
    color='lightgrey',  # Fill color
    alpha=0.5,  # Transparency
    label='_nolegend_'  # Exclude from legend
)

# Plot each curve
for idx, col in enumerate(plot3_data.columns[:0:-1]):  # Reverse column order
    x_values = plot3_data['Capacity (%)'].drop(range(2, 11))  # Exclude rows 2-10 for x-axis
    y_values = plot3_data[col].drop(range(2, 11))  # Exclude rows 2-10 for y-axis
    plt.plot(
        x_values,
        y_values,
        linestyle='-',
        marker=markers[-(idx + 1)],  # Reverse the marker order
        color=colors[-(idx + 1)],  # Reverse the color order
        label=col,
        markersize=7,
        linewidth=2
    )

# Customize x-axis
# plt.xlabel('3DP Capacity (% of Max Demand)')
plt.xlabel(r'$K$', fontsize=20)
plt.xlim([0, 20])
plt.ylim([-3, 5])

# Update x-axis ticks to display percentages
xticks = plt.gca().get_xticks()
plt.gca().set_xticklabels([f'{int(tick)}%' for tick in xticks])

# Customize y-axis
plt.ylabel('Cost Savings (%)', fontsize=18)

# Add legend with two rows
plt.legend(
    # title='Disruption - Yield Loss',
    title=r"($p_j$ , $\alpha_j$)",
    loc='lower left',
    ncol=2,  # Three-column layout for the legend
    fontsize=12, 
    title_fontsize=15,
    columnspacing=0.3  # Reduce spacing between columns
)

# Adjust tick parameters for larger font size
plt.tick_params(axis='both', which='major', labelsize=14)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=12)  # Minor ticks (if applicable)


# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

plt.ylim([-5.5,5.5])
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}%'))

# Save the plot
plt.savefig('Python_Plots/CostSavings/Varying_DisruptDistr_HIGH_MID_LOW_(Fixed_C3DP_Case11)_CostSavings.pdf', bbox_inches='tight')
plt.close()
