import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# Load Excel file 
file_path = "your_excel_file.xlsx"
df = pd.read_excel(file_path)

# Parameters for KDE
x_param = 'Bq'  # X-axis
y_param = 'Qt'  # Y-axis 

# Unique soil types
soil_column = "SoilTypeColumn"
soil_types = df['SoilTypeColumn'].dropna().unique()

# Create individual KDE plots for each soil type
for soil in soil_types:
    # Extract and clean data
    data = df[df['SoilTypeColumn'] == soil][[x_param, y_param]].dropna()
    if data.empty:
        print(f"No data available for soil type: {soil}")
        continue

    # Log-transform Qt values using log10
    data[y_param] = np.log10(data[y_param])  # Use log10 

    # Fit Gaussian KDE model
    kde = gaussian_kde(data.T)
    
    # Find mode 
    density_values = kde(data.T)
    mode_idx = np.argmax(density_values)
    mode_x, mode_y = data.iloc[mode_idx]

    # Joint KDE Plot with Marginal Density
    g = sns.jointplot(
        data=data, 
        x=x_param, 
        y=y_param, 
        kind="kde", 
        fill=True, 
        cmap="Greens", 
        alpha=0.8, 
        height=4, 
        marginal_kws={"fill": True}
    )

    # Add data points
    g.ax_joint.scatter(
        data[x_param], 
        data[y_param], 
        color='blue', 
        alpha=0.5, 
        s=10, 
        label='Data points'
    )

    # Mark the mode point
    g.ax_joint.scatter(
        mode_x, 
        mode_y, 
        color='red', 
        marker='o', 
        s=50, 
        edgecolors='white', 
        linewidth=1.5, 
        label='Mode'
    )

    # Set custom y-ticks to reflect original Qt values
    yticks = np.log10([0.1, 1, 10, 100, 1000])
    g.ax_joint.set_yticks(yticks)
    g.ax_joint.set_yticklabels(['0.1', '1', '10', '100', '1000'])

    # Labels and Formatting
    g.ax_joint.set_xlabel(x_param)
    g.ax_joint.set_ylabel('Qt')
    g.ax_joint.set_xlim(-0.5, 1.5)
    g.ax_joint.set_ylim(-1, 3)  # Corresponds to Qt from 0.1 to 1000

    # Add legend
    g.ax_joint.legend(loc='upper right')

    # Title
    g.fig.suptitle(f'KDE Plot for {soil}', fontsize=10)

    # Show plot
    plt.show()
