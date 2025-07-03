import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

# Load and filter data
def load_and_filter_data(excel_file, sheet_name, soil_types):
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    needed_cols = ["Qt", "Bq", "Soil type_column"]
    df.dropna(subset=needed_cols, inplace=True)
    df = df[df["Soil type_column"].isin(soil_types)]
    df = df[(df["Qt"] > 0) & (df["Bq"] > -0.5)]
    df["logQt"] = np.log10(df["Qt"])
    return df
def plot_kde_solid_colors_with_marginals(df, x_col="Bq", y_col="logQt", hue_col="Soil type_column"):
    soil_types_list = df[hue_col].unique()
    colors = plt.get_cmap('tab10').colors
    color_dict = {soil: colors[i % len(colors)] for i, soil in enumerate(soil_types_list)}

    # Set gridspec for marginal plots
    fig = plt.figure(figsize=(12, 10))
    grid = plt.GridSpec(4, 4, hspace=0.4, wspace=0.4)

    main_ax = fig.add_subplot(grid[1:, :-1])
    x_margin_ax = fig.add_subplot(grid[0, :-1], sharex=main_ax)
    y_margin_ax = fig.add_subplot(grid[1:, -1], sharey=main_ax)

    # Axis settings
    main_ax.set_facecolor('white')
    main_ax.set_xlim(-0.5, 1.5)
    main_ax.set_ylim(-1, 3)
    main_ax.set_xlabel(r"$B_q = \frac{u_2 - u_0}{q_t - \sigma_{vo}}[-]$")
    main_ax.set_ylabel(r"$Q_t = \frac{q_t - \sigma_{vo}}{\sigma_{vo}'}[-]$")
    main_ax.set_yticks(np.linspace(-1, 3, 5))
    main_ax.set_yticklabels(['0.1', '1', '10', '100', '1000'])

    # Grid setup for KDE evaluation
    Bq_grid = np.linspace(-0.5, 1.5, 1000)
    logQt_grid = np.linspace(-1, 3, 1000)
    Bq_mesh, logQt_mesh = np.meshgrid(Bq_grid, logQt_grid)
    positions = np.vstack([Bq_mesh.ravel(), logQt_mesh.ravel()])

    densities = np.zeros((len(soil_types_list), 1000, 1000))

    # Calculate KDE densities per soil type
    for idx, soil_type in enumerate(soil_types_list):
        data = df[df[hue_col] == soil_type][[x_col, y_col]].values.T
        if data.shape[1] < 2:
            continue

        kde = gaussian_kde(data)
        density = kde(positions).reshape(1000, 1000)

        # Thresholding low-density points (below 10% of max density)
        density_threshold = 0.10 * density.max()
        density[density < density_threshold] = 0
        densities[idx] = density

    # Determine soil type with maximum density at each grid point
    max_density_indices = np.argmax(densities, axis=0)
    max_density_values = np.max(densities, axis=0)

    #remove points where density is 0 after thresholding
    overall_mask = max_density_values > 0

    plot_array = np.where(overall_mask, max_density_indices + 1, 0)

    # Prepare colormap with white for background
    cmap_colors = ['white'] + [color_dict[soil] for soil in soil_types_list]
    cmap = ListedColormap(cmap_colors)

    # Plot KDE solid colors
    main_ax.imshow(plot_array, extent=[-0.5, 1.5, -1, 3], origin='lower', cmap=cmap, aspect='auto')

    # Plot modes for each soil type and marginals
    legend_patches = []
    for idx, soil_type in enumerate(soil_types_list):
        density = densities[idx]

        if np.all(density == 0):
            continue  # skip soil types fully removed by threshold

        idx_max = np.unravel_index(np.argmax(density), density.shape)
        mode_x = Bq_grid[idx_max[1]]
        mode_y = logQt_grid[idx_max[0]]
        main_ax.plot(mode_x, mode_y, 'o', markersize=8, markeredgecolor='black', color=color_dict[soil_type])

        legend_patches.append(Patch(color=color_dict[soil_type], label=soil_type))

        # Marginal plots
        data = df[df[hue_col] == soil_type]
        kde_x = gaussian_kde(data[x_col])
        kde_y = gaussian_kde(data[y_col])

        x_margin_ax.plot(Bq_grid, kde_x(Bq_grid), color=color_dict[soil_type])
        y_margin_ax.plot(kde_y(logQt_grid), logQt_grid, color=color_dict[soil_type])

    # Remove marginal plot axes for clarity
    x_margin_ax.axis('off')
    y_margin_ax.axis('off')

    # Add legend and title
    main_ax.legend(handles=legend_patches, title="Soil Type", loc='upper right')
    main_ax.set_title("KDE Regions ")

    plt.tight_layout()
    plt.show()
# needed files
excel_file = r""
sheet_name = ""
soil_types = [""]
