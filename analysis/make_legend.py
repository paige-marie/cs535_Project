import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase


# Create figure and axis for just the colorbar
fig, ax = plt.subplots(figsize=(0.5, 6))  # Adjust size as needed (width, height)

norm = Normalize(vmin=-1, vmax=1)
cb = ColorbarBase(ax, cmap='viridis', norm=norm, orientation='vertical')
cb.set_label('NDVI', rotation=270, labelpad=15)

plt.savefig("ndvi_lgend.png", bbox_inches='tight', dpi=300)
plt.close()