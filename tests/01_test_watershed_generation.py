import numpy as np
import matplotlib.pyplot as plt
import rasterio
import rasterio.plot
from pyproj import Transformer
from config import OUTPUT_POINT, TARGET_CRS_EPSG

# ── Load clipped flow accumulation ──────────────────────────────────────────
with rasterio.open("output/clipped_flow_accumulation.tif") as src:
    fa_data      = src.read(1).astype(float)   # float so we can mask nodata
    fa_transform = src.transform
    fa_nodata    = src.nodata                   # -1.0

# Mask nodata (outside watershed) so they are transparent in the plot
fa_masked = np.ma.masked_where(fa_data <= 0, fa_data)

# ── Reproject outlet point to the raster CRS ────────────────────────────────
transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS_EPSG, always_xy=True)
outlet_x, outlet_y = transformer.transform(OUTPUT_POINT[1], OUTPUT_POINT[0])
print(f"Outlet (projected): ({outlet_x:.2f}, {outlet_y:.2f})")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

img = rasterio.plot.show(
    fa_masked,
    transform=fa_transform,
    cmap="Blues",
    ax=ax,
    title="Clipped Flow Accumulation (Contributing Area) — Watershed Only",
)

plt.colorbar(img.get_images()[0], ax=ax, label="Flow Accumulation (# upstream cells)")

ax.scatter(outlet_x, outlet_y, color="red", marker="*", s=200,
           zorder=5, label="Outlet point")
ax.legend()
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")

plt.tight_layout()
plt.savefig("output/clipped_flow_accumulation_check.png", dpi=150)
print("Plot saved to output/clipped_flow_accumulation_check.png")
plt.show()
