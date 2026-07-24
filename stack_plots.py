from PIL import Image
import os

folder = r"C:\Users\agdse\Documents\University\Artificial_Intelligence\Thesis\MentalModel_Generator_SymbolicAI\model_growth_plots_multi"

# Load in requested order: multibound, env, pure_env
names = ["multibound.png", "env.png", "pure_env.png"]
images = [Image.open(os.path.join(folder, n)) for n in names]

single_w, single_h = images[0].size

# --- Vertical stack ---
total_width_v = max(img.width for img in images)
total_height_v = sum(img.height for img in images)
combined_v = Image.new("RGB", (total_width_v, total_height_v), "white")
y_offset = 0
for img in images:
    combined_v.paste(img, (0, y_offset))
    y_offset += img.height

out_v = os.path.join(folder, "stacked_plots.png")
combined_v.save(out_v, optimize=True)
print(f"Vertical:   {out_v} ({combined_v.width}x{combined_v.height}px)")

# --- Horizontal stack ---
total_width_h = sum(img.width for img in images)
total_height_h = max(img.height for img in images)
combined_h = Image.new("RGB", (total_width_h, total_height_h), "white")
x_offset = 0
for img in images:
    combined_h.paste(img, (x_offset, 0))
    x_offset += img.width

out_h = os.path.join(folder, "horizontal_plots.png")
combined_h.save(out_h, optimize=True)
print(f"Horizontal: {out_h} ({combined_h.width}x{combined_h.height}px)")
print(f"Each original was {single_w}x{single_h}px")
