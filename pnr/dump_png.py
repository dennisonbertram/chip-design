import pya
view = pya.LayoutView()
cv_idx = view.load_layout("/work/pnr/results/nangate45/nano_accel/base/6_final.gds")
cv = view.cellview(cv_idx)
tops = [c.name for c in cv.layout().top_cells()]
print("top cells:", tops)
target = "nano_accel" if "nano_accel" in tops else tops[0]
cv.cell = cv.layout().cell(target)
view.load_layer_props("/OpenROAD-flow-scripts/flow/platforms/nangate45/FreePDK45.lyp")
view.max_hier()
view.zoom_fit()
view.save_image("/work/out/gds_final.png", 1200, 1200)
print("saved /work/out/gds_final.png, cell =", target)
