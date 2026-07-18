import pya

GDS = "/work/pnr/results/nangate45/nano_accel/base/6_final.gds"
LYP = "/OpenROAD-flow-scripts/flow/platforms/nangate45/FreePDK45.lyp"

view = pya.LayoutView()
cv_idx = view.load_layout(GDS)
cv = view.cellview(cv_idx)
cv.cell = cv.layout().cell("nano_accel")
view.load_layer_props(LYP)
view.max_hier()

crops = {
    "zoomA": (245, 415, 325, 495),   # heart of the dense compute blob
    "zoomB": (485, 530, 565, 610),   # upper-right cluster
}
for name, (x0, y0, x1, y1) in crops.items():
    view.zoom_box(pya.DBox(x0, y0, x1, y1))
    view.save_image(f"/work/out/{name}.png", 1200, 1200)
    print(name, "saved")
