import pya

GDS = "/work/pnr/results/nangate45/nano_accel/base/6_final.gds"
LYP = "/OpenROAD-flow-scripts/flow/platforms/nangate45/FreePDK45.lyp"

view = pya.LayoutView()
cv_idx = view.load_layout(GDS)
cv = view.cellview(cv_idx)
cv.cell = cv.layout().cell("nano_accel")
view.load_layer_props(LYP)
view.max_hier()

bb = cv.cell.bbox()          # in database units
dbu = cv.layout().dbu
W = bb.width() * dbu
H = bb.height() * dbu
print(f"die bbox: {W:.1f} x {H:.1f} um, origin ({bb.left*dbu:.1f}, {bb.bottom*dbu:.1f})")

# 3x3 tile survey with 15% overlap; tiles named by grid position
n = 3
for gy in range(n):
    for gx in range(n):
        cx = bb.left * dbu + W * (gx + 0.5) / n
        cy = bb.bottom * dbu + H * (gy + 0.5) / n
        tw = W / n * 1.15
        th = H / n * 1.15
        box = pya.DBox(cx - tw/2, cy - th/2, cx + tw/2, cy + th/2)
        view.zoom_box(box)
        fn = f"/work/out/tile_{gx}_{gy}.png"
        view.save_image(fn, 700, 700)
print("tiles written: out/tile_{0..2}_{0..2}.png (gy=0 is BOTTOM row)")
