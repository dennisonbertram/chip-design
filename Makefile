# nano_accel: a chip that serves a nano language model
#
# make vectors                     - train + quantize model, emit out/*.hex
# make sim                         - run RTL sim, check bit-exact vs golden model
# make synth                       - synthesize on Nangate 45nm, report area/cells
# make pnr                         - place & route to GDSII (Docker, ORFS)
#
# play with it:
# make sim PROMPT="To be or" G=48  - your own 8-char prompt, 48 tokens, streamed live
# make sim STEPS=8000              - train longer (better prose)
# note: G <= 50 (instruction RAM is 256 entries: 5 ops/token)

STEPS  ?= 2000
PROMPT ?= First Ci
G      ?= 32

RTL = rtl/ram.sv rtl/dot64.sv rtl/nano_accel.sv

vectors:
	python3 sw/build.py $(STEPS) "$(PROMPT)" $(G)

sim: out/tb
	vvp out/tb

out/tb: tb/tb_nano.sv $(RTL) vectors
	iverilog -g2012 -DGEN_TOKENS=$(G) -o $@ tb/tb_nano.sv $(RTL)

synth:
	yosys -s syn/synth.ys

pnr:
	bash pnr/prep.sh
	docker run --rm --platform linux/amd64 \
	  -v $(CURDIR):/work \
	  -v $(CURDIR)/pnr/results:/OpenROAD-flow-scripts/flow/results \
	  -v $(CURDIR)/pnr/logs:/OpenROAD-flow-scripts/flow/logs \
	  -v $(CURDIR)/pnr/reports:/OpenROAD-flow-scripts/flow/reports \
	  -w /OpenROAD-flow-scripts/flow \
	  openroad/orfs:latest \
	  make DESIGN_CONFIG=/work/pnr/config.mk

# --- Tiny Tapeout variant (tt/): streams the model from QSPI PSRAM ---
# make tt-sim   - small model (H=128), pack PSRAM image, verify bit-exact
tt-sim:
	NANO_H=128 python3 sw/build.py 400 "$(PROMPT)" 8
	python3 tt/sw/pack.py
	cd tt/test && iverilog -g2012 -o tb_local_sim \
	  ../src/project.v ../src/qspi_ctrl.v ../src/nano_tt_core.v \
	  psram_model.v tb_local.v && vvp tb_local_sim

clean:
	rm -f out/tb out/tb_compile_check tt/test/tb_local_sim

.PHONY: vectors sim synth pnr tt-sim clean
