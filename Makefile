# nano_accel: a chip that serves a nano language model
#
# make vectors  - train + quantize model, emit out/*.hex (sw/build.py)
# make sim      - run RTL sim, check bit-exact vs golden model
# make synth    - synthesize on Nangate 45nm, report area/cells
# make pnr      - place & route to GDSII (Docker; see pnr/README)
# make clean

RTL = rtl/ram.sv rtl/dot64.sv rtl/nano_accel.sv

vectors:
	python3 sw/build.py

sim: out/tb
	vvp out/tb

out/tb: tb/tb_nano.sv $(RTL) vectors
	iverilog -g2012 -o $@ tb/tb_nano.sv $(RTL)

synth:
	yosys -s syn/synth.ys

pnr: 
	bash pnr/prep.sh
	docker run --rm --platform linux/amd64 \
	  -v $(CURDIR):/work \
	  -v $(CURDIR)/pnr/results:/OpenROAD-flow-scripts/flow/results \
	  -w /OpenROAD-flow-scripts/flow \
	  openroad/orfs:latest \
	  make DESIGN_CONFIG=/work/pnr/config.mk

clean:
	rm -f out/tb out/tb_compile_check

.PHONY: vectors sim synth pnr clean
