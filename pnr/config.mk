export DESIGN_NICKNAME = nano_accel
export DESIGN_NAME     = nano_accel
export PLATFORM        = nangate45

export VERILOG_FILES = /work/pnr/rtl/ram.sv /work/pnr/rtl/dot64.sv /work/pnr/rtl/nano_accel.sv
export SDC_FILE      = /work/pnr/constraint.sdc

export ABC_AREA        = 1
export CORE_UTILIZATION = 55
export PLACE_DENSITY_LB_ADDON = 0.20
export TNS_END_PERCENT = 100
# placeholder FF SRAMs (depth-capped) exceed the default 4096-bit guard
export SYNTH_MEMORY_MAX_BITS = 32768
# 2nd detailed_placement in CTS crashes under amd64 emulation (SIGILL);
# setup is clean pre-route and hold repair re-runs in the route stage
export SKIP_CTS_REPAIR_TIMING = 1
