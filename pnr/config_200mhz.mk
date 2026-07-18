# 200 MHz overclock attempt: same design, 5.0 ns constraint.
# fmax measured at 194.65 MHz, so post-route setup should FAIL.
export DESIGN_NICKNAME = nano_accel_200mhz
export DESIGN_NAME     = nano_accel
export PLATFORM        = nangate45

export VERILOG_FILES = /work/pnr/rtl/ram.sv /work/pnr/rtl/dot64.sv /work/pnr/rtl/nano_accel.sv
export SDC_FILE      = /work/pnr/constraint_200.sdc

export ABC_AREA        = 1
export CORE_UTILIZATION = 55
export PLACE_DENSITY_LB_ADDON = 0.20
export TNS_END_PERCENT = 100
export SYNTH_MEMORY_MAX_BITS = 32768
export SKIP_CTS_REPAIR_TIMING = 1
