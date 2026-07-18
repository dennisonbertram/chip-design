# 200 MHz overclock attempt (fmax measured 194.65 MHz -> expect fail)
create_clock -name core_clk -period 5.0 [get_ports clk]
set_clock_uncertainty 0.1 [get_clocks core_clk]
set_false_path -from [get_ports rst_n]
set_input_delay 0.5 -clock core_clk [all_inputs]
set_output_delay 0.5 -clock core_clk [all_outputs]
