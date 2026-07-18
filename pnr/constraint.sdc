# 100 MHz target, matching the blog post's clock
create_clock -name core_clk -period 10.0 [get_ports clk]
set_clock_uncertainty 0.1 [get_clocks core_clk]
set_false_path -from [get_ports rst_n]
set_input_delay 1.0 -clock core_clk [all_inputs]
set_output_delay 1.0 -clock core_clk [all_outputs]
