#!/bin/bash
# Prepare PnR-variant RTL:
#  - SRAMs become small FF arrays (depth-capped to 32) so the netlist is
#    placeable without SRAM macros (absent from open PDKs).
#  - Program/weight RAMs get a live host load interface (host_we/host_addr/
#    host_data as real input pins) so synthesis cannot constant-fold memory
#    contents; without this the whole datapath optimizes away.
# Functional verification is done at RTL level with full memories (tb/).
set -e
mkdir -p pnr/rtl
grep -v 'blackbox' rtl/ram.sv > pnr/rtl/ram.sv
sed -E \
  -e 's/\.DEPTH\([0-9]+\)/.DEPTH(32)/' \
  -e "s/\.we\(1'b0\)/.we(host_we)/g" \
  -e "s/\.waddr\(8'd0\)/.waddr(host_addr[7:0])/g" \
  -e "s/\.waddr\(13'd0\)/.waddr(host_addr[12:0])/g" \
  -e "s/\.waddr\(10'd0\)/.waddr(host_addr[9:0])/g" \
  -e "s/\.waddr\(11'd0\)/.waddr(host_addr[10:0])/g" \
  -e "s/\.wdata\(128'd0\)/.wdata(host_data[127:0])/g" \
  -e "s/\.wdata\(256'd0\)/.wdata(host_data[255:0])/g" \
  -e "s/\.wdata\(32'd0\)/.wdata(host_data[31:0])/g" \
  -e "s/\.wdata\(16'd0\)/.wdata(host_data[15:0])/g" \
  -e 's/input  wire       rst_n,/input  wire       rst_n,\n    input  wire       host_we,\n    input  wire [12:0] host_addr,\n    input  wire [255:0] host_data,/' \
  rtl/nano_accel.sv > pnr/rtl/nano_accel.sv
cp rtl/dot64.sv pnr/rtl/
echo "pnr/rtl prepared (DEPTH=32 FF RAMs, live host load interface)"
