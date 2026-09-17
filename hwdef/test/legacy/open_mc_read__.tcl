# open_ila.sh 생성. 1개를 파형 창에 띄운다.  패턴: mc_read_*.ila
open_hw_manager
puts "opening mc_read_axis_ila_0"
display_hw_ila_data [read_hw_ila_data {/home/kjy/pim/emulator_top/hwdef/test/legacy/mc_read_axis_ila_0.ila}]
puts "1 opened"
