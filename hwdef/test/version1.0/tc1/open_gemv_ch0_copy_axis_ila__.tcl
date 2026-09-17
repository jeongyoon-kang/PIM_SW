# open_ila.sh 생성. 5개를 파형 창에 띄운다.  패턴: gemv_ch0_copy_axis_ila**.ila
open_hw_manager
puts "opening gemv_ch0_copy_axis_ila_0"
display_hw_ila_data [read_hw_ila_data {/home/kjy/pim/emulator_top/hwdef/test/version1.0/tc1/gemv_ch0_copy_axis_ila_0.ila}]
puts "opening gemv_ch0_copy_axis_ila_1"
display_hw_ila_data [read_hw_ila_data {/home/kjy/pim/emulator_top/hwdef/test/version1.0/tc1/gemv_ch0_copy_axis_ila_1.ila}]
puts "opening gemv_ch0_copy_axis_ila_2"
display_hw_ila_data [read_hw_ila_data {/home/kjy/pim/emulator_top/hwdef/test/version1.0/tc1/gemv_ch0_copy_axis_ila_2.ila}]
puts "opening gemv_ch0_copy_axis_ila_3"
display_hw_ila_data [read_hw_ila_data {/home/kjy/pim/emulator_top/hwdef/test/version1.0/tc1/gemv_ch0_copy_axis_ila_3.ila}]
puts "opening gemv_ch0_copy_axis_ila_4"
display_hw_ila_data [read_hw_ila_data {/home/kjy/pim/emulator_top/hwdef/test/version1.0/tc1/gemv_ch0_copy_axis_ila_4.ila}]
puts "5 opened"
