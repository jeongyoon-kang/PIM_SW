# capture.py 가 생성한 스크립트. 직접 수정하지 말 것.
set OUT {/home/kjy/pim/emulator_top/hwdef/test/version1.0/tc1}
file mkdir $OUT
set T0 [clock milliseconds]
proc mark {t} { global OUT T0
    set dt [expr {([clock milliseconds]-$T0)/1000.0}]
    set f [open $OUT/.progress a]; puts $f [format "%7.1fs  %s" $dt $t]; close $f }
mark "start"
open_hw_manager
connect_hw_server -url localhost:3121 -allow_non_jtag
open_hw_target
current_hw_device [get_hw_devices xcvh1582_1]
set DEV [get_hw_devices xcvh1582_1]
set_property PROBES.FILE      {/home/kjy/pim/emulator_top/hw/ch2/version1.0/emu_top_fpga_wrapper.ltx} $DEV
set_property FULL_PROBES.FILE {/home/kjy/pim/emulator_top/hw/ch2/version1.0/emu_top_fpga_wrapper.ltx} $DEV
refresh_hw_device $DEV
proc ila {n} { return [lindex [get_hw_ilas -filter "CELL_NAME=~*$n"] 0] }
set SRC [ila {axis_ila_0}]
set NAMES {}
foreach i [get_hw_ilas] { lappend NAMES [get_property CELL_NAME $i] }
set ALL {}
foreach n $NAMES { lappend ALL [ila $n] }
set DEPTH 1024
set NWIN 1
set TPOS [expr {($DEPTH / $NWIN) / 16}]
foreach i $ALL {
    reset_hw_ila $i
    catch { set_property CONTROL.DATA_DEPTH $DEPTH $i }
    set_property CONTROL.WINDOW_COUNT     $NWIN $i
    set_property CONTROL.TRIGGER_POSITION $TPOS $i
}
set_property CONTROL.TRIGGER_MODE      BASIC_ONLY $SRC
set_property CONTROL.TRIGGER_CONDITION OR $SRC
if {[get_property STATIC.IS_TRIG_OUT_SUPPORTED $SRC]} {
    set_property CONTROL.TRIG_OUT_MODE TRIGGER_ONLY $SRC
}
set_property TRIGGER_COMPARE_VALUE eq1'b1 [get_hw_probes {emu_top_fpga_i/axis_ila_0/SLOT_3_AXI_arvalid} -of_objects $SRC]
set_property TRIGGER_COMPARE_VALUE eq1'b1 [get_hw_probes {emu_top_fpga_i/axis_ila_0/SLOT_4_AXI_arvalid} -of_objects $SRC]
set CHAIN {}
foreach i $ALL {
    if {$i eq $SRC} { continue }
    if {![get_property STATIC.IS_TRIG_IN_SUPPORTED $i]} { continue }
    set_property CONTROL.TRIGGER_MODE TRIG_IN_ONLY $i
    if {[get_property STATIC.IS_TRIG_OUT_SUPPORTED $i]} {
        set_property CONTROL.TRIG_OUT_MODE TRIG_IN_ONLY $i
    }
    lappend CHAIN $i
}
foreach i [lreverse $CHAIN] { run_hw_ila $i }
run_hw_ila $SRC
foreach i $ALL { get_property STATUS.CORE_STATUS $i }
after 2000
foreach i $ALL { mark "  arm: [lindex [split [get_property CELL_NAME $i] /] end] = [get_property STATUS.CORE_STATUS $i]" }
mark "armed"
set PFX {mc_read_}
set LOG $OUT/${PFX}workload.log
set _cmd {/home/kjy/pim/emulator_top/hwdef/test/emu_mc --read --mib 4}
if {[catch {exec {/home/kjy/pim/emulator_top/hwdef/test/emu_mc} {--read} {--mib} {4} >& $LOG} _e]} { mark "workload 비정상 종료: $_e" }
set _f [open $LOG a]; puts $_f "
--- 실행: $_cmd"; close $_f
after 1000
mark "workload done"
foreach i $ALL {
    set cn [get_property CELL_NAME $i]
    set nm $PFX[lindex [split $cn /] end]
    catch {wait_on_hw_ila_data -timeout 60 $i}
    set st [get_property STATUS.CORE_STATUS $i]
    if {$st ne "FULL"} {
        mark "$nm 미완 status=$st samples=[get_property STATUS.SAMPLE_COUNT $i] win=[get_property STATUS.WINDOW_COUNT $i]"
        continue
    }
    mark "$nm upload 시작"
    set d [upload_hw_ila_data $i]
    mark "$nm upload 끝"
    display_hw_ila_data $d
    after 8000
    write_hw_ila_data -force $OUT/$nm.ila $d
    write_hw_ila_data -force -csv_file $OUT/$nm.csv $d
    mark "$nm written"
}
close_hw_target
disconnect_hw_server
mark "DONE"
exit 0
