
�	
v++_link_vdot.sw_emu$b865609d-4fbe-4bdb-b4d3-1b317a246aed�v++ --xp param:compiler.lockFlowCritSlackThreshold=0 --xp vivado_param:hd.routingContainmentAreaExpansion=true --xp vivado_param:hd.supportClockNetCrossDiffReconfigurablePartitions=1 --xp vivado_param:bitstream.enablePR=4123 --xp vivado_param:physynth.ultraRAMOptOutput=false --xp vivado_prop:run.__KERNEL__.{STEPS.SYNTH_DESIGN.ARGS.MAX_URAM_CASCADE_HEIGHT}={1} --xp vivado_param:synth.elaboration.rodinMoreOptions={rt::set_parameter disableOregPackingUram true}  -t sw_emu --config ./configs/design.cfg --log_dir ./logs --report_dir ./reports --platform /home/centos/aws-fpga/Vitis/aws_platform/xilinx_aws-vu9p-f1_shell-v04261818_201920_2/xilinx_aws-vu9p-f1_shell-v04261818_201920_2.xpfm --link -o build/vdot.sw_emu.xclbin build/vdot.sw_emu.xo *U"Q/home/centos/FPGA_accelerated_CNN/reports/link/v++_link_vdot.sw_emu_guidance.html2I"E/home/centos/FPGA_accelerated_CNN/_x/v++_link_vdot.sw_emu_guidance.pbB�
./configs/design.cfg�debug=1
profile_kernel=data:all:all:all

[connectivity]
nk=vdot:2:vdot_1.vdot_2

sp = vdot_1.input1:DDR[0]
sp = vdot_1.input2:DDR[1]
sp = vdot_1.output:DDR[2]

sp = vdot_2.input1:DDR[2]
sp = vdot_2.input2:DDR[3]
sp = vdot_2.output:DDR[1]