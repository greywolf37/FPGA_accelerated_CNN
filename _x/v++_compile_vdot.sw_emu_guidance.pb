
¦
v++_compile_vdot.sw_emu$d40b9589-76b2-483b-98fb-d6b91260e79b¬v++  -t sw_emu --config ./configs/design.cfg --log_dir ./logs --report_dir ./reports --platform /home/centos/aws-fpga/Vitis/aws_platform/xilinx_aws-vu9p-f1_shell-v04261818_201920_2/xilinx_aws-vu9p-f1_shell-v04261818_201920_2.xpfm --compile --kernel vdot -I ./src -o build/vdot.sw_emu.xo src/vdot.cpp *_"[/home/centos/FPGA_accelerated_CNN/reports/vdot.sw_emu/v++_compile_vdot.sw_emu_guidance.html2L"H/home/centos/FPGA_accelerated_CNN/_x/v++_compile_vdot.sw_emu_guidance.pbB†
./configs/design.cfgídebug=1
profile_kernel=data:all:all:all

[connectivity]
nk=vdot:2:vdot_1.vdot_2

sp = vdot_1.input1:DDR[0]
sp = vdot_1.input2:DDR[1]
sp = vdot_1.output:DDR[2]

sp = vdot_2.input1:DDR[2]
sp = vdot_2.input2:DDR[3]
sp = vdot_2.output:DDR[1]