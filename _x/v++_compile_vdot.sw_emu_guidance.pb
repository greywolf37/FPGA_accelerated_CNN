
Ï
v++_compile_vdot.sw_emu$2798abb5-e4b5-43ea-bff0-4579448df590¬v++  -t sw_emu --config ./configs/design.cfg --log_dir ./logs --report_dir ./reports --platform /home/centos/aws-fpga/Vitis/aws_platform/xilinx_aws-vu9p-f1_shell-v04261818_201920_2/xilinx_aws-vu9p-f1_shell-v04261818_201920_2.xpfm --compile --kernel vdot -I ./src -o build/vdot.sw_emu.xo src/vdot.cpp *_"[/home/centos/FPGA_accelerated_CNN/reports/vdot.sw_emu/v++_compile_vdot.sw_emu_guidance.html2L"H/home/centos/FPGA_accelerated_CNN/_x/v++_compile_vdot.sw_emu_guidance.pbB¯
./configs/design.cfg–debug=1
profile_kernel=data:all:all:all

[connectivity]
nk=vdot:1:vdot_1
sp = vdot_1.input1:DDR[0]
sp = vdot_1.input2:DDR[1]
sp = vdot_1.output:DDR[2]