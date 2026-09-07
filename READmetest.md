param set MC_ROLLRATE_MAX 720
param set MC_PITCHRATE_MAX 720
param set MC_YAWRATE_MAX 540

sudo apt-get install python3-gz-transport13 python3-gz-msgs10

# 1. Bỏ qua kiểm tra GPS (Cho phép cất cánh trong nhà/không có GPS)

param set COM_ARM_WO_GPS 1

# 2. Thay đổi cơ chế kiểm tra an toàn từ "Nghiêm ngặt" sang "Cảnh báo thôi vẫn cho bay"

param set COM_ARM_CHK_EN 0


```
export PATH=$PATH:/opt/xtensa-esp-elf/bin/
export LIBGL_ALWAYS_SOFTWARE=0
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export GALLIUM_DRIVER=d3d12
export PATH=$(echo $PATH | tr ':' '\n' | grep -v '/mnt/c/' | tr '\n' ':')
```

make px4_sitl gz_x500_mono_cam

nano ~/.bashrc
