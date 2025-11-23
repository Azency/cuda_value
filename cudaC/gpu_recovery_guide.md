# GPU 恢复指南

## 当前状态
所有 GPU 显示 `[GPU requires reset]`，表明 GPU 处于错误状态需要重置。

## 问题原因
1. **GPU 驱动崩溃**：长时间运行或高负载导致驱动重置
2. **硬件问题**：过热、电源不足、硬件故障
3. **驱动问题**：驱动版本或配置问题

## 恢复步骤

### 方法 1: 重新加载 NVIDIA 驱动模块（推荐，无需重启系统）
```bash
# 卸载驱动模块
sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia

# 重新加载驱动模块
sudo modprobe nvidia
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm
sudo modprobe nvidia_uvm

# 验证 GPU 状态
nvidia-smi
```

### 方法 2: 重启 NVIDIA 持久化守护进程
```bash
sudo systemctl restart nvidia-persistenced
nvidia-smi
```

### 方法 3: 系统重启（如果上述方法无效）
```bash
sudo reboot
```

## 预防措施

### 1. 监控 GPU 状态
```bash
# 持续监控 GPU 温度和状态
watch -n 1 nvidia-smi

# 或使用 nvidia-smi 的守护模式
nvidia-smi -l 1
```

### 2. 设置 GPU 温度限制
```bash
# 设置 GPU 最大温度（例如 80°C）
sudo nvidia-smi -pl 250  # 设置功耗限制
sudo nvidia-smi -lgc 210,210  # 设置时钟频率（需要根据GPU型号调整）
```

### 3. 检查系统日志
```bash
# 查看 GPU 相关错误
sudo dmesg | grep -i nvidia
sudo journalctl -u nvidia-persistenced
```

### 4. 代码层面的改进
- ✅ 已添加设备检查机制
- ✅ 已添加同步机制防止命令队列堆积
- 建议：在长时间运行的循环中定期检查设备状态

## 代码改进建议

在长时间运行的循环中，可以添加定期设备检查：

```cuda
for (int t = T-1; t >= 0; t--) {
    // 每 100 次迭代检查一次设备状态
    if (t % 100 == 0) {
        check_and_recover_device();
    }
    // ... 原有代码 ...
}
```

## 如果问题持续

1. **检查硬件**：
   - GPU 温度是否过高
   - 电源供应是否充足
   - PCIe 连接是否正常

2. **更新驱动**：
   ```bash
   # 检查最新驱动版本
   ubuntu-drivers devices
   # 或访问 NVIDIA 官网下载最新驱动
   ```

3. **降低工作负载**：
   - 减少并发任务
   - 降低 GPU 使用率
   - 增加任务间隔时间

4. **联系技术支持**：
   - 如果硬件故障，可能需要 RMA

