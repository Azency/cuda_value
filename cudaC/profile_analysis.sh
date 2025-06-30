#!/bin/bash

# CUDA性能分析脚本 - 生成GUI可查看的报告
# 使用方法: ./profile_analysis.sh

echo "=========================================="
echo "CUDA性能分析工具 - GUI报告生成器"
echo "=========================================="

# 确保testmain已编译
rm -rf ./testmain
if [ ! -f "./testmain" ]; then
    echo "编译程序..."
    nvcc -lineinfo -g -O2 -Xcompiler -Wall -dc src/cuda_value.cu -o cuda_value.o
    nvcc -lineinfo -g -O2 -Xcompiler -Wall -dc src/testmain.cu -o testmain.o
    nvcc cuda_value.o testmain.o -o ./testmain -lcudadevrt
    rm -rf cuda_value.o testmain.o
    echo "编译完成!"
fi

# 创建分析结果目录
mkdir -p analysis_reports
cd analysis_reports

echo "1. 生成完整性能分析报告..."
sudo CUDA_VISIBLE_DEVICES=0 /usr/local/cuda/bin/ncu \
  --target-processes all \
  --force-overwrite \
  --export full_analysis \
  --set full \
  ../testmain

echo "2. 生成内存分析报告..."
sudo CUDA_VISIBLE_DEVICES=0 /usr/local/cuda/bin/ncu \
  --target-processes all \
  --force-overwrite \
  --export memory_analysis \
  --set roofline \
  ../testmain

echo "3. 生成计算分析报告..."
sudo CUDA_VISIBLE_DEVICES=0 /usr/local/cuda/bin/ncu \
  --target-processes all \
  --force-overwrite \
  --export compute_analysis \
  --set detailed \
  ../testmain

echo "4. 生成XYZEW_kernel详细分析..."
sudo CUDA_VISIBLE_DEVICES=0 /usr/local/cuda/bin/ncu \
  --target-processes all \
  --force-overwrite \
  --export xyzew_kernel_detailed \
  --kernel-name "XYZEW_kernel" \
  --set full \
  ../testmain

echo "5. 生成V_tp1_kernel详细分析..."
sudo CUDA_VISIBLE_DEVICES=0 /usr/local/cuda/bin/ncu \
  --target-processes all \
  --force-overwrite \
  --export v_tp1_kernel_detailed \
  --kernel-name "V_tp1_kernel" \
  --set full \
  ../testmain

cd ..

echo "=========================================="
echo "分析报告生成完成! 文件位置:"
echo "=========================================="
echo "📊 完整分析: analysis_reports/full_analysis.ncu-rep"
echo "📊 内存分析: analysis_reports/memory_analysis.ncu-rep"
echo "📊 计算分析: analysis_reports/compute_analysis.ncu-rep"
echo "📊 XYZEW内核: analysis_reports/xyzew_kernel_detailed.ncu-rep"
echo "📊 V_tp1内核: analysis_reports/v_tp1_kernel_detailed.ncu-rep"
echo ""
echo "🖥️  在GUI中查看报告的方法:"
echo "=========================================="
echo "方法1 - 使用ncu-ui (NVIDIA Nsight Compute GUI):"
echo "  ncu-ui analysis_reports/full_analysis.ncu-rep"
echo ""
echo "方法2 - 在浏览器中查看:"
echo "  /usr/local/cuda/bin/ncu-ui --port 8080"
echo "  然后打开浏览器访问: http://localhost:8080"
echo ""
echo "方法3 - 转换为HTML格式:"
echo "  /usr/local/cuda/bin/ncu --import analysis_reports/full_analysis.ncu-rep --page details --export analysis_reports/report.html"
echo ""
echo "🔍 快速命令行查看摘要:"
echo "  /usr/local/cuda/bin/ncu --import analysis_reports/full_analysis.ncu-rep --page details"
