# 在 Apple Silicon Mac 上用 Docker 运行

使用仓库的 Ubuntu 20.04 配置，运行 ARM64 Linux 容器。相机与 IMU 标定入口均为无界面批处理，不需要 XQuartz。此镜像用于离线数据集，不包含 RealSense 驱动。

## 构建和启动

先启动 Docker Desktop，在仓库根目录执行：

```sh
docker compose build
docker compose up -d
docker compose exec calib cmake -S . -B build/docker \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCXX_MARCH=armv8-a
docker compose exec calib cmake --build build/docker --parallel 2 \
  --target basalt_calibrate basalt_calibrate_imu
```

`armv8-a` 用于 Apple Silicon 的 ARM64 容器；Intel Mac 请改为 `-DCXX_MARCH=x86-64`。不要在同一个构建目录混用 macOS、ARM64 Linux 和 x86 Linux 产物。首次编译较久，建议保留 Docker 至少 8 GB 内存，并发从 2 开始。

Compose 将整个仓库挂载到 `/workspace`，源码、数据、日志和结果在 Mac 上都能看到。`build/` 已被 Git 忽略。

## 下载示例数据

在 Mac 的仓库根目录执行，文件约 1.09 GB：

```sh
mkdir -p build/datasets build/results
curl -fL --retry 3 -C - \
  -o build/datasets/dataset-calib-imu1_512_16.bag \
  https://cdn2.vision.in.tum.de/tumvi/calibrated/512_16/dataset-calib-imu1_512_16.bag
```

## 先标定相机，再标定 IMU

```sh
docker compose exec -T calib build/docker/basalt_calibrate \
  --dataset-path /workspace/build/datasets/dataset-calib-imu1_512_16.bag \
  --dataset-type bag --aprilgrid /workspace/data/aprilgrid_6x6.json \
  --result-path /workspace/build/results/ --cam-types ds ds \
  > build/camera-calibration.log 2>&1

docker compose exec -T calib build/docker/basalt_calibrate_imu \
  --dataset-path /workspace/build/datasets/dataset-calib-imu1_512_16.bag \
  --dataset-type bag --aprilgrid /workspace/data/aprilgrid_6x6.json \
  --result-path /workspace/build/results/ \
  --gyro-noise-std 0.000282 --accel-noise-std 0.016 --skip-images 1 \
  > build/imu-calibration.log 2>&1
```

`--result-path` 末尾的 `/` 必须保留，程序直接用它拼接文件名。第二步从这个目录读取第一步生成的 `calibration.json`。更换数据集、标定板或跳帧参数时使用新的结果目录，避免复用旧角点和位姿缓存。重复相机标定会覆盖同目录中的 `calibration.json`。

IMU 入口默认注入 50 ms 时间偏移，可在命令后添加 `--time-offset-ms 10` 改成 10 ms，或用 `--time-offset-ms 0` 不注入。它改变读取后的 IMU 时间戳，不修改原始 bag。最终变换、误差和耗时打印在日志中，入口没有调用 `saveCalib()`，所以不能把相机阶段的 `calibration.json` 当作 IMU 最终标定文件。

10、20、30、40、50 ms 各重复三次的结果、逐轮收敛 CSV、原始日志和曲线见 [时间延迟实验记录](../experiments/time_delay_2026-11-12/README.md)。

当前源码的优化停止阈值已改为 `1e-8`；与 `1e-4` 的单次对照见 [严格阈值诊断记录](../experiments/time_delay_2026-11-12_threshold_1e-8/README.md)。严格阈值运行触发了上游拒绝步之后仍报告收敛的问题，诊断记录已将这些运行标为未验证，未修改优化器的接受/拒绝逻辑。

## 查看和停止

```sh
docker compose ps
docker compose exec calib bash
tail -f build/imu-calibration.log
docker compose down
```

停止容器会保留 Mac 上的 `build/` 数据与编译结果。再次使用执行 `docker compose up -d`。

## 本机验证记录

2026-11-11，在 Apple Silicon Mac 的 Docker Desktop 中实测：

- Ubuntu 20.04 / aarch64、Clang 10，两个标定目标均以 Release 构建通过。
- 完整 `dataset-calib-imu1_512_16.bag`：1,038 组双目图像，两个阶段均正常退出并报告收敛。
- 相机标定日志的平均重投影误差为 `0.0854259` 像素，结果为 `build/results/calibration.json`。
- IMU 联合标定日志的平均重投影误差为 `0.0869407` 像素，估计时间偏移为 `0.050165 s`，其中包含入口注入的 50 ms 偏移。
- 日志报告 IMU 优化循环耗时 `105 ms`，不包含读取数据、角点检测和初始化；该数值只代表本次运行。

为使此配置编译通过，仅修复了两处源码兼容问题：BRIEF 采样坐标表显式使用 `signed char`；IMU 的两个循环通过常量引用读取智能指针，避免 Clang 的 `-Werror=range-loop-construct`。

本次验证为离线示例流程，未验证实时相机接入、GUI、全部单元测试或实机标定精度。
