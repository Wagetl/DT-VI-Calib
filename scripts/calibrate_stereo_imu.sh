#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法
  scripts/calibrate_stereo_imu.sh <双目数据集> <联合标定数据集> <新结果目录>

两阶段可以使用同一个数据集。结果目录必须尚未创建。

可通过环境变量覆盖以下默认值
  BASALT_BIN_DIR   仓库的 build/ 目录，Docker 构建使用 build/docker
  DATASET_TYPE     bag（ROS bag 文件）或 euroc（数据集目录）
  APRILGRID        仓库的 data/aprilgrid_6x6.json，需与实物标定板匹配
  CAM0_TYPE        ds，可选 ds、eucm、kb4、pinhole
  CAM1_TYPE        ds，可选 ds、eucm、kb4、pinhole
  SKIP_IMAGES      1，两阶段均使用该取帧间隔
  GYRO_NOISE_STD   0.000282，TUM VI 示例值
  ACCEL_NOISE_STD  0.016，TUM VI 示例值

双目视觉结果保存为 calibration.json，联合标定结果见 imu-calibration.log。
EOF
}

fail() {
  printf '%s\n' "$*" >&2
  exit 1
}

if [[ $# -eq 1 && ( $1 == -h || $1 == --help ) ]]; then
  usage
  exit 0
fi
if [[ $# -ne 3 ]]; then
  usage >&2
  exit 2
fi

repo_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
bin_dir=${BASALT_BIN_DIR:-$repo_dir/build}
dataset_type=${DATASET_TYPE:-bag}
aprilgrid=${APRILGRID:-$repo_dir/data/aprilgrid_6x6.json}
cam0_type=${CAM0_TYPE:-ds}
cam1_type=${CAM1_TYPE:-ds}
skip_images=${SKIP_IMAGES:-1}
gyro_noise_std=${GYRO_NOISE_STD:-0.000282}
accel_noise_std=${ACCEL_NOISE_STD:-0.016}
result_dir=${3%/}

[[ -n $result_dir ]] || fail '请指定一个新的结果目录。'
[[ -f $aprilgrid && -r $aprilgrid ]] || fail "无法读取标定板配置 $aprilgrid"
[[ $skip_images =~ ^[1-9][0-9]*$ ]] || fail 'SKIP_IMAGES 必须是正整数。'

for binary in "$bin_dir/basalt_calibrate" "$bin_dir/basalt_calibrate_imu"; do
  [[ -f $binary && -x $binary ]] || fail "找不到可执行文件 $binary，请先构建标定程序。"
done
for cam_type in "$cam0_type" "$cam1_type"; do
  case "$cam_type" in
    ds|eucm|kb4|pinhole) ;;
    *) fail "不支持的相机模型 $cam_type" ;;
  esac
done
for dataset in "$1" "$2"; do
  case "$dataset_type" in
    bag) [[ -f $dataset && -r $dataset ]] || fail "无法读取 bag 文件 $dataset" ;;
    euroc) [[ -d $dataset && -r $dataset ]] || fail "无法读取 EuRoC 目录 $dataset" ;;
    *) fail 'DATASET_TYPE 必须是 bag 或 euroc。' ;;
  esac
done

# 两个阶段共享结果目录，已有目录可能含旧角点或位姿缓存。
mkdir -p -- "$(dirname -- "$result_dir")"
mkdir -- "$result_dir" || fail "无法创建新结果目录 $result_dir"

common_args=(--dataset-type "$dataset_type" --aprilgrid "$aprilgrid"
  --result-path "$result_dir/" --skip-images "$skip_images")
camera_log="$result_dir/camera-calibration.log"
imu_log="$result_dir/imu-calibration.log"

printf '开始双目视觉标定，日志写入 %s\n' "$camera_log"
"$bin_dir/basalt_calibrate" --dataset-path "$1" "${common_args[@]}" \
  --cam-types "$cam0_type" "$cam1_type" 2>&1 | tee -- "$camera_log" \
  || fail "双目视觉标定失败，请查看 $camera_log"
[[ -s "$result_dir/calibration.json" ]] && grep -Fqx -- 'Optimization Converged !!' "$camera_log" \
  || fail "双目视觉标定未报告收敛或缺少 calibration.json，请查看 $camera_log"

printf '开始双目惯性联合标定，日志写入 %s\n' "$imu_log"
# 常规标定保留原始时间戳，覆盖程序默认注入的 50 ms 偏移。
"$bin_dir/basalt_calibrate_imu" --dataset-path "$2" "${common_args[@]}" \
  --gyro-noise-std "$gyro_noise_std" --accel-noise-std "$accel_noise_std" \
  --time-offset-ms 0 2>&1 | tee -- "$imu_log" \
  || fail "双目惯性联合标定失败，请查看 $imu_log"
grep -Fqx -- 'Optimization Converged !!' "$imu_log" \
  || fail "双目惯性联合标定未报告收敛，请查看 $imu_log"

printf '双目标定结果见 %s，联合标定结果见 %s\n' "$result_dir/calibration.json" "$imu_log"
