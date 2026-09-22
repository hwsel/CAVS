#!/usr/bin/env bash
#!/bin/bash

source /home/miniconda3/etc/profile.d/conda.sh
conda activate cavs
declare -A VIDEO_MAP

VIDEO_MAP["Beauty"]="Beauty_3840x2160.yuv"
VIDEO_MAP["Bosphorus"]="Bosphorus_3840x2160.yuv"
VIDEO_MAP["CityAlley"]="CityAlley_3840x2160_50fps_8bit.yuv"
VIDEO_MAP["FlowerFocus"]="FlowerFocus_3840x2160_50fps_8bit.yuv"
VIDEO_MAP["FlowerKids"]="FlowerKids_3840x2160_50fps_8bit.yuv"
VIDEO_MAP["FlowerPan"]="FlowerPan_3840x2160_50fps_8bit.yuv"
VIDEO_MAP["HoneyBee"]="HoneyBee_3840x2160.yuv"
VIDEO_MAP["Jockey"]="Jockey_3840x2160.yuv"
VIDEO_MAP["Lips"]="Lips_3840x2160_120fps_8bit.yuv"
VIDEO_MAP["RaceNight"]="RaceNight_3840x2160_50fps_8bit.yuv"
VIDEO_MAP["ReadySteadyGo"]="ReadySteadyGo_3840x2160.yuv"
VIDEO_MAP["RiverBank"]="RiverBank_3840x2160_50fps_8bit.yuv"
VIDEO_MAP["ShakeNDry"]="ShakeNDry_3840x2160.yuv"
VIDEO_MAP["SunBath"]="SunBath_3840x2160_50fps_8bit.yuv"
VIDEO_MAP["Twilight"]="Twilight_3840x2160_50fps_8bit.yuv"
VIDEO_MAP["YachtRide"]="YachtRide_3840x2160.yuv"

#function case1() {
#  {
#    python live_demo.py ${VIDEO} -skp ${SKP} -skd ${SKD} -sth ${STH} -stl ${STL} --log --comment "full"
#  } &
#
#  wait
#  sudo nvidia-smi -rgc
#}
#
#
#function case2() {
#  sudo nvidia-smi -lgc 0,210
#
#  {
#    python live_demo.py ${VIDEO} -skp ${SKP} -skd ${SKD} -sth ${STH} -stl ${STL} --log --comment "210"
#  } &
#
#  wait
#  sudo nvidia-smi -rgc
#}
#
#function case3() {
#  {
#    python live_demo.py ${VIDEO} -skp ${SKP} -skd ${SKD} -sth ${STH} -stl ${STL} --log --comment "14s,full,400"
#  } &
#
#  sudo nvidia-smi -rgc \
#  && sleep 14 \
#  && sudo nvidia-smi -lgc 0,400
#
#  wait
#  sudo nvidia-smi -rgc
#}
#
function case4() {
  {
   python live_demo.py ${VIDEO} -skp ${SKP} -skd ${SKD} -sth ${STH} -stl ${STL} --log --comment "7s,full,210,500,full"
  } &

  sudo nvidia-smi -rgc \
  && sleep 7 \
  && sudo nvidia-smi -lgc 0,210 \
  && sleep 7 \
  && sudo nvidia-smi -lgc 0,500 \
  && sleep 7 \
  && sudo nvidia-smi -rgc
#
  wait
  sudo nvidia-smi -rgc
}
#
#function case600() {
#  {
#    python live_demo.py ${VIDEO} -skp ${SKP} -skd ${SKD} -sth ${STH} -stl ${STL} --log --comment "14s,full,600"
#  } &
#
#  sudo nvidia-smi -rgc \
#  && sleep 14 \
#  && sudo nvidia-smi -lgc 0,600
#
#  wait
#  sudo nvidia-smi -rgc
#}
#
#function case500() {
#  {
#    python live_demo.py ${VIDEO} -skp ${SKP} -skd ${SKD} -sth ${STH} -stl ${STL} --log --comment "14s,full,500"
#  } &
#
#  sudo nvidia-smi -rgc \
#  && sleep 14 \
#  && sudo nvidia-smi -lgc 0,500
#
#  wait
#  sudo nvidia-smi -rgc
#}
#
#function case400() {
#  {
#    python live_demo.py ${VIDEO} -skp ${SKP} -skd ${SKD} -sth ${STH} -stl ${STL} --log --comment "14s,full,400"
#  } &
#
#  sudo nvidia-smi -rgc \
#  && sleep 14 \
#  && sudo nvidia-smi -lgc 0,400
#
#  wait
#  sudo nvidia-smi -rgc
#}
#
#function case300() {
#  {
#    python live_demo.py ${VIDEO} -skp ${SKP} -skd ${SKD} -sth ${STH} -stl ${STL} --log --comment "14s,full,300"
#  } &
#
#  sudo nvidia-smi -rgc \
#  && sleep 14 \
#  && sudo nvidia-smi -lgc 0,300
#
#  wait
#  sudo nvidia-smi -rgc
#}
#
#function case200() {
#  {
#    python live_demo.py ${VIDEO} -skp ${SKP} -skd ${SKD} -sth ${STH} -stl ${STL} --log --comment "14s,full,200"
#  } &
#
#  sudo nvidia-smi -rgc \
#  && sleep 14 \
#  && sudo nvidia-smi -lgc 0,200
#
#  wait
#  sudo nvidia-smi -rgc
#}

function case_eval() {
  sudo nvidia-smi -rgc
 # if [[ $1 =~ ^-?[0-9]+$ ]]; then
  #  sudo nvidia-smi -lgc 0,"$1"
  #fi
  if [[ $QP = "None" ]]; then
    python live_demo.py "${VIDEO}" -sta "${STA}" -skp "${SKP}" -skd "${SKD}" -sth "${STH}" -stl "${STL}" --backend "eval" --log --comment "${1}" &
    pid_main=$!
  elif [[ $QP =~ ^-?[0-9]+$ ]]; then
    python live_demo.py "${VIDEO}" -sta "${STA}" -skp "${SKP}" -skd "${SKD}" -sth "${STH}" -stl "${STL}" --backend "eval" --log --comment "${1}" --qp "${QP}" &
    pid_main=$!
  else
    return
  fi

#  ========CONST,DROP,DROP-REC========
# sleep 12
 #if [[ $1 =~ ^-?[0-9]+$ ]]; then
  #  sudo nvidia-smi -lgc 0,"$1"
  #fi
  #sleep 5
  #sudo nvidia-smi -rgc
#  ========CONST,DROP,DROP-REC========

# ========Low-Mid-High========
#sleep 12
#sudo nvidia-smi -lgc 0,400
#sleep 5
#sudo nvidia-smi -lgc 0,800
#sleep 5
#sudo nvidia-smi -rgc
# vLow-Mid-High========

# ========RANDOM========
sudo nvidia-smi -lgc 0,400
sleep 1
sudo nvidia-smi -rgc
sleep 1
sudo nvidia-smi -lgc 0,400
sleep 1
sudo nvidia-smi -rgc
sleep 1
sudo nvidia-smi -lgc 0,400
sleep 1
sudo nvidia-smi -lgc 0,400
sleep 1
sudo nvidia-smi -lgc 0,800
sleep 1
sudo nvidia-smi -lgc 0,400
sleep 1
sudo nvidia-smi -rgc
sleep 1
sudo nvidia-smi -rgc
sleep 1
sudo nvidia-smi -lgc 0,400
sleep 1
sudo nvidia-smi -rgc
sleep 1
sudo nvidia-smi -lgc 0,400
sleep 1
sudo nvidia-smi -lgc 0,800
sleep 1
sudo nvidia-smi -lgc 0,400
sleep 1
sudo nvidia-smi -lgc 0,400
sleep 1
sudo nvidia-smi -lgc 0,400
sleep 1
sudo nvidia-smi -rgc
sleep 1
sudo nvidia-smi -lgc 0,400
sleep 1
sudo nvidia-smi -rgc
sleep 1
sudo nvidia-smi -lgc 0,400
sleep 1
sudo nvidia-smi -lgc 0,400
sleep 1
sudo nvidia-smi -lgc 0,800
sleep 1
sudo nvidia-smi -lgc 0,800
sleep 1
sudo nvidia-smi -lgc 0,800
sleep 1
sudo nvidia-smi -lgc 0,400
sleep 1
sudo nvidia-smi -rgc
sleep 1
sudo nvidia-smi -lgc 0,800
sleep 1
sudo nvidia-smi -rgc
sleep 1
sudo nvidia-smi -lgc 0,400
sleep 1
# ========RANDOM========



  wait "${pid_main}"

  sudo nvidia-smi -rgc

  TS="$(date +%s)"

  pids=()

#  for model in vmaf_4k_v0.6.1 vmaf_4k_v0.6.1neg vmaf_v0.6.1 vmaf_v0.6.1neg; do
#    docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,video -v /home/Projects:/data --rm vmaf_cuda -r "/data/UVG/${VIDEO_MAP[$VIDEO]}" -d /data/cavs2/WIP/WIP.yuv -w 3840 -h 2160 -p 420 -b 8 --json -o "/data/cavs2/results/set1/vmaf_${model}_${VIDEO}_${1}_${QP}_${TS}.json" --model "path=/vmaf/model/${model}.json" &
#    pids+=($!)

#    docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,video -v /home/Projects:/data --rm vmaf_cuda -r /data/cavs2/WIP/UPPER.yuv -d /data/cavs2/WIP/WIP.yuv -w 3840 -h 2160 -p 420 -b 8 --json -o "/data/cavs2/results/set1/UPPER_vmaf_${model}_${VIDEO}_${1}_${QP}_${TS}.json" --model "path=/vmaf/model/${model}.json" &
#    pids+=($!)
#  done

#  ffmpeg -s 3840x2160 -pix_fmt yuv420p -i "/home/Projects/UVG/${VIDEO_MAP[$VIDEO]}" -s 3840x2160 -pix_fmt yuv420p -i /home/Projects/cavs2/WIP/WIP.yuv -lavfi "ssim=stats_file=/home/Projects/cavs2/results/set1/ssim_${VIDEO}_${1}_${QP}_${TS}.log" -f null - &
#  pids+=($!)
#  ffmpeg -s 3840x2160 -pix_fmt yuv420p -i "/home/Projects/UVG/${VIDEO_MAP[$VIDEO]}" -s 3840x2160 -pix_fmt yuv420p -i /home/Projects/cavs2/WIP/WIP.yuv -lavfi "psnr=stats_file=/home/Projects/cavs2/results/set1/psnr_${VIDEO}_${1}_${QP}_${TS}.log" -f null - &
#  pids+=($!)

#  ffmpeg -s 3840x2160 -pix_fmt yuv420p -i /home/Projects/cavs2/WIP/UPPER.yuv -s 3840x2160 -pix_fmt yuv420p -i /home/Projects/cavs2/WIP/WIP.yuv -lavfi "ssim=stats_file=/home/Projects/cavs2/results/set1/UPPER_ssim_${VIDEO}_${1}_${QP}_${TS}.log" -f null - &
#  pids+=($!)
#  ffmpeg -s 3840x2160 -pix_fmt yuv420p -i /home/Projects/cavs2/WIP/UPPER.yuv -s 3840x2160 -pix_fmt yuv420p -i /home/Projects/cavs2/WIP/WIP.yuv -lavfi "psnr=stats_file=/home/Projects/cavs2/results/set1/UPPER_psnr_${VIDEO}_${1}_${QP}_${TS}.log" -f null - &
#  pids+=($!)

#  wait "${pids[@]}"


}

#function case_eval_freerun() {
#  tmp_log=$(mktemp)
#
#  script -q -c "vmtouch -v -l /home/Projects/UVG/${VIDEO_MAP[$VIDEO]}" | tee "$tmp_log" &
#
#  while ! grep -q "LOCKED" "$tmp_log"; do
#    sleep 0.1
#  done
#
#  sudo nvidia-smi -rgc
#
#  python live_demo.py "${VIDEO}" -skp "${SKP}" -skd "${SKD}" -sth "${STH}" -stl "${STL}" --backend "eval" --log --comment "full"# --qp "${QP}"
#
#  sudo nvidia-smi -rgc
#
#  docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,video -v /home/Projects:/data -it --rm vmaf_cuda -r "/data/UVG/${VIDEO_MAP[$VIDEO]}" -d /data/cavs2/WIP/WIP.yuv -w 3840 -h 2160 -p 420 -b 8 --json -o "/data/cavs2/results/set1/${VIDEO}_full_${QP}_$(date +%s)".json
#
#  sudo pkill vmtouch
#  rm "$tmp_log"
#}

SKP=0.5
SKD=0.25
STA=33
STH=3
STL=0
QP=28
#VIDEO=Beauty

#case1
#case2
#case3
#case4

#case600
#case500
#case400
#case300
#case200
for VIDEO in "${!VIDEO_MAP[@]}"; do
#  LOCK THE FILE
  tmp_log=$(mktemp)

  script -q -c "vmtouch -v -l /home/Projects/UVG/${VIDEO_MAP[$VIDEO]}" | tee "$tmp_log" &

  while ! grep -q "LOCKED" "$tmp_log"; do
    sleep 0.1
  done

# PREPARE POST CODEC

if [[ $QP =~ ^-?[0-9]+$ ]]; then
  ffmpeg -y -f rawvideo -s 3840x2160 -pix_fmt yuv420p -i "/home/Projects/UVG/${VIDEO_MAP[$VIDEO]}" -frames:v 597 -vf "scale=960:540" -c:v hevc_nvenc -preset medium -rc constqp -qp "${QP}" -bf 0 -delay 0 -f hevc WIP/post-codec.hevc
  
#  ffmpeg -y -f rawvideo -s 3840x2160 -pix_fmt yuv420p -i "/home/Projects/UVG/${VIDEO_MAP[$VIDEO]}" -c:v hevc_nvenc -preset medium -rc constqp -qp "${QP}" -f hevc WIP/post-codec.hevc
fi

#echo "${VIDEO}"
#ffprobe -v quiet -select_streams v -show_entries packet=size -of compact=p=0:nk=1 WIP/post-codec.hevc | awk '{s+=$1} END {print s}' 



#  PREPARE UPPER
#  sudo nvidia-smi -rgc
#
#  python live_demo_generate_full_frame_sr.py "${VIDEO}" -sta "${STA}" -skp "${SKP}" -skd "${SKD}" -sth "${STH}" -stl "${STL}" --backend "eval" --log --comment "UPPER" --qp "${QP}"
#
#  sudo nvidia-smi -rgc
#
#  TS="$(date +%s)"
#
#  pids=()
#
#  for model in vmaf_4k_v0.6.1 vmaf_4k_v0.6.1neg vmaf_v0.6.1 vmaf_v0.6.1neg; do
#    docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,video -v /home/Projects:/data --rm vmaf_cuda -r "/data/UVG/${VIDEO_MAP[$VIDEO]}" -d /data/cavs2/WIP/UPPER.yuv -w 3840 -h 2160 -p 420 -b 8 --json -o "/data/cavs2/results/set1/vmaf_${model}_${VIDEO}_UPPER_${QP}_${TS}.json" --model "path=/vmaf/model/${model}.json" &
#    pids+=($!)
#  done
#
#  ffmpeg -s 3840x2160 -pix_fmt yuv420p -i "/home/Projects/UVG/${VIDEO_MAP[$VIDEO]}" -s 3840x2160 -pix_fmt yuv420p -i /home/Projects/cavs2/WIP/UPPER.yuv -lavfi "ssim=stats_file=/home/Projects/cavs2/results/set1/ssim_${VIDEO}_UPPER_${QP}_${TS}.log" -f null - &
#  pids+=($!)
#  ffmpeg -s 3840x2160 -pix_fmt yuv420p -i "/home/Projects/UVG/${VIDEO_MAP[$VIDEO]}" -s 3840x2160 -pix_fmt yuv420p -i /home/Projects/cavs2/WIP/UPPER.yuv -lavfi "psnr=stats_file=/home/Projects/cavs2/results/set1/psnr_${VIDEO}_UPPER_${QP}_${TS}.log" -f null - &
#  pids+=($!)
#
#  wait "${pids[@]}"


#  EXPERIMENTS

 # for freq in 400 800; do
 #for freq in 200 300 400 500 600 700 800 900 1000 full; do
# for freq in 200 400 600 800 1000 full; do
#for freq in 400 800 full; do
#for freq in 400; do
#for freq in 800; do
#for freq in lmh; do
#for freq in 200 300 400 500 600 800 full; do
#  case_eval $freq
  # done

#case4
case_eval "random"

  sudo pkill vmtouch
  rm "$tmp_log"
done

