#!/usr/bin/env bash
# Multi-domain physical-AI corpora with VISION-INDEPENDENT truth.
#
# Why these: the encoder ceiling measured on the robot-arm sim (action
# AUC 0.721 -> yield ~0.49) is only interesting if it is a PHYSICAL AI
# ceiling rather than an arm artifact. That needs other embodiments with
# truth that does not come from the camera, so events can be derived the
# way Oxford's are (INS stop/start/turn), with no circularity.
#
#   car    KITTI raw  - per-frame OXTS GPS/IMU (lat/lon, vel, yaw rate)
#   drone  see notes at the bottom; most aerial sets are form-gated
#
# Nothing here needs registration. Run sections independently.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/data"

# ---------------------------------------------------------------- car
# KITTI raw. _sync zips bundle Velodyne, so budget ~4 MB per frame; a
# full hour would be ~144 GB. This is a diverse SUBSET - city, road and
# residential - which is what a domain-generality test needs, not more
# minutes of the same street.
KITTI_BASE="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
KITTI_DAY="2011_09_26"
KITTI_DRIVES=(0001 0009 0011 0013 0015 0017 0018 0027 0028 0029 0032 0035 0036 0039 0046 0051 0056 0059 0064 0084 0091 0096)

get_kitti() {
  mkdir -p "${OUT}/kitti"
  curl -fL --retry 3 -C - -o "${OUT}/kitti/${KITTI_DAY}_calib.zip" \
    "${KITTI_BASE}/${KITTI_DAY}_calib.zip" || true
  for d in "${KITTI_DRIVES[@]}"; do
    f="${KITTI_DAY}_drive_${d}_sync.zip"
    [ -s "${OUT}/kitti/${f}" ] && { echo "have ${f}"; continue; }
    echo "=> ${f}"
    curl -fL --retry 3 -C - -o "${OUT}/kitti/${f}" \
      "${KITTI_BASE}/${KITTI_DAY}_drive_${d}/${f}" \
      || echo "MISS ${f}"
  done
  ( cd "${OUT}/kitti" && for z in *.zip; do
      unzip -n -q "$z" && echo "unzipped $z"; done ) || true
  echo "car frames: $(find "${OUT}/kitti" -name '*.png' -path '*image_02*' | wc -l)"
}

# -------------------------------------------------------------- drone
# Aerial sets with truth are mostly behind forms or Google Drive, so
# these are attempted and reported rather than assumed. See the README
# note printed at the end for the ones needing a human.
get_drone() {
  mkdir -p "${OUT}/drone"
  # All direct HTTP, no registration. Truth is vision-independent in
  # every case (GPS/IMU or laser-tracker pose), which is what lets
  # events be derived without the camera - the same discipline used for
  # Oxford's INS boundaries.
  #
  #   AGZ  Air-Ground Zurich: ~2 km flight over a city, 81k frames with
  #        GPS + IMU. The aerial analogue of Oxford, and the one that
  #        matters most for a domain-generality claim: OUTDOOR, moving
  #        camera, real scene.
  #   UZH-FPV  aggressive flight, laser-tracker ground-truth pose.
  #   racing   fast indoor flight.
  D=(
    "agz_subset.zip|https://download.ifi.uzh.ch/rpg/AGZ_data/AGZ_subset.zip"
    "uzhfpv_outdoor_forward_1.zip|http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/outdoor_forward_1_snapdragon_with_gt.zip"
    "uzhfpv_indoor_forward_3.zip|http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_3_snapdragon_with_gt.zip"
    "rpg_race_1.zip|https://download.ifi.uzh.ch/rpg/drone_racing_data/race_1.zip"
  )
  for e in "${D[@]}"; do
    f="${e%%|*}"; u="${e#*|}"
    [ -s "${OUT}/drone/${f}" ] && { echo "have ${f}"; continue; }
    echo "=> ${f}"
    curl -fL --retry 3 -C - -o "${OUT}/drone/${f}" "$u" || echo "MISS ${f}"
  done
  ls -lh "${OUT}/drone" 2>/dev/null
}

# AGZ full is 29.8 GB - the whole 45-minute flight. Separate target so
# it is an explicit choice, not a surprise.
get_drone_full() {
  mkdir -p "${OUT}/drone"
  curl -fL --retry 3 -C - -o "${OUT}/drone/AGZ.zip" \
    "https://download.ifi.uzh.ch/rpg/AGZ_data/AGZ.zip"
}

case "${1:-all}" in
  car) get_kitti ;;
  drone) get_drone ;;
  drone-full) get_drone_full ;;
  *) get_kitti; get_drone ;;
esac
