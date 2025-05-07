#!/bin/bash
used_vram=0
total_vram=0

# Get the list of PIDs and their VRAM usage
pids_and_vram=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
# Iterate over each line of the output
while IFS=, read -r pid used_memory; do
  #Sum up the VRAM usage
  used_vram=$((used_vram + used_memory))
done <<< "$pids_and_vram"

all_vram=$(nvidia-smi --query-gpu=memory.total,memory.used,memory.free,memory.reserved --format=csv,noheader,nounits)
while IFS=, read -r total used free reserved; do
  total_vram=$((total_vram + total))
  total_usable_vram=$((total_vram - reserved))
  done <<< "$all_vram"

echo "$used_vram / $total_vram ($total_usable_vram usable)"
