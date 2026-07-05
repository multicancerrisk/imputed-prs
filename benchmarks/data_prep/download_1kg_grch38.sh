#!/usr/bin/env bash
# Download the 1000 Genomes NYGC high-coverage 30x **GRCh38** phased panel (3,202 samples).
#
# This is the production-representative reference for the scaling benchmark (closest to the
# All of Us GRCh38 high-coverage target). It replaces the GRCh37 Phase-3 data that the old
# analysis/prs313_evaluation/download_1kg.sh fetched.
#
# Usage:  benchmarks/data_prep/download_1kg_grch38.sh [TARGET_DIR] [CHROMS]
#   TARGET_DIR  default: benchmarks/data/1kg_grch38
#   CHROMS      default: 1..22  (e.g. "22 21 20" to fetch a subset first)
#
# ~20-30 GB total. Resumable (wget -c / curl -C -). .tbi indices are published and fetched.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${1:-${SCRIPT_DIR}/../data/1kg_grch38}"
CHROMS="${2:-$(seq 1 22)}"

BASE_URL="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV"
FILE_TMPL="1kGP_high_coverage_Illumina.chr%s.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"

mkdir -p "${TARGET_DIR}"
echo "Target dir : ${TARGET_DIR}"
echo "Source     : ${BASE_URL}"
echo "Chromosomes: ${CHROMS}"
echo

fetch() {  # fetch <url> <dest>  (resumable, skip if already complete)
  local url="$1" dest="$2"
  if [[ -s "${dest}" ]]; then
    echo "  exists, skipping: $(basename "${dest}")"
    return 0
  fi
  echo "  downloading: $(basename "${dest}")"
  if command -v wget >/dev/null 2>&1; then
    wget -c -q --show-progress -O "${dest}.part" "${url}" && mv "${dest}.part" "${dest}"
  elif command -v curl >/dev/null 2>&1; then
    curl -fL -C - -o "${dest}.part" "${url}" && mv "${dest}.part" "${dest}"
  else
    echo "ERROR: need wget or curl on PATH" >&2
    exit 1
  fi
}

for chr in ${CHROMS}; do
  fname="$(printf "${FILE_TMPL}" "${chr}")"
  echo "chr${chr}:"
  fetch "${BASE_URL}/${fname}"     "${TARGET_DIR}/${fname}"
  fetch "${BASE_URL}/${fname}.tbi" "${TARGET_DIR}/${fname}.tbi"
done

echo
echo "Done. Files in ${TARGET_DIR}:"
ls -lh "${TARGET_DIR}" | awk '{print "  "$5"\t"$9}'
echo
echo "Sanity check a sample count with:  bcftools query -l ${TARGET_DIR}/$(printf "${FILE_TMPL}" 22) | wc -l   # expect 3202"
