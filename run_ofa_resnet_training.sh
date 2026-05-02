#!/usr/bin/env bash
# Orchestrates the full OFA-ResNet50 supernet training pipeline.
#
# Stages (run in order):
#   expand/phase1  →  expand/phase2
#   width/phase1   →  width/phase2
#   depth/phase1   →  depth/phase2
#
# Completed phases are skipped automatically (checkpoint already exists).
# Re-run with --force to overwrite.
#
# Usage:
#   bash run_ofa_resnet_training.sh --imagenet_path /path/to/imagenet
#   bash run_ofa_resnet_training.sh --imagenet_path /data/imagenet \
#       --nproc_per_node 4 \
#       --checkpoint_dir /coc/scratch/dgarg/ofa_checkpoints

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
NPROC_PER_NODE=8
CHECKPOINT_DIR="/coc/scratch/dgarg/ofa_checkpoints"
IMAGENET_PATH=""
FORCE=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Argument parsing ─────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 --imagenet_path PATH [options]

Required:
  --imagenet_path PATH    ImageNet root (must contain train/ and val/)

Options:
  --nproc_per_node N      GPUs to use per node (default: 8)
  --checkpoint_dir DIR    Root dir for all checkpoints
                          (default: /coc/scratch/dgarg/ofa_checkpoints)
  --force                 Re-run phases even if their checkpoint already exists
  -h, --help              Show this message
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nproc_per_node)  NPROC_PER_NODE="$2"; shift 2 ;;
        --checkpoint_dir)  CHECKPOINT_DIR="$2";  shift 2 ;;
        --imagenet_path)   IMAGENET_PATH="$2";   shift 2 ;;
        --force)           FORCE=1;              shift   ;;
        -h|--help)         usage ;;
        *) echo "ERROR: Unknown argument: $1"; usage ;;
    esac
done

if [[ -z "$IMAGENET_PATH" ]]; then
    echo "ERROR: --imagenet_path is required"
    usage
fi

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$CHECKPOINT_DIR"

PRETRAINED="$CHECKPOINT_DIR/resnet50d_pretrained.pth.tar"

if [[ ! -f "$PRETRAINED" ]]; then
    echo "────────────────────────────────────────────────────────────────────"
    echo "  WARNING: No pretrained checkpoint found at:"
    echo "    $PRETRAINED"
    echo "  expand/phase1 will start from random init."
    echo "  To use the released OFA-ResNet50 weights as the seed, run:"
    echo "    python - <<'PY'"
    echo "    from ofa.model_zoo import ofa_net; import torch"
    echo "    net = ofa_net('ofa_resnet50', pretrained=True)"
    echo "    torch.save({'state_dict': net.state_dict()}, '$PRETRAINED')"
    echo "    PY"
    echo "────────────────────────────────────────────────────────────────────"
fi

TOTAL_START=$SECONDS

# ── Helpers ──────────────────────────────────────────────────────────────────
fmt_duration() {
    local S=$1
    printf "%dh %02dm %02ds" $(( S/3600 )) $(( (S%3600)/60 )) $(( S%60 ))
}

PHASE_RAN=0   # set to 1 inside run_phase when a phase actually trains

run_phase() {
    local TASK="$1"
    local PHASE="$2"
    local INPUT_CKPT="$3"   # empty string = no checkpoint arg passed
    local OUT_CKPT="$CHECKPOINT_DIR/$TASK/phase$PHASE/checkpoint/model_best.pth.tar"
    local LABEL="$TASK / phase $PHASE"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── Skip if already done ─────────────────────────────────────────────────
    if [[ $FORCE -eq 0 && -f "$OUT_CKPT" ]]; then
        echo "  SKIP  $LABEL"
        echo "        checkpoint exists → $OUT_CKPT"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        PHASE_RAN=0
        return 0
    fi

    echo "  START  $LABEL  │  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local PHASE_START=$SECONDS

    # Build the checkpoint arg only when the file actually exists
    local CKPT_ARG=""
    if [[ -n "$INPUT_CKPT" && -f "$INPUT_CKPT" ]]; then
        CKPT_ARG="--ofa_checkpoint_path $INPUT_CKPT"
    fi

    torchrun --nproc_per_node="$NPROC_PER_NODE" \
        "$SCRIPT_DIR/train_ofa_resnet.py" \
        --task          "$TASK" \
        --phase         "$PHASE" \
        --imagenet_path "$IMAGENET_PATH" \
        --base_checkpoint_dir "$CHECKPOINT_DIR" \
        $CKPT_ARG

    local ELAPSED=$(( SECONDS - PHASE_START ))

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  DONE   $LABEL  │  $(fmt_duration $ELAPSED)  │  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "         checkpoint → $OUT_CKPT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    PHASE_RAN=1
}

pause_between_phases() {
    # Only pause when a phase actually ran (not when it was skipped)
    if [[ $PHASE_RAN -eq 1 ]]; then
        echo ""
        echo "  ↳ Waiting 20s before next phase ..."
        sleep 20
    fi
}

# ── Checkpoint chain ─────────────────────────────────────────────────────────
E1="$CHECKPOINT_DIR/expand/phase1/checkpoint/model_best.pth.tar"
E2="$CHECKPOINT_DIR/expand/phase2/checkpoint/model_best.pth.tar"
W1="$CHECKPOINT_DIR/width/phase1/checkpoint/model_best.pth.tar"
W2="$CHECKPOINT_DIR/width/phase2/checkpoint/model_best.pth.tar"
D1="$CHECKPOINT_DIR/depth/phase1/checkpoint/model_best.pth.tar"

# ── Run all phases ───────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  OFA-ResNet50 training pipeline"
echo "  checkpoint dir : $CHECKPOINT_DIR"
echo "  imagenet       : $IMAGENET_PATH"
echo "  GPUs per node  : $NPROC_PER_NODE"
echo "  force re-run   : $( [[ $FORCE -eq 1 ]] && echo yes || echo no )"
echo "════════════════════════════════════════════════════════════════════"

run_phase  expand  1  "$PRETRAINED";  pause_between_phases
run_phase  expand  2  "$E1";          pause_between_phases
run_phase  width   1  "$E2";          pause_between_phases
run_phase  width   2  "$W1";          pause_between_phases
run_phase  depth   1  "$W2";          pause_between_phases
run_phase  depth   2  "$D1"           # no pause after the last phase

# ── Summary ──────────────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ALL STAGES COMPLETE"
echo "  Total time      : $(fmt_duration $TOTAL_ELAPSED)"
echo "  Final supernet  : $CHECKPOINT_DIR/depth/phase2/checkpoint/model_best.pth.tar"
echo "════════════════════════════════════════════════════════════════════"
