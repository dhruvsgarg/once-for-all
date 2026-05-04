#!/usr/bin/env bash
# Orchestrates OFA-ResNet50 training.  Two modes:
#
# MODE A ─ Full supernet (progressive-shrinking, 6 phases)
#   expand/phase1 → expand/phase2 → width/phase1 → width/phase2
#   → depth/phase1 → depth/phase2
#   Completed phases are skipped automatically.  Re-run with --force.
#
# MODE B ─ Subnet fine-tuning (--train_subnets)
#   Fine-tunes 6 specific subnets from a base ResNet50D checkpoint.
#   No progressive-shrinking stages required.
#   Outputs go to --subnet_out_dir (default: /coc/scratch/dgarg/finetuned_subnets),
#   completely separate from the supernet checkpoint tree.
#
# ── Supernet training ─────────────────────────────────────────────────────────
#   bash run_ofa_resnet_training.sh --cluster firefly
#   bash run_ofa_resnet_training.sh --cluster firefly --nproc_per_node 4
#   bash run_ofa_resnet_training.sh \
#       --imagenet_path /data/imagenet \
#       --checkpoint_dir /scratch/ofa_checkpoints \
#       --nproc_per_node 4
#
# ── Subnet fine-tuning ────────────────────────────────────────────────────────
#   bash run_ofa_resnet_training.sh --train_subnets \
#       --pretrained_ckpt  /coc/scratch/dgarg/resnet50d_base.pth.tar \
#       --subnet_config_json latency_curves_supernet_resnet_A40_with_stages_29apr26.json \
#       --subnet_out_dir   /coc/scratch/dgarg/finetuned_subnets \
#       --nproc_per_node   4
#
#   Optional fine-tuning knobs (shown with defaults):
#       --subnet_epochs 30
#       --subnet_lr     2.5e-3
#
# Cluster presets live in cluster-configs.yaml next to this script.
# Explicit CLI flags always override cluster preset values.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Mode flag ────────────────────────────────────────────────────────────────
TRAIN_SUBNETS=0

# ── Common defaults ───────────────────────────────────────────────────────────
CLUSTER=""
NPROC_PER_NODE=8
MASTER_PORT=29500
FORCE=0

IMAGENET_PATH=""
CHECKPOINT_DIR=""
MASTER_ADDR=""

# ── Subnet-mode defaults ──────────────────────────────────────────────────────
PRETRAINED_CKPT=""     # INPUT base ResNet50D checkpoint (read-only)
SUBNET_CONFIG_JSON=""
SUBNET_OUT_DIR=""
SUBNET_EPOCHS=30
SUBNET_LR="2.5e-3"

# ── Override-tracking (so cluster preset doesn't clobber explicit flags) ──────
_USER_IMAGENET=0
_USER_CKPT=0
_USER_ADDR=0
_USER_NPROC=0
_USER_PORT=0

# ── Argument parsing ─────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 [options]

MODE A – Supernet training (default):
  --cluster NAME          Load imagenet_path, checkpoint_dir, master_addr from
                          cluster-configs.yaml.  Known clusters: firefly, sysml
  --imagenet_path PATH    ImageNet root (must contain train/ and val/)
  --checkpoint_dir DIR    Root dir for supernet checkpoints
  --nproc_per_node N      GPUs to use (default: 8)
  --master_port PORT      torchrun rendezvous port (default: 29500)
  --force                 Re-run phases even if checkpoint already exists

MODE B – Subnet fine-tuning:
  --train_subnets                   Enable subnet fine-tuning mode
  --pretrained_ckpt PATH            INPUT base ResNet50D checkpoint (required)
  --subnet_config_json PATH         Subnet config JSON (required)
  --subnet_out_dir DIR              Output dir for fine-tuned subnets
                                    (default: /coc/scratch/dgarg/finetuned_subnets)
  --subnet_epochs N                 Epochs per subnet (default: 30)
  --subnet_lr F                     Base LR per GPU (default: 2.5e-3)
  --nproc_per_node N                GPUs to use (default: 8)
  --imagenet_path PATH              ImageNet root (required for data loading)

  -h, --help              Show this message
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train_subnets)      TRAIN_SUBNETS=1;                                  shift   ;;
        --cluster)            CLUSTER="$2";                                     shift 2 ;;
        --imagenet_path)      IMAGENET_PATH="$2";   _USER_IMAGENET=1;          shift 2 ;;
        --checkpoint_dir)     CHECKPOINT_DIR="$2";  _USER_CKPT=1;              shift 2 ;;
        --nproc_per_node)     NPROC_PER_NODE="$2";  _USER_NPROC=1;             shift 2 ;;
        --master_addr)        MASTER_ADDR="$2";     _USER_ADDR=1;              shift 2 ;;
        --master_port)        MASTER_PORT="$2";     _USER_PORT=1;              shift 2 ;;
        --pretrained_ckpt)    PRETRAINED_CKPT="$2";                            shift 2 ;;
        --subnet_config_json) SUBNET_CONFIG_JSON="$2";                         shift 2 ;;
        --subnet_out_dir)     SUBNET_OUT_DIR="$2";                             shift 2 ;;
        --subnet_epochs)      SUBNET_EPOCHS="$2";                              shift 2 ;;
        --subnet_lr)          SUBNET_LR="$2";                                  shift 2 ;;
        --force)              FORCE=1;                                          shift   ;;
        -h|--help)            usage ;;
        *) echo "ERROR: Unknown argument: $1"; usage ;;
    esac
done

# ── Load cluster preset ───────────────────────────────────────────────────────
_cfg_get() {
    local key="$1"
    python3 - <<PYEOF
import sys, os
try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed — run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)
cfg_path = os.path.join('${SCRIPT_DIR}', 'cluster-configs.yaml')
try:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
except FileNotFoundError:
    print(f"WARNING: cluster-configs.yaml not found at {cfg_path}", file=sys.stderr)
    sys.exit(0)
cluster = '${CLUSTER}'
if cluster not in cfg:
    print(f"ERROR: Cluster '{cluster}' not found in cluster-configs.yaml.", file=sys.stderr)
    print(f"       Known clusters: {', '.join(cfg.keys())}", file=sys.stderr)
    sys.exit(1)
val = cfg[cluster].get('${key}', '')
print(val if val else '', end='')
PYEOF
}

if [[ -n "$CLUSTER" ]]; then
    [[ $_USER_IMAGENET -eq 0 ]] && IMAGENET_PATH="$(_cfg_get imagenet_path)"
    [[ $_USER_CKPT     -eq 0 ]] && CHECKPOINT_DIR="$(_cfg_get checkpoint_dir)"
    [[ $_USER_ADDR     -eq 0 ]] && MASTER_ADDR="$(_cfg_get master_addr)"
fi

# ── Apply fallback defaults ───────────────────────────────────────────────────
[[ -z "$IMAGENET_PATH"  ]] && IMAGENET_PATH="/coc/data/datasets/ImageNet"
[[ -z "$CHECKPOINT_DIR" ]] && CHECKPOINT_DIR="/coc/scratch/dgarg/ofa_checkpoints"
[[ -z "$MASTER_ADDR"    ]] && MASTER_ADDR="127.0.0.1"
[[ -z "$SUBNET_OUT_DIR" ]] && SUBNET_OUT_DIR="/coc/scratch/dgarg/finetuned_subnets"

# ── Validate common paths ─────────────────────────────────────────────────────
if [[ ! -d "$IMAGENET_PATH" ]]; then
    echo "ERROR: ImageNet path does not exist: $IMAGENET_PATH"
    [[ -n "$CLUSTER" ]] && echo "       (loaded from cluster '$CLUSTER'; override with --imagenet_path)"
    exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
TOTAL_START=$SECONDS

fmt_duration() {
    local S=$1
    printf "%dh %02dm %02ds" $(( S/3600 )) $(( (S%3600)/60 )) $(( S%60 ))
}

_src_label() {
    if   [[ "$1" -eq 1 ]]; then echo "cli flag"
    elif [[ -n "$2" ]];    then echo "cluster: $2"
    else                        echo "default"
    fi
}

# ── Configuration summary ─────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════"
if [[ $TRAIN_SUBNETS -eq 1 ]]; then
    echo "  OFA-ResNet50  ─  subnet fine-tuning mode"
else
    echo "  OFA-ResNet50  ─  supernet training mode"
fi
[[ -n "$CLUSTER" ]] && echo "  Cluster preset   : $CLUSTER  (cluster-configs.yaml)"
echo "────────────────────────────────────────────────────────────────────"
printf "  %-22s %s  [%s]\n" "imagenet_path:"    "$IMAGENET_PATH"   "$(_src_label $_USER_IMAGENET "$CLUSTER")"
printf "  %-22s %s  [%s]\n" "nproc_per_node:"   "$NPROC_PER_NODE"  "$(_src_label $_USER_NPROC '')"

if [[ $TRAIN_SUBNETS -eq 1 ]]; then
    printf "  %-22s %s\n" "pretrained_ckpt:"  "$PRETRAINED_CKPT"
    printf "  %-22s %s\n" "subnet_config_json:" "$SUBNET_CONFIG_JSON"
    printf "  %-22s %s\n" "subnet_out_dir:"   "$SUBNET_OUT_DIR"
    printf "  %-22s %s\n" "subnet_epochs:"    "$SUBNET_EPOCHS"
    printf "  %-22s %s\n" "subnet_lr:"        "$SUBNET_LR"
else
    printf "  %-22s %s  [%s]\n" "checkpoint_dir:"   "$CHECKPOINT_DIR"  "$(_src_label $_USER_CKPT "$CLUSTER")"
    printf "  %-22s %s\n"       "force:"            "$( [[ $FORCE -eq 1 ]] && echo yes || echo no )"
fi
echo "════════════════════════════════════════════════════════════════════"

# ═════════════════════════════════════════════════════════════════════════════
# MODE B — Subnet fine-tuning
# ═════════════════════════════════════════════════════════════════════════════

run_subnet_finetuning() {
    if [[ -z "$PRETRAINED_CKPT" ]]; then
        echo "ERROR: --pretrained_ckpt is required in subnet fine-tuning mode."
        echo "       Provide the path to your base ResNet50D checkpoint (.pth.tar)."
        exit 1
    fi
    if [[ ! -f "$PRETRAINED_CKPT" ]]; then
        echo "ERROR: Pretrained checkpoint not found: $PRETRAINED_CKPT"
        exit 1
    fi
    if [[ -z "$SUBNET_CONFIG_JSON" ]]; then
        echo "ERROR: --subnet_config_json is required in subnet fine-tuning mode."
        exit 1
    fi
    if [[ ! -f "$SUBNET_CONFIG_JSON" ]]; then
        echo "ERROR: Subnet config JSON not found: $SUBNET_CONFIG_JSON"
        exit 1
    fi

    mkdir -p "$SUBNET_OUT_DIR"

    local START=$SECONDS
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  START  subnet fine-tuning  │  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    torchrun \
        --nproc_per_node="$NPROC_PER_NODE" \
        --master_port="$MASTER_PORT" \
        "$SCRIPT_DIR/train_ofa_resnet.py" \
            --train_subnets \
            --ofa_checkpoint_path "$PRETRAINED_CKPT" \
            --subnet_config_json  "$SUBNET_CONFIG_JSON" \
            --subnet_out_dir      "$SUBNET_OUT_DIR" \
            --subnet_epochs       "$SUBNET_EPOCHS" \
            --subnet_lr           "$SUBNET_LR" \
            --imagenet_path       "$IMAGENET_PATH"

    local ELAPSED=$(( SECONDS - START ))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  DONE   subnet fine-tuning  │  $(fmt_duration $ELAPSED)  │  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "         fine-tuned subnets → $SUBNET_OUT_DIR/subnet_<id>/subnet_<id>_final.pth.tar"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ═════════════════════════════════════════════════════════════════════════════
# MODE A — Supernet progressive-shrinking
# ═════════════════════════════════════════════════════════════════════════════

PHASE_RAN=0

run_phase() {
    local TASK="$1"
    local PHASE="$2"
    local INPUT_CKPT="$3"   # empty string = no --ofa_checkpoint_path passed
    local OUT_CKPT="$CHECKPOINT_DIR/$TASK/phase$PHASE/checkpoint/model_best.pth.tar"
    local PARTIAL_CKPT="$CHECKPOINT_DIR/$TASK/phase$PHASE/checkpoint/checkpoint.pth.tar"
    local LABEL="$TASK / phase $PHASE"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

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

    local CKPT_ARG=""
    if [[ -n "$INPUT_CKPT" && -f "$INPUT_CKPT" ]]; then
        CKPT_ARG="--ofa_checkpoint_path $INPUT_CKPT"
    fi

    local RESUME_ARG=""
    if [[ -f "$PARTIAL_CKPT" ]]; then
        RESUME_ARG="--resume"
        echo "  INFO  partial checkpoint found, resuming from epoch in $PARTIAL_CKPT"
    fi

    torchrun \
        --nproc_per_node="$NPROC_PER_NODE" \
        --master_port="$MASTER_PORT" \
        "$SCRIPT_DIR/train_ofa_resnet.py" \
            --task                "$TASK" \
            --phase               "$PHASE" \
            --imagenet_path       "$IMAGENET_PATH" \
            --base_checkpoint_dir "$CHECKPOINT_DIR" \
            $CKPT_ARG \
            $RESUME_ARG

    local ELAPSED=$(( SECONDS - PHASE_START ))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  DONE   $LABEL  │  $(fmt_duration $ELAPSED)  │  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "         checkpoint → $OUT_CKPT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    PHASE_RAN=1
}

pause_between_phases() {
    if [[ $PHASE_RAN -eq 1 ]]; then
        echo ""
        echo "  ↳ Waiting 20s before next phase ..."
        sleep 20
    fi
}

run_supernet() {
    mkdir -p "$CHECKPOINT_DIR"

    PRETRAINED="$CHECKPOINT_DIR/resnet50d_pretrained.pth.tar"
    if [[ ! -f "$PRETRAINED" ]]; then
        echo "────────────────────────────────────────────────────────────────────"
        echo "  WARNING: No pretrained checkpoint found at:"
        echo "    $PRETRAINED"
        echo "  expand/phase1 will start from random init."
        echo "  To seed from the released OFA-ResNet50 weights, run:"
        echo "    python - <<'PY'"
        echo "    from ofa.model_zoo import ofa_net; import torch"
        echo "    net = ofa_net('ofa_resnet50', pretrained=True)"
        echo "    torch.save({'state_dict': net.state_dict()}, '$PRETRAINED')"
        echo "    PY"
        echo "────────────────────────────────────────────────────────────────────"
    fi

    # Checkpoint chain passed as INPUT to each phase
    local E1="$CHECKPOINT_DIR/expand/phase1/checkpoint/model_best.pth.tar"
    local E2="$CHECKPOINT_DIR/expand/phase2/checkpoint/model_best.pth.tar"
    local W1="$CHECKPOINT_DIR/width/phase1/checkpoint/model_best.pth.tar"
    local W2="$CHECKPOINT_DIR/width/phase2/checkpoint/model_best.pth.tar"
    local D1="$CHECKPOINT_DIR/depth/phase1/checkpoint/model_best.pth.tar"

    run_phase  expand  1  "$PRETRAINED";  pause_between_phases
    run_phase  expand  2  "$E1";          pause_between_phases
    run_phase  width   1  "$E2";          pause_between_phases
    run_phase  width   2  "$W1";          pause_between_phases
    run_phase  depth   1  "$W2";          pause_between_phases
    run_phase  depth   2  "$D1"           # no pause after last phase

    local TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  ALL STAGES COMPLETE"
    echo "  Total time      : $(fmt_duration $TOTAL_ELAPSED)"
    echo "  Final supernet  : $CHECKPOINT_DIR/depth/phase2/checkpoint/model_best.pth.tar"
    echo "════════════════════════════════════════════════════════════════════"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
if [[ $TRAIN_SUBNETS -eq 1 ]]; then
    run_subnet_finetuning
    TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Total time : $(fmt_duration $TOTAL_ELAPSED)"
    echo "════════════════════════════════════════════════════════════════════"
else
    run_supernet
fi
