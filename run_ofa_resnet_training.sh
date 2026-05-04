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
# ── Single-node, cluster preset ──────────────────────────────────────────────
#   bash run_ofa_resnet_training.sh --cluster sysml
#   bash run_ofa_resnet_training.sh --cluster firefly --nproc_per_node 4
#
# ── Single-node, manual paths ────────────────────────────────────────────────
#   bash run_ofa_resnet_training.sh \
#       --imagenet_path /data/imagenet \
#       --checkpoint_dir /scratch/checkpoints
#
# ── Multi-node, cluster preset ───────────────────────────────────────────────
#   # Run on EVERY node; change --node_rank per node (0, 1, 2, …)
#   bash run_ofa_resnet_training.sh --cluster sysml \
#       --nnodes 2 --nproc_per_node 8 --node_rank 0   # master node
#   bash run_ofa_resnet_training.sh --cluster sysml \
#       --nnodes 2 --nproc_per_node 8 --node_rank 1   # worker node
#
# ── Multi-node, fully manual ─────────────────────────────────────────────────
#   bash run_ofa_resnet_training.sh \
#       --imagenet_path /data/imagenet \
#       --nnodes 2 --nproc_per_node 8 \
#       --master_addr 10.0.0.1 --node_rank 0
#
# Cluster presets live in cluster-configs.yaml next to this script.
# Explicit CLI flags always override cluster preset values.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Sentinel defaults (empty = not set yet) ───────────────────────────────────
CLUSTER=""
NPROC_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_PORT=29500
FORCE=0

# These three are filled by cluster config or explicit flags; empty = not yet set
IMAGENET_PATH=""
CHECKPOINT_DIR=""
MASTER_ADDR=""

# Track which were explicitly set by the user (overrides cluster config)
_USER_IMAGENET=0
_USER_CKPT=0
_USER_ADDR=0
_USER_NPROC=0
_USER_NNODES=0
_USER_RANK=0
_USER_PORT=0

# ── Argument parsing ─────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 [options]

Cluster presets (from cluster-configs.yaml):
  --cluster NAME          Load imagenet_path, checkpoint_dir, and master_addr
                          from cluster-configs.yaml for this cluster name.
                          Explicit flags below always override preset values.
                          Known clusters: firefly, sysml

Per-run overrides (these take precedence over --cluster):
  --imagenet_path PATH    ImageNet root (must contain train/ and val/)
  --checkpoint_dir DIR    Root dir for all checkpoints
  --nproc_per_node N      GPUs per node (default: 8)
  --nnodes N              Total number of nodes (default: 1)
  --node_rank N           Rank of this node, 0-based (default: 0)
  --master_addr ADDR      IP/hostname of rank-0 node (default: 127.0.0.1)
  --master_port PORT      Port for rendezvous (default: 29500)
  --force                 Re-run phases even if checkpoint already exists
  -h, --help              Show this message

Multi-node example:
  # master node (rank 0):
  bash $0 --cluster sysml --nnodes 2 --nproc_per_node 8 --node_rank 0
  # worker node (rank 1):
  bash $0 --cluster sysml --nnodes 2 --nproc_per_node 8 --node_rank 1
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster)        CLUSTER="$2";                           shift 2 ;;
        --imagenet_path)  IMAGENET_PATH="$2";  _USER_IMAGENET=1; shift 2 ;;
        --checkpoint_dir) CHECKPOINT_DIR="$2"; _USER_CKPT=1;     shift 2 ;;
        --nproc_per_node) NPROC_PER_NODE="$2"; _USER_NPROC=1;    shift 2 ;;
        --nnodes)         NNODES="$2";         _USER_NNODES=1;   shift 2 ;;
        --node_rank)      NODE_RANK="$2";      _USER_RANK=1;     shift 2 ;;
        --master_addr)    MASTER_ADDR="$2";    _USER_ADDR=1;     shift 2 ;;
        --master_port)    MASTER_PORT="$2";    _USER_PORT=1;     shift 2 ;;
        --force)          FORCE=1;                                shift   ;;
        -h|--help)        usage ;;
        *) echo "ERROR: Unknown argument: $1"; usage ;;
    esac
done

# ── Load cluster preset ───────────────────────────────────────────────────────
# Uses Python (already required) to parse cluster-configs.yaml.
# Only fills in values that were not set by explicit CLI flags.
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

# ── Apply built-in fallback defaults for anything still unset ─────────────────
[[ -z "$IMAGENET_PATH"  ]] && IMAGENET_PATH="/coc/data/datasets/ImageNet"
[[ -z "$CHECKPOINT_DIR" ]] && CHECKPOINT_DIR="/coc/scratch/dgarg/ofa_checkpoints"
[[ -z "$MASTER_ADDR"    ]] && MASTER_ADDR="127.0.0.1"

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ ! -d "$IMAGENET_PATH" ]]; then
    echo "ERROR: ImageNet path does not exist: $IMAGENET_PATH"
    [[ -n "$CLUSTER" ]] && echo "       (loaded from cluster '$CLUSTER'; override with --imagenet_path)"
    exit 1
fi

if [[ $NNODES -gt 1 && "$MASTER_ADDR" == "127.0.0.1" ]]; then
    echo "WARNING: --nnodes=$NNODES but master_addr is still 127.0.0.1."
    echo "         Set master_addr in cluster-configs.yaml for cluster '$CLUSTER'"
    echo "         or pass --master_addr explicitly."
fi

# ── Setup ─────────────────────────────────────────────────────────────────────
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
TOTAL_GPUS=$(( NNODES * NPROC_PER_NODE ))

# ── Helpers ───────────────────────────────────────────────────────────────────
fmt_duration() {
    local S=$1
    printf "%dh %02dm %02ds" $(( S/3600 )) $(( (S%3600)/60 )) $(( S%60 ))
}

PHASE_RAN=0

run_phase() {
    local TASK="$1"
    local PHASE="$2"
    local INPUT_CKPT="$3"   # empty string = no checkpoint arg passed
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

    # Build the checkpoint arg only when the file actually exists
    local CKPT_ARG=""
    if [[ -n "$INPUT_CKPT" && -f "$INPUT_CKPT" ]]; then
        CKPT_ARG="--ofa_checkpoint_path $INPUT_CKPT"
    fi

    # Pass --resume only when this phase already has a partial checkpoint
    local RESUME_ARG=""
    if [[ -f "$PARTIAL_CKPT" ]]; then
        RESUME_ARG="--resume"
        echo "  INFO  partial checkpoint found, resuming from epoch in $PARTIAL_CKPT"
    fi

    torchrun \
        --nproc_per_node="$NPROC_PER_NODE" \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        "$SCRIPT_DIR/train_ofa_resnet.py" \
            --task          "$TASK" \
            --phase         "$PHASE" \
            --imagenet_path "$IMAGENET_PATH" \
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

# ── Configuration summary ─────────────────────────────────────────────────────
_src_label() {
    if   [[ "$1" -eq 1 ]];       then echo "cli flag"
    elif [[ -n "$2" ]];          then echo "cluster: $2"
    else                              echo "default"
    fi
}

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  OFA-ResNet50 training pipeline"
[[ -n "$CLUSTER" ]] && echo "  Cluster preset   : $CLUSTER  (cluster-configs.yaml)"
echo "────────────────────────────────────────────────────────────────────"
printf "  %-18s %s  [%s]\n" "imagenet_path:"   "$IMAGENET_PATH"   "$(_src_label $_USER_IMAGENET "$CLUSTER")"
printf "  %-18s %s  [%s]\n" "checkpoint_dir:"  "$CHECKPOINT_DIR"  "$(_src_label $_USER_CKPT     "$CLUSTER")"
printf "  %-18s %s  [%s]\n" "master_addr:"     "$MASTER_ADDR"     "$(_src_label $_USER_ADDR     "$CLUSTER")"
printf "  %-18s %s  [%s]\n" "master_port:"     "$MASTER_PORT"     "$(_src_label $_USER_PORT     '')"
printf "  %-18s %s  [%s]\n" "nproc_per_node:"  "$NPROC_PER_NODE"  "$(_src_label $_USER_NPROC    '')"
printf "  %-18s %s  [%s]\n" "nnodes:"          "$NNODES"          "$(_src_label $_USER_NNODES   '')"
printf "  %-18s %s  [%s]\n" "node_rank:"       "$NODE_RANK"       "$(_src_label $_USER_RANK     '')"
printf "  %-18s %s\n"       "total_gpus:"      "$TOTAL_GPUS  (${NNODES} node(s) × ${NPROC_PER_NODE} GPU(s))"
printf "  %-18s %s\n"       "force:"           "$( [[ $FORCE -eq 1 ]] && echo yes || echo no )"
echo "════════════════════════════════════════════════════════════════════"

# ── Checkpoint chain ─────────────────────────────────────────────────────────
E1="$CHECKPOINT_DIR/expand/phase1/checkpoint/model_best.pth.tar"
E2="$CHECKPOINT_DIR/expand/phase2/checkpoint/model_best.pth.tar"
W1="$CHECKPOINT_DIR/width/phase1/checkpoint/model_best.pth.tar"
W2="$CHECKPOINT_DIR/width/phase2/checkpoint/model_best.pth.tar"
D1="$CHECKPOINT_DIR/depth/phase1/checkpoint/model_best.pth.tar"

# ── Run all phases ───────────────────────────────────────────────────────────
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
