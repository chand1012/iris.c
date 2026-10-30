#!/bin/sh
# Qwen-Image 2.1 parity test. Builds the CPU parity harnesses and compares
# transformer text2img latents and editing step-0 velocity against reference
# dumps produced by debug/qwen21_reference.py (the official diffusers
# pipeline). Exits non-zero if either check fails.
#
# Environment overrides:
#   QW21_TF       transformer dir        (default qwen-image-2.1/transformer)
#   QW21_REF_T2I  t2i dump dir           (default /tmp/qw21_ref40_512)
#   QW21_REF_EDIT edit dump dir          (default /tmp/qw21_edit_ref)
set -e
cd "$(dirname "$0")/.."
TF=${QW21_TF:-qwen-image-2.1/transformer}
T2I=${QW21_REF_T2I:-/tmp/qw21_ref40_512}
EDIT=${QW21_REF_EDIT:-/tmp/qw21_edit_ref}
CC=${CC:-cc}
CFLAGS="-O2 -I. -DUSE_BLAS -DACCELERATE_NEW_LAPACK -DIRIS_USE_ACCELERATE"
LIBS="-framework Accelerate -lm"
UNAME=$(uname)
if [ "$UNAME" = Darwin ]; then
    :
else
    echo "test-qwen21: CPU BLAS harnesses currently target macOS/Accelerate" >&2
    exit 77
fi
for d in "$T2I" "$EDIT"; do
    if [ ! -f "$d/emb.bin" ]; then
        echo "missing reference dump in $d" >&2
        echo "generate with: flux_env/bin/python debug/qwen21_reference.py ..." >&2
        exit 77
    fi
done
rm -rf /tmp/qw21_parity_build && mkdir -p /tmp/qw21_parity_build
$CC $CFLAGS -o /tmp/qw21_parity_build/t2i  debug/qw21_parity_t2i.c  iris_transformer_qwen21.c iris_safetensors.c $LIBS
$CC $CFLAGS -o /tmp/qw21_parity_build/edit debug/qw21_parity_edit.c iris_transformer_qwen21.c iris_safetensors.c $LIBS
/tmp/qw21_parity_build/t2i  "$TF" "$T2I"
/tmp/qw21_parity_build/edit "$TF" "$EDIT"
echo "test-qwen21: all checks passed"
