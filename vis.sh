
# CUDA_VISIBLE_DEVICES=2 \
# python visualize_attention.py \
#     --patch_size 8 \
#     --pretrained_weights pretrained/mit_b5.pth \
#     --image_path /home/aghosh/Projects/2PCNet/Datasets/cityscapes/leftImg8bit/train/aachen/aachen_000003_000019_leftImg8bit.png \
#     --output_dir .


function visualize_attention() {
    local exp_id=$1
    local attn_stage=$2
    local ref_index_end=$3
    local weights_pattern="work_dirs/local-exp${exp_id}/*/best_mIoU_iter_*.pth"
    local pretrained_weights=( $weights_pattern )

    # Check if weights files are found
    if [ -z "${pretrained_weights}" ]; then
        echo "No pretrained weights found with pattern: $weights_pattern"
        return 1
    fi

    # Now use the first matching file
    echo "Using pretrained weights: ${pretrained_weights[0]}"

    CUDA_VISIBLE_DEVICES=2 \
    python visualize_attention.py \
        --exp_id "${exp_id}" \
        --attn_stage "${attn_stage}" \
        --ref_index "${ref_index_end}" \
        --patch_size 8 \
        --pretrained_weights "${pretrained_weights[0]}" \
        --image_path "/home/aghosh/Projects/2PCNet/Datasets/acdc/debug" \
        --output_dir "attention_maps/${exp_id}/attn_stage_${attn_stage}"
}

# NOTE: stick to attn_stage=1 for now

# TODO: process a folder of images
# /home/aghosh/Projects/2PCNet/Datasets/cityscapes/leftImg8bit/val/frankfurt

# # To use the function, call it with an experiment ID
# # Example: visualize_attention 80

visualize_attention 80 1 225
visualize_attention 81 1 225
visualize_attention 82 1 225
visualize_attention 83 1 225
visualize_attention 84 1 225
visualize_attention 85 1 225

# visualize_attention 90 1 225
# visualize_attention 91 1 225
# visualize_attention 92 1 225
# visualize_attention 93 1 225
# visualize_attention 94 1 225
# visualize_attention 95 1 225

# visualize_attention 85 1 1
# visualize_attention 85 1 2
# visualize_attention 85 1 3

# visualize_attention 90
# visualize_attention 91
# visualize_attention 92
# visualize_attention 93
# visualize_attention 94
# visualize_attention 95