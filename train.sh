# NOTE: hardcode v2 now for debugs
run_experiment() {
    GPU_ID="$1"
    EXP_ID="$2"
    
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    nohup python run_experiments.py \
    --exp "$EXP_ID" \
    > "outs/${EXP_ID}_v4.out" 2>&1 &
}

# run_experiment 0 80
# run_experiment 0 90
# run_experiment 0 98
# run_experiment 0 81
# run_experiment 1 82
# run_experiment 1 83
# run_experiment 1 84
# run_experiment 0 85
# run_experiment 1 86

# run_experiment 0 91
# run_experiment 1 93

# run_experiment 0 95
# run_experiment 1 94

# run_experiment 0 87

# run_experiment 2 92

# run_experiment 0 97

# run_experiment 1 96

# run_experiment 0 88
# run_experiment 1 98

# run_experiment 0 88
# run_experiment 1 98

# DOING these experiments for now
# run_experiment 1 89
# run_experiment 2 99

# run_experiment 0 96
# run_experiment 1 97


# run_experiment 0 105
# run_experiment 2 106


# cs, 25% images
# run_experiment 0 110
# run_experiment 1 118

# cs, 25% images (REDO it, by setting validation images to cs)
# run_experiment 0 110
# run_experiment 1 118

# # cs, 50% images
# run_experiment 0 130
# run_experiment 1 138

# # cs, 75% images
# run_experiment 0 150
# run_experiment 1 158

# # cs, uniform sampled images (50% images)
# run_experiment 0 170
# run_experiment 1 178

# # cs, uniform (25% images)
# run_experiment 0 190
# run_experiment 1 198


# NOTE; use this to help debug image shape stuffs
# run_experiment 1 90

# gta to cs
# run_experiment 0 210
# run_experiment 1 215

# synthia to cs
# run_experiment 0 220
# run_experiment 1 225
# run_experiment 0 228


# CS2DZ (full-size)
# run_experiment 1 230
# run_experiment 1 238

# CS2ACDC (full-size)
# run_experiment 0 240
# run_experiment 1 248

# CS2IDD
# run_experiment 2 250
# run_experiment 1 258

# # CS2FCS
# run_experiment 0 260
# run_experiment 1 268

# IDD2CS
# run_experiment 0 270
# run_experiment 1 275

# NOTE: train two at a time; leave some GPUs!

# NOTE: use batch size of 1 below!!!

# # roadwork (train_ALL, test_REST)
# run_experiment 3 280

# # roadwork (train_PIT, test_REST)
# run_experiment 1 290

# # # roadwork (train_PIT, adapt to train_REST, test_REST)
# run_experiment 2 300

# # # roadwork (train_PIT, adapt to train_REST, test_REST; using InstWarp)
# run_experiment 3 305

# NOTE: this for debug only! => DO NOT rerun this!!!
# roadwork (train_DEBUG, adapt to train_DEBUG, test_DEBUG)
# run_experiment 3 310