# ####################### These for test-stage only #######################
# bash test.sh work_dirs/local-exp80/test_11_11
# bash test.sh work_dirs/local-exp83/test_11_11
# bash test.sh work_dirs/local-exp84/test_11_11
# bash test.sh work_dirs/local-exp88/test_11_11

# bash test.sh work_dirs/local-exp110/test_11_11
# bash test.sh work_dirs/local-exp118/test_11_11

# TODO: train on src, test directly on tgt
# cityscapes => darkzurich

# cityscapes => acdc

# # 80
# bash test.sh work_dirs/local-exp80/231025_1229_cs2dzur_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_c4e21



# ####################### These for train-stage only #######################

# # 81
# bash test.sh work_dirs/local-exp81/231026_0127_cs2dzur_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_597d4

# # 82
# bash test.sh work_dirs/local-exp82/231026_0127_cs2dzur_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_7e208

# # 83
# bash test.sh work_dirs/local-exp83/231028_1132_cs2dzur_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_87fc8


# # 84
# bash test.sh work_dirs/local-exp84/231026_1531_cs2dzur_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_b3cce


# # 85
# bash test.sh work_dirs/local-exp85/231028_0100_cs2dzur_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_69c9b

# 88
# bash test.sh work_dirs/local-exp88/231108_0047_cs2dzur_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_d1027


# # 90
# bash test.sh work_dirs/local-exp90/231025_1230_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_87f3d

# # 91
# bash test.sh work_dirs/local-exp91/231028_1352_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_f5375

# # 92
# bash test.sh work_dirs/local-exp92/231030_0012_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_e09d7

# # 93
# bash test.sh work_dirs/local-exp93/231029_0116_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_df01b

# # 94
# bash test.sh work_dirs/local-exp94/231029_1551_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_592db

# # 95
# bash test.sh work_dirs/local-exp95/231029_1105_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_231df


# # 98
# bash test.sh work_dirs/local-exp98/231107_0010_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_119f2


# NOTE; this for DZ test-set generatioon only (used for public leaderboard)
# bash test_test.sh  /longdata/anurag_storage/2PCNet/work_dirs/local-exp80/231025_1229_cs2dzur_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_c4e21
# bash test_test.sh  /longdata/anurag_storage/2PCNet/work_dirs/local-exp83/231028_1132_cs2dzur_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_87fc8
# bash test_test.sh  /longdata/anurag_storage/2PCNet/work_dirs/local-exp84/231026_1531_cs2dzur_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_b3cce
# bash test_test.sh  /longdata/anurag_storage/2PCNet/work_dirs/local-exp88/231108_0047_cs2dzur_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_d1027

# # NOTE: this for ACDCtest-set generation only (used for public leaderboard)
# bash test_test.sh  /longdata/anurag_storage/2PCNet/work_dirs/local-exp90/231025_1230_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_87f3d
# bash test_test.sh  /longdata/anurag_storage/2PCNet/work_dirs/local-exp93/231029_0116_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_df01b
# bash test_test.sh  /longdata/anurag_storage/2PCNet/work_dirs/local-exp94/231029_1551_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_592db
# bash test_test.sh  /longdata/anurag_storage/2PCNet/work_dirs/local-exp98/231107_0010_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_119f2

# 98
# bash test.sh /longdata/anurag_storage/2PCNet/work_dirs/local-exp98/test_full

# NOTE: for test-time warp, set is_training = True!!!!!!!!!!! for mmseg/modes/segmentors/encoder_decoder.py
# NOTE: after this, re-run previous testing without is_training = True, to go back to normal results

# # 82 (NOTE: add test-time warp)
# bash test.sh /longdata/anurag_storage/2PCNet/work_dirs/local-exp82/231026_0127_cs2dzur_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_7e208

# # 81 (NOTE: add test-time warp)
# bash test.sh /longdata/anurag_storage/2PCNet/work_dirs/local-exp81/231026_0127_cs2dzur_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_597d4

# # 92 (NOTE: add test-time warp)
# bash test.sh /longdata/anurag_storage/2PCNet/work_dirs/local-exp92/231030_0012_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_e09d7

# # 91 (NOTE: add test-time warp)
# bash test.sh /longdata/anurag_storage/2PCNet/work_dirs/local-exp91/231028_1352_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_f5375




# Base='/longdata/anurag_storage/2PCNet/DAFormer/work_dirs'

# bash test.sh ${Base}/local-exp90/231025_1230_cs2acdc_dacs_a999_fdthings_rcs001_cpl_daformer_sepaspp_mitb5_poly10warm_s0_87f3d

# # 90
# bash test.sh ${Base}/local-exp90/night
# bash test.sh ${Base}/local-exp90/fog
# bash test.sh ${Base}/local-exp90/rain
# bash test.sh ${Base}/local-exp90/snow

# # # 98
# bash test.sh ${Base}/local-exp98/night
# bash test.sh ${Base}/local-exp98/fog
# bash test.sh ${Base}/local-exp98/rain
# bash test.sh ${Base}/local-exp98/snow


Base='/home/aghosh/Projects/2PCNet/Methods/Instance-Warp/DAFormer/work_dirs'

# # 270 (NOTE: test on idd)
bash test.sh ${Base}/local-exp270/test_12_18

# # 275  (NOTE: test on idd)
bash test.sh ${Base}/local-exp275/test_12_18