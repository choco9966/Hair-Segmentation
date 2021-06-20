# ====================================================
# CFG  
# ====================================================
class CFG:
    debug=False
    img_size=512
    max_len=275
    print_freq=1000
    num_workers=0
    encoder_type='timm-efficientnet-b2'
    decoder_type='Unet'
    size=512 # [512, 1024]
    freeze_epo = 0
    warmup_epo = 1
    cosine_epo = 9 #14 #19
    warmup_factor=10
    scheduler='GradualWarmupSchedulerV2' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'GradualWarmupSchedulerV2', 'get_linear_schedule_with_warmup']
    epochs=freeze_epo + warmup_epo + cosine_epo # not to exceed 9h #[1, 5, 10]
    factor=0.2 # ReduceLROnPlateau
    patience=4 # ReduceLROnPlateau
    eps=1e-6 # ReduceLROnPlateau
    T_max=4 # CosineAnnealingLR
    T_0=4 # CosineAnnealingWarmRestarts
    encoder_lr=3e-5 #[1e-4, 3e-5]
    min_lr=1e-6
    batch_size= 36 + 0 #[64, 256 + 128, 512, 1024, 512 + 256 + 128, 2048]
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=5
    dropout=0.5
    seed=42
    n_fold=1
    trn_fold=[0]
    #trn_fold=[0, 1, 2, 3, 4] # [0, 1, 2, 3, 4]
    train=True
    apex=False
    log_day='0618'
    version='v1-1'
    load_state=False
    light='light'
    data_type='celeb'