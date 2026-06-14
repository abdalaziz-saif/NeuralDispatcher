

class Config :
    
    # Data 
    DATASET_NAME = "yousefsaeedian/databricksdatabricks-dolly-15k"
    MAX_LENGTH = 512 
    BATCH_SIZE = 16 


    # MODEL 
    MODEL_NAME= "roberta-base"
    NUM_LABELS      = 8
    LAYERS_TO_TRAIN = 6


    # Training
    EPOCHS           = 6
    LEARNING_RATE    = 2e-5
    CLASSIFIER_LR    = 3e-5
    WARMUP_RATIO     = 0.06

    # path 
    SAVE_PATH = "./saved_model"