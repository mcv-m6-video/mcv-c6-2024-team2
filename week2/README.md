We are Team 2 which is composed by:
- Diana Tat : dianatat120@gmail.com
- Miruna-Diana Jarda : mirunajrda@gmail.com
- Beatrice Anamaria Peptenaru : peptenaru.beatrice@gmail.com 
- Gunjan Paul : gunjan.mtbpaul@gmail.com


### Week 2

Our code is organized as:
```
├── task1_1.py
└──task 1_3.py fine-tune the model from task1_1
├── task_1_2 : annotation result          
├──task_1_4: k-fold method
```

- Task 1.1 & 1.3
Run task1_1.py and task_1_3finetune.py. The latter needs to be run after the first since the first one includes code to extract the frames from the given video. Due to the total size of the extracted frames, these were not added directly to this repository. The user needs to specify the path to the video in the python file.

- Task 1.4
```
.
├── B_KFold_<x>                  # Training results for Strategy A, B (plots, some annotated frames, weights) , x = number between 0-4
├── random_kfold_<x>             # Training results for Strategy C (plots, some annotated frames, weights) , x = number between 0-4
└── datasets
    └── data
        ├── vdo.avi              # Original input video
        └── scaled
            ├── data.yaml        # Configuration file from RoboFlow (YOLO8 format)
            ├── vdo-scaled.mp4   # Video scaled to 640x384 (keep ratio)
            └── train
                ├── images       # All frames scaled to 640x640 (YOLO recommended size)(comes from RoboFlow)
                ├── labels       # All labels (comes from RoboFlow)
                ├── B_KFold/R-KFold                  # B_KFold = Strategy B Kfold, R-KFold = Strategy C (random) KFold 
                │   └── split_<x>                    # x = 0-3 for B_KFold, x = 1-4 for R-KFold
                │       ├── output.avi               # Annotated output video from training the model on this dataset
                │       ├── split_0_dataset.yaml     # Configuration file (YOLO8 format)
                │       ├── train                    # Training dataset
                │       │   ├── images
                │       │   └── labels
                │       └── val                      # Testing/Validating dataset
                │           ├── images               
                │           └── labels
                └── No-fold
                    └── pre-trained-output.avi      # Annotated output video from pre-trained model
```
Task 2.1 - Can be run on its own. It uses the json files containing the predictions (found in this repository) to perform tracking.
