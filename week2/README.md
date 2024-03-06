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


Task 2.2 

The object tracking system consists of two main components: `KalmanFilter` and `Tracker`. 

### KalmanFilter

The `KalmanFilter` class implements the Kalman Filter algorithm, which predicts and updates the state of objects being tracked. It includes the following components:

- **Initialization**: Initializes the Kalman Filter with parameters such as time step, process noise covariance, measurement noise covariance, initial state, and initial uncertainty covariance.

- **Prediction**: Predicts the next state of the object based on the current state and the system dynamics. This involves updating the state estimate $(\(\hat{x}_{k|k-1}\))$ and state covariance $(\(P_{k|k-1}\))$ using the state transition model $(\(A\))$, process noise covariance (\(Q\)), and control-input model $(\(B\))$.

- **Update**: Corrects the predicted state based on the measurement obtained from sensors. This involves calculating the measurement pre-fit residual $(\(y_k\))$, residual covariance (\(S_k\)), Kalman gain (\(K_k\)), and updating the state estimate $(\(\hat{x}_k\))$ and state covariance $(\(P_k\))$ using the measurement model $(\(H\))$ and measurement noise covariance $(\(R\))$.

### Tracker

The `Tracker` class maintains individual trackers for each object being tracked. It includes the following components:

- **Initialization**: Initializes a tracker with a unique ID, a Kalman Filter instance, and optional bounding box information.

- **Update**: Updates the tracker's state based on the associated measurement obtained from the Kalman Filter.

- **Trail**: Maintains a trail of the object's previous positions for visualization purposes.

## Flow

The overall flow of the object tracking system is as follows:

1. **Initialization**: 
   - Initialize Kalman Filters for each detected object in the first frame.
   - Create corresponding trackers for each Kalman Filter.

2. **Frame Processing**:
   - Iterate through each frame of the video sequence.
   - Obtain detections or measurements from each frame.

3. **Prediction**:
   - Predict the next state for each tracker using the Kalman Filter's `predict()` method.

4. **Data Association**:
   - Match detections to trackers using the Hungarian algorithm based on distance metrics between predicted and detected object positions.

5. **Update**:
   - For each matched pair of tracker and detection, update the tracker's state using the Kalman Filter's `update()` method.
   - Create new trackers for unmatched detections.

6. **Trail Update**:
   - Update the object's trail based on its current estimated position.

7. **Trackers Maintenance**:
   - Remove trackers that have been lost for too long or have consecutive absence.
   - Reset trails of removed trackers.

8. **Visualization**:
   - Draw bounding boxes and IDs on the frame based on tracker information.
   - Display or save the frame with tracking information.

9. **Loop**:
   - Repeat the above steps for each frame until the end of the video sequence.

## Conclusion

This implementation provides a robust framework for object tracking using the Kalman Filter algorithm. By combining prediction and update steps with data association and tracker maintenance, it effectively tracks objects in dynamic environments.

For more details, refer to the source code and comments provided in the repository.
