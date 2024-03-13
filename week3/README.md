We are Team 2 which is composed by:
- Diana Tat : dianatat120@gmail.com
- Miruna-Diana Jarda : mirunajrda@gmail.com
- Beatrice Anamaria Peptenaru : peptenaru.beatrice@gmail.com 
- Gunjan Paul : gunjan.mtbpaul@gmail.com


### Week 3
``` 
Our code is organized as:
├── task1_1.py : optical flow with block matching
├── studies-info.txt - contains the parameters and results of the latest trials for the exhaustive optical flow.
                        The time includes wandb and optuna (wandb taking the most time)
                        Sortable results can be found here: https://wandb.ai/c5/block_matching_bea/table?nw=nwuserteam10cvc
├── task1_2_pyflow.py : Can be run as a normal python file. 
├── task_2_(1, 2, 3):
- In order to evaluate the results of tracking, one can run evaluate_tracking/convert_evaluate.sh with two arguments: path to tracking results as json file and path to gt.txt. These will convert the files to the right formats, place the files in the required directories and run the evaluation.
``` 


In order to run optical flow with unimatch, please use the command:
```console
python unimatch/main_flow.py --inference_dir unimatch/demo/seq45 --resume unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth --output_path output/gmflow-scale2-regrefine6-davis --padding_factor 32 --upsample_factor 4 --num_scales 2 --attn_splits_list 2 8 --corr_radius_list -1 4 --prop_radius_list -1 1 --reg_refine --num_reg_refine 6
``` 

Image _000045_10_flow.png_ shows the result of optical flow by unimatch on the image (uploaded here because the document became uneditable before the deadline :) )

### Results for task 2

Due to lack of time, we haven't finished in time to post in the slides.

| IDF1 / HOTA (SEQ01) | c10   | c11   | c012  | c013  | c014  | c015  | Average |
|---------------------|-------|-------|-------|-------|-------|-------|---------|
| Team 2              | 18.72 | 12.36 | 8.97  | 6.83  | 1.94  | 4.22  | 10.773  |


| IDF1 / HOTA (SEQ03) | c10   | c11   | c012  | c013  | c014  | c015  | Average |
|---------------------|-------|-------|-------|-------|-------|-------|---------|
| Team 2              | 32.57 | 20.92 | 14.38 | 11.75 | 3.26  | 7.91  | 31.808  |

| IDF1 / HOTA (SEQ04) | c10   | c11   | c012  | c013  | c014  | c015  | Average |
|---------------------|-------|-------|-------|-------|-------|-------|---------|
| Team 2              | 37.29 | 24.47 | 14.14 | 13.27 | 3.58  | 7.91  | 20.101  |


# Task 1.1: Optical flow with block matching 

Overview
This repository contains a Python implementation of optical flow estimation using block matching techniques. Optical flow refers to the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between an observer and the scene. The block matching method divides images into blocks and searches for corresponding blocks in consecutive frames to estimate motion.

**Features**
**Exhaustive Search**: Implements an exhaustive search strategy where each block in the first frame is compared with all possible blocks within a specified search range in the next frame to find the best match.

**Three Step Search**: Utilizes a more efficient approach by reducing the search area in steps, significantly decreasing the computational cost while maintaining accuracy.


### Parameters

**block_size**: This parameter determines the size of the blocks that will be compared between frames during the block matching process.
Larger block sizes may capture more global motion but may also lead to less accurate estimates in areas with significant local motion.

**search_range**: Specifies the maximum distance (in pixels) that the algorithm will search for a matching block in the next frame.
A larger search range allows for capturing larger motions between frames but increases computational complexity.

**step_size**: Determines the spacing between the centers of adjacent blocks during the search process.
A smaller step size results in a denser search grid and potentially more accurate motion estimation but requires more computational resources.

**similarity_method**: Specifies the method used to measure the similarity between blocks.
In the provided code snippet, "SSD" stands for Sum of Squared Differences, which computes the sum of squared differences between corresponding pixel values in the blocks.
Other similarity measures, such as Sum of Absolute Differences (SAD), can also be used depending on the application requirements.

Supported similarity methods
  1. Sum of Absolute Differences (SAD)
  2. Sum of Squared Differences (SSD)
  3. Normalized Cross-Correlation (NCC)
  4. Mean Squared Error (MSE)
  5. Structural Similarity Index (SSIM)

### To run the code
```
block_size = 16
search_range = 20
step_size = 2
similarity_method = "SSD"

# Compute optical flow using three-step search block matching
flow_three_step = block_matching(
    prev_frame,
    next_frame,
    block_size=block_size,
    search_range=search_range,
    step_size=step_size,
    method="three_step",
    similarity_method=similarity_method,
)


```
