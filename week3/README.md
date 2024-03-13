We are Team 2 which is composed by:
- Diana Tat : dianatat120@gmail.com
- Miruna-Diana Jarda : mirunajrda@gmail.com
- Beatrice Anamaria Peptenaru : peptenaru.beatrice@gmail.com 
- Gunjan Paul : gunjan.mtbpaul@gmail.com


### Week 3

Our code is organized as:
├── task1_1.py : optical flow with block matching 
├── task1_2_pyflow.py : Can be run as a normal python file. In order to run the unimatch, please use the command:
```console
python unimatch/main_flow.py --inference_dir unimatch/demo/seq45 --resume unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth --output_path output/gmflow-scale2-regrefine6-davis --padding_factor 32 --upsample_factor 4 --num_scales 2 --attn_splits_list 2 8 --corr_radius_list -1 4 --prop_radius_list -1 1 --reg_refine --num_reg_refine 6
``` 
├── task1_3_optical_flow.ipynb : object tracking with optical flow : just run the notebook 



├──task_2_(1, 2, 3):
- In order to evaluate the results of tracking, one can run evaluate_tracking/convert_evaluate.sh with two arguments: path to tracking results as json file and path to gt.txt. These will convert the files to the right formats, place the files in the required directories and run the evaluation.


