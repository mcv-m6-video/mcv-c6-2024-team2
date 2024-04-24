We are Team 2 which is composed by:
- Diana Tat : dianatat120@gmail.com
- Miruna-Diana Jarda : mirunajrda@gmail.com
- Beatrice Anamaria Peptenaru : peptenaru.beatrice@gmail.com 
- Gunjan Paul : gunjan.mtbpaul@gmail.com

Instructions to run the code:

* For Task 1
```
run task1_train.py - this will run the training and testing for all of the models
```
* For Task 2 (running ResNet & VGG experiments):
```
run train_task2.py with: --frames_dir <frames directory>
                         --model-name resnet/slow_r50/vgg/vgg_3d
                         --es-start-epoch 0
                         --patience 10
                         --min_delta 0.1
                         --epochs 50
                         --validate-every 1 
            *(OPTIONAL)* --clip-length 1 for 2D models
```
