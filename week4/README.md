We are Team 2 which is composed by:
- Diana Tat : dianatat120@gmail.com
- Miruna-Diana Jarda : mirunajrda@gmail.com
- Beatrice Anamaria Peptenaru : peptenaru.beatrice@gmail.com 
- Gunjan Paul : gunjan.mtbpaul@gmail.com


Link to the presentation: https://docs.google.com/presentation/d/1HLrGZbPCgnSNtdCfUbMQ8fEZIdAIzIc4XQESc2qzx6w/edit#slide=id.g2c4f6c3a97a_0_98

## Week 4
### Task 1.1 : Car Speed Estimation
We applied a mask to isolate the foreground, ensuring that only relevant objects, such as nearby vehicles, are included for annotations, while distant vehicles are excluded.
Subsequently, we proceed with vehicle annotation by assessing if the lower right corner of a bounding box falls within the defined ROI, incrementing a time counter and setting a flag indicating the vehicle's presence within the region. We check if the lower right corner of a bounding box is between the 2 lines. If it is, I add 1 to the time and set a flag that it was inside, and then we calculate the speed.
Here we have some examples in which we can see that the cars are detected correctly. Using YOLOv8, had some issues with the bigs cars and with an extra class (truck) that the code was not identifying.
