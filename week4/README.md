We are Team 2 which is composed by:
- Diana Tat : dianatat120@gmail.com
- Miruna-Diana Jarda : mirunajrda@gmail.com
- Beatrice Anamaria Peptenaru : peptenaru.beatrice@gmail.com 
- Gunjan Paul : gunjan.mtbpaul@gmail.com


Link to the presentation: https://docs.google.com/presentation/d/1HLrGZbPCgnSNtdCfUbMQ8fEZIdAIzIc4XQESc2qzx6w/edit#slide=id.g2c4f6c3a97a_0_98

## Week 4
### Task 1.1 & 1.2 : Car Speed Estimation
We applied a mask to isolate the foreground, ensuring that only relevant objects, such as nearby vehicles, are included for annotations, while distant vehicles are excluded.
Subsequently, we proceed with vehicle annotation by assessing if the lower right corner of a bounding box falls within the defined ROI (between the 2 lines), incrementing a time counter and setting a flag indicating the vehicle's presence within the region. When the corner leaves the region and it was previously flagged as inside, the speed can be calculated based on the real measurement of the distance between the two lines and the time spent between them.

To run the code, we run python task1.py and put input video path, output paths for video and speed data, mask path, real life measurements between the 2 lines and line coordinates, using a command-line interface.

To replicate the same results use:
python task1.py -i ./data/vdo.avi -o ./data/out-orig.avi -ot ./data/seq3c10_speed.txt -m ./data/roi-edited.png -r 62 -lt 0 855 1919 855 -lb 0 297 1919 297

python task1.py -i ./data/custom/tunnel_10fps.mp4 -o ./data/custom/out_tunnel10_20.avi -ot ./data/custom/out_tunnel10_20.txt -m ./data/custom/mask_tunnel.png -r 20 -lt 0 625 1919 625 -lb 0 825 1919 825

python task1.py -i ./data/custom/zoom_10fps.mp4 -o ./data/custom/out_zoom10fps_15.avi -ot ./data/custom/out_zoom_15.txt -m ./data/custom/mask_zoom.png -r 15 -lb 0 615 1919 615 -lt 0 950 1919 950

python task1.py -i ./data/custom/radar.mov -o ./data/custom/out_radar.avi -ot ./data/custom/out_radar_speed.txt -m ./data/custom/mask_radar.png -r 50 -lb 1019 1079 1919 285 -lt 0 1015 1260 0

### Task 2 : Multi Camera Tracking
For this task we have created an offline pipeline consisting of two parts. First, we create the database containing all the identified objects (cars) as feature vectors and their corresponding frame_id, track_id sequence, camera and whether they have been reidentified or not. In the second part, pe perform reidentification based on a few assumptions:

- Tracking within one camera works perfectly 
- A car seen by a camera in a frame should be visible to another camera within a certain timeframe
- All cars are similar to each other to a certain degree
- Mapping of tracking IDs is transitive

Thus, our algorithm does the reidentification of the cars from one camera to another by performing similarity search between each object in a camera and all the other objects within a timeframe in all the other camera. The tracking ID of the object that yields the highest similarity score, if the highest similarity score is above a certain threshold, gets assigned to the current object.

In order to perform multitracking, one must run the track.py file with the desired sequence to perform the initial tracking, then multitrack.py with the same sequence to perform the actual multitracking. The results will be in the directory  given to us used to retrieve data from for this course. For multiple reasons, the directory is not present here.

An example on how to run the code can be seen below:
```
python track.py seq
python multitrack.py seq
```
