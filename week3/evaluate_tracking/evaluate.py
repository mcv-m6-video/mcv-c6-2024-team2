from tqdm import tqdm
import convertor as cnv
import os
import shutil
import argparse

tracked_dir = './TrackEval/data/trackers/mot_challenge/S03-train/ioutrack/data'
tracked_temp = 'out.csv'
tracked_out = 'S03.txt'

gt_dir = './TrackEval/data/gt/mot_challenge/S03-train/S03/gt'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracked_in', type=str, help='Path to tracked input file.')
    parser.add_argument('--gt_in', type=str, help='Path to ground truth input file.')
    args = parser.parse_args().__dict__
    
    if args['tracked_in'] is not None:
        tracked_in = args['tracked_in']

    if args['gt_in'] is not None:
        gt_in = args['gt_in']

    out_temp = os.path.join(tracked_dir, tracked_temp)
    out_fin = os.path.join(tracked_dir, tracked_out)

    if os.path.exists(out_fin):
        os.remove(out_fin)

    cnv.track_to_csv(tracked_in, out_temp)
    os.rename(out_temp, out_fin)

    out_fin = os.path.join(gt_dir, gt_in)

    if os.path.exists(out_fin):
        os.remove(out_fin)

    shutil.copy()
