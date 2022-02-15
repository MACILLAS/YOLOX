import cv2
import pandas as pd

import wrapper as detector
from sort import *

class LolTracker():
    def __init__(self):
        self.last_bbs = None
        self.current_bbs = None
        self.lost_det_counter = 0
        self.tracking = False
        self.index = 0
        self.missThresh = 30

    def isTracking(self):
        return self.tracking

    def bbsInThresh(self):
        """
        This function checks if the centroid of the current_bbs is within the last_bbs
        if last_bbs is None return True
        """
        if self.last_bbs is None:
            return True
        centroid_x = self.current_bbs[0] + self.current_bbs[2]/2
        centroid_y = self.current_bbs[1] + self.current_bbs[3]/2
        if centroid_x > self.last_bbs[0] and centroid_x < (self.last_bbs[0]+self.last_bbs[2]):
            if centroid_y > self.last_bbs[1] and centroid_y < (self.last_bbs[1]+self.last_bbs[3]):
                return True
        else:
            return False

    def reset(self):
        """
        When detection is lost or missing for too many frames. The system vars are reset.
        """
        self.tracking = False
        self.lost_det_counter = 0
        self.last_bbs = None
        self.current_bbs = None

    def update(self, bbs=None):
        self.current_bbs = bbs
        if self.current_bbs is None:
            # if not tracking then do nothing
            # if tracking then return
            if self.isTracking():
                self.lost_det_counter = self.lost_det_counter + 1
                if self.lost_det_counter > self.missThresh:
                    self.reset()
                    return 0
                # if false-negative then return last bbs position
                self.last_bbs = self.last_bbs
                return self.index
            else:
                return 0
        elif self.bbsInThresh():
            # if not tracking, switch tracking to True, index=index+1
            if not self.tracking:
                self.tracking = True
                self.index = self.index + 1
            self.last_bbs = self.current_bbs
            self.lost_det_counter = 0
            return self.index
        else:
            self.lost_det_counter = self.lost_det_counter + 1
            if self.lost_det_counter > self.missThresh:
                self.reset()
            return self.index

class detspallv2():

    def __init__(self, model='yolox.onnx'):
        self.session = detector.open_sess(model=model)
        #self.mot_tracker = Sort()
        self.tracker = LolTracker()
        self.bb_df = pd.DataFrame(columns=['file', 'x1', 'y1', 'x2', 'y2', 'conf', 'track'])

    def detect(self, img):
        final_boxes, final_scores, final_cls_inds = detector.run(sess=self.session, img=img, visual=False)
        if final_boxes is not None:
            final_boxes = final_boxes[final_scores.argmax(), :]
            x1 = final_boxes[0]
            y1 = final_boxes[1]
            x2 = final_boxes[2]
            y2 = final_boxes[3]
            conf = final_scores[final_scores.argmax()]
        else:
            return None
        return [x1, y1, x2, y2, conf]

    def detAndTrack(self, img):
        # Get detections [x1, y1, x2, y2, conf]
        detections = self.detect(img)
        if detections is not None:
            ids = self.tracker.update(detections)
            return np.concatenate((detections[:-1], [ids]))
        else:
            ids = self.tracker.update(None)
            return np.empty((0, 5))
        # ids [track_idx]
        # track_bbs_ids [x1, y1, x2, y2, track_idx]

class detspall():

    def __init__(self, model='yolox.onnx'):
        self.session = detector.open_sess(model=model)
        self.mot_tracker = Sort()
        self.bb_df = pd.DataFrame(columns=['file', 'x1', 'y1', 'x2', 'y2', 'conf', 'track'])

    def detect(self, img):
        final_boxes, final_scores, final_cls_inds = detector.run(sess=self.session, img=img, visual=False)
        if final_boxes is not None:
            final_boxes = final_boxes[final_scores.argmax(), :]
            x1 = final_boxes[0]
            y1 = final_boxes[1]
            x2 = final_boxes[2]
            y2 = final_boxes[3]
            conf = final_scores[final_scores.argmax()]
        else:
            return None
        return [x1, y1, x2, y2, conf]

    def detAndTrack(self, img):
        # Get detections [x1, y1, x2, y2, conf]
        detections = self.detect(img)
        if detections is not None:
            track_bbs_ids = self.mot_tracker.update(np.array([detections]))
        else:
            track_bbs_ids = self.mot_tracker.update(np.empty((0, 5)))
        # track_bbs_ids [x1, y1, x2, y2, track_idx]
        return track_bbs_ids


if __name__ == "__main__":
    DATADIR = '../YOLOX/datasets/ig_sim_closeup/'
    lst = os.listdir(DATADIR)

    session = detector.open_sess(model='yolox.onnx')

    mot_tracker = Sort()

    bb_df = pd.DataFrame(columns=['file', 'x1', 'y1', 'x2', 'y2', 'conf', 'track'])

    start = time.time()
    for i in range(len(lst)):
        imgFile = os.path.join(DATADIR, str(i) + '_rgb.jpg')
        img = cv2.imread(imgFile, -1)
        final_boxes, final_scores, final_cls_inds = detector.run(sess=session, img=img, visual=False)
        if final_boxes is not None:
            final_boxes = final_boxes[final_scores.argmax(), :]
            x1 = final_boxes[0]
            y1 = final_boxes[1]
            x2 = final_boxes[2]
            y2 = final_boxes[3]
            conf = final_scores[final_scores.argmax()]
            track_bbs_ids = mot_tracker.update(np.array([[x1, y1, x2, y2, conf]]))
            if track_bbs_ids.size > 0:
                entry = pd.DataFrame({'file': imgFile, 'x1': [x1], 'y1': [y1], 'x2': [x2], 'y2': [y2], 'conf': [conf],
                                      'track': track_bbs_ids[0, 4]}, index=[i])
            else:
                entry = pd.DataFrame(
                    {'file': imgFile, 'x1': [x1], 'y1': [y1], 'x2': [x2], 'y2': [y2], 'conf': [conf], 'track': False},
                    index=[i])

            bb_df = bb_df.append(entry)
        else:
            track_bbs_ids = mot_tracker.update(np.empty((0, 5)))
    end = time.time()
    ex_time = end - start

    print(f"Execution Time: {ex_time} s")

    bb_df.to_csv('dets.csv', index=False)
