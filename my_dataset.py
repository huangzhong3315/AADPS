from PIL import Image
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from utils import pad_sequence
from padding import select_frames_with_cycling
from torch.nn import utils as nn_utils


class MyDataSet(Dataset):
    """自定义数据集"""
    def __init__(self, videos_path: list, videos_class: list, points_path: list, transform=None):
        self.videos_path = videos_path
        self.videos_class = videos_class
        self.points_path = points_path
        self.transform = transform

    def __len__(self):
        return len(self.videos_path)

    def __getitem__(self, item):
        frame, face_point, point, lh_point, rh_point = [], [], [], [], []

        for f in os.listdir(self.videos_path[item]):
            image_path = os.path.join(self.videos_path[item], f)
            img = Image.open(image_path)
            image = cv2.imread(image_path)
            # RGB为彩色图片，L为灰度图片
            if img.mode != 'RGB':
                raise ValueError("image: {} isn't RGB mode.".format(self.videos_path[item]))

            # # json_data = self.points_path[item]
            frame_points = os.path.splitext(f)[0] + ".json"
            # # 保证图片与骨骼点对应
            point_path = os.path.join(self.points_path[item], frame_points)
            # # point_path = os.path.join(self.points_path[item], frame_points)

            with open(point_path) as p:
                # 把一个json文件返回一个python对象
                json_data = json.load(p)

            keypoints = np.asarray(json_data['people'][0]['pose_keypoints_2d'], dtype=np.float32).reshape((-1, 3))
            face_keypoints = np.asarray(json_data['people'][0]['face_keypoints_2d'], dtype=np.float32).reshape((-1, 3))
            hand_left_keypoints = np.asarray(json_data['people'][0]['hand_left_keypoints_2d'], dtype=np.float32).reshape((-1, 3))
            hand_right_keypoints = np.asarray(json_data['people'][0]['hand_right_keypoints_2d'], dtype=np.float32).reshape((-1, 3))
            # ======================================标准化坐标点============================================
            normalize_point_x = keypoints[8, 0]
            normalize_point_y = keypoints[8, 1]
            keypoints[:, 0] -= normalize_point_x
            keypoints[:, 1] -= normalize_point_y

            normalize_face_x = face_keypoints[30, 0]
            normalize_face_y = face_keypoints[30, 1]
            face_keypoints[:, 0] -= normalize_face_x
            face_keypoints[:, 1] -= normalize_face_y

            hand_left_keypoints[:, 0] = hand_left_keypoints[:, 0] - hand_left_keypoints[0, 0]
            hand_left_keypoints[:, 1] = hand_left_keypoints[:, 1] - hand_left_keypoints[0, 1]

            hand_right_keypoints[:, 0] = hand_right_keypoints[:, 0] - hand_right_keypoints[0, 0]
            hand_right_keypoints[:, 1] = hand_right_keypoints[:, 1] - hand_right_keypoints[0, 1]

            # keypoints = np.reshape(keypoints, (-1))
            # face_keypoints = np.reshape(face_keypoints, (-1))
            # hand_left_keypoints = np.reshape(hand_left_keypoints, (-1))
            # hand_right_keypoints = np.reshape(hand_right_keypoints, (-1))
            # ======================================标准化坐标点============================================
            # label = self.images_class[item]
            label = self.videos_class[item]

            sample = {'frame': image, 'face_point': face_keypoints, 'point': keypoints,
                      'left_hand_point': hand_left_keypoints, 'right_hand_point': hand_right_keypoints, "Y": label}
            if self.transform is not None:
                sample = self.transform(sample)

            frame.append(sample['frame'])
            face_point.append(sample['face_point'])
            point.append(sample['point'])
            lh_point.append(sample['left_hand_point'])
            rh_point.append(sample['right_hand_point'])

        frames = torch.from_numpy(np.stack(frame).astype(np.float32))
        face_points = torch.from_numpy(np.stack(face_point).astype(np.float32))
        points = torch.from_numpy(np.stack(point).astype(np.float32))
        lh_points = torch.from_numpy(np.stack(lh_point).astype(np.float32))
        rh_points = torch.from_numpy(np.stack(rh_point).astype(np.float32))

        label = self.videos_class[item]
        label = torch.as_tensor(label)

        sample = {'frames': frames, 'face_points': face_points, 'points': points,
                  'lh_points': lh_points, 'rh_points': rh_points, "Y": label}

        return sample['frames'], sample['face_points'], sample['points'], sample['lh_points'], sample['rh_points'], sample['Y']

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        img, f_point, b_point, l_point, r_point, y = tuple(zip(*batch))    # 将数据与标签压缩
        #
        img = select_frames_with_cycling(img, num_frames=64)
        f_point = select_frames_with_cycling(f_point, num_frames=64)
        b_point = select_frames_with_cycling(b_point, num_frames=64)
        l_point = select_frames_with_cycling(l_point, num_frames=64)
        r_point = select_frames_with_cycling(r_point, num_frames=64)
        # img = pad_sequence(img, batch_first=True)
        # f_point = pad_sequence(f_point, batch_first=True)
        # b_point = pad_sequence(b_point, batch_first=True)
        # l_point = pad_sequence(l_point, batch_first=True)
        # r_point = pad_sequence(r_point, batch_first=True)

        return {'frames': img,
                'face_points': f_point,
                'points': b_point,
                'lh_points': l_point,
                'rh_points': r_point,
                'Y': torch.stack(y, dim=0)
                }
