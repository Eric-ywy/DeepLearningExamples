# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import json
import math
import os
import pickle

import monai.transforms as transforms
import nibabel
import numpy as np
from joblib import Parallel, delayed
from skimage.morphology import dilation, erosion, square
from skimage.transform import resize
from skimage.io import imread
from utils.utils import get_task_code, make_empty_dir, get_fpath_pairs

from data_preprocessing.configs import ct_max, ct_mean, ct_min, ct_std, med_patch_size, med_spacings, med_task
from data_preprocessing.configs import ind_patch_size, ind_task, ind_mean, ind_std


class med_Preprocessor:
    def __init__(self, args):
        self.args = args
        self.ct_min = 0
        self.ct_max = 0
        self.ct_mean = 0
        self.ct_std = 0
        self.target_spacing = None
        self.task = args.task
        self.task_code = get_task_code(args)
        self.patch_size = med_patch_size[self.task_code]
        self.training = args.exec_mode == "training"
        self.data_path = os.path.join(args.data, med_task[args.task])
        self.results = os.path.join(args.results, self.task_code)
        if not self.training:
            self.results = os.path.join(self.results, self.args.exec_mode)
        self.crop_foreg = transforms.CropForegroundd(keys=["image", "label"], source_key="image")
        self.normalize_intensity = transforms.NormalizeIntensity(nonzero=False, channel_wise=True)
        metadata_path = os.path.join(self.data_path, "dataset.json")
        if self.args.exec_mode == "val":
            dataset_json = json.load(open(metadata_path, "r"))
            dataset_json["val"] = dataset_json["training"]
            with open(metadata_path, "w") as outfile:
                json.dump(dataset_json, outfile)
        self.metadata = json.load(open(metadata_path, "r"))
        self.modality = self.metadata["modality"]["0"]

    def run(self):
        make_empty_dir(self.results)

        print(f"Preprocessing {self.data_path}")
        try:
            self.target_spacing = med_spacings[self.task_code]
        except:
            self.collect_spacings()
        print(f"Target spacing {self.target_spacing}")

        if self.modality == "CT":
            try:
                self.ct_min = ct_min[self.task]
                self.ct_max = ct_max[self.task]
                self.ct_mean = ct_mean[self.task]
                self.ct_std = ct_std[self.task]
            except:
                self.collect_intensities()

            _mean = round(self.ct_mean, 2)
            _std = round(self.ct_std, 2)
            print(f"[CT] min: {self.ct_min}, max: {self.ct_max}, mean: {_mean}, std: {_std}")

        self.run_parallel(self.preprocess_pair, self.args.exec_mode)

        pickle.dump(
            {
                "patch_size": self.patch_size,
                "spacings": self.target_spacing,
                "n_class": len(self.metadata["labels"]),
                "in_channels": len(self.metadata["modality"]),
            },
            open(os.path.join(self.results, "config.pkl"), "wb"),
        )

    def preprocess_pair(self, pair):
        fname = os.path.basename(pair["image"] if isinstance(pair, dict) else pair)
        image, label, image_spacings = self.load_pair(pair)
        if self.training:
            data = self.crop_foreg({"image": image, "label": label})
            image, label = data["image"], data["label"]
            test_metadata = None
        else:
            bbox = transforms.utils.generate_spatial_bounding_box(image)
            test_metadata = np.vstack([bbox, image.shape[1:]])
            image = transforms.SpatialCrop(roi_start=bbox[0], roi_end=bbox[1])(image)
            if label is not None:
                label = transforms.SpatialCrop(roi_start=bbox[0], roi_end=bbox[1])(label)
        if self.args.dim == 3:
            image, label = self.resample(image, label, image_spacings)
        if self.modality == "CT":
            image = np.clip(image, self.ct_min, self.ct_max)
        image = self.normalize(image)
        if self.training:
            image, label = self.standardize(image, label)
            if self.args.dilation:
                new_lbl = np.zeros(label.shape, dtype=np.uint8)
                for depth in range(label.shape[1]):
                    new_lbl[0, depth] = erosion(dilation(label[0, depth], square(3)), square(3))
                label = new_lbl
        self.save(image, label, fname, test_metadata)

    def resample(self, image, label, image_spacings):
        if self.target_spacing != image_spacings:
            image, label = self.resample_pair(image, label, image_spacings)
        return image, label

    def standardize(self, image, label):
        pad_shape = self.calculate_pad_shape(image)
        image_shape = image.shape[1:]
        if pad_shape != image_shape:
            paddings = [(pad_sh - image_sh) / 2 for (pad_sh, image_sh) in zip(pad_shape, image_shape)]
            image = self.pad(image, paddings)
            label = self.pad(label, paddings)
        if self.args.dim == 2:  # Center cropping 2D images.
            _, _, height, weight = image.shape
            start_h = (height - self.patch_size[0]) // 2
            start_w = (weight - self.patch_size[1]) // 2
            image = image[:, :, start_h : start_h + self.patch_size[0], start_w : start_w + self.patch_size[1]]
            label = label[:, :, start_h : start_h + self.patch_size[0], start_w : start_w + self.patch_size[1]]
        return image, label

    def normalize(self, image):
        if self.modality == "CT":
            return (image - self.ct_mean) / self.ct_std
        return self.normalize_intensity(image)

    def save(self, image, label, fname, test_metadata):
        mean, std = np.round(np.mean(image, (1, 2, 3)), 2), np.round(np.std(image, (1, 2, 3)), 2)
        print(f"Saving {fname} shape {image.shape} mean {mean} std {std}")
        self.save_npy(image, fname, "_x.npy")
        if label is not None:
            self.save_npy(label, fname, "_y.npy")
        if test_metadata is not None:
            self.save_npy(test_metadata, fname, "_meta.npy")

    def load_pair(self, pair):
        image = self.load_nifty(pair["image"] if isinstance(pair, dict) else pair)
        image_spacing = self.load_spacing(image)
        image = image.get_fdata().astype(np.float32)
        image = self.standardize_layout(image)

        if self.training:
            label = self.load_nifty(pair["label"]).get_fdata().astype(np.uint8)
            label = self.standardize_layout(label)
        else:
            label = None

        return image, label, image_spacing

    def resample_pair(self, image, label, spacing):
        shape = self.calculate_new_shape(spacing, image.shape[1:])
        if self.check_anisotrophy(spacing):
            image = self.resample_anisotrophic_image(image, shape)
            if label is not None:
                label = self.resample_anisotrophic_label(label, shape)
        else:
            image = self.resample_regular_image(image, shape)
            if label is not None:
                label = self.resample_regular_label(label, shape)
        image = image.astype(np.float32)
        if label is not None:
            label = label.astype(np.uint8)
        return image, label

    def calculate_pad_shape(self, image):
        min_shape = self.patch_size[:]
        image_shape = image.shape[1:]
        if len(min_shape) == 2:  # In 2D case we don't want to pad depth axis.
            min_shape.insert(0, image_shape[0])
        pad_shape = [max(mshape, ishape) for mshape, ishape in zip(min_shape, image_shape)]
        return pad_shape

    def get_intensities(self, pair):
        image = self.load_nifty(pair["image"]).get_fdata().astype(np.float32)
        label = self.load_nifty(pair["label"]).get_fdata().astype(np.uint8)
        foreground_idx = np.where(label > 0)
        intensities = image[foreground_idx].tolist()
        return intensities

    def collect_intensities(self):
        intensities = self.run_parallel(self.get_intensities, "training")
        intensities = list(itertools.chain(*intensities))
        self.ct_min, self.ct_max = np.percentile(intensities, [0.5, 99.5])
        self.ct_mean, self.ct_std = np.mean(intensities), np.std(intensities)

    def get_spacing(self, pair):
        image = nibabel.load(os.path.join(self.data_path, pair["image"]))
        spacing = self.load_spacing(image)
        return spacing

    def collect_spacings(self):
        spacing = self.run_parallel(self.get_spacing, "training")
        spacing = np.array(spacing)
        target_spacing = np.median(spacing, axis=0)
        if max(target_spacing) / min(target_spacing) >= 3:
            lowres_axis = np.argmin(target_spacing)
            target_spacing[lowres_axis] = np.percentile(spacing[:, lowres_axis], 10)
        self.target_spacing = list(target_spacing)

    def check_anisotrophy(self, spacing):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(spacing) or check(self.target_spacing)

    def calculate_new_shape(self, spacing, shape):
        spacing_ratio = np.array(spacing) / np.array(self.target_spacing)
        new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
        return new_shape

    def save_npy(self, image, fname, suffix):
        np.save(os.path.join(self.results, fname.replace(".nii.gz", suffix)), image, allow_pickle=False)

    def run_parallel(self, func, exec_mode):
        return Parallel(n_jobs=self.args.n_jobs)(delayed(func)(pair) for pair in self.metadata[exec_mode])

    def load_nifty(self, fname):
        return nibabel.load(os.path.join(self.data_path, fname))

    @staticmethod
    def load_spacing(image):
        return image.header["pixdim"][1:4].tolist()[::-1]

    @staticmethod
    def pad(image, padding):
        pad_d, pad_w, pad_h = padding
        return np.pad(
            image,
            (
                (0, 0),
                (math.floor(pad_d), math.ceil(pad_d)),
                (math.floor(pad_w), math.ceil(pad_w)),
                (math.floor(pad_h), math.ceil(pad_h)),
            ),
        )

    @staticmethod
    def standardize_layout(data):
        if len(data.shape) == 3:
            data = np.expand_dims(data, 3)
        return np.transpose(data, (3, 2, 1, 0))

    @staticmethod
    def resize_fn(image, shape, order, mode):
        return resize(image, shape, order=order, mode=mode, cval=0, clip=True, anti_aliasing=False)

    def resample_anisotrophic_image(self, image, shape):
        resized_channels = []
        for image_c in image:
            resized = [self.resize_fn(i, shape[1:], 3, "edge") for i in image_c]
            resized = np.stack(resized, axis=0)
            resized = self.resize_fn(resized, shape, 0, "constant")
            resized_channels.append(resized)
        resized = np.stack(resized_channels, axis=0)
        return resized

    def resample_regular_image(self, image, shape):
        resized_channels = []
        for image_c in image:
            resized_channels.append(self.resize_fn(image_c, shape, 3, "edge"))
        resized = np.stack(resized_channels, axis=0)
        return resized

    def resample_anisotrophic_label(self, label, shape):
        depth = label.shape[1]
        reshaped = np.zeros(shape, dtype=np.uint8)
        shape_2d = shape[1:]
        reshaped_2d = np.zeros((depth, *shape_2d), dtype=np.uint8)
        n_class = np.max(label)
        for class_ in range(1, n_class + 1):
            for depth_ in range(depth):
                mask = label[0, depth_] == class_
                resized_2d = self.resize_fn(mask.astype(float), shape_2d, 1, "edge")
                reshaped_2d[depth_][resized_2d >= 0.5] = class_

        for class_ in range(1, n_class + 1):
            mask = reshaped_2d == class_
            resized = self.resize_fn(mask.astype(float), shape, 0, "constant")
            reshaped[resized >= 0.5] = class_
        reshaped = np.expand_dims(reshaped, 0)
        return reshaped

    def resample_regular_label(self, label, shape):
        reshaped = np.zeros(shape, dtype=np.uint8)
        n_class = np.max(label)
        for class_ in range(1, n_class + 1):
            mask = label[0] == class_
            resized = self.resize_fn(mask.astype(float), shape, 1, "edge")
            reshaped[resized >= 0.5] = class_
        reshaped = np.expand_dims(reshaped, 0)
        return reshaped


class ind_Preprocessor:
    def __init__(self, args):
        self.args = args
        self._mean = 0
        self._std = 0
        self.task = args.task
        self.task_code = get_task_code(args)
        self.patch_size = ind_patch_size[self.task_code]
        self.training = args.exec_mode == "training"
        self.data_path = os.path.join(args.data, ind_task[args.task])
        self.results = os.path.join(args.results, self.task_code)
        if not self.training:
            self.results = os.path.join(self.results, self.args.exec_mode)
        

    def run(self):
        make_empty_dir(self.results)

        print(f"Preprocessing {self.data_path}")
        
        if self.training:
            self.collect_intensities()
        else:
            self._mean = ind_mean[self.task]
            self._std  = ind_std[self.task]

        self.run_parallel(self.preprocess_pair, self.args.exec_mode)

        pickle.dump(
            {
                "patch_size": self.patch_size,
                "n_class": 2,
                "in_channels": 1,
            },
            open(os.path.join(self.results, "config.pkl"), "wb"),
        )

    def preprocess_pair(self, pair):
        fname = os.path.basename(pair["image"] if isinstance(pair, dict) else pair)
        image, label = self.load_pair(pair)
        test_metadata = None
        
        image = self.normalize(image)
        if self.training:
            image, label = self.standardize(image, label)
            if self.args.dilation:
                new_lbl = np.zeros(label.shape, dtype=np.uint8)
                for depth in range(label.shape[1]):
                    new_lbl[0, depth] = erosion(dilation(label[0, depth], square(3)), square(3))
                label = new_lbl
        self.save(image, label, fname, test_metadata)


    def standardize(self, image, label):
        pad_shape = self.calculate_pad_shape(image)
        image_shape = image.shape[1:]
        if pad_shape != image_shape:
            paddings = [(pad_sh - image_sh) / 2 for (pad_sh, image_sh) in zip(pad_shape, image_shape)]
            image = self.pad(image, paddings)
            label = self.pad(label, paddings)
        if self.args.dim == 2:  # Center cropping 2D images.
            _, _, height, weight = image.shape
            start_h = (height - self.patch_size[0]) // 2
            start_w = (weight - self.patch_size[1]) // 2
            image = image[:, :, start_h : start_h + self.patch_size[0], start_w : start_w + self.patch_size[1]]
            label = label[:, :, start_h : start_h + self.patch_size[0], start_w : start_w + self.patch_size[1]]
        return image, label

    def normalize(self, image):
        return (image - self._mean) / self._std
        

    def save(self, image, label, fname, test_metadata):
        mean, std = np.round(np.mean(image, (1, 2, 3)), 2), np.round(np.std(image, (1, 2, 3)), 2)
        print(f"Saving {fname} shape {image.shape} mean {mean} std {std}")
        self.save_npy(image, fname, "_x.npy")
        if label is not None:
            self.save_npy(label, fname, "_y.npy")
        if test_metadata is not None:
            self.save_npy(test_metadata, fname, "_meta.npy")

    def load_pair(self, pair):

        image = imread(pair["image"] if isinstance(pair, dict) else pair)

        image = self.standardize_layout(image.astype(np.float32))

        if len(pair["label"]):
            label = (imread(pair["label"]) > 127).astype(np.uint8)
            label = self.standardize_layout(label)
        else:
            label = np.zeros(image.shape, dtype=np.uint8)
        
        return image, label

    def calculate_pad_shape(self, image):
        min_shape = self.patch_size[:]
        image_shape = image.shape[1:]
        if len(min_shape) == 2:  # In 2D case we don't want to pad depth axis.
            min_shape.insert(0, image_shape[0])
        pad_shape = [max(mshape, ishape) for mshape, ishape in zip(min_shape, image_shape)]
        return pad_shape

    def get_intensities(self, pair):
        image = imread(pair["image"]).astype(np.float32)
        # label = imread(pair["label"]).astype(np.uint8)
        # foreground_idx = np.where(label > 0)
        intensities = image.tolist() #image[foreground_idx].tolist()
        return intensities

    def collect_intensities(self):
        intensities = self.run_parallel(self.get_intensities, "training")
        intensities = list(itertools.chain(*intensities))
        # self._min, self._max = np.percentile(intensities, [0.5, 99.5])
        self._mean, self._std = np.mean(intensities), np.std(intensities)
        print(f"overall {ind_task[self.task]} : mean {self._mean} std {self._std}")

    # def get_spacing(self, pair):
    #     image = nibabel.load(os.path.join(self.data_path, pair["image"]))
    #     spacing = self.load_spacing(image)
    #     return spacing

    # def collect_spacings(self):
    #     spacing = self.run_parallel(self.get_spacing, "training")
    #     spacing = np.array(spacing)
    #     target_spacing = np.median(spacing, axis=0)
    #     if max(target_spacing) / min(target_spacing) >= 3:
    #         lowres_axis = np.argmin(target_spacing)
    #         target_spacing[lowres_axis] = np.percentile(spacing[:, lowres_axis], 10)
    #     self.target_spacing = list(target_spacing)

    # def check_anisotrophy(self, spacing):
    #     def check(spacing):
    #         return np.max(spacing) / np.min(spacing) >= 3

    #     return check(spacing) or check(self.target_spacing)

    # def calculate_new_shape(self, spacing, shape):
    #     spacing_ratio = np.array(spacing) / np.array(self.target_spacing)
    #     new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
    #     return new_shape

    def save_npy(self, image, fname, suffix):
        np.save(os.path.join(self.results, fname.replace(".PNG", suffix)), image, allow_pickle=False)

    def run_parallel(self, func, exec_mode):
        metafiles = get_fpath_pairs(self.data_path, exec_mode)
        return Parallel(n_jobs=self.args.n_jobs)(delayed(func)(pair) for pair in metafiles)

    # def load_nifty(self, fname):
    #     return nibabel.load(os.path.join(self.data_path, fname))

    # @staticmethod
    # def load_spacing(image):
    #     return image.header["pixdim"][1:4].tolist()[::-1]

    @staticmethod
    def pad(image, padding):
        pad_d, pad_w, pad_h = padding
        return np.pad(
            image,
            (
                (0, 0),
                (math.floor(pad_d), math.ceil(pad_d)),
                (math.floor(pad_w), math.ceil(pad_w)),
                (math.floor(pad_h), math.ceil(pad_h)),
            ),
        )

    @staticmethod
    def standardize_layout(data):
        if len(data.shape) == 3:
            data = np.expand_dims(data, 3)
        elif len(data.shape) == 2:
            data = np.expand_dims(data, (2, 3))
        return np.transpose(data, (3, 2, 0, 1))

    # @staticmethod
    # def resize_fn(image, shape, order, mode):
    #     return resize(image, shape, order=order, mode=mode, cval=0, clip=True, anti_aliasing=False)

    # def resample_anisotrophic_image(self, image, shape):
    #     resized_channels = []
    #     for image_c in image:
    #         resized = [self.resize_fn(i, shape[1:], 3, "edge") for i in image_c]
    #         resized = np.stack(resized, axis=0)
    #         resized = self.resize_fn(resized, shape, 0, "constant")
    #         resized_channels.append(resized)
    #     resized = np.stack(resized_channels, axis=0)
    #     return resized

    # def resample_regular_image(self, image, shape):
    #     resized_channels = []
    #     for image_c in image:
    #         resized_channels.append(self.resize_fn(image_c, shape, 3, "edge"))
    #     resized = np.stack(resized_channels, axis=0)
    #     return resized

    # def resample_anisotrophic_label(self, label, shape):
    #     depth = label.shape[1]
    #     reshaped = np.zeros(shape, dtype=np.uint8)
    #     shape_2d = shape[1:]
    #     reshaped_2d = np.zeros((depth, *shape_2d), dtype=np.uint8)
    #     n_class = np.max(label)
    #     for class_ in range(1, n_class + 1):
    #         for depth_ in range(depth):
    #             mask = label[0, depth_] == class_
    #             resized_2d = self.resize_fn(mask.astype(float), shape_2d, 1, "edge")
    #             reshaped_2d[depth_][resized_2d >= 0.5] = class_

    #     for class_ in range(1, n_class + 1):
    #         mask = reshaped_2d == class_
    #         resized = self.resize_fn(mask.astype(float), shape, 0, "constant")
    #         reshaped[resized >= 0.5] = class_
    #     reshaped = np.expand_dims(reshaped, 0)
    #     return reshaped

    # def resample_regular_label(self, label, shape):
    #     reshaped = np.zeros(shape, dtype=np.uint8)
    #     n_class = np.max(label)
    #     for class_ in range(1, n_class + 1):
    #         mask = label[0] == class_
    #         resized = self.resize_fn(mask.astype(float), shape, 1, "edge")
    #         reshaped[resized >= 0.5] = class_
    #     reshaped = np.expand_dims(reshaped, 0)
    #     return reshaped
