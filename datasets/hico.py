"""
HICO detection dataset.
"""
from pathlib import Path

import torchvision.transforms
from PIL import Image
import json
from collections import defaultdict
import numpy as np

import torch
import torch.utils.data
import clip
#biomed_clip
# from open_clip import create_model_from_pretrained, get_tokenizer

import torchvision.transforms.functional as F
import datasets.transforms as T
from .hico_text_label import hico_text_label, hico_unseen_index
from util.box_ops import box_cxcywh_to_xyxy


class HICODetection(torch.utils.data.Dataset):
    def __init__(self, img_set, img_folder, anno_file, clip_feats_folder, transforms, num_queries, args):
        self.img_set = img_set
        self.img_folder = img_folder
        self.clip_feates_folder = clip_feats_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

        self.num_queries = num_queries

        self.unseen_index = hico_unseen_index.get(args.zero_shot_type, [])
        self._valid_obj_ids = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)
        self._valid_verb_ids = list(range(0, 10))

        self.text_label_dict = hico_text_label
        self.text_label_ids = list(self.text_label_dict.keys())
        if img_set == 'train' and len(self.unseen_index) != 0 and args.del_unseen:
            tmp = []
            for idx, k in enumerate(self.text_label_ids):
                if idx in self.unseen_index:
                    continue
                else:
                    tmp.append(k)
            self.text_label_ids = tmp

        if img_set == 'train':
            self.ids = []
            for idx, img_anno in enumerate(self.annotations):
                new_img_anno = []
                skip_pair = []
                for hoi in img_anno['hoi_annotation']:
                    if hoi['hoi_category_id'] - 1 in self.unseen_index:
                        skip_pair.append((hoi['subject_id'], hoi['object_id']))
                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(
                            img_anno['annotations']):
                        new_img_anno = []
                        break
                    if (hoi['subject_id'], hoi['object_id']) not in skip_pair:
                        new_img_anno.append(hoi)
                if len(new_img_anno) > 0:
                    self.ids.append(idx)
                    img_anno['hoi_annotation'] = new_img_anno
        else:
            self.ids = list(range(len(self.annotations)))
        print("{} contains {} images".format(img_set, len(self.ids)))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.clip_preprocess = clip.load(args.clip_model, device)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        nframes = 5
        
        img_anno = self.annotations[self.ids[idx]]

        image = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        w, h = image.size

        ##mamba
        img_anno_dict = {}
        image_dict = {}
        for n in range(1, nframes):
            if idx-nframes+n >= 0:
                img_anno_dict[n] = self.annotations[self.ids[idx-nframes+n]]
                image_dict[n] = Image.open(self.img_folder / img_anno_dict[n]['file_name']).convert('RGB')
            else:
                img_anno_dict[n] = self.annotations[self.ids[idx]]
                image_dict[n] = Image.open(self.img_folder / img_anno_dict[n]['file_name']).convert('RGB')
            
        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        if self.img_set == 'train':
            # Add index for confirming which boxes are kept after image transformation
            classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in
                       enumerate(img_anno['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(obj['category_id']) for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        if self.img_set == 'train':
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            target['boxes'] = boxes
            target['labels'] = classes
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            
            
            if self._transforms is not None:
                if nframes == 1:
                    img_0, target_0 = self._transforms[0](image, target)
                    img, target = self._transforms[1](img_0, target_0)
                if nframes == 3:
                    img_0, image_dict[1], image_dict[2], target_0 = self._transforms[0](image, image_dict[1], image_dict[2], target)
                    img, image_dict[1], image_dict[2], target = self._transforms[1](img_0, image_dict[1], image_dict[2], target_0) ##box -> cxcywh
                if nframes == 5:
                    img_0, image_dict[1], image_dict[2], image_dict[3], image_dict[4], target_0 = self._transforms[0](image, image_dict[1], image_dict[2], image_dict[3], image_dict[4], target)
                    img, image_dict[1], image_dict[2], image_dict[3], image_dict[4], target = self._transforms[1](img_0, image_dict[1], image_dict[2], image_dict[3], image_dict[4], target_0) ##box -> cxcywh
                if nframes == 10:
                    img_0, image_dict[1], image_dict[2], image_dict[3], image_dict[4], image_dict[5], image_dict[6], image_dict[7], image_dict[8], image_dict[9], target_0 = self._transforms[0](
                        image, image_dict[1], image_dict[2], image_dict[3], image_dict[4], image_dict[5], image_dict[6], image_dict[7], image_dict[8], image_dict[9], target)
                    img, image_dict[1], image_dict[2], image_dict[3], image_dict[4], image_dict[5], image_dict[6], image_dict[7], image_dict[8], image_dict[9], target = self._transforms[1](
                        img_0, image_dict[1], image_dict[2], image_dict[3], image_dict[4], image_dict[5], image_dict[6], image_dict[7], image_dict[8], image_dict[9], target_0) ##box -> cxcywh
            for n in range(1,nframes):
                if image_dict[n].shape != img.shape:
                    image_dict[n] = self.align_images(img, image_dict[n])
            img_mix = torch.stack([image_dict[n] for n in range(1,nframes)] + [img], dim=0)

            clip_inputs = self.clip_preprocess(img_0)
            target['clip_inputs'] = clip_inputs
            kept_box_indices = [label[0] for label in target['labels']]

            target['labels'] = target['labels'][:, 1]

            sub_labels, obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], [], []
            sub_obj_pairs = []
            hoi_labels = []
            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                    continue
                ins_verb_tar_pair = (
                    target["labels"][kept_box_indices.index(hoi["subject_id"])],
                    self._valid_verb_ids.index(hoi['category_id']),
                    target['labels'][kept_box_indices.index(hoi['object_id'])]
                )
                if ins_verb_tar_pair not in self.text_label_ids:
                    continue

                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                    hoi_labels[sub_obj_pairs.index(sub_obj_pair)][self.text_label_ids.index(ins_verb_tar_pair)] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    sub_labels.append(target['labels'][kept_box_indices.index(hoi['subject_id'])])
                    obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    hoi_label = [0] * len(self.text_label_ids)
                    hoi_label[self.text_label_ids.index(ins_verb_tar_pair)] = 1
                    verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1
                    sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                    obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                    verb_labels.append(verb_label)
                    hoi_labels.append(hoi_label)
                    sub_boxes.append(sub_box)
                    obj_boxes.append(obj_box)
            
            target['filename'] = img_anno['file_name']
            if len(sub_obj_pairs) == 0:
                target['sub_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['hoi_labels'] = torch.zeros((0, len(self.text_label_ids)), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            else:
                target['sub_labels'] = torch.stack(sub_labels)
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['hoi_labels'] = torch.as_tensor(hoi_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes) #cxcywh 0-1 scale
                target['obj_boxes'] = torch.stack(obj_boxes)
 
        else:
            target['filename'] = img_anno['file_name']
            target['boxes'] = boxes
            target['labels'] = classes
            target['id'] = idx
                
            if self._transforms is not None:
                if nframes == 1:
                    img, _ = self._transforms(image, None)
                if nframes == 3:
                    img, image_dict[1], image_dict[2], _ = self._transforms(image, image_dict[1], image_dict[2], None)
                if nframes == 5:
                    img, image_dict[1], image_dict[2], image_dict[3], image_dict[4], _ = self._transforms(image, image_dict[1], image_dict[2], image_dict[3], image_dict[4], None)
                if nframes == 10:
                    img, image_dict[1], image_dict[2], image_dict[3], image_dict[4], image_dict[5], image_dict[6], image_dict[7], image_dict[8], image_dict[9], _ = self._transforms(
                        image, image_dict[1], image_dict[2], image_dict[3], image_dict[4], image_dict[5], image_dict[6], image_dict[7], image_dict[8], image_dict[9], None)
            for n in range(1, nframes):
                if image_dict[n].shape != img.shape:
                    image_dict[n] = self.align_images(img, image_dict[n])
            img_mix = torch.stack([image_dict[n] for n in range(1,nframes)] + [img], dim=0)

            hois = []
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)
                
        return img_mix, target

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)
    
    def align_images(self, image1, image2):
        _, h1, w1 = image1.size()
        _, h2, w2 = image2.size()

        if h2 > h1 or w2 > w1:
            image2 = F.center_crop(image2, (h1, w1))
            return image2
        else:
            aligned_image2 = torch.zeros_like(image1)

            h_diff = (h1 - h2) // 2
            w_diff = (w1 - w2) // 2
            aligned_image2[:, h_diff:h_diff+h2, w_diff:w_diff+w2] = image2

            return aligned_image2

def make_hico_transforms(image_set):
    normalize = T.Compose_five([
        T.ToTensor_five(),
        T.Normalize_five([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400]

    if image_set == 'train':
        return [T.Compose_five([
            T.RandomHorizontalFlip_five(),
            T.ColorJitter_five(.4, .4, .4),
            T.RandomSelect_five(
                T.RandomResize_five(scales, max_size=1333),
                T.Compose_five([
                    T.RandomResize_five([200, 250, 300]),
                    T.RandomSizeCrop_five(192, 300),
                    T.RandomResize_five(scales, max_size=1333),
                ]))]
            ),
            normalize
            ]

    if image_set == 'val':
        return T.Compose_five([
            T.RandomResize_five([400], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (root / 'images' / 'train', root / 'annotations' / 'train.json',
                  root / 'clip_feats_pool' / 'train2015'),
        'val': (
            root / 'images' / 'test', root / 'annotations' / 'eval.json',
            root / 'clip_feats_pool' / 'test2015')
    }
    
    CORRECT_MAT_PATH = root / 'annotations' / 'corre_hico.npy'

    img_folder, anno_file, clip_feats_folder = PATHS[image_set]
   
    dataset = HICODetection(image_set, img_folder, anno_file, clip_feats_folder,
                            transforms=make_hico_transforms(image_set),
                            num_queries=args.num_queries, args=args)
    if image_set == 'val':
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset