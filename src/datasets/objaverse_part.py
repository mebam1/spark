from src.utils.typing_utils import *

import json
import os
import random

import accelerate
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.data_utils import load_surface, load_surfaces

class ObjaversePartDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs: DictConfig, 
        training: bool = True, 
    ):
        super().__init__()
        self.configs = configs
        self.training = training

        self.min_num_parts = configs['dataset']['min_num_parts']
        self.max_num_parts = configs['dataset']['max_num_parts']
        self.val_min_num_parts = configs['val']['min_num_parts']
        self.val_max_num_parts = configs['val']['max_num_parts']

        self.max_iou_mean = configs['dataset'].get('max_iou_mean', None)
        self.max_iou_max = configs['dataset'].get('max_iou_max', None)

        self.shuffle_parts = configs['dataset']['shuffle_parts']
        self.training_ratio = configs['dataset']['training_ratio']
        self.balance_object_and_parts = configs['dataset'].get('balance_object_and_parts', False)

        self.rotating_ratio = configs['dataset'].get('rotating_ratio', 0.0)
        self.rotating_degree = configs['dataset'].get('rotating_degree', 10.0)
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-self.rotating_degree, self.rotating_degree), fill=(255, 255, 255)),
        ])

        if isinstance(configs['dataset']['config'], ListConfig):
            data_configs = []
            for config in configs['dataset']['config']:
                local_data_configs = json.load(open(config))
                if self.balance_object_and_parts:
                    if self.training:
                        local_data_configs = local_data_configs[:int(len(local_data_configs) * self.training_ratio)]
                    else:
                        local_data_configs = local_data_configs[int(len(local_data_configs) * self.training_ratio):]
                        local_data_configs = [config for config in local_data_configs if self.val_min_num_parts <= config['num_parts'] <= self.val_max_num_parts]
                data_configs += local_data_configs
        else:
            data_configs = json.load(open(configs['dataset']['config']))
        data_configs = [config for config in data_configs if config['valid']]
        data_configs = [config for config in data_configs if self.min_num_parts <= config['num_parts'] <= self.max_num_parts]
        if self.max_iou_mean is not None and self.max_iou_max is not None:
            data_configs = [config for config in data_configs if config['iou_mean'] <= self.max_iou_mean]
            data_configs = [config for config in data_configs if config['iou_max'] <= self.max_iou_max]
        if not self.balance_object_and_parts:
            if self.training:
                data_configs = data_configs[:int(len(data_configs) * self.training_ratio)]
            else:
                data_configs = data_configs[int(len(data_configs) * self.training_ratio):]
                data_configs = [config for config in data_configs if self.val_min_num_parts <= config['num_parts'] <= self.val_max_num_parts]
        self.data_configs = data_configs
        self.image_size = (512, 512)

    def __len__(self) -> int:
        return len(self.data_configs)
    
    def _get_data_by_config(self, data_config):
        if 'surface_path' in data_config:
            surface_path = data_config['surface_path']
            surface_data = np.load(surface_path, allow_pickle=True).item()
            # If parts is empty, the object is the only part
            part_surfaces = surface_data['parts'] if len(surface_data['parts']) > 0 else [surface_data['object']]
            num_parts = len(part_surfaces)

            # Create part indices for tracking shuffle order
            part_indices = list(range(num_parts))

            if self.shuffle_parts:
                # Shuffle both parts and indices together to maintain correspondence
                combined = list(zip(part_surfaces, part_indices))
                random.shuffle(combined)
                part_surfaces, part_indices = zip(*combined)
                part_surfaces = list(part_surfaces)
                part_indices = list(part_indices)

            part_surfaces = load_surfaces(part_surfaces) # [N, P, 6]
        else:
            part_surfaces = []
            for surface_path in data_config['surface_paths']:
                surface_data = np.load(surface_path, allow_pickle=True).item()
                part_surfaces.append(load_surface(surface_data))
            part_surfaces = torch.stack(part_surfaces, dim=0) # [N, P, 6]
            num_parts = part_surfaces.shape[0]
            part_indices = list(range(num_parts))

        # Load per-part images
        # Extract base directory from image_path (remove filename)
        base_dir = os.path.dirname(data_config['image_path'])

        # Note: With position_embedding, shuffle_parts is now safe
        # Each image gets correct position embedding based on part_positions

        images = []
        for i, part_idx in enumerate(part_indices):
            # Load image for each part: link_0.png, link_1.png, etc.
            part_image_path = os.path.join(base_dir, f"link_{part_idx}.png")

            if os.path.exists(part_image_path):
                # Load part-specific image
                image = Image.open(part_image_path).resize(self.image_size)
                # Debug logging removed to prevent stdout buffer issues in multi-GPU training
            else:
                # Fallback to global image if part image doesn't exist
                # Only log missing files in single process to avoid stdout issues
                image = Image.open(data_config['image_path']).resize(self.image_size)

            if random.random() < self.rotating_ratio:
                image = self.transform(image)
            image = np.array(image)
            image = torch.from_numpy(image).to(torch.uint8) # [H, W, 3]
            images.append(image)

        images = torch.stack(images, dim=0) # [N, H, W, 3]

        # Load parent_indices for hierarchy attention
        base_dir = os.path.dirname(data_config['surface_path'])
        parent_indices_path = os.path.join(base_dir, 'parent_indices.json')

        if os.path.exists(parent_indices_path):
            with open(parent_indices_path, 'r') as f:
                parent_data = json.load(f)
                original_parent_indices = parent_data['parent_indices']

            # If parts were shuffled, remap parent_indices to match new order
            if self.shuffle_parts:
                # Create mapping: old_index -> new_index
                old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(part_indices)}

                # Remap parent indices
                parent_indices = []
                for new_idx, old_idx in enumerate(part_indices):
                    old_parent = original_parent_indices[old_idx]
                    if old_parent == -1:
                        # Root node stays as -1
                        parent_indices.append(-1)
                    else:
                        # Map old parent index to new parent index
                        new_parent = old_to_new[old_parent]
                        parent_indices.append(new_parent)
            else:
                parent_indices = original_parent_indices[:num_parts]
        else:
            # Fallback: all parts are roots (no hierarchy)
            parent_indices = [-1] * num_parts

        result = {
            "images": images,           # Per-part images [N, H, W, 3] - for model input
            "part_surfaces": part_surfaces,
            "part_positions": part_indices,  # Absolute position indices (e.g., [0, 1, 2, ...] or shuffled [2, 0, 1, ...])
            "parent_indices": parent_indices,  # Parent index for each part (-1 for root)
        }

        # Load global image - for validation display and global attention during training
        global_image = Image.open(data_config['image_path']).resize(self.image_size)
        if random.random() < self.rotating_ratio:
            global_image = self.transform(global_image)
        global_image = np.array(global_image)
        global_image = torch.from_numpy(global_image).to(torch.uint8) # [H, W, 3]

        # Replicate global image to match part images dimensions: [H, W, 3] -> [N, H, W, 3]
        global_image = global_image.unsqueeze(0).repeat(num_parts, 1, 1, 1) # [N, H, W, 3]
        result["global_image"] = global_image  # Global image [N, H, W, 3] - aligned with per-part images

        # Debug logging removed to prevent stdout buffer overflow in multi-GPU training
        # Shapes: images [N, H, W, 3], part_surfaces [N, P, 6], global_image [N, H, W, 3]

        return result
    
    def __getitem__(self, idx: int):
        # The dataset can only support batchsize == 1 training. 
        # Because the number of parts is not fixed.
        # Please see BatchedObjaversePartDataset for batched training.
        data_config = self.data_configs[idx]
        data = self._get_data_by_config(data_config)
        return data
        
class BatchedObjaversePartDataset(ObjaversePartDataset):
    def __init__(
        self,
        configs: DictConfig,
        batch_size: int,
        is_main_process: bool = False,
        shuffle: bool = True,
        training: bool = True,
    ):
        assert training
        assert batch_size > 1
        super().__init__(configs, training)
        self.batch_size = batch_size
        self.is_main_process = is_main_process
        if batch_size < self.max_num_parts:
            self.data_configs = [config for config in self.data_configs if config['num_parts'] <= batch_size]
        
        if shuffle:
            random.shuffle(self.data_configs)

        self.object_configs = [config for config in self.data_configs if config['num_parts'] == 1]
        self.parts_configs = [config for config in self.data_configs if config['num_parts'] > 1]
        
        self.object_ratio = configs['dataset']['object_ratio']
        # Here we keep the ratio of object to parts
        self.object_configs = self.object_configs[:int(len(self.parts_configs) * self.object_ratio)]

        dropped_data_configs = self.parts_configs + self.object_configs
        if shuffle:
            random.shuffle(dropped_data_configs)

        self.data_configs = self._get_batched_configs(dropped_data_configs, batch_size)
    
    def _get_batched_configs(self, data_configs, batch_size):
        batched_data_configs = []
        num_data_configs = len(data_configs)
        progress_bar = tqdm(
            range(len(data_configs)),
            desc="Batching Dataset",
            ncols=125,
            disable=not self.is_main_process,
        )
        while len(data_configs) > 0:
            temp_batch = []
            temp_num_parts = 0
            unchosen_configs = []
            while temp_num_parts < batch_size and len(data_configs) > 0:
                config = data_configs.pop() # pop the last config
                num_parts = config['num_parts']
                if temp_num_parts + num_parts <= batch_size:
                    temp_batch.append(config)
                    temp_num_parts += num_parts
                    progress_bar.update(1)
                else:
                    unchosen_configs.append(config) # add back to the end
            data_configs = data_configs + unchosen_configs # concat the unchosen configs
            if temp_num_parts == batch_size:
                # Successfully get a batch
                if len(temp_batch) < batch_size:
                    # pad the batch
                    temp_batch += [{}] * (batch_size - len(temp_batch))
                batched_data_configs += temp_batch
                # Else, the code enters here because len(data_configs) == 0
                # which means in the left data_configs, there are no enough 
                # "suitable" configs to form a batch. 
                # Thus, drop the uncompleted batch.
        progress_bar.close()
        return batched_data_configs
        
    def __getitem__(self, idx: int):
        data_config = self.data_configs[idx]
        if len(data_config) == 0:
            # placeholder
            return {}
        data = self._get_data_by_config(data_config)
        return data
    
    def collate_fn(self, batch):
        batch = [data for data in batch if len(data) > 0]

        # Collect all images and part surfaces
        all_images = []
        all_global_images = []
        all_surfaces = []
        all_part_positions = []
        all_parent_indices = []
        num_parts_list = []

        # Debug logging removed to prevent stdout buffer issues in multi-GPU training
        # Process each object in the batch
        offset = 0  # Track global part index offset for parent_indices
        for obj_idx, data in enumerate(batch):
            obj_images = data['images']
            obj_global_image = data['global_image']
            obj_surfaces = data['part_surfaces']
            obj_part_positions = data['part_positions']
            obj_parent_indices = data['parent_indices']
            obj_num_parts = obj_surfaces.shape[0]

            all_images.append(obj_images)
            all_global_images.append(obj_global_image)
            all_surfaces.append(obj_surfaces)
            all_part_positions.extend(obj_part_positions)
            num_parts_list.append(obj_num_parts)

            # Adjust parent indices by offset (for batching multiple objects)
            for parent_id in obj_parent_indices:
                if parent_id >= 0:
                    # Add offset to map to global batch indices
                    all_parent_indices.append(parent_id + offset)
                else:
                    # Root node, keep as -1
                    all_parent_indices.append(-1)

            offset += obj_num_parts

        images = torch.cat(all_images, dim=0) # [N, H, W, 3] - per-part images
        global_images = torch.cat(all_global_images, dim=0) # [N, H, W, 3] - global images (replicated per part)
        surfaces = torch.cat(all_surfaces, dim=0) # [N, P, 6]
        num_parts = torch.LongTensor(num_parts_list)
        part_positions = torch.LongTensor(all_part_positions)  # [N] - absolute position indices
        parent_indices = torch.LongTensor(all_parent_indices)  # [N] - parent index for each part

        total_parts = num_parts.sum().item()

        assert images.shape[0] == surfaces.shape[0] == total_parts == self.batch_size, \
            f"Shape mismatch: per-part images={images.shape[0]}, surfaces={surfaces.shape[0]}, total_parts={total_parts}, batch_size={self.batch_size}"
        assert global_images.shape[0] == images.shape[0] == total_parts == self.batch_size, \
            f"Global images batch mismatch: global_images={global_images.shape[0]}, per-part images={images.shape[0]}, total_parts={total_parts}, batch_size={self.batch_size}"
        assert part_positions.shape[0] == total_parts == self.batch_size, \
            f"Part positions batch mismatch: part_positions={part_positions.shape[0]}, total_parts={total_parts}, batch_size={self.batch_size}"
        assert parent_indices.shape[0] == total_parts == self.batch_size, \
            f"Parent indices batch mismatch: parent_indices={parent_indices.shape[0]}, total_parts={total_parts}, batch_size={self.batch_size}"

        batch = {
            "images": images,                # Per-part images [N, H, W, 3] for local attention
            "global_image": global_images,   # Global images [N, H, W, 3] for global attention (replicated per part)
            "part_surfaces": surfaces,
            "num_parts": num_parts,
            "part_positions": part_positions,  # Absolute position indices [N]
            "parent_indices": parent_indices,  # Parent index for each part [N] (-1 for root)
        }
        return batch