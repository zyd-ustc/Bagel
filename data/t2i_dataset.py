# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import pyarrow.parquet as pq
import random
from PIL import Image

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths, init_arrow_pf_fs

Image.MAX_IMAGE_PIXELS = 20_000_000


class T2IIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None, vit_transform=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data):
        return get_parquet_data_paths(data_dir_list, num_used_data)

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]

                        for row_idx, row in df.iterrows():
                            num_tokens = 0
                            try:
                                # Load ground truth image (for VAE)
                                gt_image_byte = row['ground_truth_image']
                                gt_image = pil_img2rgb(Image.open(io.BytesIO(gt_image_byte)))
                                vae_image_tensor = self.transform(gt_image)
                                
                                # Load generated image (for ViT/Understanding)
                                gen_image_byte = row['generated_image']
                                gen_image = pil_img2rgb(Image.open(io.BytesIO(gen_image_byte)))
                                vit_image_tensor = self.vit_transform(gen_image)

                            except Exception as e:
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue
                            
                            # Calculate tokens for VAE image
                            height, width = vae_image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2
                            
                            # Calculate tokens for ViT image (approximate, actual calculation in PackedDataset)
                            # Assuming standard patch size for token count estimation if needed, 
                            # but PackedDataset handles it. We just need to ensure max_num_tokens check works.
                            # For now, adding a rough estimate or relying on PackedDataset to handle overflow.
                            # Let's add ViT tokens to num_tokens estimate.
                            # Assuming 14x14 patch size and 224x224 image -> 256 tokens.
                            # Better to use the actual vit_transform size if available, but for now just add a safe buffer or 0.
                            # The PackedDataset will check actual length.
                            
                            try:
                                prompt = row['prompt']
                            except Exception as e:
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue

                            caption_token = self.tokenizer.encode(prompt)
                            
                            sequence_plan, text_ids_list, image_tensor_list = [], [], []
                            
                            # 1. Text (Prompt)
                            text_ids = caption_token
                            num_tokens += len(caption_token)
                            text_ids_list.append(text_ids)
                            sequence_plan.append({
                                'type': 'text',
                                'enable_cfg': 1,
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            # 2. ViT Image (Generated Image) - for understanding/past_key_values
                            image_tensor_list.append(vit_image_tensor)
                            sequence_plan.append({
                                'type': 'vit_image',
                                'enable_cfg': 1, # or 0? Usually we want to condition on it.
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            # 3. VAE Image (Ground Truth) - for generation target
                            image_tensor_list.append(vae_image_tensor)
                            sequence_plan.append({
                                'type': 'vae_image',
                                'enable_cfg': 0,
                                'loss': 1,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            sample = dict(
                                image_tensor_list=image_tensor_list, 
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            )
                            yield sample

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
