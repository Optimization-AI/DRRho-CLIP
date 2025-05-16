import logging
import math
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value
import functools
import glob

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

from .webdata import WebDataset, pytorch_local_worker_info

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def expand_urls(urls, weights=None):
    if weights is None:
        wds_expanded_urls = wds.shardlists.expand_urls(urls)
        # wds.shardlists.expand_urls leverages braceexpand, which does not support wildcards
        expanded_urls = []
        for url in wds_expanded_urls:
            if '*' in url:
                expanded_urls.extend(glob.glob(url))
            else:
                expanded_urls.append(url)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def extract_data_index(data_url: str):
    index = -1
    try:
        index = int(data_url.split('/')[-1].split('.')[0])
    except:
        pass
    return index


def build_cached_data_index(cached_dir):
    cached_data_index = None
    if cached_dir:
        cached_data_list = expand_urls(cached_dir)[0]
        if len(cached_data_list) > 0:
            cached_data_list.sort(key=extract_data_index)
            max_data_id = extract_data_index(cached_data_list[-1])
            cached_data_index = ['' for _ in range(max_data_id + 1)]
            for cached_data in cached_data_list:
                id = extract_data_index(cached_data)
                cached_data_index[id] = cached_data
    return cached_data_index


def get_wds_dataset(args, preprocess_fns, is_train, epoch=0, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    preprocess_train, preprocess_val = preprocess_fns
    preprocess_img = preprocess_train if is_train else preprocess_val
    return_index = True

    shards_list = expand_urls(input_shards)[0]
    shards_list.sort(key=extract_data_index)
    shards_id_list = None
    if args.cache_ref_model_features or args.cache_syn_texts:
        preprocess_img = preprocess_val
        if args.ref_features_offset < 0:
            args.ref_features_offset = extract_data_index(shards_list[0]) * args.num_samples_per_shard
        if args.syn_texts_offset < 0:
            args.syn_texts_offset = extract_data_index(shards_list[0]) * args.num_samples_per_shard
        if args.cache_group_size > 1:
            assert args.world_size == 1, "World size must be 1 if cache_group_size is specified"
            group_size = args.cache_group_size
            rank = args.cache_rank
        else:
            group_size = args.world_size
            rank = args.rank
        shards_list_rank = []
        num_shards_per_rank = (len(shards_list) + group_size - 1) // group_size
        for i in range(group_size):
            shards_list_rank.append(shards_list[i * num_shards_per_rank: min(len(shards_list), (i+ 1) * num_shards_per_rank)])
        shards_list = shards_list_rank[rank]
        shards_id_list = [extract_data_index(shard) for shard in shards_list]
        args.data_size = len(shards_list) * args.num_samples_per_shard
        args.train_num_samples = args.data_size
        args.ref_features_offset = args.ref_features_offset + rank * num_shards_per_rank * args.num_samples_per_shard
        args.syn_texts_offset = args.syn_texts_offset + rank * num_shards_per_rank * args.num_samples_per_shard
        logging.info("Using validation image augmentation for caching.")
    logging.info((f"Length of expanded urls: {len(shards_list)}, "
                  f"smallest shard id: {extract_data_index(shards_list[0])}, "
                  f"largest shard id: {extract_data_index(shards_list[-1])}"))

    num_samples = args.train_num_samples if is_train else args.val_num_samples or 0

    def tokenize(tokenizer, text):
        return tokenizer(text)[0]

    if not args.skip_webdataset_split_by_node:
        rank, world_size, _, _ = wds.utils.pytorch_worker_info()
        logging.info(f"Rank {rank} in a {world_size}-GPU group is training on {len(shards_list)} shards: {input_shards}")
    else:
        local_rank, local_world_size = pytorch_local_worker_info()
        logging.info(f"Rank {local_rank} in a {local_world_size}-GPU group is training on {len(shards_list)} shards: {input_shards}")
    if args.cache_ref_model_features:
        cached_ref_features_index = None
    else:
        cached_ref_features_index = build_cached_data_index(args.cached_ref_features_dir)
    if args.cache_syn_texts:
        cached_syn_texts_index = None
    else:
        cached_syn_texts_index = build_cached_data_index(args.cached_syn_texts_dir)
    dataset = WebDataset(shards_list, is_train and not (args.cache_ref_model_features or args.cache_syn_texts),
                         args.batch_size, preprocess_img, args.seed, epoch,
                         functools.partial(tokenize, tokenizer) if tokenizer is not None else None,
                         return_index, not args.cache_ref_model_features, cached_ref_features_index,
                         not args.cache_syn_texts, cached_syn_texts_index,
                         num_samples_per_shard=args.num_samples_per_shard,
                         skip_split_by_node=args.skip_webdataset_split_by_node)
    if is_train:
        num_shards = len(shards_list)
        assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.ceil
        global_batch_size = args.batch_size * args.world_size
        if args.cache_ref_model_features or args.cache_syn_texts:
            num_batches = round_fn(num_samples / args.batch_size)
        else:
            num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        if args.cache_ref_model_features or args.cache_syn_texts:
            num_samples = num_batches * args.batch_size
        else:
            num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    data = DataInfo(dataloader=dataloader, shared_epoch=dataset.shared_epoch)
    data.shards_id_list = shards_id_list
    return data


def get_imagenet_wds(args, preprocess_fns, split):
    assert split == "val"
    batch_size = 64
    num_workers = 4
    preprocess_train, preprocess_val = preprocess_fns
    input_shards = args.imagenet_val
    preprocess_fn = preprocess_val
    shards_list = expand_urls(input_shards)[0]
    dataset = (
        wds.WebDataset(shards_list, nodesplitter=lambda src: src)
        .decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"]))
        .to_tuple(["webp", "png", "jpg", "jpeg"], "cls")
        .map_tuple(preprocess_fn, None)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )

    return DataInfo(dataloader=dataloader)


def get_csv_dataset(args, preprocess_fns, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    preprocess_train, preprocess_val = preprocess_fns
    preprocess_fn = preprocess_train if is_train else preprocess_val
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fns, is_train, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    preprocess_fn = preprocess_train if is_train else preprocess_val
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_fns, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_fns, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        if args.imagenet_val.endswith(".tar"):
            data["imagenet-val"] = get_imagenet_wds(args, preprocess_fns, "val")
        else:
            data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
