### data loading 19/03

When type in and execute

```shell
python examples/test.py Yolov2
```

in examples/test.py, set config into

```shell
train_flag = 2
model_name = Yolov2
```

then execute

```python
vn.engine.VOCTest(hyper_params)
```

which is defined in yolo/vedanet/engine/\_voc_test.py, initialize dataloader as

```python
loader = torch.utils.data.DataLoader(
    CustomDataset(hyper_params),
    batch_size = batch, # 16, defined in yolo/cfgs/yolov2.yml
    shuffle = False,
    drop_last = False,
    num_workers = nworkers if use_cuda else 0, # 0 in my case
    pin_memory = pin_mem if use_cuda else False, # False in my case
    collate_fn = vn_data.list_collate,
)
```

where the list_collate function defined in yolo/vedanet/data/\_dataloading.py

```python
def list_collate(batch):
    """ Function that collates lists or tuples together into one list (of lists/tuples).
    Use this as the collate function in a Dataloader, if you want to have a list of items as an output, as opposed to tensors (eg. Brambox.boxes).
    """
    items = list(zip(*batch))

    for i in range(len(items)):
        if isinstance(items[i][0], (list, tuple)):
            items[i] = list(items[i])
        else:
            items[i] = default_collate(items[i])

    return items
```

The CustomDataset is defined just above, in the same file. I copy and comment the code below.

```python
class CustomDataset(vn_data.BramboxDataset):
    def __init__(self, hyper_params):
        anno = hyper_params.testfile
        # for reference: self.testfile = f'{self.data_root}/{dataset}.pkl'
        root = hyper_params.data_root
        # reference: self.data_root = config['data_root_dir']
        network_size = hyper_params.network_size
        # reference: self.network_size = cur_cfg['input_shape']
        labels = hyper_params.labels
        # reference: self.labels = config['labels']

        # defined in yolo\vedanet\data\transform\_preprocess.py
        # with arguments being: dimension(width, height), dataset
        # both are optional
        lb  = vn_data.transform.Letterbox(network_size)
        # Convert a PIL Image or numpy.ndarray to tensor.
        # as said in pytorch official document
        it  = tf.ToTensor()
        # seems just combine these transforms together, details in
        # yolo\vedanet\data\transform\util.py
        img_tf = vn_data.transform.Compose([lb, it])
        anno_tf = vn_data.transform.Compose([lb])

        def identify(img_id):
            # well, formating
            return f'{img_id}'

        super(CustomDataset, self).__init__('anno_pickle', anno, network_size, labels, identify, img_tf, anno_tf)

    def __getitem__(self, index):
        img, anno = super(CustomDataset, self).__getitem__(index)
        for a in anno:
            a.ignore = a.difficult  # Mark difficult annotations as ignore for pr metric
        return img, anno

```

The BramboxDataset and its parent class Dataset are nothing special...yet.

### Problems left unsloved

1. how is the .pkl files constructed? -- in labels.py
2. deep into CustomDataset. -- in yolo/vedanet/data/\_dataset_brambox.py
3. deep into bbb. -- it has a document on the Internet.
4. bboxoffset / difficult, visible, truncate / labels mapping

