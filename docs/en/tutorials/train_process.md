# 深度学习Train and Eval 流程解析
共有以下8个步骤:

- 创建网络和初始化 <br>
- 创建DataLoader <br>
- 创建损失函数 <br>
- 创建优化器，并设置相关参数 <br>
- 创建Train Step函数 <br>
- 创建回调函数 <br>
- 创建Test精度计算函数 <br>
- 创建Trainer对象，并调用train方法来训练模型


## 1.创建网络和初始化
```python
    # Create Network
    args.network.recompute = args.recompute
    args.network.recompute_layers = args.recompute_layers
    network = create_model(
        model_name=args.network.model_name,
        model_cfg=args.network,
        num_classes=args.data.nc,
        sync_bn=args.sync_bn,
    )
```

## 2.创建DataLoader
- 获取训练数据的增强函数和epoch等信息：
```python
transforms = args.data.train_transforms
stage_epochs = [args.epochs,] if not isinstance(transforms, dict) else transforms['stage_epochs']
stage_transforms = [transforms,] if not isinstance(transforms, dict) else transforms['trans_list']
```
其中stage_epochs表示各阶段的epoch数。stage_transforms表示各个阶段的变换函数，是一个列表，每个元素是一个阶段的数据增强函数列表。
- 创建Training Dataset和DataLoader
```python
  for stage in range(len(stage_epochs)):
        _dataset = COCODataset(
            dataset_path=args.data.train_set,
            img_size=args.img_size,
            transforms_dict=stage_transforms[stage],
            is_training=True,
            augment=True,
            rect=args.rect,
            single_cls=args.single_cls,
            batch_size=args.total_batch_size,
            stride=max(args.network.stride),
        )
        _dataloader = create_loader(
            dataset=_dataset,
            batch_collate_fn=_dataset.train_collate_fn,
            dataset_column_names=_dataset.dataset_column_names,
            batch_size=args.per_batch_size,
            epoch_size=stage_epochs[stage],
            rank=args.rank,
            rank_size=args.rank_size,
            shuffle=True,
            drop_remainder=True,
            num_parallel_workers=args.data.num_parallel_workers,
            python_multiprocessing=True,
        )
        stage_dataloaders.append(_dataloader)
```
- 合并DataLoader
```python
    dataloader = stage_dataloaders[0] if len(stage_dataloaders) == 1 else ms.dataset.ConcatDataset(stage_dataloaders)
```
- 创建Testing Dataset和DataLoader（可选，由run_eval参数控制）


## 3.创建损失函数
```python
    loss_fn = create_loss(
        **args.loss, anchors=args.network.get("anchors", 1), stride=args.network.stride, nc=args.data.nc
    )
    ms.amp.auto_mixed_precision(loss_fn, amp_level="O0" if args.keep_loss_fp32 else args.ms_amp_level)
```

## 4.创建优化器，并设置相关参数
```python
    # Create Optimizer
    args.optimizer.steps_per_epoch = steps_per_epoch
    lr = create_lr_scheduler(**args.optimizer)
    params = create_group_param(params=network.trainable_params(), **args.optimizer)
    optimizer = create_optimizer(params=params, lr=lr, **args.optimizer)
    warmup_momentum = create_warmup_momentum_scheduler(**args.optimizer)
```

## 5.创建Train Step函数
```python
    # Create train_step_fn
    reducer = get_gradreducer(args.is_parallel, optimizer.parameters)
    scaler = get_loss_scaler(args.ms_loss_scaler, scale_value=args.ms_loss_scaler_value)
    train_step_fn = create_train_step_fn(
        network=network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        loss_ratio=args.rank_size,
        scaler=scaler,
        reducer=reducer,
        ema=ema,
        overflow_still_update=args.overflow_still_update,
        ms_jit=args.ms_jit,
    )
```
create_train_step_fn用来接收网络、损失函数、优化器等参数，定义前向过程、梯度计算、参数更新。

## 6.创建回调函数
```python
    # Create callbacks
    if args.summary:
        args.callback.append({"name": "SummaryCallback"})
    if args.profiler:
        args.callback.append({"name": "ProfilerCallback", "profiler_step_num": args.profiler_step_num})
    callback_fns = create_callback(args.callback)
```
创建回调函数列表。其中summary参数表示是否收集训练loss信息，profiler参数表示是否收集性能数据。

##  7.创建Test精度计算函数
```python
    # Create test function for run eval while train
    if args.run_eval:
        is_coco_dataset = "coco" in args.data.dataset_name
        test_fn = partial(
            test,
            dataloader=eval_dataloader,
            anno_json_path=os.path.join(
                args.data.val_set[: -len(args.data.val_set.split("/")[-1])], "annotations/instances_val2017.json"
            ),
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            conf_free=args.conf_free,
            nms_time_limit=args.nms_time_limit,
            is_coco_dataset=is_coco_dataset,
            imgIds=None if not is_coco_dataset else eval_dataset.imgIds,
            per_batch_size=args.per_batch_size,
            rank=args.rank,
            rank_size=args.rank_size,
            save_dir=args.save_dir,
            synchronizer=Synchronizer(args.rank_size) if args.rank_size > 1 else None,
        )
    else:
        test_fn = None
```
这个测试函数用于在训练过程中进行模型评估，它将模型在测试集上的表现记录下来。


## 8.创建Trainer对象，并调用train方法来训练模型

```python
trainer = create_trainer(
    model_name=model_name,
    train_step_fn=train_step_fn,
    scaler=scaler,
    dataloader=dataloader,
    steps_per_epoch=steps_per_epoch,
    network=network,
    loss_fn=loss_fn,
    ema=ema,
    optimizer=optimizer,
    callback=callback_fns,
    reducer=reducer,
    data_sink=args.ms_datasink,
    profiler=args.profiler
)

if not args.ms_datasink:
    trainer.train(...)
else:
    trainer.train_with_datasink(...)
```
提供普通训练方式和数据下沉训练方式，由参数ms_datasink控制。



