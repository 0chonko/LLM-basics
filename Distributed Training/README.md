# Guide to Distributed Training with PyTorch DDP

This guide explains how distributed training is managed in the provided code using PyTorch's `DistributedDataParallel` (DDP) on an HPC cluster.

---

## 1. Data Distribution with `DistributedSampler`

- **Purpose**: The `DistributedSampler` is used to split the dataset into exclusive subsets, ensuring each GPU processes a unique portion of the data, thus improving parallelism and reducing redundancy.
- **How It Works**:
    - The `DistributedSampler` partitions the dataset into `world_size` splits (one per GPU). The `num_replicas` parameter is set to `world_size`, which specifies the number of processes participating in the training, and each process (`rank`) is assigned a unique split of the data using the `rank` parameter.
    - Each GPU (rank) receives a unique subset of the dataset, which is enforced by the sampler. For example, with 4 GPUs and 1,000 samples, each GPU processes 250 samples per epoch, ensuring parallel processing without data duplication.
    - **Key**: To ensure diversity in data shuffling across epochs, `sampler.set_epoch(epoch)` is used, which shuffles the data partitions for each epoch without duplication.

---

## 2. DataLoader Configuration

- **Data Loading**:
    ```python
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True)
    ```
    The `sampler` argument in the `DataLoader` ensures that each GPU gets its exclusive subset of data as determined by the `DistributedSampler`.
    `pin_memory=True` accelerates CPU-to-GPU transfers by enabling fast memory transfer between the host (CPU) and device (GPU).
    - **Per-GPU Batch Size**: The `BATCH_SIZE` specifies the batch size for each individual GPU. The effective global batch size across all GPUs is calculated as `BATCH_SIZE × world_size`.

---

## 3. Forward/Backward Pass

- **Per-GPU Processing**:
    ```python
    data, target = data.to(rank), target.to(rank)
    output = ddp_model(data)
    loss = criterion(output, target)
    loss.backward()
    ```
    The data and target tensors are moved to the GPU specified by `rank` using `.to(rank)`.
    The model (`ddp_model`) was instantiated with `DistributedDataParallel`, which handles the distribution of data across GPUs.
    Each GPU processes its batch of data independently, computes the loss for that batch, and calculates local gradients using `loss.backward()`.

---

## 4. Gradient Synchronization

- **All-Reduce Operation**:
    ```python
    ddp_model = DDP(model, device_ids=[rank])
    ```
    PyTorch’s `DistributedDataParallel` (DDP) performs an all-reduce operation across all GPUs to average the gradients. This ensures that each GPU's gradients are combined, effectively simulating a global update equivalent to training on a single GPU with a larger batch size.

---

## 5. Parameter Updates

- **Optimizer Step**:
    ```python
    optimizer.step()
    ```
    After the gradients are synchronized across all GPUs, the optimizer updates the model's parameters using the averaged gradients. This ensures that all GPUs have the same updated parameters, maintaining consistency across the distributed training environment.

---

## 6. Scale and Efficiency

- **Effective Global Batch Size**: Calculated as the product of the per-GPU batch size (`BATCH_SIZE`) and the number of GPUs (`world_size`). For example, with 8 GPUs and a per-GPU batch size of 64, the effective global batch size is 512.
- **Memory Savings**: By splitting the dataset, each GPU only needs to process its subset, reducing per-GPU memory usage while maintaining the same global batch size.
- **Why It Works**: Gradient averaging across GPUs during backpropagation ensures that the model converges similarly to a single-GPU training setup, but with the benefit of linear scaling of training speed as additional GPUs are added.

---

## Key Takeaways

- **Data Split**: Exclusive per-GPU splits are managed by `DistributedSampler`, ensuring each GPU processes a unique subset of the dataset.
- **Processing**: Independent training on each GPU’s subset with local gradient computation.
- **Sync**: Gradients are averaged across all GPUs during backpropagation to ensure synchronized updates.
- **No Data Recombination**: Parameter updates are synchronized via gradients, avoiding the need to recombine raw data from multiple GPUs.

This approach enables linear scaling of training speed with GPUs while maintaining convergence characteristics similar to single-GPU training, making it suitable for large-scale distributed training in HPC environments.