# Diffusion

## Text to Image

### SD1.5
* https://github.com/huggingface/diffusers/blob/main/examples/controlnet/README.md#performing-inference-with-the-trained-controlnet

### SDXL
Promt : https://civitai.com/images/2134452?period=AllTime&periodMode=published&sort=Most+Collected&view=categories&modelVersionId=145282&modelId=124421&postId=522227

* https://github.com/huggingface/diffusers/blob/main/examples/controlnet/README_sdxl.md
* https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl


##

### CUDA 12.1

```sh
pip install torch torchvision --pre -f https://download.pytorch.org/whl/nightly/cu121/torch_nightly.html
```

### torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 44.00 MiB. GPU 0 has a total capacty of 2.00 GiB of which 0 bytes is free. Of the allocated memory 10.85 GiB is allocated by PyTorch, and 213.71 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

* https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch