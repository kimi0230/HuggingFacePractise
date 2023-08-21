# Hugging Face Practise

Hugging Face is the home for all Machine Learning tasks. Here you can find what you need to get started with a task: demos, use cases, models, datasets, and more!

https://huggingface.co/tasks

![](assets/images/cover-1.png)
![](assets/images/cover-2.png)

## Python env

* https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/

```sh
# Installing virtualenv
python3 -m pip install --user virtualenv

# Creating a virtual environment
python3 -m venv env

# Activating a virtual environment
source env/bin/activate
```

## PyTorch & Flax
* https://huggingface.co/docs/diffusers/v0.18.2/en/installation

PyTorch
* https://pytorch.org/get-started/locally/
* https://pytorch.org/get-started/previous-versions/

```sh
# install pytorch (Mac)
pip3 install torch torchvision torchaudio

# install pytorch (Windows 11 GPU)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# install pytorch (Windows 11 CPU)
pip3 install torch torchvision torchaudio

# 🤗 Diffusers also relies on the 🤗 Transformers library, and you can install both with the following command:
pip3 install 'diffusers[torch]' 'transformers'
```

Flax
```sh
pip3 install 'diffusers[flax]'
# 🤗 Diffusers also relies on the 🤗 Transformers library, and you can install both with the following command:
pip3 install 'diffusers[flax]' 'transformers'
```

## Pipeline
* https://huggingface.co/docs/diffusers/v0.18.2/en/api/pipelines/overview


## Transformers
* https://huggingface.co/docs/transformers/index
* models: https://huggingface.co/models


## Diffusers
Installation : https://huggingface.co/docs/diffusers/installation


## Chores

### 'LayerNorm' is one of the layers in the Model.
```
'LayerNorm' is one of the layers in the Model. Looks like you're trying to load the diffusion model in float16(Half) format on CPU which is not supported. For float16 format, GPU needs to be used. For CPU run the model in float32 format.
Reference: https://github.com/pytorch/pytorch/issues/52291
```

#### RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
* https://huggingface.co/CompVis/stable-diffusion-v1-4/discussions/64

##### Method 1
change 
```python
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, torch_dtype=torch.float16)
```

to 
```python
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, torch_dtype=torch.float32)
```

##### Method 2
# https://stackoverflow.com/questions/75641074/i-run-stable-diffusion-its-wrong-runtimeerror-layernormkernelimpl-not-implem
```python 
commandline_args = os.environ.get('COMMANDLINE_ARGS', "--precision full --no-half")
sys.argv+=shlex.split(commandline_args)
```


### Potential NSFW content was detected in one or more images. A black image will be returned instead. Try again with a different prompt and/or seed.

* https://github.com/CompVis/stable-diffusion/issues/239
```py
pipe.safety_checker = lambda images, clip_input: (images, False)
```

### zsh: no matches found: diffusers[torch]

* [zsh: no matches found: ray[tune] #6696](https://github.com/ray-project/ray/issues/6696)


## Reference
* [huggingface/diffusers](https://github.com/huggingface/diffusers)
* https://huggingface.co/docs/diffusers/
* [huggingface/transformers](https://github.com/huggingface/transformers)
* https://huggingface.co/docs/transformers/index
* [AUTOMATIC1111 / stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
* [Youtube: Getting Started With Hugging Face in 15 Minutes | Transformers, Pipeline, Tokenizer, Models ](https://www.youtube.com/watch?v=QEaBAZQCtwE)