# Hugging Face Practise

Hugging Face is the home for all Machine Learning tasks. Here you can find what you need to get started with a task: demos, use cases, models, datasets, and more!

https://huggingface.co/tasks

![](assets/images/cover-1.png)
![](assets/images/cover-2.png)


Hugging Face是一個針對人工智慧的開源社群平台，使用者可以在上邊發表和共享預訓練模型、資料集和展示檔案等。目前Hugging Face上已經共享了超過10萬個預訓練模型，1萬多個資料集，包括微軟、Google、Bloomberg、英特爾等各個行業超過1萬家機構都在使用Hugging Face的產品。

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

## Transformers
* https://huggingface.co/docs/transformers/index
* models: https://huggingface.co/models


## Diffusers
Installation : https://huggingface.co/docs/diffusers/installation

### Pipeline
* https://huggingface.co/docs/diffusers/main/en/api/pipelines/overview


## Hugging Face Hub


To be able to push your code to the Hub, you’ll need to authenticate somehow. The easiest way to do this is by installing the huggingface_hub CLI and running the login command:

```sh
python -m pip install huggingface_hub
huggingface-cli login
```

### Hugging Face Space
Spaces are one of the most popular ways to share ML applications and demos with the world.

https://huggingface.co/spaces/launch

<iframe width="1280" height="720" src="https://www.youtube.com/embed/3bSVKNKb_PY" title="Build and Deploy a Machine Learning App in 2 Minutes" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

#### demo
1. Create a Space -> `Streamlit`
2. git clone git@hf.co:spaces/kimi0230/demo

Image to Story : Upload an image, get a story made by Llama2 !
https://huggingface.co/spaces/fffiloni/Image-to-Story


## Miscellaneous


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

* https://stackoverflow.com/questions/75641074/i-run-stable-diffusion-its-wrong-runtimeerror-layernormkernelimpl-not-implem

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
* [HuggingGPT爆紅，Hugging Face又是什麼？它正在拆掉OpenAI的圍牆，要當AI界的Github](https://www.techbang.com/posts/105484-hugginggpt-is-on-fire-what-is-hugging-face-hugging-face-a-2)