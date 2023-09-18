from diffusers import StableDiffusionPipeline
import torch
import numpy as np
from PIL import Image


def main():

    # 設置本地模型文件路徑
    stable_diffusion_model_path = r"D:\Stable-Diffusion\stable-diffusion-webui\models\Stable-diffusion\xxmix9realistic_v40.safetensors"

    model_id = "0.5(SDXL1.0_sd_xl_base_1.0+xxmixsdxl_v1-000008) + 0.5(SDXL1.0_XXMix_9realisticSDXL_v1.0+xxmixsdxl_v1-000008)"
    steps = 28
    eta = 0.67
    size = (720, 1024)
    seed = 3426691999
    model_version = "v1.5.1"
    sampler = "DPM++ SDE Karras"
    cfg_scale = 8
    clip_skip = 2
    model_hash = "ecb947b4ea"

    # control_net_model_path = "path_to_control_net_model.pth"

    # 加載本地的Stable Diffusion模型和ControlNet模型
    # stable_diffusion_model = torch.load(stable_diffusion_model_path)
    # control_net_model = torch.load(control_net_model_path)

    # 使用 SafeTensorsLoader 加载 SafeTensors
    # with SafeTensorsLoader(stable_diffusion_model_path) as loader:
    #     safetensors = loader.load()

    # 现在 safetensors 包含了加载的 SafeTensors 数据

    # 創建StableDiffusionPipeline，並將模型傳遞給它
    # pipe = StableDiffusionPipeline(
    #     model=stable_diffusion_model, control_net=control_net_model).to("cuda")
    # pipe = StableDiffusionPipeline(
    #     model=stable_diffusion_model_path, use_safetensors=True)
    pipe = StableDiffusionPipeline.from_single_file(
        stable_diffusion_model_path,
        model_id=model_id,
        steps=steps,
        eta=eta,
        size=size,
        seed=seed,
        model_version=model_version,
        sampler=sampler,
        cfg_scale=cfg_scale,
        clip_skip=clip_skip,
        model_hash=model_hash,
    ).to("cuda")

    # https://github.com/CompVis/stable-diffusion/issues/239
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    # 加載文本控制碼
    # pipe.load_textual_inversion("sd-concepts-library/cat-toy")

    # 設置生成的文本提示
    # prompt = "a photo of an astronaut riding a horse on mars"
    prompt = "xxmix_girl, a close up of a person with a yellow sweater on and a cell phone in hand and a light shining on her face, long hair"
    negative_prompt = "(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), tooth, open mouth, bad hand, bad fingers"

    # 加載OpenPose生成的关键点信息图像
    # openpose_image_path = "path_to_openpose_output_image.png"
    # openpose_image = Image.open(openpose_image_path)

    # 将OpenPose图像转换为numpy数组
    # openpose_data = np.array(openpose_image)

    # 将OpenPose关键点信息嵌入到文本中，以便传递给模型
    # combined_text = f"{prompt} {openpose_data.tolist()}"

    # 生成图像
    # image = pipe(combined_text, num_inference_steps=50).images[0]
    image = pipe(prompt, negative_prompt=negative_prompt).images[0]

    # 保存生成的圖像
    image.save("astronaut_rides_horse.png")


# 自定义加载器类，用于处理 SafeTensors
# class SafeTensorsLoader(object):
#     def __init__(self, file_path):
#         self.file_path = file_path

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         pass

#     def load(self):
#         # 使用 torch.load 加载模型文件
#         loaded_data = torch.load(
#             self.file_path, map_location=torch.device('cpu'))

#         # 如果 SafeTensors 存在于加载的数据中，则将它们转换为 SafeTensors 类型
#         if 'safetensors' in loaded_data:
#             from your_module import SafeTensors  # 替换为实际的 SafeTensors 类的导入方式
#             loaded_safetensors = loaded_data['safetensors']
#             safetensors = SafeTensors.from_dict(
#                 loaded_safetensors)  # 使用适当的方法来创建 SafeTensors
#             return safetensors


if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
    main()
