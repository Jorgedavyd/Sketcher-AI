import os
from models_architecture.utils import *
import torch
import time

def get_input():
    inputs = []
    path = os.path.join(os.getcwd(), 'inputs')
    for i, filename in enumerate(os.listdir(path)):
        inputs.append(filename)
        print(f'{i+1}. {filename}')
    while True:
        try:
            num = int(input('Filenumber: ')) 
        except TypeError:
            print('Choose a valid number\n ')
            continue
        if num > len(inputs):
            print('Choose a valid number\n')
            continue
        break
    real_index = num -1
    filepath = os.path.join(os.getcwd(), inputs[real_index])
    input_img = image_loader(filepath, size=256)
    return input_img, inputs[real_index]

def get_models(MODEL):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    style_subnet = torch.jit.load('models/style_subnet_' + MODEL + '.pt', map_location='cpu').eval().to(device)
    enhance_subnet = torch.jit.load('models/enhance_subnet_' + MODEL + '.pt', map_location='cpu').eval().to(device)
    refine_subnet = torch.jit.load('models/refine_subnet_' + MODEL + '.pt', map_location='cpu').eval().to(device)
    return style_subnet, enhance_subnet, refine_subnet

def stylization(input_img, style_subnet, enhance_subnet, refine_subnet, MODEL, name, diff = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Start transforming on {}..".format(device))
    start = time.time()
    with torch.no_grad():
        generated_img_256, resized_input_img_256 = style_subnet(input_img)
        generated_img_512, resized_input_img_512 = enhance_subnet(generated_img_256)
        generated_img_1024, resized_input_img_1024 = refine_subnet(generated_img_512)
    print("Image transformed. Time for pass: {:.2f}s".format(time.time() - start))

    imshow(generated_img_256)
    imshow(generated_img_512)
    imshow(generated_img_1024)
    save_image(generated_img_512, title="outputs/" + MODEL + "_" + name +"_512")
    save_image(generated_img_256, title="outputs/" + MODEL + "_" + name +"_256")
    save_image(generated_img_1024, title="outputs/" + MODEL+ "_" + name +"_1024")
    if diff:
        diff_256_orig = generated_img_256 - resized_input_img_256
        diff_512_256 = generated_img_512 - resized_input_img_512
        diff_1024_512 = generated_img_1024 - resized_input_img_1024
        imshow(diff_256_orig)
        imshow(diff_512_256)
        imshow(diff_1024_512)