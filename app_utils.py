import os
from models.utils import image_loader

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
    return input_img

