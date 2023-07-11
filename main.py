from app_utils import *

if __name__ == '__main__':
    print('==================================Sketcher A.I.================================== \n')

    print('Now, artists are feeling the presure of losing their jobs due to the generative A.I.s, but\n they should be seen as tools that let us have a wider range of expression, thats why I am \n creating this new application, which helps artists to concentrate in creativity rather than\n wasting time in mechanical things.\n\n')

    print('Creator mode: Creates a sketch of an input image of the anatomical body of the picture, helping the artists drawing on it.\n ')

    print('Sketch mode: Creates an full-sketched version of the photo.\n')

    input('Press any to start: \n')


    while True:
        mode = input('1. Creator mode \n2. Sketcher mode\n===================>(1,2,q quit)')
        if mode ==1:
            MODEL = 'creator'
            style, enhance, refine = get_models(MODEL)
            img, filename = get_input()
            stylization(img,style,enhance,refine, MODEL, filename.split('.')[0], diff = False)
            input('Your file was saved on outputs directory')
        elif mode ==2:
            MODEL = 'sketch'
            style, enhance, refine = get_models(MODEL)
            img, filename = get_input()
            stylization(img,style,enhance,refine, MODEL, filename.split('.')[0], diff = False)
            input('Your file was saved on outputs directory')
        elif mode == 'q':
            break
        else:
            print('Tiene que insertar 1 o 2, en base al modo de uso')