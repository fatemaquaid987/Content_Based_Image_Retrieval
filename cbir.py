
import os
import random
import pickle as pickle
import numpy as np
import matplotlib.pyplot
from matplotlib.pyplot import imshow
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm import tqdm
import pyglet
from tkinter import *


import math
# imports pyglets library
from pyglet.window import Window, mouse, gl, key
#from pyglet.media import avbin
from pyglet.gl import *
glEnable(GL_TEXTURE_2D)
gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
 
platform = pyglet.window.get_platform()
display = platform.get_default_display()
screen = display.get_default_screen()
 
mycbir = pyglet.window.Window(907, 655,                     # setting window
              resizable=True,  
              caption="Content Based Image Retrieval",  
              config=pyglet.gl.Config(double_buffer=True),  # Avoids flickers
              vsync=False                                   # For flicker-free animation
              )                                             # Calling base class constructor
mycbir.set_location(screen.width // 2 - 300,screen.height//2 - 350)
 


bgimage= pyglet.resource.image('CBIR.png')
query =  0
match1 = 0
match2 = 0
match3 = 0
match4 = 0
match5 = 0
draw = False
drawm = False
file =""

def get_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

def get_closest_images(query_image_idx, num_results=5):
    distances = [ distance.euclidean(pca_features[query_image_idx], feat) for feat in pca_features ]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
    return idx_closest

def get_concatenated_images(indexes, thumb_height):
    thumbs = []
    for idx in indexes:
        img = image.load_img(images[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image

images_path = 'Images'
max_num_images = 2000

images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(xrange(len(images)), max_num_images))]
#print(images)

model = keras.applications.VGG19(weights='imagenet', include_top=True)
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
pca_features = np.loadtxt("test1-VGG19.out")

def final(path):
    matchlst =[]
    global match1, match2, match3, match4, match5
    
    query_image_idx = images.index(path)
    idx_closest = get_closest_images(query_image_idx)
    query_image = get_concatenated_images([query_image_idx], 300)
    results_image = get_concatenated_images(idx_closest, 200)
    matplotlib.pyplot.figure(figsize = (5,5))
    imshow(query_image)
    matplotlib.pyplot.title("query image (%d)" % query_image_idx)
    matplotlib.pyplot.figure(figsize = (16,12))
    imshow(results_image)
    matplotlib.pyplot.title("result image")
    for i in idx_closest:
        #print(images[i].split('\\')[-1])
        images[i].split('\\')[-1]
        matchlst.append(images[i].split('\\')[-1])
        
    image1 = pyglet.resource.image(matchlst[0])
    image1.anchor_x = image1.width//2
    image1.anchor_y = image1.height//2
    match1 = pyglet.sprite.Sprite(image1, 92 , 145)
    match1.scale = 0.4
    
    image2 = pyglet.resource.image(matchlst[1])
    image2.anchor_x = image2.width//2
    image2.anchor_y = image2.height//2
    match2 = pyglet.sprite.Sprite(image2, 264 , 145)
    match2.scale =0.4

    image3 = pyglet.resource.image(matchlst[2])
    image3.anchor_x = image3.width//2
    image3.anchor_y = image3.height//2
    match3 = pyglet.sprite.Sprite(image3, 448 , 145)
    match3.scale=0.4

    image4 = pyglet.resource.image(matchlst[3])
    image4.anchor_x = image4.width//2
    image4.anchor_y = image4.height//2
    match4 = pyglet.sprite.Sprite(image4, 627 , 145)
    match4.scale =0.4

    image5 = pyglet.resource.image(matchlst[4])
    image5.anchor_x = image5.width//2
    image5.anchor_y = image5.height//2
    match5 = pyglet.sprite.Sprite(image5, 811 , 145)
    match5.scale=0.4

    

@mycbir.event

def on_draw():
    

    global query, qimage
    mycbir.clear()
    bgimage.blit(0,0)
    if draw == True:
        query.draw()
        
    if drawm == True:
        match1.draw()
        match2.draw()
        match3.draw()
        match4.draw()
        match5.draw()
    
    
@mycbir.event
def on_mouse_release(x, y, button, modifiers):

    global draw
    global drawm, query, qimage, file
   
    pyglet.resource.path = ['Images']
    pyglet.resource.reindex()
    if button == mouse.LEFT:
        
        if (373 <= x <= 554) and (337 <= y <= 382):
            draw = False
            #root = Tk()
            filename = filedialog.askopenfilename(filetypes =(("JPG files", "*.jpg"), ("All files", "*.*")))
            name = filename.split("/")
            file = name[len(name) -1]
            qimage = pyglet.resource.image(file)
            qimage.anchor_x = qimage.width//2
            qimage.anchor_y = qimage.height//2
            query =  pyglet.sprite.Sprite(qimage,455, 474)
            query.scale =0.5
            
            
            draw = True
        if (373 <= x <= 554) and (274 <= y <= 318):
            if draw== True:
                qpath = "Images\\"+file
                final(qpath)
                drawm = True
            
@mycbir.event  
def on_key_press(symbol, modifiers):
    if symbol == key.SPACE:
        print('z')
        mycbir.close()
    
def update(dt):
    dt

pyglet.clock.schedule_interval(update, 1/20.)
pyglet.app.run()
