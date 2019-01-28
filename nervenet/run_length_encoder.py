import numpy as np

def encode(x, thresh = 0):
    '''
    x: numpy array of shape (height, width)
    Returns run length as list
    '''
    pixels, = np.where(x.T.flatten()>thresh)
    run_lengths = []
    
    prev = -2
    for p in pixels:
        if (p>prev+1): run_lengths.extend((p+1, 0))
        run_lengths[-1] += 1
        prev = p
    return run_lengths

def decode(rllist, shape = (420, 580)):
    '''
    rllist: list with [pixel1, rl1, pixel2, rl2, ...]
    Returns image as numpy array
    '''
    img = np.zeros(shape).flatten()
    
    for p, rl in zip(rllist[::2], rllist[1::2]):
        img[p:p+rl] = 255
    img = img.reshape(shape, order = 'F')
    return img