import numpy as np

def get_conv_output_shape(inputs_h, inputs_w, ksize, stride=1, pad=0):
    """compute convolution's shape
    Args:
        inputs_h: int, inputs tensor height
        inputs_w: int, inputs width
        ksize: int list, filter shape of [kernel_h, kernel_w]
        stride: int, the filter convolves around the input volume by shifting one unit at a time, default value 1
        pad: int, zero padding numbers
    Returns:
        outputs height and output width
    """

    if len(ksize) != 2:
        raise ValueError('Kernel size must be a list of shape [kernel_h, kernel_w]')

    if not isinstance(inputs_h, int):
        inputs_h = int(inputs_h)
    
    if not isinstance(inputs_w, int):
        inputs_w = int(inputs_w)
    
    if not isinstance(ksize, (tuple, list)):
        ksize = [ksize]
    
    if not isinstance(stride, int):
        stride = int(stride)
    
    if not isinstance(pad, int):
        pad = int(pad)

    kernel_h, kernel_w = ksize
    # calculate output height/width
    output_h = int((inputs_h + pad*2 - kernel_h) / stride) + 1
    output_w = int((inputs_w + pad*2 - kernel_w) / stride) + 1
    return (output_h, output_w)

