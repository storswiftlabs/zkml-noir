import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from zkml.fnn.utils import *

def noir_cnn_main(in_channels, out_channels, image_size, filter_size=3, padding=1, conv_stride=1, pooling_size=2, pooling_stride=2, path=''):

    image, str_image = cnn_image_value(image_size, in_channels)
    prover_str = 'image = '+str_image+'\n'
    noir_code = 'use dep::std;\n'
    noir_code += global_const('scaling_factor', 16)
    
    global_const_code_0, main_args_code_0, main_code_0, prover_str_0, test_args_0 = cnn_layer('image', in_channels, out_channels, image_size, filter_size, padding, conv_stride, pooling_size, pooling_stride)

    global_const_code_1, main_args_code_1, main_code_1, prover_str_1, test_args_1 = cnn_layer('pooled_image_0', out_channels, out_channels*2, image_size, filter_size, padding, conv_stride, pooling_size, pooling_stride)

    global_const_code_2, main_args_code_2, main_code_2, prover_str_2, test_args_2 = fnn_layer([image_size**2*out_channels*2, 2], 'pooled_image_1')

    global_const_code = '\n'.join([global_const_code_0, global_const_code_1, global_const_code_2])
    main_args_code = ', '.join([main_args_code_0, main_args_code_1, main_args_code_2])
    main_code = ''.join([main_code_0, main_code_1, main_code_2])
    prover_str += ''.join([prover_str_0, prover_str_1, prover_str_2])

    noir_code+=global_const_code
    noir_code+='\n\
fn main(image: [u16; image_size_0*image_size_0*in_channels_0], '+main_args_code+') {\n\
'+main_code+'\n\
}\n'
    noir_code+=noir_fn_conv()
    noir_code+=noir_pad_image()
    noir_code+=noir_fn_forward()
    noir_code+=noir_fn_activation_relu()
    noir_code+=noir_fn_max_pooling()
    noir_code+=noir_fn_max()
    noir_code+=noir_fn_copy_array()
    noir_code+=noir_test_main([['image', '[u16; image_size_0*image_size_0*in_channels_0]', '[0; image_size_0*image_size_0*in_channels_0]'], test_args_0, test_args_1, test_args_2])

    with open(os.path.join(path, "src/main.nr"), "w+") as file:
        file.write(noir_code)
    with open(os.path.join(path, "Prover.toml"), "w+") as file:
        file.write(prover_str)


if __name__=="__main__":
    in_channels = 1
    out_channels = 32
    image_size = 28
    filter_size = 5
    padding = 1
    conv_stride = 1
    pooling_size = 2
    pooling_stride = 2

    noir_cnn_main(in_channels, out_channels, image_size, filter_size, padding, conv_stride, pooling_size, pooling_stride, path='/mnt/code/noir_project/cnn2')
