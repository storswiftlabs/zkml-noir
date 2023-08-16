import random,json,os

global layer_num
layer_num = 0

def fnn_input_count(network):
    num = network[0]
    for layer in range(1, len(network)):
        num += network[layer]*(network[layer-1]+1)
    print(num)
    return num

def fnn_inputs_value(network):
    inputs = []
    str_inputs = []
    for i in range(fnn_input_count(network)):
        v = random.randint(1,10)
        # v = 0
        inputs.append(v)
        str_inputs.append(str(v))
    str_inputs = str(json.dumps(str_inputs))
    return inputs, str_inputs

def cnn_layer(image: str, in_channels, out_channels, image_size, filter_size=3, padding=1, conv_stride=1, pooling_size=2, pooling_stride=2):
    global layer_num
    image_shape = [image_size, image_size, in_channels]
    filter_shape = [filter_size, filter_size, in_channels]
    neurons_per_layer = [filter_size*filter_size*in_channels, out_channels]
    padded_image_len = (image_size+2*padding)**2*in_channels
    padded_size = image_size + 2 * padding
    convolved_image_len = int((image_size + 2 * padding - filter_size)/conv_stride + 1)**2*out_channels
    num = fnn_input_count(neurons_per_layer)
    pooling_image_len = int(convolved_image_len/(pooling_size*pooling_size))
    convolved_size = int((image_size + 2 * padding - filter_size)/conv_stride + 1)
    pooled_size = int((convolved_size-pooling_size)/pooling_stride+1)
    pooling_kernel_len = pooling_size**2

    inputs, str_inputs = fnn_inputs_value(neurons_per_layer)

    global_const_code = global_const(f'in_channels_{layer_num}', in_channels)
    global_const_code += global_const(f'out_channels_{layer_num}', out_channels)
    global_const_code += global_const(f'filter_size_{layer_num}', filter_size)
    global_const_code += global_const(f'image_size_{layer_num}', image_size)
    global_const_code += global_const(f'padding_{layer_num}', padding)
    global_const_code += global_const(f'conv_stride_{layer_num}', conv_stride)
    global_const_code += global_const(f'image_shape_{layer_num}', image_shape)
    global_const_code += global_const(f'filter_shape_{layer_num}', filter_shape)
    global_const_code += global_const(f'padded_image_len_{layer_num}', padded_image_len, note='  // (image_size+2*pading)**2*in_channels')
    global_const_code += global_const(f'padded_size_{layer_num}', padded_size, note='     // image_size + 2 * padding')
    global_const_code += global_const(f'convolved_image_len_{layer_num}', convolved_image_len, note='  // ((image_size + 2 * padding - filter_size)/conv_stride + 1)**2*out_channels')
    global_const_code += global_const(f'neurons_per_layer_{layer_num}', neurons_per_layer)
    global_const_code += global_const(f'num_{layer_num}', num)
    global_const_code += global_const(f'fnn_out_len_{layer_num}', str(neurons_per_layer[-1]))
    global_const_code += global_const(f'pooling_size_{layer_num}', pooling_size)
    global_const_code += global_const(f'pooling_image_len_{layer_num}', pooling_image_len, note='    // convolved_image_len/(pooling_size*pooling_size)')
    global_const_code += global_const(f'pooling_stride_{layer_num}', pooling_stride)
    global_const_code += global_const(f'convolved_size_{layer_num}', convolved_size, note='  // (image_size + 2 * padding - filter_size)/conv_stride + 1')
    global_const_code += global_const(f'pooled_size_{layer_num}', pooled_size, note='    // (convolved_size-pooling_size)/pooling_stride+1')
    global_const_code += global_const(f'pooling_kernel_len_{layer_num}', pooling_kernel_len, note='  // pooling_size**2')

    main_args_code = f'mut inputs_{layer_num} : pub [u16; num_{layer_num}]'
    args = 'inputs, padding, in_channels, image_size, padded_size, padded_image, convolved_image, filter_size, conv_stride, neurons_per_layer, fnn_out, fnn_out_len, out_channels, pooled_size, pooling_stride, pooling_size, convolved_size, pooled_image'
    args = ', '.join([arg+'_'+str(layer_num) for arg in args.split(', ')])
    main_code='\n\
    let mut padded_image_'+str(layer_num)+' : [u16; padded_image_len_'+str(layer_num)+'] = [0; padded_image_len_'+str(layer_num)+'];\n\
    let mut convolved_image_'+str(layer_num)+' : [u16; convolved_image_len_'+str(layer_num)+'] = [0; convolved_image_len_'+str(layer_num)+'];\n\
    let mut fnn_out_'+str(layer_num)+' : [u16; fnn_out_len_'+str(layer_num)+'] = [0; fnn_out_len_'+str(layer_num)+'];\n\
    let mut pooled_image_'+str(layer_num)+':[u16; pooling_image_len_'+str(layer_num)+'] = [0; pooling_image_len_'+str(layer_num)+'];\n\
    pooled_image_'+str(layer_num)+' = conv('+image+', '+args+');\n\
    std::println(pooled_image_'+str(layer_num)+');\n\
'
    prover_str = f'inputs_{layer_num} = '+str_inputs+'\n'
    test_args = [f'inputs_{layer_num}', f'[u16; num_{layer_num}]', f'[0; num_{layer_num}]']
    layer_num+=1
    return global_const_code, main_args_code, main_code, prover_str, test_args

def fnn_layer(neurons_per_layer, neurons=''):
    global layer_num
    num = fnn_input_count(neurons_per_layer)
    inputs, str_inputs = fnn_inputs_value(neurons_per_layer)
    global_const_code = global_const(f'neurons_per_layer_{layer_num}', neurons_per_layer)
    global_const_code += global_const(f'num_{layer_num}', num)
    global_const_code += global_const(f'fnn_out_len_{layer_num}', str(neurons_per_layer[-1]))
    main_args_code = f'mut inputs_{layer_num} : pub [u16; num_{layer_num}]'
    main_code='\n\
    let mut fnn_out_'+str(layer_num)+' : [u16; fnn_out_len_'+str(layer_num)+'] = [0; fnn_out_len_'+str(layer_num)+'];\n\
    fnn_out_'+str(layer_num)+' = forward(inputs_'+str(layer_num)+', neurons_per_layer_'+str(layer_num)+', fnn_out_len_'+str(layer_num)+', fnn_out_'+str(layer_num)+');\n\
    std::println(fnn_out_'+str(layer_num)+');\n\
\n'
    if neurons:
        main_code = f'\n    inputs_{layer_num} = copy_array({neurons}, inputs_{layer_num});'+main_code
    prover_str = f'inputs_{layer_num} = '+str_inputs+'\n'
    test_args = [f'inputs_{layer_num}', f'[u16; num_{layer_num}]', f'[0; num_{layer_num}]']
    layer_num+=1
    return global_const_code, main_args_code, main_code, prover_str, test_args

def cnn_image_value(image_size, in_channels):
    image = []
    str_image = []
    for i in range(image_size*image_size*in_channels):
        v = random.randint(1,10)
        # v = 0
        image.append(v)
        str_image.append(str(v))
    str_image = str(json.dumps(str_image))
    return image, str_image

def global_const(name, value, note=''):
    return 'global '+name+' = '+str(value)+';  '+note+'\n'

def noir_fn_copy_array():
    code = '\n\
fn copy_array<T, M, N>(array1: [T; M], mut array2: [T; N]) -> [T; N] {\n\
    for i in 0..array1.len() {\n\
        array2[i] = array1[i];\n\
    }\n\
    array2\n\
}\n'
    return code

def noir_fn_max_pooling():
    code = '\n\
fn max_pooling<M, N>(convolved_image:[u16; M], out_channels: comptime u16, pooled_size: comptime u16, pooling_stride: u16, pooling_size: comptime u16, convolved_size: u16, mut pooled_image : [u16; N]) -> [u16; N] {\n\
    for c in 0..out_channels {\n\
        for h_step in 0..pooled_size{\n\
            for w_step in 0..pooled_size{\n\
                let mut w_step_index = w_step*pooling_stride;\n\
                let mut h_step_index = h_step*pooling_stride;\n\
                let mut max_value: u16 = 0;\n\
                for h in 0..pooling_size {\n\
                    for w in 0..pooling_size{\n\
                        let mut v=convolved_image[(h_step_index+h)*convolved_size*out_channels+(w_step_index+w)*out_channels+c];\n\
                        if ((h==0) & (w==0)) | (v > max_value){\n\
                            max_value = v;\n\
                        }\n\
                    }\n\
                }\n\
                // 计算池化后的索引位置\n\
                let mut pooled_index = h_step * pooled_size * out_channels + w_step * out_channels + c;\n\
                pooled_image[pooled_index] = max_value;\n\
            }\n\
        }\n\
    }\n\
    pooled_image\n\
}\n'
    return code

def noir_fn_max():
    code = '\n\
fn max<T, N>(array: [T; N]) -> T {\n\
    let mut max_value = array[0];\n\
    for i in 1..array.len() {\n\
        if array[i] > max_value {\n\
            max_value = array[i];\n\
        }\n\
    }\n\
    max_value\n\
}\n'
    return code

def noir_fn_conv():
    code = '\n\
fn conv<M, N, P, Q, R, S, T>(image: [u16; M], mut inputs: [u16; N], padding: comptime u16, in_channels: comptime u16, image_size: comptime u16, padded_size: u16, mut padded_image: [u16; Q], mut convolved_image : [u16; R], filter_size: comptime u16, conv_stride: comptime u16, neurons_per_layer : [comptime u16; T], mut fnn_out: [u16; S], fnn_out_len: comptime u16, out_channels: comptime u16, pooled_size: comptime u16, pooling_stride: u16, pooling_size: comptime u16, convolved_size: u16, mut pooled_image: [u16; P]) -> [u16; P] {\n\
    padded_image = pad_image(image, padding, in_channels, image_size, padded_size, padded_image);\n\
    let step: comptime u16 = (image_size + 2 * padding - filter_size)/conv_stride + 1;\n\
    for h_step in 0..step {\n\
        for w_step in 0..step {\n\
            let mut w_step_index = w_step*conv_stride;\n\
            let mut h_step_index = h_step*conv_stride;\n\
            for h in 0..filter_size {\n\
                for w in 0..filter_size{\n\
                    for c in 0..in_channels{\n\
                        inputs[h*filter_size*in_channels+w*in_channels+c]=padded_image[(h_step_index+h)*padded_size*in_channels+(w_step_index+w)*in_channels+c];\n\
                    }\n\
                }\n\
            }\n\
            fnn_out = forward(inputs, neurons_per_layer, fnn_out_len, fnn_out);\n\
            for n in 0..fnn_out_len {\n\
                convolved_image[h_step*step*fnn_out_len+w_step*fnn_out_len+n] = fnn_out[n];\n\
            }\n\
        }\n\
    }\n\
    pooled_image = max_pooling(convolved_image, out_channels, pooled_size, pooling_stride, pooling_size, convolved_size, pooled_image);\n\
    pooled_image\n\
}\n'
    return code

def noir_fn_forward():
    code = '\n\
fn forward<N, M, D> (mut inputs : [u16; N], neurons_per_layer : [comptime u16; M], fnn_out_len: comptime u16, mut fnn_out : [u16; D]) -> [u16; D] {\n\
    // inputs: [neuron0-0, neuron0-1, ..., w1-0-0, w1-0-1, ..., w1-n-0, w1-n-1, ..., b1-0, b1-1, ...,neuron1-0, neuron1-1, ..., w2-0-0, w2-0-1, ..., w2-n-0, w2-n-1, ..., b2-0, b2-1, ...]\n\
    let mut index_layer = 0;\n\
    for layer in 1 .. neurons_per_layer.len() {\n\
    	index_layer += neurons_per_layer[layer - 1] * (neurons_per_layer[layer]+1);\n\
        for i in 0 .. neurons_per_layer[layer] {\n\
            let mut val = 0; \n\
            let mut index_b = index_layer + i;\n\
            let mut index_j = index_layer - neurons_per_layer[layer - 1] * (neurons_per_layer[layer]+1);\n\
            for j in 0 .. neurons_per_layer[layer-1] {\n\
                let mut index_neuron = index_j+j;\n\
                let mut index_w = index_neuron+neurons_per_layer[layer-1]*(i+1);\n\
                val += (inputs[index_neuron]*inputs[index_w]/scaling_factor);\n\
            }\n\
            val += inputs[index_b];\n\
            inputs[index_b] = activation(val);\n\
        }\n\
    }\n\
    for i in 0 .. fnn_out_len {\n\
        fnn_out[i] = inputs[inputs.len() as u16 - fnn_out_len + i];\n\
    }\n\
    fnn_out\n\
}\n '
    return code

def noir_fn_activation_relu():
    code = '\n\
fn activation(x: u16) -> u16 {\n\
    let mut result = 0;\n\
    if x>result {\n\
        result = x;\n\
    }\n\
    result\n\
}\n'
    return code

def noir_pad_image():
    code = '\n\
fn pad_image<M, N>(image:[u16; M], padding: comptime u16, in_channels: comptime u16, image_size: comptime u16, padded_size: u16, mut padded_image : [u16; N]) -> [u16; N] {\n\
    // let padded_size = image_size + 2 * padding;\n\
    for c in 0..in_channels {\n\
        for h in 0..image_size{\n\
            for w in 0..image_size{\n\
                // 计算填充后的索引位置\n\
                let mut padded_index = (h + padding) * padded_size * in_channels + (w + padding) * in_channels + c;\n\
                // 填充数据\n\
                padded_image[padded_index] = image[h * image_size * in_channels + w * in_channels + c];\n\
            }\n\
        }\n\
    }\n\
    padded_image\n\
}\n'
    return code

def noir_test_main(args):
    let=''
    names=', '.join([a[0] for a in args])
    for a in args:
        let+='    let '+a[0]+':'+a[1]+'='+str(a[2])+';\n'
    code = '\n\
#[test]\n\
fn test_main() {\n'+let+'\
    main('+names+');\n\
}'
    return code


if __name__=="__main__":
    # network=[1024,512,2]  # 失败  Downloading the Ignite SRS (134.0 MB)
    network=[60,40,2] # 成功 Downloading the Ignite SRS (786.4 KB)
    # network=[600,40,2]  # 成功 Downloading the Ignite SRS (7.3 MB)
    # network=[600,400,2]  # 失败 Downloading the Ignite SRS (58.7 MB)
    # network=[712,712,2]  # 失败 Downloading the Ignite SRS (125.8 MB)
    # fnn_input_count(network)
    # fnn_inputs_value(network)
