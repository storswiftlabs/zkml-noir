import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from zkml.fnn.utils import *

def noir_fnn_main(neurons_per_layer, path=''):
    num = fnn_input_count(neurons_per_layer)
    inputs, str_inputs = fnn_inputs_value(neurons_per_layer)
    noir_code = 'use dep::std;\n'
    noir_code += global_const('neurons_per_layer', neurons_per_layer)
    noir_code += global_const('num', num)
    noir_code += global_const('scaling_factor', 16)
    noir_code += global_const('fnn_out_len', str(neurons_per_layer[-1]))
    noir_code+='fn main(mut inputs : pub [u16; num]) {\n\
    let mut result : [u16; fnn_out_len] = [0; fnn_out_len];\n\
    result = forward(inputs, neurons_per_layer, result);\n\
    std::println(result);\n\
}\n'
    noir_code+=noir_fn_forward()
    noir_code+=noir_fn_activation_relu()
    noir_code+=noir_test_main([['inputs', '[u16; num]', '[0; num]']])
    prover_str = 'inputs = '+str_inputs+'\n'
    with open(os.path.join(path, "src/main.nr"), "w+") as file:
        file.write(noir_code)
    with open(os.path.join(path, "Prover.toml"), "w+") as file:
        file.write(prover_str)


if __name__=="__main__":
    neurons_per_layer=[1024,1024,2]  # 失败  Downloading the Ignite SRS (134.0 MB)
    # neurons_per_layer=[60,40,2] # 成功 Downloading the Ignite SRS (786.4 KB)
    # neurons_per_layer=[600,40,2]  # 成功 Downloading the Ignite SRS (7.3 MB)
    # neurons_per_layer=[600,400,2]  # 失败 Downloading the Ignite SRS (58.7 MB)
    # neurons_per_layer=[768, 2304, 768, 3072, 768]  # 失败 Downloading the Ignite SRS (125.8 MB)
    # fnn_input_count(neurons_per_layer)
    # fnn_inputs_value(neurons_per_layer)
    noir_fnn_main(neurons_per_layer, path='/mnt/code/noir_project/fnn3')
