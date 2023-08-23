import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from zkml.routine_code_generate.utils import *

def noir_fnn_main(neurons_per_layer, zero_point=18446744073709551616, scale_reciprocal=4294967296, path=''):
    noir_code = 'use dep::std;\n'
    noir_code += global_const('zero_point', zero_point)
    noir_code += global_const('scale_reciprocal', scale_reciprocal)

    global_const_code, main_args_code, main_code, prover_str, test_args = fnn_layer(neurons_per_layer)
    global_const_code = '\n'.join([global_const_code])
    main_args_code = ', '.join([main_args_code])
    main_code = ''.join([main_code])
    prover_str = ''.join([prover_str])

    noir_code+=global_const_code
    noir_code+='\n\
fn main('+main_args_code+') {\n\
'+main_code+'\n\
}\n'
    noir_code+=noir_fn_forward()
    noir_code+=noir_fn_activation_relu()
    noir_code+=noir_fn_operation()
    noir_code+=noir_test_main([test_args])

    with open(os.path.join(path, "src/main.nr"), "w+") as file:
        file.write(noir_code)
    with open(os.path.join(path, "Prover.toml"), "w+") as file:
        file.write(prover_str)


if __name__=="__main__":
    # neurons_per_layer=[1024,1024,2]  # Downloading the Ignite SRS (134.0 MB)
    neurons_per_layer=[60,40,2] # Downloading the Ignite SRS (786.4 KB)
    # neurons_per_layer=[600,40,2]  # Downloading the Ignite SRS (7.3 MB)
    # neurons_per_layer=[600,400,2]  # Downloading the Ignite SRS (58.7 MB)
    # neurons_per_layer=[768, 2304, 768, 3072, 768]  # Downloading the Ignite SRS (125.8 MB)
    # fnn_input_count(neurons_per_layer)
    # fnn_inputs_value(neurons_per_layer)
    noir_fnn_main(neurons_per_layer, zero_point=18446744073709551616, scale_reciprocal=4294967296, path='/mnt/code/noir_project/fnn3')
