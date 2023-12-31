import math

import numpy as np
from transpiler.core_module.for_loop_pod import ForLoop
from transpiler.core_module.func_pod import Function, FunctionGenerics
from transpiler.others_module import Annotation
from transpiler.context.noir_context import NoirContext
from transpiler.core_module.statement_pod import Let
from transpiler.core_module.control_pod import IfControl
from transpiler.sub_module.key_words import DEP_STD_PRINTLN, PRINTLN
from transpiler.sub_module.sign import LEFT_PARENTHESIS, RIGHT_PARENTHESIS, GREATER_THAN, SEMICOLON, COMMA, \
    LEFT_BRACKET, RIGHT_BRACKET
from transpiler.sub_module.primitive_type import UINT32
from transpiler.utils.utils import table_format_control

from ..decision_tree.decision_tree_to_noir import generate_body
from ..quantization.quantize import quantize, calc_scale, calc_zero_point

MODEL_NAME = 'XGBoost'


def get_is_leaves(dfs):
    trees = []
    for df in dfs:
        is_leaves = np.zeros(len(df['Feature']))
        for i, node in enumerate(df.iloc):
            if node['Feature'] == "Leaf":
                is_leaves[i] = 1
        trees.append(is_leaves)
    return trees


def get_threshold(dfs):
    leaf_feature = -2
    trees = []
    for df in dfs:
        classes = []
        for node in df.iloc:
            if node['Feature'] == "Leaf":
                classes.append(leaf_feature)
            else:
                classes.append(node["Split"])
        trees.append(classes)
    return trees


def get_feature(dfs):
    leaf_feature = -2
    trees = []
    for df in dfs:
        classes = []
        for node in df.iloc:
            if node["Feature"] == 'Leaf':
                classes.append(leaf_feature)
            else:
                classes.append(int(node["Feature"].replace('f', '')))
        trees.append(classes)
    return trees


def get_children_left(dfs):
    leaf_children = -1
    trees = []
    for df in dfs:
        classes = []
        for node in df.iloc:
            if node["Feature"] == 'Leaf':
                classes.append(leaf_children)
            else:
                classes.append(int(node["Yes"].split('-')[-1]))
        trees.append(classes)
    return trees


def get_children_right(dfs):
    leaf_children = -1
    trees = []
    for df in dfs:
        classes = []
        for node in df.iloc:
            if node["Feature"] == 'Leaf':
                classes.append(leaf_children)
            else:
                classes.append(int(node["No"].split('-')[-1]))
        trees.append(classes)
    return trees


def get_values(dfs):
    leaf_value = -2
    trees = []
    for df in dfs:
        classes = []
        for node in df.iloc:
            if node["Feature"] != 'Leaf':
                classes.append(leaf_value)
            else:
                classes.append(node["Gain"])
        trees.append(classes)
    return trees


def data_construction(clf, is_classification, struct_name):
    dfs = []
    trees = clf.get_booster()
    n_estimators = clf.n_estimators
    n_classes = 1
    for i in range(n_estimators):
        df = trees[i].trees_to_dataframe()
        if is_classification:
            n_classes = clf.n_classes_
            for c in range(n_classes):
                class_df = df[df["Tree"] == c].reset_index(drop=True)
                if not class_df.empty:
                    dfs.append(class_df)
        else:
            dfs.append(df)
    is_leaves = get_is_leaves(dfs)
    threshold = get_threshold(dfs)
    feature = get_feature(dfs)
    children_left = get_children_left(dfs)
    children_right = get_children_right(dfs)
    values = get_values(dfs)

    return n_estimators, n_classes, is_leaves, threshold, feature, children_left, children_right, values


def generate_functions_body(is_leaves_est, threshold_est, feature_est, children_left_est,
                            children_right_est, values_est, arr_name, q_scale, q_zero_point, q_type):
    """
    description: generate functions_body by trees from XGBoost lib
    inputs: origin trees: df[],
    result: [[body,][]] class(body) list in tree list
    """
    functions_body = []
    for i in range(len(is_leaves_est)):
        is_leaves_cls = is_leaves_est[i]
        threshold_cls = quantize(threshold_est[i], q_scale, q_zero_point, q_type)
        feature_cls = feature_est[i]
        children_left_cls = children_left_est[i]
        children_right_cls = children_right_est[i]
        values_cls = values_est[i]
        # for index, value in enumerate(values_cls):
            # if value == -2:
            #     values_cls[index] = 0
            # else:
            #     values_cls[index] = (value*100).__round__()
        values_range_list = [-1,1]
        scale_molecule, scale_denominator = calc_scale(values_range_list,q_type)
        values_scale = math.ceil(scale_molecule / scale_denominator)
        values_zero_point = calc_zero_point(values_range_list, values_scale, q_type)
        values_cls = quantize(values_cls, values_scale, values_zero_point, q_type)
        sign_function_body = generate_body(children_left_cls, children_right_cls, feature_cls, threshold_cls,
                                           values_cls, arr_name, is_leaves_cls)
        sign_function_body.append(Annotation(-1,
                                   f"inputs quantization scale reciprocal: {values_scale}, zero-point: {values_zero_point}").get_content())

        functions_body.append(sign_function_body)
    return functions_body


def generate_main_body(context, first_input_arg, result_type, is_classification, n_classes):
    main_body = []
    function_list = context.function_list
    let_variate_list = []
    for func in function_list:
        if type(func) is Function:
            let_variate = func.fn_name.replace("class", "_class")
            let_variate_list.append(let_variate)
            let_variate_type = UINT32
            let_variate_body = f"{func.fn_name}{LEFT_PARENTHESIS}{first_input_arg}{RIGHT_PARENTHESIS}"
            main_body.append(Let(let_variate, let_variate_type, let_variate_body, is_mut=False).get())
    # get count function name
    count_func = ""
    for func in function_list:
        if type(func) is FunctionGenerics:
            count_func = func.fn_name
    if not is_classification:
        # for range and add all data
        res_variate = "res"
        res_variate_type = result_type
        res_variate_body = f'{count_func}{LEFT_PARENTHESIS}{LEFT_BRACKET}'
        for index, let_variate in enumerate(let_variate_list):
            res_variate_body += f"{let_variate}"
            if index != len(let_variate_list) - 1:
                res_variate_body += f"{COMMA} "
        res_variate_body += f'{RIGHT_BRACKET}{RIGHT_PARENTHESIS}'
        main_body.append(Let(res_variate, res_variate_type, res_variate_body, is_mut=False).get())
        main_body.append(res_variate)
    else:
        u32 = UINT32
        res_variate = "res"
        max_ele = "max_ele_index"
        # let: Calculate the sum of the class
        for index in range(n_classes):
            class_list = []

            for let_variate in let_variate_list:
                if f"class{index}" in let_variate:
                    class_list.append(let_variate)

            class_variate_body = f'{count_func}{LEFT_PARENTHESIS}{LEFT_BRACKET}'
            class_variate = "c" + str(index)
            class_variate_type = u32
            for class_index, let_variate in enumerate(class_list):
                class_variate_body += f"{let_variate}"
                if class_index != len(class_list) - 1:
                    class_variate_body += f"{COMMA} "
            class_variate_body += f'{RIGHT_BRACKET}{RIGHT_PARENTHESIS}'
            main_body.append(Let(class_variate, class_variate_type, class_variate_body, is_mut=False).get())

        # if else control: get max res
        main_body.append(Let(max_ele, u32, "c0", is_mut=True).get())
        main_body.append(Let(res_variate, u32, 0, is_mut=True).get())
        for index in range(1, n_classes):
            left_value = "c" + str(index)
            right_value = max_ele
            sign = GREATER_THAN
            body = f"{max_ele} = {left_value};\n{res_variate} = {str(index)};"
            main_body.append(IfControl(left_value, right_value, sign, body).get())
        # println result info
        res_debug_print = f"{PRINTLN}{LEFT_PARENTHESIS}{res_variate}{RIGHT_PARENTHESIS}{SEMICOLON}"
        main_body.append(res_debug_print)
        main_body.append(res_variate)
    return main_body





def generate_count_prob_body(count_prob_inputs):
    body = []
    # let counter
    cls = list(count_prob_inputs.keys())[0]
    count_variate = "cls"
    body.append(
        Let(variate=count_variate, variate_type='Field', variate_body=f'{cls}[0] as Field', is_mut=True).get()
    )
    # acc
    body.append(
        ForLoop(variate='i', start_variate=1, end_variate='N', body=
        f'{count_variate} = quantize_arithmetic::add({count_variate},127,128,{cls}[i] as Field,127,128,127,128);').get()
    )
    # Out of group quantization
    body.append(
        f'{count_variate} = quantize_arithmetic::mul(cls,127,128,1,0,1,125,13);'
    )
    # Add result
    body.append(f'(({count_variate}*128) as u32)/128')
    return body



def xgboost_noir_code(clf, q_scale, q_zero_point, q_type, is_classification: bool = True, Noir_name: str = "main"):
    # Only signed integer types can be used in XGBoost to Noir, Because leaf can be negative

    # Noir context maintain
    context = NoirContext()

    # Add annotation in context
    context.add_annotation(0, f"inputs quantization scale reciprocal: {q_scale}")
    context.add_annotation(1, f"inputs quantization zero-point: {q_zero_point}")
    context.add_annotation(2, f"quantize_type: {q_type}")

    # Add Noir import
    context.add_use(DEP_STD_PRINTLN)
    context.add_mod("quantize_arithmetic")
    # structure inputs array
    arr_name = "inputs"
    fn_inputs_name_and_type = {arr_name: f"[{UINT32};{clf.n_features_in_}]"}

    # Get XGBoost information
    n_estimators, n_classes, is_leaves, threshold, feature, children_left, children_right, values = data_construction(
        clf, is_classification, arr_name)

    # Add functions
    functions_body = generate_functions_body(is_leaves, threshold, feature, children_left,
                                             children_right, values, arr_name, q_scale, q_zero_point, q_type)
    for index, tree in enumerate(functions_body):
        variate = f'trees{int(index / n_classes)}class{index % n_classes}'
        result_type = UINT32
        context.add_function(variate, fn_inputs_name_and_type, result_type, body=tree)

    # Add count_prob function
    count_prob = 'count_prob'
    result_type = UINT32
    count_prob_inputs = {"class_prob": "[u32;N]"}
    context.add_function_generics(count_prob, count_prob_inputs, result_type,
                                  body=generate_count_prob_body(count_prob_inputs),
                                  generics_type=False, generics_num=True)
    # Add main function and body
    variate = 'main'
    result_type = UINT32
    input1 = arr_name.lower()
    main_body = generate_main_body(context, input1, result_type, is_classification, n_classes)
    context.add_function(variate, fn_inputs_name_and_type, result_type, main_body)

    # Noir code generate
    Noir_code_list = context.generate_noir_code_list()
    # print(Noir_code_list)
    # code table format
    return table_format_control(Noir_code_list)
