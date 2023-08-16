import math
import numpy as np
from sklearn import tree
from transpiler.context.noir_context import NoirContext
from transpiler.sub_module.primitive_type import INT32, UINT32, custom_type
from transpiler.sub_module.sign import LESS_THAN, LESS_THAN_OR_EQUAL, LEFT_BRACKET, RIGHT_BRACKET
from transpiler.core_module.control_pod import IfElseControl
from transpiler.utils.utils import table_format_control

from zkml.quantization.quantize import quantize_all, quantize, UINT


def generate_dt(model, is_negative, q_scale, q_zero_point, quantize_type):
    # Get tree information
    children_left, children_right, feature, threshold, values, is_leaves = data_construction(model)
    threshold = quantize(threshold, q_scale, q_zero_point, quantize_type)
    # Noir context maintain
    noir = NoirContext()
    # Add annotation in context
    noir.add_annotation(0, f"inputs quantization scale reciprocal: {q_scale}")
    noir.add_annotation(1, f"inputs quantization zero-point: {q_zero_point}")
    noir.add_annotation(2, f"quantize_type: {quantize_type}")


    fn_name = 'main'
    array_name = 'inputs'

    if is_negative:
        fn_inputs_name_and_type = {array_name: f"[{INT32};{model.n_features_in_}]"}
    else:
        fn_inputs_name_and_type = {array_name: f"[{UINT32};{model.n_features_in_}]"}

    # fn_inputs_name_and_type = generate_name_and_type(is_negative, model.n_features_in_)
    fn_result = custom_type('u3')
    body = generate_body(children_left, children_right, feature, threshold, values,
                         array_name, is_leaves)
    noir.add_function(fn_name, fn_inputs_name_and_type, fn_result, body)

    noir_code_list = noir.generate_noir_code_list()

    return table_format_control(noir_code_list)


def data_construction(clf: tree.DecisionTreeClassifier):
    # number of nodes
    n_nodes = clf.tree_.node_count
    # Left and right child nodes
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    # Features: columns of data, element variable names
    feature = clf.tree_.feature
    # threshold: save decision target data, access via tree's children
    threshold = clf.tree_.threshold

    values = [np.argmax(value[0]) for value in clf.tree_.value]
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    return children_left, children_right, feature, threshold, values, is_leaves


def generate_name_and_type(is_negative, feature) -> dict:
    """
    Generate Array object name_and_type of attributes
    """
    res = dict()
    if is_negative:
        for index in range(feature):
            print(f"p{str(index)}", str(INT32))
            res[f"p{str(index)}"] = str(INT32)
    else:
        for index in range(feature):
            res[f"p{str(index)}"] = str(UINT32)

    return res


def generate_body(children_left, children_right, feature, threshold, values,
                  array_name, is_leaves):

    def build_tree(head):
        control_tree = []
        # build_tree(head)
        if is_leaves[head]:
            res = str(values[head])
            control_tree.append(res)
            return control_tree

        nodes_threshold = threshold[head]
        comp = LESS_THAN if int(threshold[head]) != threshold[head] else LESS_THAN_OR_EQUAL
        left_value = f'{array_name}{LEFT_BRACKET}{feature[head]}{RIGHT_BRACKET}'
        right_value = str(nodes_threshold)
        if_else_control = IfElseControl(left_value, right_value, comp, '\n'.join(build_tree(children_left[head])),
                                        '\n'.join(build_tree(children_right[head]))).get()
        control_tree += if_else_control.split('\n')
        return control_tree

    body = build_tree(0)
    return body
