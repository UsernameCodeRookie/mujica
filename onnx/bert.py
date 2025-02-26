from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info
)

# TODO: Analyze the reduction dimension
tQ = make_tensor_value_info('Q', TensorProto.FLOAT, ['batch', 'head', 'sequence', 'hidden'])
tK = make_tensor_value_info('K', TensorProto.FLOAT, ['batch', 'head', 'sequence', 'hidden'])
tA = make_tensor_value_info('A', TensorProto.FLOAT, ['batch', 'head', 'sequence', 'sequence'])
tV = make_tensor_value_info('V', TensorProto.FLOAT, ['batch', 'head', 'sequence', 'hidden'])
tO = make_tensor_value_info('O', TensorProto.FLOAT, ['batch', 'head', 'sequence', 'hidden'])

mm1 = make_node('MatMul', ['Q', 'K'], ['A'])
mm2 = make_node('MatMul', ['A', 'V'], ['O'])

graph = make_graph(
    [mm1, mm2],
    'attention',
    [tQ, tK, tV],
    [tO]
)

model = make_model(graph, producer_name='onnx-bert')