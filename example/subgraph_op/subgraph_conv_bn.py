import mxnet as mx
import numpy as np
import time
import ctypes
from mxnet import Context
from mxnet.base import _LIB, SymbolHandle, check_call, mx_uint, c_str_array
from mxnet.symbol import Symbol
from numpy.testing import assert_allclose

np.random.seed(12345)

data = mx.symbol.Variable('data')
weight1 = mx.symbol.Variable('weight1')
weight2 = mx.symbol.Variable('weight2')

conv1 = mx.symbol.Convolution(data=data, weight=weight1, name='conv1', num_filter=64, kernel=(3,3), stride=(1,1))
bn1 = mx.symbol.BatchNorm(data=conv1, name="bn1")
act1 = mx.symbol.Activation(data=bn1, act_type='relu', name='relu1')

conv2 = mx.symbol.Convolution(data=act1, weight=weight2, name='conv2', num_filter=128, kernel=(5,5), stride=(1,1))
bn2 = mx.symbol.BatchNorm(data=conv2, name="bn2")
act2 = mx.symbol.Activation(data=bn2, act_type='relu', name='relu2')

sum1 = mx.symbol.sum(mx.symbol.slice(act2, begin=(0, 0, 0, 0), end=(1,1,1,10)))

#g = mx.viz.plot_network(sum1)
#g.format = 'png'
#g.render('sum2')

out = SymbolHandle()
check_call(_LIB.MXPartitionConvBN(sum1.handle, ctypes.byref(out)))
psym = Symbol(out)

#g = mx.viz.plot_network(psym)
#g.format = 'png'
#g.render('psym2')

shape = (32, 3, 224, 224)
exe_subgraph = psym.simple_bind(Context.default_ctx, data=shape, grad_req='null')
exe = sum1.simple_bind(Context.default_ctx, data=shape, grad_req='null')

val = np.random.rand(*shape).astype(np.float32)
val1 = np.random.normal(size=exe.arg_arrays[1].shape)
val2 = np.random.normal(size=exe.arg_arrays[2].shape)
val3 = np.random.normal(size=exe.arg_arrays[3].shape)
val4 = np.random.normal(size=exe.arg_arrays[4].shape)
val5 = np.random.normal(size=exe.arg_arrays[5].shape)
val6 = np.random.normal(size=exe.arg_arrays[6].shape)
val7 = np.random.normal(size=exe.arg_arrays[7].shape)
val8 = np.random.normal(size=exe.arg_arrays[8].shape)


exe_subgraph.arg_arrays[0][:] = val
exe_subgraph.arg_arrays[1][:] = val1 
exe_subgraph.arg_arrays[2][:] = val2 
exe_subgraph.arg_arrays[3][:] = val3 
exe_subgraph.arg_arrays[4][:] = val4 
exe_subgraph.arg_arrays[5][:] = val5 
exe_subgraph.arg_arrays[6][:] = val6 
exe_subgraph.arg_arrays[7][:] = val7 
exe_subgraph.arg_arrays[8][:] = val8 

exe.arg_arrays[0][:] = val
exe.arg_arrays[1][:] = val1 
exe.arg_arrays[2][:] = val2 
exe.arg_arrays[3][:] = val3 
exe.arg_arrays[4][:] = val4 
exe.arg_arrays[5][:] = val5 
exe.arg_arrays[6][:] = val6 
exe.arg_arrays[7][:] = val7 
exe.arg_arrays[8][:] = val8 

p = exe_subgraph.forward(is_train=False)
p[0].wait_to_read()

q = exe.forward(is_train=False)
q[0].wait_to_read()

# print(p[0])
# print(q[0])
assert_allclose(p[0].asnumpy(), q[0].asnumpy(), rtol=1e-6)
