/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/

#include <mxnet/ndarray.h>
#include <mshadow/base.h>
#include "./mkldnn_conv_bn.h"
#include "../../../imperative/imperative_utils.h"
#include "../../../imperative/cached_op.h"
#include "../../nn/convolution-inl.h"
#include "../../nn/batch_norm-inl.h"

namespace mxnet {
namespace op {

#define SUBGRAPH_DEBUG 0

template<typename DType>
void UpdateConvWeightBias(const NDArray &weight, const NDArray* bias,
                          const NDArray &gamma, const NDArray &beta, const NDArray &variance,
                          std::shared_ptr<NDArray> update_weight,
                          std::shared_ptr<NDArray> update_bias,
                          const BatchNormParam &param) {
#if SUBGRAPH_DEBUG
  printf("input weight: %f %f %f %f \n", weight.data().dptr<float>()[0],
                                         weight.data().dptr<float>()[1],
                                         weight.data().dptr<float>()[2],
                                         weight.data().dptr<float>()[3]);
  printf("bn param eps: %f \n", param.eps);
  printf("bn param fix_gamma: %d \n", param.fix_gamma);
  printf("bn param use_global_stats: %d \n", param.use_global_stats);
  printf("bn param output_mean_var: %d \n", param.output_mean_var);
  printf("bn param axis: %d \n", param.axis);
#endif
  DType* weight_ptr = weight.data().dptr<DType>();
  DType* bias_ptr = nullptr;
  DType* update_bias_ptr = nullptr;
  if (bias) {
    bias_ptr = bias->data().dptr<DType>();
    update_bias_ptr = update_bias->data().dptr<DType>();
  }
  DType* gamma_ptr  = gamma.data().dptr<DType>();
  DType* beta_ptr   = beta.data().dptr<DType>();
  DType* var_ptr    = variance.data().dptr<DType>();

  DType* update_weight_ptr = update_weight->data().dptr<DType>();

  size_t channel = gamma.shape()[0];
  size_t offset = weight.shape()[1] * weight.shape()[2] * weight.shape()[3];
#pragma omp parallel for
  for (size_t c = 0; c < channel; ++c) {
    DType* p1 = reinterpret_cast<DType*>(weight_ptr + c * offset);
    DType* p2 = reinterpret_cast<DType*>(update_weight_ptr + c * offset);
    DType alpha = (param.fix_gamma ? static_cast<DType>(1.0f) : gamma_ptr[c]) /
        sqrt(var_ptr[c] + param.eps);

    if (bias_ptr && update_bias_ptr) {
      update_bias_ptr[c] = alpha * bias_ptr[c] + beta_ptr[c];
    }

    for (size_t k = 0; k < offset; ++k) {
      p2[k] = p1[k] * alpha;
    }
  }
#if SUBGRAPH_DEBUG
  printf("update weight: %f %f %f %f \n", update_weight->data().dptr<float>()[0],
                                          update_weight->data().dptr<float>()[1],
                                          update_weight->data().dptr<float>()[2],
                                          update_weight->data().dptr<float>()[3]);
#endif
}

class ConvBNSubgraphOperator {
 public:
  explicit ConvBNSubgraphOperator(const Symbol &sym) :
      subgraph_sym_(sym), cached_weight_(nullptr), cached_bias_(nullptr) {
    auto outputs = subgraph_sym_.outputs;
    CHECK_EQ(outputs.size(), 1U);

    auto bn_node = outputs[0].node;
    auto inputs = bn_node->inputs;
    auto conv_node = inputs[0].node;

    bn_attrs_ = bn_node->attrs;
    conv_attrs_ = conv_node->attrs;
  }

  void Forward(const OpContext &ctx,
               const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<NDArray> &outputs);

  void Backward(const OpContext &ctx,
                const std::vector<NDArray> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<NDArray> &outputs) {
    LOG(FATAL) << "Not implemented: ConvBN subgraph only supports inference computation";
  }

 private:
  nnvm::Symbol subgraph_sym_;
  std::shared_ptr<NDArray> cached_weight_;
  std::shared_ptr<NDArray> cached_bias_;

  nnvm::NodeAttrs bn_attrs_;
  nnvm::NodeAttrs conv_attrs_;
};

void ConvBNSubgraphOperator::Forward(const OpContext &ctx,
                                     const std::vector<NDArray> &inputs,
                                     const std::vector<OpReqType> &req,
                                     const std::vector<NDArray> &outputs) {

  const BatchNormParam &bn_param = nnvm::get<BatchNormParam>(bn_attrs_.parsed);
  const ConvolutionParam &conv_param = nnvm::get<ConvolutionParam>(conv_attrs_.parsed);
#if SUBGRAPH_DEBUG
  LOG(INFO) << "ConvBN inputs size: " << inputs.size();
  LOG(INFO) << "ConvBN outputs size: " << outputs.size();
  LOG(INFO) << "ConvBN req size: " << req.size();
  for (size_t k = 0; k < inputs.size(); ++k) {
    auto input = inputs[k];
    printf("input %ld :", k);
    for (size_t i = 0; i < input.shape().ndim(); ++i) {
      printf("%ld ", input.shape()[i]);
    }
    printf("\n");
  }
  CHECK_EQ(ctx.is_train, false);
  printf("output:");
    for (size_t i = 0; i < outputs[0].shape().ndim(); ++i) {
      printf("%ld ", outputs[0].shape()[i]);
    }
    printf("\n");
#endif

  CHECK_EQ(inputs.size(), conv_param.no_bias ? 6U : 7U);
  NDArray output = outputs[0];

  // CHECK(!conv_weight.IsMKLDNN());
  CHECK_EQ(inputs[3].shape()[0], inputs[2].shape()[0]);

  if (nullptr == cached_weight_ || nullptr == cached_bias_) {
    cached_weight_.reset(new NDArray(inputs[1].storage_type(),
                                     inputs[1].shape(),
                                     inputs[1].ctx(),
                                     true,
                                     inputs[1].dtype()));
    if (!conv_param.no_bias) {
      cached_bias_.reset(new NDArray(inputs[2].storage_type(),
                                     inputs[2].shape(),
                                     inputs[2].ctx(),
                                     true,
                                     inputs[2].dtype()));
      MSHADOW_REAL_TYPE_SWITCH(inputs[1].dtype(), DType, {
          UpdateConvWeightBias<DType>(inputs[1], &(inputs[2]), inputs[3], inputs[4], inputs[6],
                                      cached_weight_, cached_bias_, bn_param);
      });
    } else {
      MSHADOW_REAL_TYPE_SWITCH(inputs[1].dtype(), DType, {
          UpdateConvWeightBias<DType>(inputs[1], nullptr, inputs[2], inputs[3], inputs[5],
                                      cached_weight_, cached_bias_, bn_param);
      });
    }
  }
  if (!conv_param.no_bias) {
#if MXNET_USE_MKLDNN == 1
    ConvolutionComputeExCPU(conv_attrs_, ctx,
                            {inputs[0], *cached_weight_, *cached_bias_},
                            {req[0]},
                            {output});
#else
    ConvolutionCompute<cpu>(conv_attrs_, ctx,
                            {inputs[0].data(), cached_weight_->data(), cached_bias_->data()},
                            {req[0]},
                            {output.data()});
#endif
  } else {
#if MXNET_USE_MKLDNN == 1
    ConvolutionComputeExCPU(conv_attrs_, ctx,
                            {inputs[0], *cached_weight_},
                            {req[0]},
                            {output});
#else
    ConvolutionCompute<cpu>(conv_attrs_, ctx,
                            {inputs[0].data(), cached_weight_->data()},
                            {req[0]},
                            {output.data()});
#endif
  }
}

OpStatePtr CreateConvBNSubgraphOpState(const NodeAttrs &attrs,
                                       Context ctx,
                                       const std::vector<TShape> &in_shapes,
                                       const std::vector<int> &in_types) {
  const Symbol &subgraph_sym = nnvm::get<Symbol>(attrs.parsed);
  return OpStatePtr::Create<ConvBNSubgraphOperator>(subgraph_sym);
}


void ConvBNSubgraphOpForward(const OpStatePtr &state_ptr,
                             const OpContext &ctx,
                             const std::vector<NDArray> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<NDArray> &outputs) {
  ConvBNSubgraphOperator &op = state_ptr.get_state<ConvBNSubgraphOperator>();
  op.Forward(ctx, inputs, req, outputs);
}

NNVM_REGISTER_OP(_conv_bn_subgraph_op)
.describe(R"code(_conv_bn_subgraph_op)code" ADD_FILELINE)
.set_num_inputs(DefaultSubgraphOpNumInputs)
.set_num_outputs(DefaultSubgraphOpNumOutputs)
.set_attr<nnvm::FListInputNames>("FListInputNames", DefaultSubgraphOpListInputs)
.set_attr<nnvm::FListOutputNames>("FListOutputNames", DefaultSubgraphOpListOutputs)
.set_attr<FCreateOpState>("FCreateOpState", CreateConvBNSubgraphOpState)
.set_attr<nnvm::FInferShape>("FInferShape", DefaultSubgraphOpShape)
.set_attr<nnvm::FInferType>("FInferType", DefaultSubgraphOpType)
.set_attr<FInferStorageType>("FInferStorageType", DefaultSubgraphOpStorageType)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", ConvBNSubgraphOpForward)
.set_attr<nnvm::FMutateInputs>("FMutateInputs", DefaultSubgraphOpMutableInputs)
.set_attr<FResourceRequest>("FResourceRequest", DefaultSubgraphOpResourceRequest)
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<FExecType>("FExecType", DefaultSubgraphOpExecType)
.add_argument("data", "NDArray-or-Symbol[]", "input data list");

}  // namespace op
}  // namespace mxnet
