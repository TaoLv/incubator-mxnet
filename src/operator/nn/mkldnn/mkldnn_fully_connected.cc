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

/*!
 * \file mkldnn_fully_connected.cc
 * \brief
 * \author Da Zheng
*/

#include "../fully_connected-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

inline static mkldnn::inner_product_forward::primitive_desc GetIPFwd(
    const NDArray &data, const NDArray &weight, const NDArray *bias,
    const mkldnn::memory::desc &out_md) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetMemDesc(weight);
  auto engine = CpuEngine::Get()->get_engine();
  if (bias) {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::inner_product_forward::desc ipFwd_desc(mkldnn::prop_kind::forward_training,
        data_md, weight_md, bias_md, out_md);
    return mkldnn::inner_product_forward::primitive_desc(ipFwd_desc, engine);
  } else {
    mkldnn::inner_product_forward::desc ipFwd_desc(mkldnn::prop_kind::forward_training,
        data_md, weight_md, out_md);
    return mkldnn::inner_product_forward::primitive_desc(ipFwd_desc, engine);
  }
}

inline static mkldnn::inner_product_backward_data::primitive_desc GetIpBwdData(
    const NDArray &data, const NDArray &weight, const NDArray &output,
    mkldnn::inner_product_forward::primitive_desc ipFwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetMemDesc(weight);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  mkldnn::inner_product_backward_data::desc desc(data_md, weight_md, out_md);
  return mkldnn::inner_product_backward_data::primitive_desc(desc, engine, ipFwd_pd);
}

inline static mkldnn::inner_product_backward_weights::primitive_desc GetIPBwdWeights(
    const NDArray &data, const NDArray &weight, const NDArray *bias,
    const NDArray &output, mkldnn::inner_product_forward::primitive_desc ipFwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetMemDesc(weight);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  if (bias) {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::inner_product_backward_weights::desc ipBwdWeights_desc(data_md,
        weight_md, bias_md, out_md);
    return mkldnn::inner_product_backward_weights::primitive_desc(
        ipBwdWeights_desc, engine, ipFwd_pd);
  } else {
    mkldnn::inner_product_backward_weights::desc ipBwdWeights_desc(data_md,
        weight_md, out_md);
    return mkldnn::inner_product_backward_weights::primitive_desc(
        ipBwdWeights_desc, engine, ipFwd_pd);
  }
}

typedef MKLDNNParamOpSign<FullyConnectedParam> MKLDNNFullyConnectedSignature;

class MKLDNNFullyConnectedFwd {
 public:
  std::shared_ptr<mkldnn::inner_product_forward::primitive_desc> fwd_pd;
  MKLDNNFullyConnectedFwd(const FullyConnectedParam &param,
                          NDArray *data, const NDArray &weights,
                          const NDArray *bias, const NDArray &output,
                          const OpReqType &req_out) {
    _Init(param, data, weights, bias, output, req_out);
  }
  ~MKLDNNFullyConnectedFwd() {}
  void SetDataHandle(const FullyConnectedParam &param,
                     NDArray *data,
                     const NDArray &weights,
                     const NDArray *bias,
                     const NDArray &output,
                     const OpReqType &req_out);
  void Execute(const NDArray &output, const OpReqType &req_out);

 private:
  void _Init(const FullyConnectedParam &param, NDArray *data,
             const NDArray &weights, const NDArray *bias,
             const NDArray &output, const OpReqType &req_out);

 private:
  std::shared_ptr<mkldnn::inner_product_forward> fwd;
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> weights;
  std::shared_ptr<mkldnn::memory> bias;
  std::shared_ptr<mkldnn::memory> out;
};

void MKLDNNFullyConnectedFwd::_Init(const FullyConnectedParam &param,
                                    NDArray *data,
                                    const NDArray &weights,
                                    const NDArray *bias,
                                    const NDArray &output,
                                    const OpReqType &req_out) {
  const TShape& ishape = data->shape();
  const TShape& oshape = output.shape();
  auto out_md = GetMemDesc(output);
  if (data->shape().ndim() != 2 && !param.flatten) {
    *data = data->ReshapeMKLDNN(Shape2(ishape.ProdShape(0, ishape.ndim()-1),
                                     ishape[ishape.ndim()-1]));
    mkldnn::memory::dims out_dims{static_cast<int>(oshape.ProdShape(0,
                                                   oshape.ndim()-1)),
                                  static_cast<int>(oshape[ishape.ndim()-1])};
    out_md = mkldnn::memory::desc(out_dims, get_mkldnn_type(output.dtype()),
                                  mkldnn::memory::format::any);
  } else if (data->shape().ndim() != 2) {
    *data = data->ReshapeMKLDNN(Shape2(ishape[0], ishape.ProdShape(1,
                                                              ishape.ndim())));
    mkldnn::memory::dims out_dims{static_cast<int>(oshape[0]),
                                  static_cast<int>(oshape.ProdShape(1,
                                                             oshape.ndim()))};
    out_md = mkldnn::memory::desc(out_dims, get_mkldnn_type(output.dtype()),
                                  mkldnn::memory::format::any);
  }

  auto data_md = GetMemDesc(*data);
  auto weight_md = GetMemDesc(weights);
  auto engine = CpuEngine::Get()->get_engine();
  if (bias) {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::inner_product_forward::desc
                 ipFwd_desc(mkldnn::prop_kind::forward_training,
                            data_md, weight_md, bias_md, out_md);
    this->fwd_pd.reset(new mkldnn::inner_product_forward::primitive_desc(
                                            ipFwd_desc, engine));
  } else {
    mkldnn::inner_product_forward::desc
                 ipFwd_desc(mkldnn::prop_kind::forward_training,
                            data_md, weight_md, out_md);
    this->fwd_pd.reset(new mkldnn::inner_product_forward::primitive_desc(
                                            ipFwd_desc, engine));
  }

  this->data.reset(new mkldnn::memory(this->fwd_pd->src_primitive_desc()));
  this->weights.reset(new mkldnn::memory(
                                    this->fwd_pd->weights_primitive_desc()));
  this->out.reset(new mkldnn::memory(this->fwd_pd->dst_primitive_desc()));
  if (param.no_bias) {
    this->fwd.reset(new mkldnn::inner_product_forward(
        *(this->fwd_pd), *(this->data), *(this->weights), *(this->out)));
  } else {
    this->bias.reset(new mkldnn::memory(this->fwd_pd->bias_primitive_desc()));
    this->fwd.reset(new mkldnn::inner_product_forward(*(this->fwd_pd),
        *(this->data), *(this->weights), *(this->bias), *(this->out)));
  }
}

void MKLDNNFullyConnectedFwd::SetDataHandle(const FullyConnectedParam &param,
                                            NDArray *data,
                                            const NDArray &weights,
                                            const NDArray *bias,
                                            const NDArray &output,
                                            const OpReqType &req_out) {
  const TShape& ishape = data->shape();
  if (data->shape().ndim() != 2 && !param.flatten) {
    *data = data->ReshapeMKLDNN(Shape2(ishape.ProdShape(0, ishape.ndim()-1),
                                     ishape[ishape.ndim()-1]));
  } else if (data->shape().ndim() != 2) {
    *data = data->ReshapeMKLDNN(Shape2(ishape[0], ishape.ProdShape(1,
                                                              ishape.ndim())));
  }
  auto data_mem = data->GetMKLDNNDataReorder(this->fwd_pd->src_primitive_desc());
  auto weight_mem = weights.GetMKLDNNDataReorder(
                                      this->fwd_pd->weights_primitive_desc());
  auto out = CreateMKLDNNMem(output,
                             this->fwd_pd->dst_primitive_desc(), req_out);

  this->data->set_data_handle(data_mem->get_data_handle());
  this->weights->set_data_handle(weight_mem->get_data_handle());
  if (bias) {
    auto bias_mem = bias->GetMKLDNNDataReorder(
                        this->fwd_pd->bias_primitive_desc());
    this->bias->set_data_handle(bias_mem->get_data_handle());
  }
  this->out->set_data_handle(out.second->get_data_handle());
}

void MKLDNNFullyConnectedFwd::Execute(const NDArray &output,
                                      const OpReqType &req_out) {
  MKLDNNStream::Get()->RegisterPrim(*(this->fwd));
  auto out = CreateMKLDNNMem(output,
                             this->fwd_pd->dst_primitive_desc(), req_out);
  CommitOutput(output, out);
  MKLDNNStream::Get()->Submit();
}

static MKLDNNFullyConnectedFwd
&GetFullyConnectedFwd(const FullyConnectedParam &param, const OpContext &ctx,
                      NDArray *data, const NDArray &weights,
                      const NDArray *bias, const NDArray &output,
                      const OpReqType &req_out) {
  static thread_local std::unordered_map<MKLDNNFullyConnectedSignature,
                                         MKLDNNFullyConnectedFwd,
                                         MKLDNNOpHash> fc_fwds;
  MKLDNNFullyConnectedSignature key(param);
  key.AddSign(ctx.is_train);
  key.AddSign(req_out);
  key.AddSign(*data);
  key.AddSign(weights);
  key.AddSign(output);
  if (bias) {
    key.AddSign(*bias);
  }

  auto it = fc_fwds.find(key);
  if (it == fc_fwds.end()) {
    MKLDNNFullyConnectedFwd fwd(param, data, weights, bias, output, req_out);
    auto ins_ret = fc_fwds.insert(std::pair<MKLDNNFullyConnectedSignature,
                                            MKLDNNFullyConnectedFwd>(key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

void MKLDNNFullyConnectedCompute(const nnvm::NodeAttrs& attrs,
                                 const OpContext &ctx,
                                 const std::vector<NDArray> &in_data,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<NDArray> &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[fullc::kTempSpace]);
  const FullyConnectedParam &param
                    = nnvm::get<FullyConnectedParam>(attrs.parsed);
  auto data = in_data[fullc::kData];
  auto weights = in_data[fullc::kWeight];
  auto output = out_data[fullc::kOut];
  OpReqType req_out = req[fullc::kOut];
  MKLDNNFullyConnectedFwd &fwd = GetFullyConnectedFwd(param, ctx, &data,
          weights, param.no_bias ? nullptr : &in_data[fullc::kBias],
          output, req_out);
  fwd.SetDataHandle(param, &data, weights,
                    param.no_bias ? nullptr : &in_data[fullc::kBias],
                    output, req_out);
  fwd.Execute(output, req_out);
}

void MKLDNNFullyConnectedGradCompute(const nnvm::NodeAttrs& attrs,
                                     const OpContext &ctx,
                                     const std::vector<NDArray> &inputs,
                                     const std::vector<OpReqType> &req,
                                     const std::vector<NDArray> &outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[fullc::kTempSpace]);
  const std::vector<NDArray> &in_grad = outputs;
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  const TShape& ishape = inputs[fullc::kData + 1].shape();
  const TShape& oshape = inputs[fullc::kOut].shape();

  NDArray weight = inputs[fullc::kWeight + 1];
  NDArray data = inputs[fullc::kData + 1];
  if (data.shape().ndim() != 2 && !param.flatten)
    data = data.ReshapeMKLDNN(Shape2(ishape.ProdShape(0, ishape.ndim()-1),
                                     ishape[ishape.ndim()-1]));
  else if (data.shape().ndim() != 2)
    data = data.ReshapeMKLDNN(Shape2(ishape[0],
                                     ishape.ProdShape(1, ishape.ndim())));
  NDArray out_grad = inputs[fullc::kOut];
  if (out_grad.shape().ndim() != 2 && !param.flatten)
    out_grad = out_grad.ReshapeMKLDNN(Shape2(oshape.ProdShape(0, oshape.ndim()-1),
                                             oshape[oshape.ndim()-1]));
  else if (out_grad.shape().ndim() != 2)
    out_grad = out_grad.ReshapeMKLDNN(Shape2(oshape[0],
                                             oshape.ProdShape(1, oshape.ndim())));

  mkldnn::inner_product_forward::primitive_desc ipFwd_pd = GetIPFwd(data, weight,
      param.no_bias ? nullptr : &in_grad[fullc::kBias], GetMemDesc(out_grad));

  CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
  if (req[fullc::kData]) {
    mkldnn::inner_product_backward_data::primitive_desc ipBwdData_pd = GetIpBwdData(
        data, weight, out_grad, ipFwd_pd);
    auto out_grad_mem = out_grad.GetMKLDNNDataReorder(
        ipBwdData_pd.diff_dst_primitive_desc());
    auto weight_mem = weight.GetMKLDNNDataReorder(ipBwdData_pd.weights_primitive_desc());
    auto in_grad_mem = CreateMKLDNNMem(in_grad[fullc::kData],
                                       ipBwdData_pd.diff_src_primitive_desc(),
                                       req[fullc::kData]);
    MKLDNNStream::Get()->RegisterPrim(mkldnn::inner_product_backward_data(
          ipBwdData_pd, *out_grad_mem, *weight_mem, *in_grad_mem.second));
    CommitOutput(in_grad[fullc::kData], in_grad_mem);
  }
  if (req[fullc::kWeight]) {
    mkldnn::inner_product_backward_weights::primitive_desc ipBwdWeights_pd
      = GetIPBwdWeights(data, weight, param.no_bias ? nullptr : &in_grad[fullc::kBias],
          out_grad, ipFwd_pd);
    auto out_grad_mem = out_grad.GetMKLDNNDataReorder(
        ipBwdWeights_pd.diff_dst_primitive_desc());
    auto data_mem = data.GetMKLDNNDataReorder(ipBwdWeights_pd.src_primitive_desc());
    auto in_grad_weight = CreateMKLDNNWeightGrad(in_grad[fullc::kWeight],
                                                 ipBwdWeights_pd.diff_weights_primitive_desc(),
                                                 req[fullc::kWeight]);
    mkldnn_output_t in_grad_bias;
    if (param.no_bias) {
      MKLDNNStream::Get()->RegisterPrim(mkldnn::inner_product_backward_weights(
            ipBwdWeights_pd, *data_mem, *out_grad_mem, *in_grad_weight.second));
    } else {
      in_grad_bias = CreateMKLDNNMem(in_grad[fullc::kBias],
                                     ipBwdWeights_pd.diff_bias_primitive_desc(),
                                     req[fullc::kBias]);
      MKLDNNStream::Get()->RegisterPrim(mkldnn::inner_product_backward_weights(
            ipBwdWeights_pd, *data_mem, *out_grad_mem, *in_grad_weight.second,
            *in_grad_bias.second));
    }
    CommitOutput(in_grad[fullc::kWeight], in_grad_weight);
    CommitOutput(in_grad[fullc::kBias], in_grad_bias);
  }
  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
