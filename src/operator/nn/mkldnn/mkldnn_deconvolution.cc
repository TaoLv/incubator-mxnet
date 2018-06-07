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
 * \file mkldnn_deconvolution.cc
 * \brief
 * \author Da Zheng, Rong Zhang
*/

#if MXNET_USE_MKLDNN == 1

#include "../deconvolution-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

bool SupportMKLDNNDeconv(const DeconvolutionParam& params, const NDArray &input) {
  if (params.kernel.ndim() != 2)
    return false;
  return input.dtype() == mshadow::kFloat32 && input.shape().ndim() == 4;
}

static inline mkldnn::memory::desc GetBiasDesc(mkldnn::memory::desc md) {
  mkldnn::memory::dims dims(1);
  // This is convolution on 4D data. The second dimension is the channel.
  dims[0] = md.data.dims[1];
  return mkldnn::memory::desc(dims,
      static_cast<mkldnn::memory::data_type>(md.data.data_type),
      mkldnn::memory::format::any);
}

static mkldnn::deconvolution_forward::primitive_desc CreateDeconvFwd(const DeconvolutionParam& param,
                                                                     const bool is_train, 
                                                                     const NDArray &data,
                                                                     const NDArray &weights,
                                                                     const NDArray *bias,
                                                                     const NDArray &output) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();

  CHECK_GE(param.stride.ndim(), 2U);
  CHECK_GE(param.pad.ndim(), 2U);
  CHECK_GE(param.dilate.ndim(), 2U);
  mkldnn::memory::dims strides{0, 0};
  strides[0] = param.stride[0];
  strides[1] = param.stride[1];
  mkldnn::memory::dims padding{0, 0};
  padding[0] = param.pad[0];
  padding[1] = param.pad[1];

  auto kind = mkldnn::prop_kind::forward_scoring;
  if (is_train) {
    kind = mkldnn::prop_kind::forward_training;
  }

  if (bias) {
    auto bias_md = GetMemDesc(*bias);
    auto deconv_desc = mkldnn::deconvolution_forward::desc(kind,
                                                           mkldnn::algorithm::deconvolution_direct,
                                                           data_md, weight_md, bias_md, out_md,
                                                           strides, padding, padding,
                                                           mkldnn::padding_kind::zero);
    return mkldnn::deconvolution_forward::primitive_desc(deconv_desc, engine);
  } else {
    auto deconv_desc = mkldnn::deconvolution_forward::desc(kind,
                                                           mkldnn::algorithm::deconvolution_direct,
                                                           data_md, weight_md, out_md,
                                                           strides, padding, padding,
                                                           mkldnn::padding_kind::zero);
    return mkldnn::deconvolution_forward::primitive_desc(deconv_desc, engine);
  }
}

static mkldnn::deconvolution_backward_data::primitive_desc CreateDeconvBwdData(
    const DeconvolutionParam &param,
    const NDArray &grad_out,
    const NDArray &weights,
    const NDArray &grad_data,
    const mkldnn::deconvolution_forward::primitive_desc &fwd_pd) {
  auto diff_dst_md = GetMemDesc(grad_out);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto diff_src_md = GetMemDesc(grad_data);
  auto engine = CpuEngine::Get()->get_engine();

  CHECK_GE(param.stride.ndim(), 2U);
  CHECK_GE(param.pad.ndim(), 2U);
  CHECK_GE(param.dilate.ndim(), 2U);
  mkldnn::memory::dims strides{0, 0};
  strides[0] = param.stride[0];
  strides[1] = param.stride[1];
  mkldnn::memory::dims padding{0, 0};
  padding[0] = param.pad[0];
  padding[1] = param.pad[1];

  auto bwd_desc = mkldnn::deconvolution_backward_data::desc(mkldnn::algorithm::deconvolution_direct,
                                                            diff_src_md, weight_md, diff_dst_md,
                                                            strides, padding, padding,
                                                            mkldnn::padding_kind::zero);
  return mkldnn::deconvolution_backward_data::primitive_desc(bwd_desc, engine, fwd_pd);
}

static mkldnn::deconvolution_backward_weights::primitive_desc CreateDeconvBwdWeights(
    const DeconvolutionParam& param,
    const NDArray &data,
    const NDArray &grad_out,
    const NDArray &grad_weight,
    const NDArray *grad_bias,
    const mkldnn::deconvolution_forward::primitive_desc &fwd_pd) {
  auto data_md = GetMemDesc(data);
  auto diff_dst_md = GetMemDesc(grad_out);
  auto diff_weight_md = GetWeightDesc(grad_weight, param.num_group);
  auto engine = CpuEngine::Get()->get_engine();

  CHECK_GE(param.stride.ndim(), 2U);
  CHECK_GE(param.pad.ndim(), 2U);
  CHECK_GE(param.dilate.ndim(), 2U);
  mkldnn::memory::dims strides{0, 0};
  strides[0] = param.stride[0];
  strides[1] = param.stride[1];
  mkldnn::memory::dims padding{0, 0};
  padding[0] = param.pad[0];
  padding[1] = param.pad[1];

  if (grad_bias) {
    auto diff_bias_md = GetMemDesc(*grad_bias);
    auto bwd_desc = mkldnn::deconvolution_backward_weights::desc(
        mkldnn::algorithm::deconvolution_direct,
        data_md, diff_weight_md, diff_bias_md, diff_dst_md, strides, padding, padding,
		mkldnn::padding_kind::zero);
    return mkldnn::deconvolution_backward_weights::primitive_desc(bwd_desc, engine, fwd_pd);
  } else {
    auto bwd_desc = mkldnn::deconvolution_backward_weights::desc(
        mkldnn::algorithm::deconvolution_direct,
        data_md, diff_weight_md, diff_dst_md, strides, padding, padding,
		mkldnn::padding_kind::zero);
    return mkldnn::deconvolution_backward_weights::primitive_desc(bwd_desc, engine, fwd_pd);
  }
}

class MKLDNNDeconvForward {
  std::shared_ptr<mkldnn::deconvolution_forward> fwd;
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> weights;
  std::shared_ptr<mkldnn::memory> bias;
  std::shared_ptr<mkldnn::memory> out;
  OutDataOp data_op;

 public:
  MKLDNNDeconvForward(const DeconvolutionParam &param,
                      const bool is_train,
                      const NDArray &data,
                      const NDArray &weights,
                      const NDArray *bias,
                      const NDArray &output);

  void SetDataHandle(const DeconvolutionParam &param,
                     const OpContext &ctx,
                     const NDArray &data,
                     const NDArray &weight,
                     const NDArray *bias,
                     const OpReqType &req,
                     const NDArray &output);

  void Execute(const NDArray &output);

 private:
  mkldnn::deconvolution_forward::primitive_desc fwd_pd;
};  // class MKLDNNDeconvForward

MKLDNNDeconvForward::MKLDNNDeconvForward(const DeconvolutionParam& param,
                                         const bool is_train,
                                         const NDArray &data,
                                         const NDArray &weights,
                                         const NDArray *bias,
                                         const NDArray &output) :
    fwd(nullptr), data(nullptr), weights(nullptr), bias(nullptr), out(nullptr),
    fwd_pd(CreateDeconvFwd(param, is_train, data, weights, bias, output)) {
}

void MKLDNNDeconvForward::SetDataHandle(const DeconvolutionParam &param,
                                        const OpContext &ctx,
                                        const NDArray &data,
                                        const NDArray &weight,
                                        const NDArray *bias,
                                        const OpReqType &req,
                                        const NDArray &output) {
  CHECK(data.IsMKLDNNData() && !(data.IsView()));
  auto data_mem = data.GetMKLDNNDataReorder(fwd_pd.src_primitive_desc());

  const mkldnn::memory *weight_mem;
  if (ctx.is_train) {
    // TODO(zhengda) kvstore doesn't handle MKLDNN correctly. Let's reorder it
    // to the default format for now.
    if (weight.IsMKLDNNData())
      // This asks the engine to reorder data after the weight array is used.
    const_cast<NDArray &>(weight).Reorder2DefaultAsync();
    weight_mem = GetWeights(weight, fwd_pd.weights_primitive_desc(), param.num_group);
  } else {
    // For inference, we want to reorder the weight array so we don't need to
    // reorder data every time.
    if (weight.IsDefaultData()) {
      weight_mem = GetWeights(weight, fwd_pd.weights_primitive_desc(), param.num_group);
      // We also need to modify the layout on the original weight array. The
      // data conversion happens after the weight array is used.
      const_cast<NDArray &>(weight).MKLDNNDataReorderAsync(fwd_pd.weights_primitive_desc());
    } else {
      weight_mem = weight.GetMKLDNNData();
      CHECK(weight_mem->get_primitive_desc() == fwd_pd.weights_primitive_desc());
    }
  }

  auto out_mem = CreateMKLDNNMem(output, fwd_pd.dst_primitive_desc(), req);
  this->data->set_data_handle(data_mem->get_data_handle());
  this->weights->set_data_handle(weight_mem->get_data_handle());
  this->out->set_data_handle(out_mem.second->get_data_handle());
  this->data_op = out_mem.first;
}

void MKLDNNDeconvForward::Execute(const NDArray &output) {
  MKLDNNStream::Get()->RegisterPrim(*fwd);
  CommitOutput(output, mkldnn_output_t(this->data_op, this->out.get()));
  MKLDNNStream::Get()->Submit();
}

static void MKLDNNDeconvFwdBiasPostProcess(const DeconvolutionParam& param,
                                           const OpContext &ctx,
                                           const NDArray &bias,
                                           const NDArray &output) {
  // add bias, broadcast bias to dim 1: channel
  if (!param.no_bias) {
    // MKLDNN only supports float right now.
    typedef float DType;
    Stream<cpu> *s = ctx.get_stream<cpu>();
    Tensor<cpu, 1, DType> bias_cpu = bias.data().get<cpu, 1, DType>(s);
    // If the output data is stored in a special MKLDNN format, data()
    // automatically converts its format to the default format.
    // Unfortunately, MKLDNN doesn't support broadcast.
    Tensor<cpu, 4, DType> out_cpu = output.data().get<cpu, 4, DType>(s);
    out_cpu += mshadow::expr::broadcast<1>(bias_cpu, out_cpu.shape_);
  }
}

static inline MKLDNNDeconvForward &GetDeconvFwd(const DeconvolutionParam &param,
                                                const bool is_train,
                                                const NDArray &data,
                                                const NDArray &weight,
                                                const NDArray *bias,
                                                const NDArray &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DeconvSignature, MKLDNNDeconvForward, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DeconvSignature, MKLDNNDeconvForward, OpHash> fwds;
#endif
  DeconvSignature key(param);
  // Here we can sign the conv op with NDArray because conv primitive will
  // decide the right layout for it, so we only need to get the shape and the
  // data type of the arrays.
  key.AddSign(is_train);
  key.AddSign(data);
  key.AddSign(weight);
  key.AddSign(output);
  if (bias)
    key.AddSign(*bias);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNDeconvForward fwd(param, is_train, data, weight, bias, output);
    auto ins_ret = fwds.insert(std::pair<DeconvSignature, MKLDNNDeconvForward>(key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

void MKLDNNDeconvolutionForward(const nnvm::NodeAttrs &attrs,
                                const OpContext &ctx,
                                const std::vector<NDArray> &inputs,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[deconv::kTempSpace]);
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);

  if (param.no_bias) {
    CHECK_EQ(inputs.size(), 2U);
  } else {
    CHECK_EQ(inputs.size(), 3U);
  }

  CHECK(1U == outputs.size());

  auto data = inputs[deconv::kData];
  auto weight = inputs[deconv::kWeight];
  auto out = outputs[deconv::kOut];

  if (data.IsMKLDNNData() && data.IsView()) {
    data = inputs[deconv::kData].Reorder2Default();
  }

  if (weight.IsMKLDNNData() && weight.IsView()) {
    weight = inputs[deconv::kWeight].Reorder2Default();
  }

  if (ctx.is_train) CHECK(!weight.IsMKLDNNData());

  MKLDNNDeconvForward &deconvFwd = GetDeconvFwd(param, ctx.is_train, data, weight,
                                                param.no_bias ? nullptr : &(inputs[deconv::kBias]),
                                                out);
  deconvFwd.SetDataHandle(param, ctx, data, weight,
                          param.no_bias ? nullptr : &(inputs[deconv::kBias]),
                          req[deconv::kOut], out);
  deconvFwd.Execute(out);
  // MKLDNNDeconvFwdBiasPostProcess(param, ctx, data, weight, out);
}

void MKLDNNDeconvolutionBackward(const nnvm::NodeAttrs &attrs,
                                 const OpContext &ctx,
                                 const std::vector<NDArray> &inputs,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<NDArray> &outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[deconv::kTempSpace]);
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  CHECK_NE(req[deconv::kWeight], kWriteInplace) << "cannot write weight inplace";

  auto data = inputs[deconv::kData + 1];
  auto weight = inputs[deconv::kWeight + 1];
  auto grad_out = inputs[deconv::kOut];
  auto grad_in = outputs[deconv::kData];
  auto grad_weight = outputs[deconv::kWeight];

  const NDArray *bias = param.no_bias ? nullptr : &(inputs[deconv::kBias + 1]);
  auto fwd_pd = CreateDeconvFwd(param, true, data, weight, bias, grad_out);
  auto bwd_data_pd = CreateDeconvBwdData(param, grad_out, weight, grad_in, fwd_pd);
  auto grad_out_mem = grad_out.GetMKLDNNDataReorder(bwd_data_pd.diff_dst_primitive_desc());

  if (req[deconv::kData] != kNullOp) {
    auto weight_mem = GetWeights(weight, bwd_data_pd.weights_primitive_desc(), param.num_group);
    auto grad_in_mem = CreateMKLDNNMem(grad_in,
                                       bwd_data_pd.diff_src_primitive_desc(),
                                       req[deconv::kData]);

    auto deconv = mkldnn::deconvolution_backward_data(bwd_data_pd,
                                                      *grad_out_mem,
                                                      *weight_mem,
                                                      *grad_in_mem.second);
    MKLDNNStream::Get()->RegisterPrim(deconv);
    CommitOutput(grad_in, grad_in_mem);
  }

  if (req[deconv::kWeight] != kNullOp) {
    if (param.no_bias || req[deconv::kBias] == kNullOp) {
      auto bwd_weight_pd = CreateDeconvBwdWeights(param, data, grad_out, grad_weight, nullptr, fwd_pd);
      auto data_mem = data.GetMKLDNNDataReorder(bwd_weight_pd.src_primitive_desc());
      auto grad_weight_mem = CreateMKLDNNWeightGrad(grad_weight,
                                                    bwd_weight_pd.diff_weights_primitive_desc(),
                                                    req[deconv::kWeight]);

      auto deconv = mkldnn::deconvolution_backward_weights(bwd_weight_pd,
                                                           *data_mem,
                                                           *grad_out_mem,
                                                           *grad_weight_mem.second);
      MKLDNNStream::Get()->RegisterPrim(deconv);
      CommitOutput(grad_weight, grad_weight_mem);
    } else {
      auto grad_bias = outputs[deconv::kBias];
      auto bwd_weight_pd = CreateDeconvBwdWeights(param, data, grad_out, grad_weight, &grad_bias, fwd_pd);
      auto data_mem = data.GetMKLDNNDataReorder(bwd_weight_pd.src_primitive_desc());
      auto grad_weight_mem = CreateMKLDNNWeightGrad(grad_weight,
                                                    bwd_weight_pd.diff_weights_primitive_desc(),
                                                    req[deconv::kWeight]);
      auto grad_bias_mem = CreateMKLDNNMem(grad_bias,
                                           bwd_weight_pd.diff_bias_primitive_desc(),
                                           req[deconv::kBias]);

      auto deconv = mkldnn::deconvolution_backward_weights(bwd_weight_pd,
                                                           *data_mem,
                                                           *grad_out_mem,
                                                           *grad_weight_mem.second,
                                                           *grad_bias_mem.second);
      MKLDNNStream::Get()->RegisterPrim(deconv);
      CommitOutput(grad_weight, grad_weight_mem);
      CommitOutput(grad_bias, grad_bias_mem);
    }
  }
  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
