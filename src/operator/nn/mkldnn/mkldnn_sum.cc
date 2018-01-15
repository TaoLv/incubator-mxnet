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
 * \file mkldnn_sum.cc
 * \brief
 * \author Da Zheng
*/
#include <iostream>

#include "../../tensor/elemwise_sum.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

void Sum(const mkldnn::memory &arr1, const mkldnn::memory &arr2,
         const mkldnn::memory &out) {
  std::vector<mkldnn::memory::primitive_desc> input_pds(2);
  std::vector<float> scales(2);
  std::vector<mkldnn::primitive::at> inputs;
  input_pds[0] = arr1.get_primitive_desc();
  input_pds[1] = arr2.get_primitive_desc();
  CHECK(input_pds[0] == input_pds[1]);
  scales[0] = 1;
  scales[1] = 1;
  inputs.push_back(arr1);
  inputs.push_back(arr2);
  // TODO(zhengda) I need to reorder memory here.
  mkldnn::sum::primitive_desc sum_pd(scales, input_pds);
  MKLDNNStream::Get()->RegisterPrim(mkldnn::sum(sum_pd, inputs, out));
}

typedef MKLDNNParamOpSign<ElementWiseSumParam> MKLDNNSumSignature;

class MKLDNNSumFwd {
 public:
  explicit MKLDNNSumFwd(const std::vector<NDArray> &inputs) {
    _Init(inputs);
  }
  ~MKLDNNSumFwd() {}
  void SetDataHandle(const std::vector<NDArray> &inputs,
                     const NDArray &output,
                     const OpReqType &req);
  void Execute(const NDArray &output, const OpReqType &req);

 private:
  void _Init(const std::vector<NDArray> &inputs);

 private:
  std::shared_ptr<mkldnn::sum> fwd;
  std::vector<std::shared_ptr<mkldnn::memory>> in_data;
  std::shared_ptr<mkldnn::memory> out;
  std::shared_ptr<mkldnn::sum::primitive_desc> fwd_pd;
};

void MKLDNNSumFwd::_Init(const std::vector<NDArray> &inputs) {
  std::vector<mkldnn::memory> in_mems;
  for (size_t i = 0; i < inputs.size(); i++) {
    in_mems.push_back(*inputs[i].GetMKLDNNData());
  }
  this->in_data.resize(inputs.size());
  for (size_t i = 0; i < in_mems.size(); i++) {
    this->in_data[i].reset(new mkldnn::memory(in_mems[i].get_primitive_desc(),
                                              in_mems[i].get_data_handle()));
  }

  std::vector<float> scales(inputs.size());
  std::vector<mkldnn::memory::primitive_desc> in_pds(inputs.size());
  for (size_t i = 0; i < in_mems.size(); i++) {
    in_pds[i] = in_mems[i].get_primitive_desc();
    scales[i] = 1;
  }
  this->fwd_pd.reset(new mkldnn::sum::primitive_desc(scales, in_pds));
  this->out.reset(new mkldnn::memory(this->fwd_pd->dst_primitive_desc()));

  std::vector<mkldnn::primitive::at> in_prims;
  for (size_t i = 0; i < this->in_data.size(); i++) {
    in_prims.push_back(*(this->in_data[i]));
  }

  this->fwd.reset(new mkldnn::sum(*(this->fwd_pd), in_prims,
                                  *(this->out)));
}

void MKLDNNSumFwd::SetDataHandle(const std::vector<NDArray> &inputs,
                                 const NDArray &output,
                                 const OpReqType &req) {
  std::vector<mkldnn::memory> in_mems;
  for (size_t i = 0; i < inputs.size(); i++) {
    in_mems.push_back(*inputs[i].GetMKLDNNData());
    this->in_data[i]->set_data_handle(in_mems[i].get_data_handle());
  }

  auto out_mem = CreateMKLDNNMem(output, this->fwd_pd->dst_primitive_desc(),
                                 req);
  this->out->set_data_handle(out_mem.second->get_data_handle());
}

void MKLDNNSumFwd::Execute(const NDArray &output, const OpReqType &req) {
  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrim(*(this->fwd));
  auto out_mem = CreateMKLDNNMem(output, this->fwd_pd->dst_primitive_desc(),
                                 req);
  CommitOutput(output, out_mem);
  stream->Submit();
}

static MKLDNNSumFwd
&GetSumFwd(const ElementWiseSumParam& param, const OpContext &ctx,
           const std::vector<NDArray> &inputs) {
  static thread_local std::unordered_map<MKLDNNSumSignature,
                                         MKLDNNSumFwd, MKLDNNOpHash> sum_fwds;
  MKLDNNSumSignature key(param);
  key.AddSign(ctx.is_train);
  key.AddSign(param.num_args);
  key.AddSign(inputs);

  auto it = sum_fwds.find(key);
  if (it == sum_fwds.end()) {
    MKLDNNSumFwd fwd(inputs);
    auto ins_ret = sum_fwds.insert(std::pair<MKLDNNSumSignature, MKLDNNSumFwd>(
            key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

void MKLDNNSumCompute(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                      const std::vector<NDArray> &inputs, const OpReqType &req,
                      const NDArray &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[0]);
  ElementWiseSumParam param;

  if (!attrs.parsed.empty()) {
    param = nnvm::get<ElementWiseSumParam>(attrs.parsed);
  } else {
    memset(&param, 0, sizeof(param));
  }

  MKLDNNSumFwd &fwd = GetSumFwd(param, ctx, inputs);
  fwd.SetDataHandle(inputs, out_data, req);
  fwd.Execute(out_data, req);
}

}  // namespace op
}  // namespace mxnet
#endif
