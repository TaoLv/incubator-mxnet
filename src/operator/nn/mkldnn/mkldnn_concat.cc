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
 * \file mkldnn_concat.cc
 * \brief
 * \author Wenting Jiang (wenting.jiang@intel.com)
*/
#if MXNET_USE_MKLDNN == 1
#include "../concat-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

class MKLDNNConcatFwd {
  std::shared_ptr<mkldnn::concat> fwd;
  std::vector<std::shared_ptr<mkldnn::memory>> data;
  std::shared_ptr<mkldnn::memory> out;

 public:
  MKLDNNConcatFwd(const ConcatParam& param,
                  bool is_train,
                  const std::vector<NDArray> &in_data,
                  const NDArray &out_data):
                  fwd(nullptr), data(param.num_args), out(nullptr) {
                  _Init(param, is_train, in_data, out_data);
  }
  ~MKLDNNConcatFwd() {}
  void SetDataHandle(const std::vector<NDArray> &in_data,
                     const NDArray &out_data) {
    int num_in_data = in_data.size();
    for (int i =0; i < num_in_data; i++) {
      this->data[i]->set_data_handle(in_data[i].GetMKLDNNData()->get_data_handle());
    }
    auto out_mem = const_cast<NDArray&>(out_data).CreateMKLDNNData(this->out->get_primitive_desc());
    this->out->set_data_handle(out_mem->get_data_handle());
  }
  void Execute() {
    MKLDNNStream *stream = MKLDNNStream::Get();
    stream->RegisterPrim(*fwd);
    stream->Submit();
  }

 private:
  void _Init(const ConcatParam& param,
             bool is_train,
             const std::vector<NDArray> &in_data,
             const NDArray &out_data) {
    // mkldnn::concat::primitive_desc
    int num_in_data = param.num_args;
    int concat_dim = param.dim;
    std::vector<mkldnn::memory::primitive_desc> data_md;
    for (int i =0; i < num_in_data; i++) {
        auto tmp_pd = in_data[i].GetMKLDNNData()->get_primitive_desc();
        this->data[i].reset(new mkldnn::memory(tmp_pd));
        data_md.push_back(tmp_pd);
    }
    mkldnn::concat::primitive_desc fwd_pd(concat_dim, data_md);
    // mkldnn::memory
    this->out.reset(new mkldnn::memory(fwd_pd.dst_primitive_desc()));
    // mkldnn::concat
    std::vector<mkldnn::primitive::at> data_mem;
    for (int i =0; i < num_in_data; i++) {
      data_mem.push_back(*this->data[i]);
    }
    this->fwd = std::shared_ptr<mkldnn::concat>(
            new mkldnn::concat(fwd_pd, data_mem, *this->out));
    }
};

typedef MKLDNNParamOpSign<ConcatParam> MKLDNNConcatSignature;

static MKLDNNConcatFwd &GetConcatFwd(const ConcatParam& param,
                                     const OpContext &ctx,
                                     const std::vector<NDArray> &in_data,
                                     const NDArray &out_data) {
  static thread_local std::unordered_map<MKLDNNConcatSignature, MKLDNNConcatFwd, MKLDNNOpHash> fwds;
  MKLDNNConcatSignature key(param);
  key.AddSign(ctx.is_train);
  key.AddSign(param.num_args);
  key.AddSign(param.dim);
  key.AddSign(in_data);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNConcatFwd fwd(param, ctx.is_train, in_data, out_data);
    auto ins_ret = fwds.insert(std::pair<MKLDNNConcatSignature, MKLDNNConcatFwd>(
            key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

void MKLDNNConcatCompute(const nnvm::NodeAttrs& attrs,
                         const OpContext &ctx,
                         const std::vector<NDArray> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<NDArray> &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[concat_enum::kTempSpace]);
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  MKLDNNConcatFwd &fwd = GetConcatFwd(param, ctx, in_data, out_data[concat_enum::kOut]);
  fwd.SetDataHandle(in_data, out_data[concat_enum::kOut]);
  fwd.Execute();
}

void MKLDNNConcatGradCompute(const nnvm::NodeAttrs& attrs,
                             const OpContext &ctx,
                             const std::vector<NDArray>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[concat_enum::kTempSpace]);
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  int num_in_data = param.num_args;
  int axis_ = param.dim;
  auto engine = CpuEngine::Get()->get_engine();
  auto gz_mem = inputs[0].GetMKLDNNData();
  mkldnn::memory::primitive_desc gz_pd = gz_mem->get_primitive_desc();
  /* init the offset */
  mkldnn::memory::dims offsets = {0, 0, 0, 0};
  for (int i = 0; i < num_in_data; i++) {
    mkldnn::memory::dims diff_src_tz
        = {static_cast<int>(inputs[i+1].shape()[0]),
          static_cast<int>(inputs[i+1].shape()[1]),
          static_cast<int>(inputs[i+1].shape()[2]),
          static_cast<int>(inputs[i+1].shape()[3])};
    auto diff_src_mpd = inputs[i+1].GetMKLDNNData()->get_primitive_desc();
    auto gradi_mem_ = CreateMKLDNNMem(outputs[i], diff_src_mpd, req[i]);
    // create view from gy to gxs[i]
    std::shared_ptr<mkldnn::view::primitive_desc> view_pd;
    view_pd.reset(new mkldnn::view::primitive_desc(gz_pd, diff_src_tz, offsets));
    // create reorder primitive from gy to gxs[i]
    mkldnn::reorder::primitive_desc reorder_pd(
        view_pd.get()->dst_primitive_desc(), diff_src_mpd);
    offsets[axis_] += diff_src_tz[axis_];
    MKLDNNStream::Get()->RegisterPrim(mkldnn::reorder(
            reorder_pd, *gz_mem, *gradi_mem_.second));
    CommitOutput(outputs[i], gradi_mem_);
  }
  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif
