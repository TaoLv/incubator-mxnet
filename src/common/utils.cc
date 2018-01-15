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
 * \file utils.cc
 * \brief cpu implementation of util functions
 */

#include "./utils.h"
#include "../operator/tensor/cast_storage-inl.h"

namespace mxnet {
namespace common {

template<>
void CheckFormatWrapper<cpu>(const RunContext &rctx, const NDArray &input,
                             const TBlob &err_cpu, const bool full_check) {
  CheckFormatImpl<cpu>(rctx, input, err_cpu, full_check);
}

template<>
void CastStorageDispatch<cpu>(const OpContext& ctx,
                              const NDArray& input,
                              const NDArray& output) {
  mxnet::op::CastStorageComputeImpl<cpu>(ctx, input, output);
}

std::string stype_string(const int x) {
  switch (x) {
    case kDefaultStorage:
      return "default";
    case kCSRStorage:
      return "csr";
    case kRowSparseStorage:
      return "row_sparse";
#if MXNET_USE_MKLDNN == 1
    case kMKLDNNStorage:
      return "mkldnn";
#endif
  }
  return "unknown";
}

}  // namespace common
}  // namespace mxnet
