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

#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_CONV_BN_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_CONV_BN_H_

#include <string>
#include <vector>
#include "../default_subgraph_op.h"
#include "../../nn/convolution-inl.h"

namespace mxnet {
namespace op {
namespace sg {

class SGConvBNSelector : public SubgraphSelector {
 public:
  enum SelectStatus {
    kFail = 0,
    kInProgress,
    kSuccess,
  };

  typedef bool (*ExamFunction)(const nnvm::Node &n);

 private:
  std::shared_ptr<const std::vector<std::string>> op_names_;
  std::shared_ptr<const std::vector<ExamFunction>> op_exam_funcs_;
  size_t index_;
  SelectStatus status_;

 public:
  explicit SGConvBNSelector(std::shared_ptr<const std::vector<std::string>> op_names)
      : op_names_(op_names), op_exam_funcs_(nullptr), index_(0), status_(kFail) {}

  void SetOpExamFuncs(std::shared_ptr<const std::vector<ExamFunction> > &funcs) {
    CHECK_EQ(funcs->size(), op_names_->size());
    op_exam_funcs_ = funcs;
  }

  bool Select(const nnvm::Node &n) override {
    bool match = !n.is_variable() && ((*op_names_)[0] == n.op()->name);
    if (match) {
      if (op_exam_funcs_ && (*op_exam_funcs_)[0]) {
        match = (*op_exam_funcs_)[0](n);
      }
      if (match) {
        index_ = 1;
        status_ = kInProgress;
        return true;
      }
    }
    return false;
  }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return false;
  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    // If (status_ == kSuccess), but n isn't the last node in op_names,
    // then we encoutered a internal branch, we shouldn't do fusion.
    if ((status_ == kSuccess) && (op_names_->back() != n.op()->name)) {
      status_ = kFail;
    }
    if (status_ == kSuccess || status_ == kFail) return false;
    bool match = (!new_node.is_variable()) &&
                 ((*op_names_)[index_] == new_node.op()->name);
    if (match) {
      CHECK_EQ((*op_names_)[index_ - 1], n.op()->name);
      if (op_exam_funcs_ && (*op_exam_funcs_)[index_]) {
        match = (*op_exam_funcs_)[index_](new_node);
      }
    }
    if (match) {
      index_ += 1;
      if (index_ == op_names_->size()) status_ = kSuccess;
      return true;
    }
    status_ = kFail;
    return false;
  }

  std::vector<nnvm::Node *> Filter(nnvm::Graph *g,
                                   const std::vector<nnvm::Node *> &candidates) override {
    if (status_ == kSuccess) {
      return candidates;
    } else {
      return std::vector<nnvm::Node *>(0);
    }
  }
};

class SGConvBNProperty : public SubgraphProperty {
 public:
  SGConvBNProperty()
      : op_names_(std::make_shared<const std::vector<std::string> >(std::vector<std::string>{
          "Convolution", "BatchNorm"})) {}

  nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const std::vector<SimpleNode *> &subgraph_nodes,
                                   const int subgraph_id = 0) const override {
    nnvm::NodePtr n = nnvm::Node::Create();

    // n->attrs.dict = subgraph_nodes[0]->node->attrs.dict;
    // ConvolutionParam &param = nnvm::get<ConvolutionParam>(n->attrs.parsed);
    n->attrs.name = "_conv_bn_subgraph_op" + std::to_string(subgraph_id);
    n->attrs.op = Op::Get("_conv_bn_subgraph_op");
    CHECK(n->attrs.op);
    n->attrs.parsed = sym;
    // n->attrs.op->attr_parser(&(n->attrs));
    return n;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SGConvBNSelector>(op_names_);
    // auto funcs = std::make_shared<const std::vector<SGConvBNSelector::ExamFunction> >(
    //     std::vector<SGConvBNSelector::ExamFunction>{nullptr, nullptr});
    // selector->SetOpExamFuncs(funcs);
    return selector;
  }
/*
  void ConnectSubgraphOutput(const nnvm::NodePtr n,
                             const std::vector<nnvm::NodeEntry> *output_entries) const override {
    for (size_t i = 0; i < output_entries->size(); ++i) {
      uint32_t index = (*output_entries)[i].index;
      (*output_entries)[i] = nnvm::NodeEntry{n, index, 0};
    }
  }
*/
 private:
  std::shared_ptr<const std::vector<std::string> > op_names_;
};

}  // namespace sg
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_CONV_BN_H_
