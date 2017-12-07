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
 * \file mkldnn_batch_norm.cc
 * \brief
 * \author Tao Lv
*/

//#include <mxnet/base.h>
#include "../batch_norm-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

typedef mkldnn::batch_normalization_forward::primitive_desc     t_bn_f_pdesc;
typedef mkldnn::batch_normalization_forward::desc               t_bn_f_desc;
typedef mkldnn::batch_normalization_backward::primitive_desc    t_bn_b_pdesc;
typedef mkldnn::batch_normalization_backward::desc              t_bn_b_desc;

using mkldnn::use_global_stats;
using mkldnn::use_scale_shift;
using mkldnn::forward_training;
using mkldnn::forward_inference;

/* Note:
 *
 * flags for batch normalization
 * use_global_stats: has input mean and variance data
 * use_scale_shift: has input gamma and beta data (weight)
 *
 * For input param:
 * use_global_stats: default is false
 * fix_gamma: default is true, then all elements of gamma will be treated as 1.0f
 *            and it's gradient will be set to 0.0f
 * output_mean_var: default is false
 *
 * Here mxnet always has 5 inputs and 3 outputs for bn forward computation
 * if use_global_stats if flase, then it's no need to process inMean and inVar
 * alway output mean and var, but mxnet will hide them if output_mean_var is false
 *
 */


static unsigned _GetFlags(const std::vector<NDArray> &in_data,
                          const std::vector<NDArray> &aux_states, 
                          const BatchNormParam &param, bool is_train) {
    unsigned flags = 0U;
    if (in_data.size() == 3U) {
        flags |= use_scale_shift;
    }

    // aux_states[0]: inMean
    // aux_states[1]: inVariance
    if (aux_states.size() == 2U && param.use_global_stats) {
        flags |= use_global_stats;
    }

    return flags;
}

template <typename DType>
static t_bn_f_pdesc _GetFwd(const NDArray &data, bool is_train,
                            DType eps, unsigned flags) {
    auto data_mem   = data.GetMKLDNNData();
    auto data_mpd   = data_mem->get_primitive_desc();
    auto data_md    = data_mpd.desc();
    auto engine     = CpuEngine::Instance().get_engine();

    if (is_train) {
        t_bn_f_desc bnFwd_desc(forward_training, data_md, eps, flags);
        return t_bn_f_pdesc(bnFwd_desc, engine);
    } else {
        t_bn_f_desc bnFwd_desc(forward_inference, data_md, eps, flags);
        return t_bn_f_pdesc(bnFwd_desc, engine);
    }
}

template <typename DType>
static t_bn_b_pdesc _GetBwd(const NDArray &data, const NDArray &diff_data,
                            DType eps, unsigned flags) {
    auto data_mem   = data.GetMKLDNNData();
    auto data_mpd   = data_mem->get_primitive_desc();
    auto data_md    = data_mpd.desc();
    auto diff_mem   = diff_data.GetMKLDNNData();
    auto diff_mpd   = diff_mem->get_primitive_desc();
    auto diff_md    = diff_mpd.desc();
    auto engine     = CpuEngine::Instance().get_engine();

    t_bn_b_desc  bnBwd_desc(mkldnn::prop_kind::backward, diff_md, data_md, eps, flags);
    return t_bn_b_pdesc(bnBwd_desc, engine, _GetFwd(data, true, eps, flags));
}

template <typename DType>
void MKLDNNBatchNorm_Forward(const OpContext &ctx, const BatchNormParam &param,
                             const std::vector<NDArray> &in_data,
                             const std::vector<OpReqType> &req,
                             const std::vector<NDArray> &out_data,
                             const std::vector<NDArray> &aux_states) {
    unsigned flags      = _GetFlags(in_data, aux_states, param, ctx.is_train);
    const NDArray &data = in_data[batchnorm::kData];
    CHECK_EQ (data.storage_type(), mxnet::kMKLDNNStorage);
    
    auto data_mem       = data.GetMKLDNNData();
    auto fwd_pd         = _GetFwd(data, ctx.is_train, (DType) param.eps, flags);
    const NDArray &out  = out_data[batchnorm::kOut];

    // for output memory
    std::shared_ptr<const mkldnn::memory> out_mem =
            const_cast<NDArray &>(out).CreateMKLDNNData(fwd_pd.dst_primitive_desc());

    // mxnet will always use scale shift.
    // But if fix_gamma is true, then all scale elements will be set to 1.0f
    if (flags & use_scale_shift) {
        const NDArray &gamma    = in_data[batchnorm::kGamma];
        const NDArray &beta     = in_data[batchnorm::kBeta];

        CHECK_EQ (gamma.storage_type(), mxnet::kDefaultStorage);
        CHECK_EQ (beta.storage_type(), mxnet::kDefaultStorage);

        // TODO: how to reuse this memory?
        std::shared_ptr<const mkldnn::memory> weight_mem(new mkldnn::memory(fwd_pd.weights_primitive_desc()));
        DType* weight_buf = reinterpret_cast<DType *>(weight_mem->get_data_handle());

        nnvm::dim_t channels_ = data.shape()[1];
        for (size_t i = 0; i < channels_; i++) {
            if (!param.fix_gamma)
                weight_buf[i] = (gamma.data().dptr<DType>())[i];   // weight
            else
                weight_buf[i] = (DType)1.0f;
        }

        for (size_t i = 0; i < channels_; i++) {
            weight_buf[channels_ + i] = (beta.data().dptr<DType>())[i];  // bias
        }

        if (!ctx.is_train && !(flags & use_global_stats)) { // inference & no mean and var inputs
            MKLDNNStream::Instance().RegisterPrim(
                    mkldnn::batch_normalization_forward(fwd_pd, *data_mem, *weight_mem, *out_mem));

        } else if (flags & use_global_stats) {
            const NDArray &inMean   = aux_states[batchnorm::kMovingMean];
            const NDArray &inVar    = aux_states[batchnorm::kMovingVar];
            auto mean_mem           = inMean.GetMKLDNNData();
            auto var_mem            = inVar.GetMKLDNNData();
            MKLDNNStream::Instance().RegisterPrim(
                    mkldnn::batch_normalization_forward(fwd_pd, *data_mem, *mean_mem, *var_mem, *weight_mem, *out_mem));

        } else if (ctx.is_train && (out_data.size() == 3U)) { // training
            std::cout << "bn forward here.." << std::endl;
            const NDArray &outMean  = out_data[batchnorm::kMean];
            const NDArray &outVar   = out_data[batchnorm::kVar];
            CHECK_EQ (outMean.storage_type(), mxnet::kMKLDNNStorage);
            CHECK_EQ (outVar.storage_type(), mxnet::kMKLDNNStorage);

            auto mean_mem           = const_cast<NDArray &>(outMean).CreateMKLDNNData(fwd_pd.mean_primitive_desc());
            auto var_mem            = const_cast<NDArray &>(outVar).CreateMKLDNNData(fwd_pd.variance_primitive_desc());

            MKLDNNStream::Instance().RegisterPrim(
                    mkldnn::batch_normalization_forward(fwd_pd, *data_mem, *weight_mem, *out_mem, *mean_mem, *var_mem));

        } else {
            LOG(FATAL) << "Unknown flags for MKLDNN Batch Normalization.";
        }
        MKLDNNStream::Instance().Submit();

    } else { // no input gamma and beta
        LOG(FATAL) << "MKLDNN batch normalization: should not reach here ...";
    }
    return;
}

// in_data: input, gamma, beta
// out_data: output, outmean, outvar
// out_grad: grad_output, grad_outmean, grad_outvar
// in_grad: grad_input, grad_gamma, grad_beta
// aux_states: inmean, invar
template <typename DType>
void MKLDNNBatchNorm_Backward(const OpContext &ctx, const BatchNormParam &param,
                              const std::vector<NDArray>    &out_grad,
                              const std::vector<NDArray>    &in_data,
                              const std::vector<NDArray>    &out_data,
                              const std::vector<OpReqType>  &req,
                              const std::vector<NDArray>    &in_grad,
                              const std::vector<NDArray>    &aux_states) {

    CHECK_EQ(out_grad.size(), param.output_mean_var ? 3U : 1U);
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(out_data.size(), 3U);
    CHECK_EQ(in_grad.size(), 3U);
    unsigned flags = _GetFlags(in_data, aux_states, param, ctx.is_train);

    const NDArray &data         = in_data[batchnorm::kData];
    const NDArray &diff         = out_grad[batchnorm::kOut];
    const NDArray &gradIn       = in_grad[batchnorm::kData];
    const NDArray &moving_mean  = aux_states[batchnorm::kMovingMean];
    const NDArray &moving_var   = aux_states[batchnorm::kMovingVar];
    const NDArray &out_mean     = out_data[batchnorm::kMean];
    const NDArray &out_var      = out_data[batchnorm::kVar];

    CHECK_EQ (out_mean.storage_type(), mxnet::kMKLDNNStorage);
    CHECK_EQ (out_var.storage_type(), mxnet::kMKLDNNStorage);
    CHECK_EQ (moving_mean.storage_type(), mxnet::kDefaultStorage);
    CHECK_EQ (moving_var.storage_type(), mxnet::kDefaultStorage);

    auto data_mem  = data.GetMKLDNNData();
    auto diff_mem  = diff.GetMKLDNNData();
    auto omean_mem = out_mean.GetMKLDNNData();
    auto ovar_mem  = out_var.GetMKLDNNData();

    auto fwd_pd = _GetFwd(data, ctx.is_train, param.eps, flags);
    auto bwd_pd = _GetBwd(data, diff, param.eps, flags);
    std::shared_ptr<const mkldnn::memory> gradi_mem =
            const_cast<NDArray &>(gradIn).CreateMKLDNNData(data_mem->get_primitive_desc());
    // mxnet will always use scale shift.
    // But if fix_gamma is true, then all scale elements will be set to 1.0f and
    // gradient of it will be set to 0.0f.
    if (flags & use_scale_shift) {
        const NDArray &gamma    = in_data[batchnorm::kGamma];
        const NDArray &beta     = in_data[batchnorm::kBeta];
        // TODO: how to reuse this memory?
        std::shared_ptr<const mkldnn::memory> weight_mem(new mkldnn::memory(bwd_pd.weights_primitive_desc()));
        DType* weight_buf = reinterpret_cast<DType *>(weight_mem->get_data_handle());

        nnvm::dim_t channels_ = data.shape()[1];
        for (size_t i = 0; i < channels_; i++) {
            if (!param.fix_gamma)
                weight_buf[i] = (gamma.data().dptr<DType>())[i];   // weight
            else
                weight_buf[i] = (DType)1.0f;
        }

        for (size_t i = 0; i < channels_; i++) {
            weight_buf[channels_ + i] = (beta.data().dptr<DType>())[i];  // bias
        }

        std::shared_ptr<const mkldnn::memory> gradw_mem(new mkldnn::memory(bwd_pd.diff_weights_primitive_desc()));
        // training but no input mean and variance
        if (ctx.is_train && !param.use_global_stats) {
            std::cout << "bn backward here .." << std::endl;
            DType* imean_ptr = reinterpret_cast<DType *>(moving_mean.data().dptr<DType>());
            DType* ivar_ptr  = reinterpret_cast<DType *>(moving_var.data().dptr<DType>());

            DType* omean_ptr = reinterpret_cast<DType *>(omean_mem->get_data_handle());
            DType* ovar_ptr  = reinterpret_cast<DType *>(ovar_mem->get_data_handle());

            DType minus_mom = (1.0f - param.momentum);
            for (size_t i = 0; i < channels_; i++) {
                imean_ptr[i] = imean_ptr[i] * param.momentum + omean_ptr[i] * minus_mom;
            }
            for (size_t i = 0; i < channels_; i++) {
                ivar_ptr[i] = ivar_ptr[i] * param.momentum + ovar_ptr[i] * minus_mom;
            }

            MKLDNNStream::Instance().RegisterPrim(
                mkldnn::batch_normalization_backward(bwd_pd, *data_mem, *omean_mem,
                    *ovar_mem, *diff_mem, *weight_mem, *gradi_mem, *gradw_mem));
            MKLDNNStream::Instance().Submit();
 
        } else {
            std::shared_ptr<const mkldnn::memory> imean_mem(new mkldnn::memory(bwd_pd.mean_primitive_desc(),
                                                            moving_mean.data().dptr<DType>()));
            std::shared_ptr<const mkldnn::memory> ivar_mem(new mkldnn::memory(bwd_pd.variance_primitive_desc(),
                                                            moving_var.data().dptr<DType>()));
            MKLDNNStream::Instance().RegisterPrim(
                mkldnn::batch_normalization_backward(bwd_pd, *data_mem, *imean_mem,
                    *ivar_mem, *diff_mem, *weight_mem, *gradi_mem, *gradw_mem));
            MKLDNNStream::Instance().Submit();
        }

       // copy data from gradw_mem to in_grad[1] and in_grad[2]
        DType* gw_buf = reinterpret_cast<DType *>(gradw_mem->get_data_handle());
        for (size_t i = 0; i < channels_; i++) {
            if (!param.fix_gamma)
                (in_grad[1].data().dptr<DType>())[i] = gw_buf[i];
            else
                (in_grad[1].data().dptr<DType>())[i] = 0.0f;
        }

        for (size_t i = 0; i < channels_; i++) {
            (in_grad[2].data().dptr<DType>())[i] = gw_buf[i + channels_];
        }
/*
        if (req[batchnorm::kGamma] == mxnet::kAddTo) {
            for (int i = 0; i < channels_; i++) {
                (gamma.data().dptr<DType>())[i] += (in_grad[1].data().dptr<DType>())[i];
                (beta.data().dptr<DType>())[i] += (in_grad[2].data().dptr<DType>())[i];
            }
        } else if (req[batchnorm::kGamma] == mxnet::kWriteTo) {
             for (int i = 0; i < channels_; i++) {
                (gamma.data().dptr<DType>())[i] = (in_grad[1].data().dptr<DType>())[i];
                (beta.data().dptr<DType>())[i] = (in_grad[2].data().dptr<DType>())[i];
            }
        } else {
            LOG(FATAL) << "MKLDNN batch normalization: should not reach here. req=" << req[batchnorm::kGamma];
        }
*/
        return;
    } else {
        return;
    }

}
}
}
#endif
