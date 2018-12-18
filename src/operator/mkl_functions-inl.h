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
 * Copyright (c) 2018 by Contributors
 * \file mkl_functions-inl.h
 * \brief
 * \author
*/
#ifndef MXNET_OPERATOR_MKL_FUNCTIONS_H_
#define MXNET_OPERATOR_MKL_FUNCTIONS_H_

#if MSHADOW_USE_MKL == 1
#include "mkl.h"

namespace mxnet {
namespace op {
namespace mkl_func {

MSHADOW_XINLINE
static bool check_size(const size_t n) {
  const size_t MKL_INT_MAX = (sizeof(MKL_INT) == sizeof(int)) ? INT_MAX : LLONG_MAX;
  return (n <= MKL_INT_MAX);
}

MSHADOW_XINLINE
static bool check_type(const int t) {
  return (t == mshadow::kFloat32 || t == mshadow::kFloat64); 
}

#define MXNET_MKL_UNARY_MATH_FUNC(name) \
template<typename DType> MSHADOW_XINLINE \
void MKL##name(const index_t n, const DType* src, float* dst) { \
  vs##name(static_cast<MKL_INT>(n), reinterpret_cast<const float*>(src), dst); \
} \
MSHADOW_XINLINE \
void MKL##name(const index_t n, const double* src, double* dst) { \
  vd##name(static_cast<MKL_INT>(n), src, dst); \
}

#define MXNET_MKL_BINARY_MATH_FUNC(name) \
template<typename DType> MSHADOW_XINLINE \
void MKL##name(const index_t n, const DType* a, const DType* b, float* c) { \
  vs##name(static_cast<MKL_INT>(n), \
           reinterpret_cast<const float*>(a), \
           reinterpret_cast<const float*>(b), \
           c); \
} \
MSHADOW_XINLINE \
void MKL##name(const index_t n, const double* a, const double* b, double* c) { \
  vd##name(static_cast<MKL_INT>(n), a, b, c); \
}

MXNET_MKL_UNARY_MATH_FUNC(Erf);
MXNET_MKL_UNARY_MATH_FUNC(Exp);
MXNET_MKL_UNARY_MATH_FUNC(Exp2);
MXNET_MKL_UNARY_MATH_FUNC(Exp10);
MXNET_MKL_UNARY_MATH_FUNC(Expm1);
MXNET_MKL_UNARY_MATH_FUNC(Ln);
MXNET_MKL_UNARY_MATH_FUNC(Log2);
MXNET_MKL_UNARY_MATH_FUNC(Log10);
MXNET_MKL_UNARY_MATH_FUNC(Log1p);

MXNET_MKL_UNARY_MATH_FUNC(Sin);
MXNET_MKL_UNARY_MATH_FUNC(Cos);
MXNET_MKL_UNARY_MATH_FUNC(Tan);
MXNET_MKL_UNARY_MATH_FUNC(Asin);
MXNET_MKL_UNARY_MATH_FUNC(Acos);
MXNET_MKL_UNARY_MATH_FUNC(Atan);

MXNET_MKL_UNARY_MATH_FUNC(Sinh);
MXNET_MKL_UNARY_MATH_FUNC(Cosh);
MXNET_MKL_UNARY_MATH_FUNC(Tanh);
MXNET_MKL_UNARY_MATH_FUNC(Asinh);
MXNET_MKL_UNARY_MATH_FUNC(Acosh);
MXNET_MKL_UNARY_MATH_FUNC(Atanh);

MXNET_MKL_UNARY_MATH_FUNC(Sqrt);
MXNET_MKL_UNARY_MATH_FUNC(Abs);
MXNET_MKL_UNARY_MATH_FUNC(Cbrt);
MXNET_MKL_UNARY_MATH_FUNC(Round);
MXNET_MKL_UNARY_MATH_FUNC(Ceil);
MXNET_MKL_UNARY_MATH_FUNC(Floor);
MXNET_MKL_UNARY_MATH_FUNC(Trunc);

MXNET_MKL_UNARY_MATH_FUNC(LGamma);
MXNET_MKL_UNARY_MATH_FUNC(TGamma);
MXNET_MKL_UNARY_MATH_FUNC(Sqr);

MXNET_MKL_BINARY_MATH_FUNC(Add);
MXNET_MKL_BINARY_MATH_FUNC(Sub);
MXNET_MKL_BINARY_MATH_FUNC(Mul);
MXNET_MKL_BINARY_MATH_FUNC(Pow);
MXNET_MKL_BINARY_MATH_FUNC(Hypot);

}  // namespace mkl_func
}  // namespace op
}  // namespace mxnet
#endif  // MSHADOW_USE_MKL == 1
#endif  // MXNET_OPERATOR_MKL_FUNCTIONS_H_
