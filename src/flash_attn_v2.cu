// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:08:30 on Sun, Aug 27, 2023
//
// Description: flash attn v2.1.0

#include "flash_attn_v2/flash.h"
#include "flash_attn_v2/static_switch.h"
#include "tensor.h"

#define M_LOG2E 1.4426950408889634074  // log_2 e

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    FWD_HEADDIM_SWITCH(params.d, [&] { run_mha_fwd_<cutlass::half_t, kHeadDim>(params, stream); });
}

Flash_fwd_params set_mha_fwd_params(Tensor<cutlass::half_t> *Q, Tensor<cutlass::half_t> *K, Tensor<cutlass::half_t> *V,
                                    Tensor<cutlass::half_t> *O, int *cu_seq_q, int *cu_seq_k, bool is_causal,
                                    cudaDeviceProp *dev_prop) {
    size_t batch = Q->getShape()[0];
    size_t seq_q = Q->getShape()[1];
    size_t head = Q->getShape()[2];
    size_t dim = Q->getShape()[3];
    size_t seq_k = K->getShape()[1];
    size_t head_k = K->getShape()[2];

    FAI_CHECK_LE(dim, 256);
    FAI_CHECK_EQ(head % head_k, 0);

    Flash_fwd_params params;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = reinterpret_cast<void *>(Q->getDevPtr());
    params.k_ptr = reinterpret_cast<void *>(K->getDevPtr());
    params.v_ptr = reinterpret_cast<void *>(V->getDevPtr());

    params.q_batch_stride = seq_q * head * dim;
    params.q_row_stride = head * dim;
    params.q_head_stride = dim;

    params.k_batch_stride = seq_k * head_k * dim;
    params.k_row_stride = head_k * dim;
    params.k_head_stride = dim;

    params.v_batch_stride = seq_k * head_k * dim;
    params.v_row_stride = head_k * dim;
    params.v_head_stride = dim;

    params.cu_seqlens_q = cu_seq_q;
    params.cu_seqlens_k = cu_seq_k;

    params.h = head;
    params.h_k = head_k;
    params.h_h_k_ratio = params.h / params.h_k;

    params.o_ptr = reinterpret_cast<void *>(O->getDevPtr());

    params.o_batch_stride = seq_q * head * dim;
    params.o_row_stride = head * dim;
    params.o_head_stride = dim;

    // Softmax sum
    Tensor<float> *softmax_lse = new Tensor<float>({batch, head, seq_q});
    params.softmax_lse_ptr = reinterpret_cast<void *>(softmax_lse->getDevPtr());

    // Set the dimensions.
    params.b = batch;
    params.seqlen_q = seq_q;
    params.seqlen_k = seq_k;
    params.d = dim;

    params.scale_softmax = 1.0 / sqrtf(dim);
    params.scale_softmax_log2 = params.scale_softmax * M_LOG2E;

    params.is_causal = is_causal;

    params.props = dev_prop;
    params.is_sm8x = params.props->major == 8 && params.props->minor > 0;

    return params;
}

void flash_attn_v2(Tensor<cutlass::half_t> *Q, Tensor<cutlass::half_t> *K, Tensor<cutlass::half_t> *V,
                   Tensor<cutlass::half_t> *O, int *cu_seq_q, int *cu_seq_k, bool is_causal, int num_splits,
                   cudaDeviceProp *dev_prop) {
    static Flash_fwd_params params = set_mha_fwd_params(Q, K, V, O, cu_seq_q, cu_seq_k, is_causal, dev_prop);
    run_mha_fwd(params, nullptr);
}
