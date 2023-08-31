// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:08:30 on Sun, Aug 27, 2023
//
// Description: flash attn v1.0.9

#include "flash_attn/fmha.h"
#include "flash_attn/static_switch.h"
#include "tensor.h"

void run_fmha_fwd(Launch_params<FMHA_fprop_params> &launch_params) {
    if (launch_params.params.d <= 32) {
        run_fmha_fwd_hdim32(launch_params);
    } else if (launch_params.params.d <= 64) {
        run_fmha_fwd_hdim64(launch_params);
    } else if (launch_params.params.d <= 128) {
        run_fmha_fwd_hdim128(launch_params);
    }
}

Launch_params<FMHA_fprop_params> set_fmha_fwd_params(Tensor<cutlass::half_t> *Q, Tensor<cutlass::half_t> *K,
                                                     Tensor<cutlass::half_t> *V, Tensor<cutlass::half_t> *O,
                                                     int *cu_seq_q, int *cu_seq_k, bool is_causal, int num_splits,
                                                     cudaDeviceProp *dev_prop) {
    size_t batch = Q->getShape()[0];
    size_t seq_q = Q->getShape()[1];
    size_t head = Q->getShape()[2];
    size_t dim = Q->getShape()[3];
    size_t seq_k = K->getShape()[1];

    FAI_CHECK_LE(dim, 128);
    FAI_CHECK_EQ(dim % 8, 0);

    Launch_params<FMHA_fprop_params> launch_params(dev_prop, nullptr);

    // Reset the parameters
    memset(&launch_params.params, 0, sizeof(launch_params.params));

    int max_seq_q = ((seq_q + 16 - 1) / 16) * 16;

    int blocksize_c = dim > 64 ? 128 : 256;
    // Need to round max_seqlen_k to multiples of blocksize_c
    int max_seq_k = ((seq_k + blocksize_c - 1) / blocksize_c) * blocksize_c;
    if (max_seq_k <= 128) {
        max_seq_k = 128;
    } else if (max_seq_k <= 256) {
        max_seq_k = 256;
    }

    // Set the pointers and strides.
    launch_params.params.q_ptr = reinterpret_cast<void *>(Q->getDevPtr());
    launch_params.params.k_ptr = reinterpret_cast<void *>(K->getDevPtr());
    launch_params.params.v_ptr = reinterpret_cast<void *>(V->getDevPtr());

    launch_params.params.q_row_stride_in_elts = head * dim;
    launch_params.params.k_row_stride_in_elts = head * dim;
    launch_params.params.v_row_stride_in_elts = head * dim;
    launch_params.params.q_head_stride_in_elts = dim;
    launch_params.params.k_head_stride_in_elts = dim;
    launch_params.params.v_head_stride_in_elts = dim;

    launch_params.params.h = head;

    launch_params.params.o_ptr = reinterpret_cast<void *>(O->getDevPtr());

    launch_params.params.o_row_stride_in_elts = head * dim;
    launch_params.params.o_head_stride_in_elts = dim;
    launch_params.params.o_tmp_row_stride_in_elts = head * dim;
    launch_params.params.o_tmp_head_stride_in_elts = dim;

    launch_params.params.o_tmp_ptr = nullptr;
    if (max_seq_k > blocksize_c) {
        Tensor<float> *o_tmp = new Tensor<float>({batch, seq_q, head, dim});
        launch_params.params.o_tmp_ptr = reinterpret_cast<void *>(o_tmp->getDevPtr());
    }

    // Softmax sum
    Tensor<float> *softmax_lse = new Tensor<float>({batch, head, static_cast<size_t>(max_seq_q)});
    launch_params.params.softmax_lse_ptr = reinterpret_cast<void *>(softmax_lse->getDevPtr());

    // Set the dimensions.
    launch_params.params.b = batch;
    launch_params.params.seqlen_q = max_seq_q;
    launch_params.params.seqlen_k = max_seq_k;
    launch_params.params.d = dim;

    launch_params.params.scale_bmm1f = 1.0 / sqrtf(dim);
    set_alpha(launch_params.params.scale_bmm1, launch_params.params.scale_bmm1f, DATA_TYPE_FP16);

    launch_params.params.cu_seqlens_q = cu_seq_q;
    launch_params.params.cu_seqlens_k = cu_seq_k;

    launch_params.params.is_causal = is_causal;

    launch_params.params.num_splits = num_splits;

    return launch_params;
}

void flash_attn(Tensor<cutlass::half_t> *Q, Tensor<cutlass::half_t> *K, Tensor<cutlass::half_t> *V,
                Tensor<cutlass::half_t> *O, int *cu_seq_q, int *cu_seq_k, bool is_causal, int num_splits,
                cudaDeviceProp *dev_prop) {
    static Launch_params<FMHA_fprop_params> launch_params =
        set_fmha_fwd_params(Q, K, V, O, cu_seq_q, cu_seq_k, is_causal, num_splits, dev_prop);
    run_fmha_fwd(launch_params);
}
