// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:08:30 on Sun, Aug 27, 2023
//
// Description: tester

#ifndef __FLASH_ATTENTION_INFERENCE_TESTER_H__
#define __FLASH_ATTENTION_INFERENCE_TESTER_H__

#include "cuda_timer.h"
#include "tensor.h"

class Tester {
public:
    explicit Tester(size_t batch = 2, size_t seq_q = 256, size_t seq_k = 256, size_t head_q = 32, size_t head_k = 32,
                    size_t dim = 128, bool is_causal = true, int num_splits = 0, cudaDeviceProp *dev_prop = nullptr,
                    size_t warmup_iterations = 1, size_t profiling_iterations = 10, size_t sleep_duration = 100,
                    bool enable_check = false)
        : m_batch(batch),
          m_seq_q(seq_q),
          m_seq_k(seq_k),
          m_head_q(head_q),
          m_head_k(head_k),
          m_dim(dim),
          m_is_causal(is_causal),
          m_num_splits(num_splits),
          m_dev_prop(dev_prop),
          m_warmup_iterations(warmup_iterations),
          m_profiling_iterations(profiling_iterations),
          m_sleep_duration(sleep_duration),
          m_enable_check(enable_check) {
        FAI_CHECK_GT(m_batch, 0);
        FAI_CHECK_GT(m_seq_q, 0);
        FAI_CHECK_GT(m_seq_k, 0);
        FAI_CHECK_GT(m_head_q, 0);
        FAI_CHECK_GT(m_head_k, 0);
        FAI_CHECK_GT(m_dim, 0);
        FAI_CHECK_GT(m_warmup_iterations, 0);
        FAI_CHECK_GT(m_profiling_iterations, 0);
        FAI_CHECK_GT(m_sleep_duration, 0);

        m_Q = new Tensor<cutlass::half_t>({m_batch, m_seq_q, m_head_q, m_dim}, "Tensor Q");
        FAI_CHECK(m_Q);
        m_K = new Tensor<cutlass::half_t>({m_batch, m_seq_k, m_head_k, m_dim}, "Tensor K");
        FAI_CHECK(m_K);
        m_V = new Tensor<cutlass::half_t>({m_batch, m_seq_k, m_head_k, m_dim}, "Tensor V");
        FAI_CHECK(m_V);
        m_O = new Tensor<cutlass::half_t>({m_batch, m_seq_q, m_head_q, m_dim}, "Tensor O");
        FAI_CHECK(m_O);
        m_base = new Tensor<cutlass::half_t>({m_batch, m_seq_q, m_head_q, m_dim}, "Tensor Base");
        FAI_CHECK(m_base);

        get_cu_seq(m_cu_seq_q, m_batch, m_seq_q);
        FAI_CHECK_CUDART_ERROR(cudaMalloc((void **)&m_cu_seq_q_dev, m_cu_seq_q.size() * sizeof(int)));
        FAI_CHECK(m_cu_seq_q_dev);
        FAI_CHECK_CUDART_ERROR(
            cudaMemcpy(m_cu_seq_q_dev, m_cu_seq_q.data(), m_cu_seq_q.size() * sizeof(int), cudaMemcpyHostToDevice));

        get_cu_seq(m_cu_seq_k, m_batch, m_seq_k);
        FAI_CHECK_CUDART_ERROR(cudaMalloc((void **)&m_cu_seq_k_dev, m_cu_seq_k.size() * sizeof(int)));
        FAI_CHECK(m_cu_seq_k_dev);
        FAI_CHECK_CUDART_ERROR(
            cudaMemcpy(m_cu_seq_k_dev, m_cu_seq_k.data(), m_cu_seq_k.size() * sizeof(int), cudaMemcpyHostToDevice));

        if (m_enable_check) {
            clock_t begin, end;
            begin = clock();
            attention_cpu(m_Q, m_K, m_V, m_base, m_cu_seq_q.data(), m_cu_seq_k.data(), m_is_causal);
            end = clock();
            FLOG("CPU use: %.3f ms", float(end - begin) / CLOCKS_PER_SEC * 1e3);
        }
    }

    ~Tester() {
        if (m_Q) {
            delete m_Q;
            m_Q = nullptr;
        }

        if (m_K) {
            delete m_K;
            m_K = nullptr;
        }

        if (m_V) {
            delete m_V;
            m_V = nullptr;
        }

        if (m_O) {
            delete m_O;
            m_O = nullptr;
        }

        if (m_base) {
            delete m_base;
            m_base = nullptr;
        }

        if (m_cu_seq_q_dev) {
            FAI_CHECK_CUDART_ERROR(cudaFree((void *)m_cu_seq_q_dev));
            m_cu_seq_q_dev = nullptr;
        }

        if (m_cu_seq_k_dev) {
            FAI_CHECK_CUDART_ERROR(cudaFree((void *)m_cu_seq_k_dev));
            m_cu_seq_k_dev = nullptr;
        }
    }

    template <typename Func>
    void evaluate(Func &&flash_attention, const std::string &name) {
        FLOG("----------------- Evaluating %s -----------------", name.c_str());
        usleep(m_sleep_duration * 1000);
        m_O->tearUp(m_base);

        // warm up
        m_cuda_timer.start();
        for (size_t i = 0; i < m_warmup_iterations; ++i) {
            flash_attention(m_Q, m_K, m_V, m_O, m_cu_seq_q_dev, m_cu_seq_k_dev, m_is_causal, m_num_splits, m_dev_prop);
        }
        m_warmup_time = static_cast<double>(m_cuda_timer.end()) / static_cast<double>(m_warmup_iterations);
        FLOG("Warm up time: %.3f ms", m_warmup_time);

        if (m_enable_check) {
            m_O->moveToHost();
            m_O->checkValue(m_base);
        }

        profile(std::forward<Func>(flash_attention), name);
    }

private:
    void get_cu_seq(std::vector<int> &cu_seq, size_t batch, size_t seq) {
        cu_seq.resize(batch + 1);
        for (size_t i = 0; i < cu_seq.size(); ++i) {
            cu_seq[i] = i * seq;
        }
    }

    void attention_cpu(Tensor<cutlass::half_t> *Q, Tensor<cutlass::half_t> *K, Tensor<cutlass::half_t> *V,
                       Tensor<cutlass::half_t> *O, int *cu_seq_q, int *cu_seq_k, bool is_causal) {
        size_t batch = Q->getShape()[0];
        size_t seq_q = Q->getShape()[1];
        size_t head_q = Q->getShape()[2];
        size_t dim = Q->getShape()[3];
        size_t seq_k = K->getShape()[1];
        size_t head_k = K->getShape()[2];

        FAI_CHECK_GE(head_q, head_k);
        FAI_CHECK_EQ(head_q % head_k, 0);
        size_t head_ratio = head_q / head_k;

        FAI_CHECK_GE(seq_k, seq_q);
        const size_t row_shift = seq_k - seq_q;

        cutlass::half_t *q_ptr = Q->getHostPtr();
        cutlass::half_t *k_ptr = K->getHostPtr();
        cutlass::half_t *v_ptr = V->getHostPtr();
        cutlass::half_t *o_ptr = O->getHostPtr();

        // S = Q * K^T
        Tensor<float> *S = new Tensor<float>({batch, seq_q, head_q, seq_k});
        float *s_ptr = S->getHostPtr();
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < head_q; ++h) {
                for (size_t sq = 0; sq < seq_q; ++sq) {
                    for (size_t sk = 0; sk < seq_k; ++sk) {
                        float tmp = 0.0;
                        for (size_t d = 0; d < dim; ++d) {
                            tmp += static_cast<cutlass::half_t>(
                                       q_ptr[b * (seq_q * head_q * dim) + sq * (head_q * dim) + h * dim + d]) *
                                   static_cast<cutlass::half_t>(k_ptr[b * (seq_k * head_k * dim) + sk * (head_k * dim) +
                                                                      (h / head_ratio) * dim + d]);
                        }
                        s_ptr[b * (seq_q * head_q * seq_k) + sq * (head_q * seq_k) + h * seq_k + sk] = tmp;
                    }
                }
            }
        }

        // P = Softmax(S)
        Tensor<cutlass::half_t> *P = new Tensor<cutlass::half_t>({batch, seq_q, head_q, seq_k});
        cutlass::half_t *p_ptr = P->getHostPtr();
        float scale = 1.0 / std::sqrt(dim);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < head_q; ++h) {
                for (size_t sq = 0; sq < seq_q; ++sq) {
                    size_t row = seq_q;
                    if (is_causal) {
                        row = sq + row_shift + 1;
                    }

                    // Max(S)
                    std::vector<float> tmp_s(seq_k, 0.0);
                    float max_s = -std::numeric_limits<float>::max();
                    for (size_t sk = 0; sk < row; ++sk) {
                        tmp_s[sk] =
                            s_ptr[b * (seq_q * head_q * seq_k) + sq * (head_q * seq_k) + h * seq_k + sk] * scale;
                        max_s = std::max(max_s, tmp_s[sk]);
                    }

                    // Sum(S)
                    float sum_s = 0.0;
                    for (size_t sk = 0; sk < row; ++sk) {
                        tmp_s[sk] = std::exp(tmp_s[sk] - max_s);
                        sum_s += tmp_s[sk];
                    }

                    // Softmax(S)
                    for (size_t sk = 0; sk < row; ++sk) {
                        p_ptr[b * (seq_q * head_q * seq_k) + sq * (head_q * seq_k) + h * seq_k + sk] =
                            static_cast<cutlass::half_t>(tmp_s[sk] / sum_s);
                    }

                    // Causal(S)
                    if (is_causal) {
                        for (size_t sk = row; sk < seq_q; ++sk) {
                            p_ptr[b * (seq_q * head_q * seq_k) + sq * (head_q * seq_k) + h * seq_k + sk] = 0_hf;
                        }
                    }
                }
            }
        }

        // O = P * V
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < head_q; ++h) {
                for (size_t sq = 0; sq < seq_q; ++sq) {
                    for (size_t d = 0; d < dim; ++d) {
                        float tmp = 0.0;
                        for (size_t sk = 0; sk < seq_k; ++sk) {
                            tmp += static_cast<cutlass::half_t>(
                                       p_ptr[b * (seq_q * head_q * seq_k) + sq * (head_q * seq_k) + h * seq_k + sk]) *
                                   static_cast<cutlass::half_t>(v_ptr[b * (seq_k * head_k * dim) + sk * (head_k * dim) +
                                                                      (h / head_ratio) * dim + d]);
                        }
                        o_ptr[b * (seq_q * head_q * dim) + sq * (head_q * dim) + h * dim + d] =
                            static_cast<cutlass::half_t>(tmp);
                    }
                }
            }
        }
    }

    template <typename Func>
    void profile(Func &&flash_attention, const std::string &name) {
        m_cuda_timer.start();
        for (size_t i = 0; i < m_profiling_iterations; ++i) {
            flash_attention(m_Q, m_K, m_V, m_O, m_cu_seq_q_dev, m_cu_seq_k_dev, m_is_causal, m_num_splits, m_dev_prop);
        }
        m_profiling_time = static_cast<double>(m_cuda_timer.end()) / static_cast<double>(m_profiling_iterations);
        m_throughput = static_cast<double>(m_batch * m_seq_q * m_seq_k * m_seq_q * m_dim * 4) * 1e-12 /
                       (static_cast<double>(m_profiling_time) * 1e-3);

        if (m_is_causal) {
            m_throughput /= 2;
        }

        if ((std::abs(m_base_time) <= 1e-6) && (std::abs(m_base_throughput) <= 1e-6)) {
            m_base_time = m_profiling_time;
            m_base_throughput = m_throughput;
        }

        FLOG("%s exit, profiling time: %.3f ms (%.2f%%), throughput: %.3f TFLOPS (%.2f%%)", name.c_str(),
             m_profiling_time, m_profiling_time / m_base_time * 100, m_throughput,
             m_throughput / m_base_throughput * 100);
    }

    const size_t m_batch = 2;
    const size_t m_seq_q = 256;
    const size_t m_seq_k = 256;
    const size_t m_head_q = 32;
    const size_t m_head_k = 32;
    const size_t m_dim = 128;
    const bool m_is_causal = true;
    cudaDeviceProp *m_dev_prop = nullptr;
    const int m_num_splits = 0;
    const size_t m_warmup_iterations = 1;
    const size_t m_profiling_iterations = 10;
    const size_t m_sleep_duration = 100;
    const bool m_enable_check = false;

    Tensor<cutlass::half_t> *m_Q = nullptr;  // batch * seq_q * head_q * dim
    Tensor<cutlass::half_t> *m_K = nullptr;  // batch * seq_k * head_k * dim
    Tensor<cutlass::half_t> *m_V = nullptr;  // batch * seq_k * head_k * dim
    Tensor<cutlass::half_t> *m_O = nullptr;  // batch * seq_q * head_q * dim
    Tensor<cutlass::half_t> *m_base =
        nullptr;  // batch * seq_q * head_q * dim, base result, init tensor O before each attention

    std::vector<int> m_cu_seq_q;
    int *m_cu_seq_q_dev = nullptr;
    std::vector<int> m_cu_seq_k;
    int *m_cu_seq_k_dev = nullptr;

    CudaTimer m_cuda_timer;

    double m_warmup_time = 0.0;
    double m_profiling_time = 0.0;
    double m_throughput = 0.0;
    double m_base_time = 0.0;        // flash attn op
    double m_base_throughput = 0.0;  // flash attn op

    FAI_DISALLOW_COPY_AND_ASSIGN(Tester);
};

#endif  // __FLASH_ATTENTION_INFERENCE_TESTER_H__
