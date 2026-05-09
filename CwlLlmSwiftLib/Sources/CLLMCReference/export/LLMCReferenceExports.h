#ifndef LLMCReferenceExports_h
#define LLMCReferenceExports_h

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#ifndef _WIN32
#include <glob.h>
#endif

void gelu_forward(float *out, float *inp, int N);
void gelu_backward(float *dinp, float *inp, float *dout, int N);

void layernorm_forward(
    float *out,
    float *mean,
    float *rstd,
    float *inp,
    float *weight,
    float *bias,
    int B,
    int T,
    int C
);

void layernorm_backward(
    float *dinp,
    float *dweight,
    float *dbias,
    float *dout,
    float *inp,
    float *weight,
    float *mean,
    float *rstd,
    int B,
    int T,
    int C
);

typedef struct {
    int max_seq_len;
    int vocab_size;
    int padded_vocab_size;
    int num_layers;
    int num_heads;
    int channels;
} GPT2Config;

#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float *wte;
    float *wpe;
    float *ln1w;
    float *ln1b;
    float *qkvw;
    float *qkvb;
    float *attprojw;
    float *attprojb;
    float *ln2w;
    float *ln2b;
    float *fcw;
    float *fcb;
    float *fcprojw;
    float *fcprojb;
    float *lnfw;
    float *lnfb;
} ParameterTensors;

#define NUM_ACTIVATION_TENSORS 23
typedef struct {
    float *encoded;
    float *ln1;
    float *ln1_mean;
    float *ln1_rstd;
    float *qkv;
    float *atty;
    float *preatt;
    float *att;
    float *attproj;
    float *residual2;
    float *ln2;
    float *ln2_mean;
    float *ln2_rstd;
    float *fch;
    float *fch_gelu;
    float *fcproj;
    float *residual3;
    float *lnf;
    float *lnf_mean;
    float *lnf_rstd;
    float *logits;
    float *probs;
    float *losses;
} ActivationTensors;

typedef struct {
    GPT2Config config;
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float *params_memory;
    size_t num_parameters;
    ParameterTensors grads;
    float *grads_memory;
    float *m_memory;
    float *v_memory;
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float *acts_memory;
    size_t num_activations;
    ActivationTensors grads_acts;
    float *grads_acts_memory;
    int batch_size;
    int seq_len;
    int *inputs;
    int *targets;
    float mean_loss;
} GPT2;

void gpt2_build_from_checkpoint(GPT2 *model, const char *checkpoint_path);
void gpt2_forward(GPT2 *model, int *inputs, int *targets, size_t B, size_t T);
void gpt2_zero_grad(GPT2 *model);
void gpt2_backward(GPT2 *model);
void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t);
void gpt2_free(GPT2 *model);

#endif
typedef struct {
    unsigned long long seed_;
    int left_;
    unsigned int next_;
    unsigned int state_[624];
    unsigned int MATRIX_A[2];
} mt19937_state;

typedef struct {
    int process_rank;
    int num_processes;
    size_t B;
    size_t T;
    size_t num_tokens;
    size_t shard_num_samples;
    glob_t glob_result;
    size_t current_shard_idx;
    size_t current_sample_idx;
    FILE *tokens_file;
    uint16_t *buffer;
    int *inputs;
    int *targets;
    mt19937_state shuffle_rng;
    int should_shuffle;
    int *shard_indices;
    int *intra_shard_indices;
    size_t total_batch_size_bytes;
    size_t local_batch_offset_bytes;
    size_t header_bytes;
    int64_t file_size_bytes;
} DataLoader;

void dataloader_init(DataLoader *loader, const char *filename_pattern, size_t B, size_t T, int process_rank, int num_processes, int should_shuffle);
void dataloader_reset(DataLoader *loader);
void dataloader_next_batch(DataLoader *loader);
void dataloader_free(DataLoader *loader);
