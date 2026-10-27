/*
 * Qwen3 Text Encoder for Iris
 *
 * Implements the Qwen3 text encoder that produces embeddings for image
 * generation. Supports multiple model sizes (architecture read from config.json).
 */

#ifndef IRIS_QWEN3_H
#define IRIS_QWEN3_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Architecture Constants
 * ======================================================================== */

/* Fixed constants (same across model sizes) */
#define QWEN3_VOCAB_SIZE       151936
#define QWEN3_MAX_SEQ_LEN      512
#define QWEN3_VL_MAX_SEQ_LEN   1280  /* Flux3 (Qwen3-VL): 16 windows of 80 tokens */
#define QWEN3_FLUX3_PAD_MULTIPLE 80  /* Flux3 pads the prompt to a multiple of this */
#define QWEN3_RMS_NORM_EPS     1e-6f
#define QWEN3_ROPE_THETA       1000000.0f

/* Maximum number of hidden-state layers concatenated into the embeddings
 * (Flux3 stacks 8 of them). */
#define QWEN3_MAX_OUTPUT_LAYERS 8

/* Extraction modes, see qwen3_set_extraction_mode(). */
#define QWEN3_EXTRACT_FLUX     0
#define QWEN3_EXTRACT_ZIMAGE   1
#define QWEN3_EXTRACT_FLUX3    2
#define QWEN3_EXTRACT_QWEN21   3

/* ========================================================================
 * Opaque Types
 * ======================================================================== */

typedef struct qwen3_model qwen3_model_t;
typedef struct qwen3_tokenizer qwen3_tokenizer_t;

/* ========================================================================
 * Tokenizer API
 * ======================================================================== */

/*
 * Load tokenizer from HuggingFace tokenizer.json file.
 */
qwen3_tokenizer_t *qwen3_tokenizer_load(const char *tokenizer_json_path);

/*
 * Free tokenizer resources.
 */
void qwen3_tokenizer_free(qwen3_tokenizer_t *tok);

/*
 * Tokenize text with chat template.
 * Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
 * skip_think_tags: when non-zero, omit <think>...</think> suffix (used by Z-Image mode)
 */
int *qwen3_tokenize_chat(qwen3_tokenizer_t *tok, const char *prompt,
                         int *num_tokens, int max_len, int skip_think_tags);

/*
 * Tokenize text with a system message (Qwen-Image 2.1's text encoder):
 * Format: </think>
 */
int *qwen3_tokenize_chat_with_system(qwen3_tokenizer_t *tok, const char *prompt,
                                     const char *system_prompt,
                                     int *num_tokens, int max_len);

/*
 * Pad tokens to max_len with PAD token.
 * Returns new array, caller must free original.
 */
int *qwen3_pad_tokens(int *tokens, int num_tokens, int max_len, int *attention_mask);

/*
 * Get token string by ID.
 */
const char *qwen3_get_token(qwen3_tokenizer_t *tok, int id);

/*
 * Get token ID by string.
 */
int qwen3_get_id(qwen3_tokenizer_t *tok, const char *token);

/* ========================================================================
 * Model API
 * ======================================================================== */

/*
 * Load Qwen3 model weights from HuggingFace model directory.
 * Directory should contain model-00001-of-00002.safetensors, etc.
 * Also accepts the text model of a Qwen3-VL checkpoint (Flux3), whose
 * tensors live under "model.language_model.".
 */
qwen3_model_t *qwen3_model_load(const char *model_dir);

/*
 * Free model resources.
 */
void qwen3_model_free(qwen3_model_t *model);

/*
 * Run forward pass to generate text embeddings.
 *
 * input_ids: Token IDs [seq_len]
 * attention_mask: Attention mask [seq_len] (1 for real tokens, 0 for padding)
 * seq_len: Length of input sequences
 *
 * Returns: Embedding array [seq_len, text_dim] (caller must free)
 * Concatenates the hidden states selected by the extraction mode.
 */
float *qwen3_forward(qwen3_model_t *model,
                     const int *input_ids,
                     const int *attention_mask,
                     int seq_len);

/* ========================================================================
 * Combined Text Encoder API
 * ======================================================================== */

typedef struct qwen3_encoder {
    qwen3_tokenizer_t *tokenizer;
    qwen3_model_t *model;
} qwen3_encoder_t;

/*
 * Load complete text encoder (tokenizer + model).
 * model_dir should contain text_encoder/ and the tokenizer, either in
 * tokenizer/ or (Flux3) inside text_encoder/.
 * use_mmap: if true, use memory-mapped bf16 weights (saves ~8GB, slower inference)
 */
qwen3_encoder_t *qwen3_encoder_load(const char *model_dir, int use_mmap);

/*
 * Set extraction mode (Python hidden_states[] indices):
 *   QWEN3_EXTRACT_FLUX:   hidden_states[9,18,27] concatenated -> [seq, 3*hidden]
 *   QWEN3_EXTRACT_ZIMAGE: hidden_states[-2] only -> [seq, hidden]
 *   QWEN3_EXTRACT_FLUX3:  hidden_states[4,8,...,32] -> [seq, 8*hidden]
 * Flux and Z-Image pad prompts to 512 tokens; Flux3 pads to a multiple of
 * 80 tokens and places the padding at RoPE position 1, like the
 * transformers version it was trained with.
 */
void qwen3_set_extraction_mode(qwen3_encoder_t *enc, int mode);

/*
 * Free encoder resources.
 */
void qwen3_encoder_free(qwen3_encoder_t *enc);

/*
 * Encode text prompt to embeddings.
 * Returns: Embedding array [seq_len, text_dim] (caller must free), where
 * seq_len is 512 except in Flux3 mode (see qwen3_encode_text_ex) and
 * text_dim is hidden_size times the number of extracted layers.
 */
float *qwen3_encode_text(qwen3_encoder_t *enc, const char *prompt);

/*
 * Encode text prompt to embeddings, returning token counts.
 * Same as qwen3_encode_text but also returns the number of real
 * (non-padding) tokens in *out_num_tokens and the padded sequence
 * length (rows of the returned array) in *out_seq_len. Both may be NULL.
 * Returns: Embedding array [seq_len, text_dim] (caller must free)
 */
float *qwen3_encode_text_ex(qwen3_encoder_t *enc, const char *prompt,
                              int *out_num_tokens, int *out_seq_len);

#ifdef __cplusplus
}
#endif

#endif /* IRIS_QWEN3_H */
