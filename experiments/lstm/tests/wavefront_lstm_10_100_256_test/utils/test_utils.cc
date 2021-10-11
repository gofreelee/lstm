#include "test_utils.h"

namespace mica::experiments::lstm {

std::vector<const float **> readInputParams(int fd, int hidden_size,
                                            int batch_size, int num_layer) {
    float *all_one_input =
        (float *)malloc(sizeof(float) * batch_size * hidden_size);
    for (int i = 0; i < batch_size * hidden_size; ++i)
        all_one_input[i] = 1.0000f;
    const float **_input = (const float **)malloc(sizeof(float *) * num_layer);
    for (int i = 0; i < num_layer; ++i)
        _input[i] = all_one_input;

    float *all_zero_state =
        (float *)malloc(sizeof(float) * batch_size * hidden_size);
    for (int i = 0; i < batch_size * hidden_size; ++i)
        all_zero_state[i] = 0.000f;
    const float **_init_state =
        (const float **)malloc(sizeof(float *) * num_layer);
    for (int i = 0; i < num_layer; ++i)
        _init_state[i] = all_zero_state;

    void *buf =
        (void *)malloc(sizeof(float) * (hidden_size * hidden_size * 40 * 2 +
                                        batch_size * hidden_size * 40));
    int sum = sizeof(float) * (hidden_size * hidden_size * 40 * 2 +
                               batch_size * hidden_size * 40);
    int readed_bytes = 0, onetime_bytes = 0;
    void *base = buf;
    while (readed_bytes < sum) {
        onetime_bytes = read(fd, buf, sum - readed_bytes);
        if (onetime_bytes == -1 && errno == EINTR)
            continue;
        readed_bytes += onetime_bytes;
        buf = (char *)base + readed_bytes;
    }
    float *data = reinterpret_cast<float *>(base);
    const float **_W = (const float **)malloc(sizeof(float *) * 40);
    const float **_U = (const float **)malloc(sizeof(float *) * 40);
    const float **_bias = (const float **)malloc(sizeof(float *) * 40);
    for (int i = 0; i < 40; ++i) {
        _W[i] = data + hidden_size * hidden_size * i;
        _U[i] = data + hidden_size * hidden_size * (i + 40);
        _bias[i] = data + hidden_size * hidden_size * 80 +
                   i * batch_size * hidden_size;
    }
    std::vector<const float **> params{_input, _init_state, _W, _U, _bias};
    return params;
}

std::vector<LSTMNetHostParams> readInputParamsFuse(int fd, int hidden_size,
                                                   int batch_size,
                                                   int num_layer,
                                                   int num_step) {
    float *all_one_input =
        (float *)malloc(sizeof(float) * batch_size * hidden_size);
    for (int i = 0; i < batch_size * hidden_size; ++i)
        all_one_input[i] = 1.0000f;

    float *all_zero_state =
        (float *)malloc(sizeof(float) * batch_size * hidden_size);
    for (int i = 0; i < batch_size * hidden_size; ++i)
        all_zero_state[i] = 0.000f; // state_c  state_h  init

    void *buf =
        (void *)malloc(sizeof(float) * (hidden_size * hidden_size * 40 * 2 +
                                        batch_size * hidden_size * 40));
    int sum =
        sizeof(float) * (hidden_size * hidden_size * 40 * 2 + hidden_size * 40);
    int readed_bytes = 0, onetime_bytes = 0;
    void *base = buf;
    while (readed_bytes < sum) {
        onetime_bytes = read(fd, buf, sum - readed_bytes);
        if (onetime_bytes == -1 && errno == EINTR)
            continue;
        readed_bytes += onetime_bytes;
        buf = (char *)base + readed_bytes;
    }
    float *data = reinterpret_cast<float *>(base);
    std::vector<LSTMNetHostParams> params;
    for (int i = 0; i < num_layer; ++i) {

        float *w_ptr = data + hidden_size * hidden_size * 4 * i;
        float *u_ptr = data + hidden_size * hidden_size * 40 +
                       hidden_size * hidden_size * 4 * i;
        float *bias_ptr =
            data + hidden_size * hidden_size * 80 + hidden_size * 4 * i;
        float *w_fuse_ptr =
            (float *)malloc(sizeof(float) * hidden_size * hidden_size * 4);
        float *u_fuse_ptr =
            (float *)malloc(sizeof(float) * hidden_size * hidden_size * 4);
        float *bias_fuse_ptr = (float *)malloc(sizeof(float) * hidden_size * 4);
        for (int j = 0; j < hidden_size * hidden_size; ++j) {
            w_fuse_ptr[j * 4] = w_ptr[j];
            w_fuse_ptr[j * 4 + 1] = w_ptr[j + hidden_size * hidden_size];
            w_fuse_ptr[j * 4 + 2] = w_ptr[j + hidden_size * hidden_size * 2];
            w_fuse_ptr[j * 4 + 3] = w_ptr[j + hidden_size * hidden_size * 3];
            u_fuse_ptr[j * 4] = u_ptr[j];
            u_fuse_ptr[j * 4 + 1] = u_ptr[j + hidden_size * hidden_size];
            u_fuse_ptr[j * 4 + 2] = u_ptr[j + hidden_size * hidden_size * 2];
            u_fuse_ptr[j * 4 + 3] = u_ptr[j + hidden_size * hidden_size * 3];
        }
        for (int j = 0; j < hidden_size; ++j) {
            bias_fuse_ptr[j * 4] = bias_ptr[j];
            bias_fuse_ptr[j * 4 + 1] = bias_ptr[j + hidden_size];
            bias_fuse_ptr[j * 4 + 2] = bias_ptr[j + hidden_size * 2];
            bias_fuse_ptr[j * 4 + 3] = bias_ptr[j + hidden_size * 3];
        }
        LSTMNetHostParams param = {all_one_input,  all_zero_state,
                                   all_zero_state, w_fuse_ptr,
                                   u_fuse_ptr,     bias_fuse_ptr};
        params.push_back(param);
    }
    // free(data);
    return params;

} // namespace mica::experiments::lstm

void freeParams(const std::vector<const float **> &params) {
    free((void *)params[0][0]);
    free((void *)params[1][0]);
    free((void *)params[2][0]); // free data
    for (auto item : params)
        free(item);
}

float *readExpectedResult(int fd, int hidden_size) {
    void *buf = (void *)malloc(sizeof(float) * hidden_size);
    int sum = sizeof(float) * hidden_size;

    int readed_bytes = 0, onetime_bytes = 0;
    void *base = buf;
    while (readed_bytes < sum) {
        onetime_bytes = read(fd, buf, sum - readed_bytes);
        if (onetime_bytes == -1 && errno == EINTR)
            continue;
        readed_bytes += onetime_bytes;
        buf = (char *)base + readed_bytes;
    }
    return reinterpret_cast<float *>(base);
}

void freeParamsFuse(const std::vector<LSTMNetHostParams> &parmas) {
    for (auto &param : parmas) {
        free(param.weights_w);
        free(param.weights_u);
        free(param.bias_s);
    }
    if (parmas.size() > 0) {
        free(parmas[0].inputs);
        free(parmas[0].state_c_s);
        if (parmas[0].state_c_s != parmas[1].state_h_s)
            free(parmas[0].state_h_s);
    }
}

} // namespace mica::experiments::lstm