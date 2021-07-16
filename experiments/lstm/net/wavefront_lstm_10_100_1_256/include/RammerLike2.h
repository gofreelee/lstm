#ifndef _LSTMNet_RAMMERLIKE2_H_
#define _LSTMNet_RAMMERLIKE2_H_

#include "LstmCompute.h"
#include "RammerLikeArgs.h"

namespace mica::experiments::lstm {

template <size_t t_num_step, size_t t_num_layer, size_t t_batch_size,
          size_t t_hidden_size>
class LstmRammerLike2Net : public LstmCompute {
  public:
    LstmRammerLike2Net(size_t x = 0, size_t y = 0, size_t z = 0, size_t k = 0)
        : LstmCompute(t_num_step, t_num_layer, t_batch_size, t_hidden_size) {}

    void init(const float *input[], const float *init_state[], const float *W[],
              const float *U[], const float *bias[]) final;

    void finalize() final;

    void compute(const float *input[], const float *init_state[],
                 const float *W[], const float *U[], const float *bias[]) final;

    const float *getOutput() final;

  private:
    void initGraph();

    bool hasFetchOutput{false};
    cudaStream_t stream;
    RammerLikeCellInput<t_hidden_size> *cell_inputs;
    RammerLikeCellOutput *cell_outputs;
    RammerLikeCellModel<t_hidden_size> *models;

    float4 *input_dev, *output_host;
    float4 *state_c_dev[t_num_layer * t_num_step];
    float4 *state_h_dev[(t_num_layer + 1) * t_num_step];
};

} // namespace mica::experiments::lstm

#endif