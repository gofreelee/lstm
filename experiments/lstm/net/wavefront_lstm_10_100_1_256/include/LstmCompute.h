#ifndef _LSTMCOMPUTE_H_
#define _LSTMCOMPUTE_H_

#include <cstdlib>

namespace mica::experiments::lstm {

class LstmCompute {
  protected:
    LstmCompute(size_t num_step, size_t num_layer, size_t batch_size,
                size_t hidden_size)
        : num_step(num_step), num_layer(num_layer), batch_size(batch_size),
          hidden_size(hidden_size) {}

    size_t num_step, num_layer;
    size_t batch_size, hidden_size;

  public:
    virtual void init(const float *input[], const float *init_state[],
                      const float *W[], const float *U[],
                      const float *bias[]) = 0;

    virtual void finalize() = 0;

    virtual void compute(const float *input[], const float *init_state[],
                         const float *W[], const float *U[],
                         const float *bias[]) = 0;

    virtual const float *getOutput() = 0;
};
} // namespace mica::experiments::lstm

#endif