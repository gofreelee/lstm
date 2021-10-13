

__global__  void Dot_float_float_float_cuda_Dot_8157_block_kernel(float* input0, float* input1, float* output0, int thread_id, int block_id, char *shared_buffer)
{
    if (thread_id >= 256){
        return;
    }
    const dim3 gridDim(8, 1, 1);
    const dim3 blockDim(256, 1, 1);
    const dim3 blockIdx(block_id, 0, 0);
    {
        {
            int warp_id = threadIdx.x >> 5;
            int lane_id = threadIdx.x & 31;
            int col_id = blockIdx.x * blockDim.x / 4 + lane_id;
            if (col_id < 256)
            {
                float val = 0;
                int k_start = warp_id * 32;
                int k_end = (warp_id + 1) * 32;
                for (int i = k_start; i < k_end; i++)
                {
                    val = fma(input0[i], input1[i * 256 + col_id], val);
                }
                if (warp_id == 0)
                {
                    output0[col_id]=0;
                }
                __syncthreads();
                atomicAdd(output0 + col_id, val);
            }

        }

    }

}
