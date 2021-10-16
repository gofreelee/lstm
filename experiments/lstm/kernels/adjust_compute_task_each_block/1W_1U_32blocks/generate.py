def genarate_function_call():
    fd = open("call.txt", "w+")
    for i in range(109):
        if i < 10:
            fd.write("cudaLaunchKernel((void *)wave_compute_" + str(i) + ", dim3("+ str(32 * (i + 1)) +"), dim3(256), (void **)arg_s, 0,stream);\n")
            fd.write("cudaLaunchKernel((void *)wave_solve_" + str(i) + ", dim3("+ str(8 * (i + 1)) +"), dim3(256), (void **)arg_s, 0,stream);\n")
        if i >= 10 and i <= 99:
            fd.write("cudaLaunchKernel((void *)wave_compute_" + str(i) + ", dim3(320), dim3(256), (void **)arg_s, 0,stream);\n")
            fd.write("cudaLaunchKernel((void *)wave_solve_" + str(i) + ", dim3(80), dim3(256), (void **)arg_s, 0,stream);\n")
        if i >99 and i < 109:
            fd.write("cudaLaunchKernel((void *)wave_compute_" + str(i) + ", dim3("+ str(32 * (109 - i)) +"), dim3(256), (void **)arg_s, 0,stream);\n")
            fd.write("cudaLaunchKernel((void *)wave_solve_" + str(i) + ", dim3("+ str(8 * (109 - i)) +"), dim3(256), (void **)arg_s, 0,stream);\n")

def generate_function_call_seq2seq():
    fd = open("seq2seq.cc", "w+")
    for i in range(107):
        if i < 7:
           fd.write("cudaLaunchKernel((void *)seq2seq_enc_wave_compute_" + str(i) + ", dim3("+ str(4 * (i + 1)) +"), dim3(256), (void **)arg_s, 0,stream);\n")
           fd.write("cudaLaunchKernel((void *)seq2seq_enc_wave_solve_" + str(i) + ", dim3("+ str(4 * (i + 1)) +"), dim3(256), (void **)arg_s, 0,stream);\n")
        if i >= 7 and i <= 99:
           fd.write("cudaLaunchKernel((void *)seq2seq_enc_wave_compute_" + str(i) + ", dim3(32), dim3(256), (void **)arg_s, 0,stream);\n")
           fd.write("cudaLaunchKernel((void *)seq2seq_enc_wave_solve_" + str(i) + ", dim3(32), dim3(256), (void **)arg_s, 0,stream);\n")
        if i > 99 and i < 107:
           fd.write("cudaLaunchKernel((void *)seq2seq_enc_wave_compute_" + str(i) + ", dim3("+ str(4 * (107 - i)) +"), dim3(256), (void **)arg_s, 0,stream);\n")
           fd.write("cudaLaunchKernel((void *)seq2seq_enc_wave_solve_" + str(i) + ", dim3("+ str(4 * (107 - i)) +"), dim3(256), (void **)arg_s, 0,stream);\n")

    for i in range(120):
           fd.write("cudaLaunchKernel((void *)seq2seq_dec_wave_compute_" + str(i) + ", dim3(4), dim3(256), (void **)arg_s, 0,stream);\n")
           fd.write("cudaLaunchKernel((void *)seq2seq_dec_wave_solve_" + str(i) + ", dim3(4), dim3(256), (void **)arg_s, 0,stream);\n")




def generate_1W_1U_32blocks_waves():
    for i in range(109):
        fd = open("wave_" + str(i) + ".cu", "w+")
        fd.write('#include "LstmExperimentLib.h"\n')
        if i <= 9:
            # for j in range(i + 1):
            fd.write(
                "__global__ void __launch_bounds__(256, 4) wave_compute_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 3) {\n")
            for m in range(i + 1):
                fd.writelines("case " + str(m) + ":")
                fd.writelines("call_onekernel_compute_wi_uh_0(" + str(m) + ", " + str(i - m) + ");")
                fd.writelines("break;")
            for m in range(i + 1):
                fd.writelines("case " + str(m + i + 1) + ":")
                fd.writelines("call_onekernel_compute_wi_uh_1(" + str(m) + ", " + str(i - m) + ");")
                fd.writelines("break;")
            for m in range(i + 1):
                fd.writelines("case " + str(m + 2*(i + 1)) + ":")
                fd.writelines("call_onekernel_compute_wi_uh_2(" + str(m) + ", " + str(i - m) + ");")
                fd.writelines("break;")
            for m in range(i + 1):
                fd.writelines("case " + str(m + 3*(i + 1)) + ":")
                fd.writelines("call_onekernel_compute_wi_uh_3(" + str(m) + ", " + str(i - m) + ");")
                fd.writelines("break;")
            fd.write("}\n")
            fd.write("}")
            #
            fd.write(
                "__global__ void __launch_bounds__(256, 4) wave_solve_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 3) {\n")
            for m in range(i + 1):
                fd.writelines("case " + str(m) + ":")
                fd.writelines("call_onekernel_solve(" + str(m) + ", " + str(i - m) + ");")
                fd.writelines("break;")
            fd.write("}\n")
            fd.write("}")
        if i > 9 and i <= 99:
            fd.write(
                "__global__ void __launch_bounds__(256, 4) wave_compute_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 3) {\n")
            index = 0
            while index < 10:
                fd.writelines("case " + str(index) + ":")
                fd.writelines("call_onekernel_compute_wi_uh_0(" + str(index) + ", " + str(i - index) + ");")
                fd.writelines("break;")
                index = index + 1
            
            index = 0
            while index < 10:
                fd.writelines("case " + str(index + 10) + ":")
                fd.writelines("call_onekernel_compute_wi_uh_1(" + str(index) + ", " + str(i - index) + ");")
                fd.writelines("break;")
                index = index + 1

            index = 0
            while index < 10:
                fd.writelines("case " + str(index + 20) + ":")
                fd.writelines("call_onekernel_compute_wi_uh_2(" + str(index) + ", " + str(i - index) + ");")
                fd.writelines("break;")
                index = index + 1

            index = 0
            while index < 10:
                fd.writelines("case " + str(index + 30) + ":")
                fd.writelines("call_onekernel_compute_wi_uh_3(" + str(index) + ", " + str(i - index) + ");")
                fd.writelines("break;")
                index = index + 1
            fd.write("}\n")
            fd.write("}")
            #
            fd.write(
                "__global__ void __launch_bounds__(256, 4) wave_solve_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 3) {\n")
            index = 0
            while index < 10:
                fd.writelines("case " + str(index) + ":")
                fd.writelines("call_onekernel_solve(" + str(index) + ", " + str(i - index) + ");")
                fd.writelines("break;")
                index = index + 1
            fd.write("}\n")
            fd.write("}")
        if i > 99:
            fd.write(
                "__global__ void __launch_bounds__(256, 4) wave_compute_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 3) {\n")
            for m in range(108 - i + 1):
                fd.writelines("case " + str(m) + ":")
                fd.writelines("call_onekernel_compute_wi_uh_0(" + str(i - 99 + m) + ", " + str(99 - m) + ");")
                fd.writelines("break;")
            for m in range(108 - i + 1):
                fd.writelines("case " + str(m + 108 - i + 1) + ":")
                fd.writelines("call_onekernel_compute_wi_uh_1(" + str(i - 99 + m) + ", " + str(99 - m) + ");")
                fd.writelines("break;")
            
            for m in range(108 - i + 1):
                fd.writelines("case " + str(m + 2 * (108 - i + 1)) + ":")
                fd.writelines("call_onekernel_compute_wi_uh_2(" + str(i - 99 + m) + ", " + str(99 - m) + ");")
                fd.writelines("break;")
            for m in range(108 - i + 1):
                fd.writelines("case " + str(m + 3 * (108 - i + 1)) + ":")
                fd.writelines("call_onekernel_compute_wi_uh_3(" + str(i - 99 + m) + ", " + str(99 - m) + ");")
                fd.writelines("break;")
            fd.write("}\n")
            fd.write("}")
            #
            fd.write(
                "__global__ void __launch_bounds__(256, 4) wave_solve_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 3) {\n")
            for m in range(108 - i + 1):
                fd.writelines("case " + str(m) + ":")
                fd.writelines("call_onekernel_solve(" + str(i - 99 + m) + ", " + str(99 - m) + ");")
                fd.writelines("break;")
          
            fd.write("}\n")
            fd.write("}")
        fd.close()
generate_1W_1U_32blocks_waves()
