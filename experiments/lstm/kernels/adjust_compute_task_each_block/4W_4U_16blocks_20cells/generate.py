def genarate_function_call():
    fd = open("call.txt", "w+")
    for i in range(119):
        if i < 10:
            fd.write("cudaLaunchKernel((void *)wave_compute_" + str(i) + ", dim3("+ str(16 * (i + 1)) +"), dim3(256), (void **)arg_s, 0,stream);\n")
            fd.write("cudaLaunchKernel((void *)wave_solve_" + str(i) + ", dim3("+ str(8 * (i + 1)) +"), dim3(256), (void **)arg_s, 0,stream);\n")
        if i >= 10 and i <= 99:
            fd.write("cudaLaunchKernel((void *)wave_compute_" + str(i) + ", dim3(320), dim3(256), (void **)arg_s, 0,stream);\n")
            fd.write("cudaLaunchKernel((void *)wave_solve_" + str(i) + ", dim3(160), dim3(256), (void **)arg_s, 0,stream);\n")
        if i >99 and i < 119:
            fd.write("cudaLaunchKernel((void *)wave_compute_" + str(i) + ", dim3("+ str(16 * (119 - i)) +"), dim3(256), (void **)arg_s, 0,stream);\n")
            fd.write("cudaLaunchKernel((void *)wave_solve_" + str(i) + ", dim3("+ str(8 * (119 - i)) +"), dim3(256), (void **)arg_s, 0,stream);\n")




def generate_waves_4W_4U_16blocks():
    for i in range(119):
        fd = open("wave_"+  str(i) + ".cu", "w+")
        fd.write('#include "LstmExperimentLib.h"\n')
        if i <= 19:
            # for j in range(i + 1):
            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave_compute_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 3) {\n")
            for m in range(i + 1):
                fd.writelines("case " + str(m) + ":")
                fd.writelines("call_onekernel_compute_wi(" + str(m) + ", " + str(i - m) + ");")
                fd.writelines("break;")
            for m in range(i + 1):
                fd.writelines("case " + str(m + i + 1) + ":")
                fd.writelines("call_onekernel_compute_uh(" + str(m) + ", " + str(i - m) + ");")
                fd.writelines("break;")
            fd.write("}\n")
            fd.write("}")
            #
            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave_solve_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 3) {\n")
            for m in range(i + 1):
                fd.writelines("case " + str(m) + ":")
                fd.writelines("call_onekernel_solve(" + str(m) + ", " + str(i - m) + ");")
                fd.writelines("break;")
            fd.write("}\n")
            fd.write("}")
        if i > 19 and i <= 99:
            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave_compute_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 3) {\n")
            index = 0
            while index < 20:
                fd.writelines("case " + str(index) + ":")
                fd.writelines("call_onekernel_compute_wi(" + str(index) + ", " + str(i - index) + ");")
                fd.writelines("break;")
                index = index + 1
            
            index = 0
            while index < 20:
                fd.writelines("case " + str(index + 20) + ":")
                fd.writelines("call_onekernel_compute_uh(" + str(index) + ", " + str(i - index) + ");")
                fd.writelines("break;")
                index = index + 1
            fd.write("}\n")
            fd.write("}")
            #
            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave_solve_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 3) {\n")
            index = 0
            while index < 20:
                fd.writelines("case " + str(index) + ":")
                fd.writelines("call_onekernel_solve(" + str(index) + ", " + str(i - index) + ");")
                fd.writelines("break;")
                index = index + 1
            fd.write("}\n")
            fd.write("}")
        if i > 99:
            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave_compute_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 3) {\n")
            for m in range(118 - i + 1):
                fd.writelines("case " + str(m) + ":")
                fd.writelines("call_onekernel_compute_wi(" + str(i - 99 + m) + ", " + str(99 - m) + ");")
                fd.writelines("break;")
            for m in range(118 - i + 1):
                fd.writelines("case " + str(m + 118 - i + 1) + ":")
                fd.writelines("call_onekernel_compute_uh(" + str(i - 99 + m) + ", " + str(99 - m) + ");")
                fd.writelines("break;")
            fd.write("}\n")
            fd.write("}")
            #
            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave_solve_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 3) {\n")
            for m in range(118 - i + 1):
                fd.writelines("case " + str(m) + ":")
                fd.writelines("call_onekernel_solve(" + str(i - 99 + m) + ", " + str(99 - m) + ");")
                fd.writelines("break;")
         
            fd.write("}\n")
            fd.write("}")


genarate_function_call()