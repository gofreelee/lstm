def generate_structparam_function_define():
    fd = open("wavefunction.cc", "w+")
    for i in range(107):
        if i <= 7:
            # for j in range(i + 1):
            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 2) {\n")
            for m in range(i + 1):
                fd.writelines("case " + str(m) + ":")
                fd.writelines("call_onekernel_seq2seq_enc(" + str(m) + ", " + str(i - m) + ");")
                fd.writelines("break;")
            fd.write("}\n")
            fd.write("}")
        if i > 7 and i <= 99:
            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 2) {\n")
            index = 0
            while index < 8:
                fd.writelines("case " + str(index) + ":")
                fd.writelines("call_onekernel_seq2seq_enc(" + str(index) + ", " + str(i - index) + ");")
                fd.writelines("break;")
                index = index + 1
            fd.write("}\n")
            fd.write("}")
        if i > 99:
            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 2) {\n")
            for m in range(106 - i + 1):
                fd.writelines("case " + str(m) + ":")
                fd.writelines("call_onekernel_seq2seq_enc(" + str(i - 99 + m) + ", " + str(99 - m) + ");")
                fd.writelines("break;")
            fd.write("}\n")
            fd.write("}")

    for i in range(120):
        fd.write(
                "__global__ void __launch_bounds__(256, 1)seq2seq_dec_wave"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
        fd.write("){")
        fd.writelines("call_onekernel_seq2seq_dec(" + str(i % 4) + "," + str(int(i / 4)) + ");")
        fd.write("}\n")

def generate_function_call_struct_params():
    fd = open("function_call_struct_params.cc", "w+")
    for i in range(107):
        if i <= 7:
            # for j in range(i + 1):
            fd.write("cudaLaunchKernel((void *)wave_compute_"+str(i)+", dim3("+str(4 * (i + 1))+"), dim3(256), (void **)arg_s"+", 0,stream);")
            fd.write("cudaLaunchKernel((void *)wave_solve_"+str(i)+", dim3("+str(4 * (i + 1))+"), dim3(256), (void **)arg_s"+", 0,stream);")
            fd.write("\n\n")

        if i > 7 and i <= 99:
            fd.write("cudaLaunchKernel((void *)wave_compute_"+str(i)+", dim3(32), dim3(256), (void **)arg_s"+", 0,stream);")
            fd.write("cudaLaunchKernel((void *)wave_solve_"+str(i)+", dim3(32), dim3(256), (void **)arg_s"+", 0,stream);")
            fd.write("\n\n")


        if i > 99:
            fd.write("cudaLaunchKernel((void *)wave_compute_"+str(i)+", dim3("+str(4 * (107 - i))+"), dim3(256), (void **)arg_s"+", 0,stream);")
            fd.write("cudaLaunchKernel((void *)wave_solve_"+str(i)+", dim3("+str(4 * (107 - i))+"), dim3(256), (void **)arg_s"+", 0,stream);")

            fd.write("\n\n")
    for i in range(120):
        fd.write("cudaLaunchKernel((void *)wave_compute_" + str(i + 107) + ", dim3(4), dim3(256), (void **)arg_s" + ", 0,stream);")
        fd.write("cudaLaunchKernel((void *)wave_solve_" + str(i + 107) + ", dim3(4), dim3(256), (void **)arg_s" + ", 0,stream);")


def function_declaraion():
    fd = open("function_declaration.cc", "w+")
    for i in range(107):
            # for j in range(i + 1):
            fd.write(
                " void wave_compute_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write(");") 

            fd.write(
                " void wave_solve_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write(");") 
    for i in range(120):
        fd.write(
                "void wave_compute_"+str(i + 107)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
        fd.write(");")

        fd.write(
                "void wave_solve_"+str(i + 107)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
        fd.write(");")


function_declaraion()