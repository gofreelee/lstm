def generate_waves():
    for i in range(107):
        fd = open("wave_" + str(i) + ".cu", "w+")
        fd.write('#include "kernels/include/lstmlib.cuh"\n')
        if i <= 7:
            # for j in range(i + 1):            
            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave_compute_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 2) {\n")
            for m in range(i + 1):
                fd.writelines("case " + str(m) + ":")
                fd.writelines("call_onekernel_compute_naivefuse(" + 
                str(i - m) + "* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + " + str(m) +", " +
                str(m) + ", " +
                str(i - m) + "* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + " + str(m) +", " +
                "LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);")
                fd.writelines("break;")
            fd.write("}\n")
            fd.write("}")

            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave_solve_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 2) {\n")
            for m in range(i + 1):
                fd.writelines("case " + str(m) + ":")
                fd.writelines("call_onekernel_solve_naivefuse(" + 
                str(i - m) + "* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + " + str(m) +", " +
                str(m) + ", " +
                str(i - m) + "* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + " + str(m) +", " +
                "LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);")
                fd.writelines("break;")
            fd.write("}\n")
            fd.write("}")
        if i > 7 and i <= 99:
            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave_compute_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 2) {\n")
            index = 0
            while index < 10:
                fd.writelines("case " + str(index) + ":")
                fd.writelines("call_onekernel_compute_naivefuse(" + 
                str(i - index) + "* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + " + str(index) +", " +
                str(index) + ", " +
                str(i - index) + "* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + " + str(index) +", " +
                "LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);")
                fd.writelines("break;")
                index = index + 1
            fd.write("}\n")
            fd.write("}")

            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave_solve_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 2) {\n")
            index = 0
            while index < 10:
                fd.writelines("case " + str(index) + ":")
                fd.writelines("call_onekernel_solve_naivefuse(" + 
                str(i - index) + "* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + " + str(index) +", " +
                str(index) + ", " +
                str(i - index) + "* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + " + str(index) +", " +
                "LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);")
                fd.writelines("break;")
                index = index + 1
            fd.write("}\n")
            fd.write("}")

        if i > 99:

            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave_compute_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 2) {\n")
            for m in range(108 - i + 1):
                fd.writelines("case " + str(m) + ":")
                fd.writelines("call_onekernel_compute_naivefuse(" + 
                str(99 - m) + "* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + " + str(i - 99 + m) +", " +
                str(i - 99 + m) + ", " +
                str(99 - m) + "* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + " + str(i - 99 + m) +", " +
                "LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);")
                fd.writelines("break;")
            fd.write("}\n")
            fd.write("}")


            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave_solve_"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 2) {\n")
            for m in range(108 - i + 1):
                fd.writelines("case " + str(m) + ":")
                fd.writelines("call_onekernel_solve_naivefuse(" + 
                str(99 - m) + "* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + " + str(i - 99 + m) +", " +
                str(i - 99 + m) + ", " +
                str(99 - m) + "* LstmScaleParams::kSeq2SeqEncodeCellNumber8 + " + str(i - 99 + m) +", " +
                "LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);")
                fd.writelines("break;")
            fd.write("}\n")
            fd.write("}")

    for i in range(120):
        fd = open("wave_" + str(i + 107) + ".cu", "w+")
        fd.write('#include "kernels/include/lstmlib.cuh"\n')
        fd.write(
                "__global__ void __launch_bounds__(256, 1)wave_compute_"+str(i + 107)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
        fd.write("){")
        fd.writelines("call_onekernel_compute_naivefuse(" + 
        str(int(i / 4)) + "* LstmScaleParams::kSeq2SeqDecodeCellNumber4 + " + str(int(i % 4)) + "+ LstmScaleParams::kSeq2SeqEncodeCellNumber8 * LstmScaleParams::kSeq2SeqEncodeTimestep100"   +", " +
        str(int(i % 4)) + "+ LstmScaleParams::kSeq2SeqEncodeCellNumber8"  + ", " +
        str(int(i / 4)) + "* LstmScaleParams::kSeq2SeqDecodeCellNumber4 + " + str(int(i % 4)) + "+ LstmScaleParams::kSeq2SeqEncodeCellNumber8 * LstmScaleParams::kSeq2SeqEncodeTimestep100"   +", " +
        "LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);")
        fd.write("}\n")

        fd.write(
                "__global__ void __launch_bounds__(256, 1)wave_solve_"+str(i + 107)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
        fd.write("){")
        fd.writelines("call_onekernel_solve_naivefuse(" + 
        str(int(i / 4)) + "* LstmScaleParams::kSeq2SeqDecodeCellNumber4 + " + str(int(i % 4)) + "+ LstmScaleParams::kSeq2SeqEncodeCellNumber8 * LstmScaleParams::kSeq2SeqEncodeTimestep100"   +", " +
        str(int(i % 4)) + "+ LstmScaleParams::kSeq2SeqEncodeCellNumber8"  + ", " +
        str(int(i / 4)) + "* LstmScaleParams::kSeq2SeqDecodeCellNumber4 + " + str(int(i % 4)) + "+ LstmScaleParams::kSeq2SeqEncodeCellNumber8 * LstmScaleParams::kSeq2SeqEncodeTimestep100"   +", " +
        "LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize128, LstmScaleParams::kInputSize128, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask3);")
        fd.write("}\n")
    fd.close()


generate_waves()