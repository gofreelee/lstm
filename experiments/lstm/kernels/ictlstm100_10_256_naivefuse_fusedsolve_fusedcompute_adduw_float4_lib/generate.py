from os import close


def genarate_waves_defination():
     for i in range(109):
        fd = open("wave_" + str(i) + ".cu", "w+")
        fd.write('#include "kernels/include/lstmlib.cuh"\n')
        if i <= 9:
            # for j in range(i + 1):
            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 3) {\n")
            for m in range(i + 1):
                fd.writelines("case " + str(m) + ":")
                fd.writelines("call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(" + 
                str(i - m) + "* LstmScaleParams::kCellNumber10 + " + str(m) +", " +
                str(m) + ", " +
                str(i - m) + "* LstmScaleParams::kCellNumber10 + " + str(m) +", " +
                "LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);")
                fd.writelines("break;")
            fd.write("}\n")
            fd.write("}")
           
        if i > 9 and i <= 99:
            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 3) {\n")
            index = 0
            while index < 10:
                fd.writelines("case " + str(index) + ":")
                fd.writelines("call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(" + 
                str(i - index) + "* LstmScaleParams::kCellNumber10 + " + str(index) +", " +
                str(index) + ", " +
                str(i - index) + "* LstmScaleParams::kCellNumber10 + " + str(index) +", " +
                "LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);")
                fd.writelines("break;")
                index = index + 1
            fd.write("}\n")
            fd.write("}")
           
        if i > 99:
            fd.write(
                "__global__ void __launch_bounds__(256, 1)wave"+str(i)+"(" + "WaveInputParams *__restrict__ input, WaveModelParams *__restrict__ model,WaveOutputParams *__restrict__ output")
            fd.write("){")
            fd.write("switch (blockIdx.x >> 3) {\n")
            for m in range(108 - i + 1):
                fd.writelines("case " + str(m) + ":")
                fd.writelines("call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(" + 
                str(99 - m) + "* LstmScaleParams::kCellNumber10 + " + str(i - 99 + m) +", " +
                str(i - 99 + m) + ", " +
                str(99 - m) + "* LstmScaleParams::kCellNumber10 + " + str(i - 99 + m) +", " +
                "LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256, LstmScaleParams::kInputSize256, LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);")
                fd.writelines("break;")
            fd.write("}\n")
            fd.write("}")
        fd.close()





genarate_waves_defination()