template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok_1(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {

    const int step = blockIdx.x >> 5;
    const int idx = (blockIdx.x & 0x1f) >> 3;
    WMulData(0, step);
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(256, 1)
    ok0(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {

    point_to_point_func(&inputs[0], &models[0], &outputs[0]);
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok1(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 1);
        break;
    case 1:
        ok_hasu_update_c(1, 0);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok2(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 2);
        break;
    case 1:
        ok_update_c(1, 1);
        break;
    case 2:
        ok_hasu_update_c(2, 0);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok3(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 3);
        break;
    case 1:
        ok_update_c(1, 2);
        break;
    case 2:
        ok_update_c(2, 1);
        break;
    case 3:
        ok_hasu_update_c(3, 0);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok4(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 4);
        break;
    case 1:
        ok_update_c(1, 3);
        break;
    case 2:
        ok_update_c(2, 2);
        break;
    case 3:
        ok_update_c(3, 1);
        break;
    case 4:
        ok_hasu_update_c(4, 0);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok5(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 5);
        break;
    case 1:
        ok_update_c(1, 4);
        break;
    case 2:
        ok_update_c(2, 3);
        break;
    case 3:
        ok_update_c(3, 2);
        break;
    case 4:
        ok_update_c(4, 1);
        break;
    case 5:
        ok_hasu_update_c(5, 0);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok6(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 6);
        break;
    case 1:
        ok_update_c(1, 5);
        break;
    case 2:
        ok_update_c(2, 4);
        break;
    case 3:
        ok_update_c(3, 3);
        break;
    case 4:
        ok_update_c(4, 2);
        break;
    case 5:
        ok_update_c(5, 1);
        break;
    case 6:
        ok_hasu_update_c(6, 0);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok7(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 7);
        break;
    case 1:
        ok_update_c(1, 6);
        break;
    case 2:
        ok_update_c(2, 5);
        break;
    case 3:
        ok_update_c(3, 4);
        break;
    case 4:
        ok_update_c(4, 3);
        break;
    case 5:
        ok_update_c(5, 2);
        break;
    case 6:
        ok_update_c(6, 1);
        break;
    case 7:
        ok_hasu_update_c(7, 0);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok8(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 8);
        break;
    case 1:
        ok_update_c(1, 7);
        break;
    case 2:
        ok_update_c(2, 6);
        break;
    case 3:
        ok_update_c(3, 5);
        break;
    case 4:
        ok_update_c(4, 4);
        break;
    case 5:
        ok_update_c(5, 3);
        break;
    case 6:
        ok_update_c(6, 2);
        break;
    case 7:
        ok_update_c(7, 1);
        break;
    case 8:
        ok_hasu_update_c(8, 0);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok9(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
        RammerLikeCellModel<t_hidden_size> *__restrict__ models,
        RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 9);
        break;
    case 1:
        ok_update_c(1, 8);
        break;
    case 2:
        ok_update_c(2, 7);
        break;
    case 3:
        ok_update_c(3, 6);
        break;
    case 4:
        ok_update_c(4, 5);
        break;
    case 5:
        ok_update_c(5, 4);
        break;
    case 6:
        ok_update_c(6, 3);
        break;
    case 7:
        ok_update_c(7, 2);
        break;
    case 8:
        ok_update_c(8, 1);
        break;
    case 9:
        ok_hasu_update_c(9, 0);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok10(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 10);
        break;
    case 1:
        ok_update_c(1, 9);
        break;
    case 2:
        ok_update_c(2, 8);
        break;
    case 3:
        ok_update_c(3, 7);
        break;
    case 4:
        ok_update_c(4, 6);
        break;
    case 5:
        ok_update_c(5, 5);
        break;
    case 6:
        ok_update_c(6, 4);
        break;
    case 7:
        ok_update_c(7, 3);
        break;
    case 8:
        ok_update_c(8, 2);
        break;
    case 9:
        ok_update_c(9, 1);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok11(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 11);
        break;
    case 1:
        ok_update_c(1, 10);
        break;
    case 2:
        ok_update_c(2, 9);
        break;
    case 3:
        ok_update_c(3, 8);
        break;
    case 4:
        ok_update_c(4, 7);
        break;
    case 5:
        ok_update_c(5, 6);
        break;
    case 6:
        ok_update_c(6, 5);
        break;
    case 7:
        ok_update_c(7, 4);
        break;
    case 8:
        ok_update_c(8, 3);
        break;
    case 9:
        ok_update_c(9, 2);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok12(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 12);
        break;
    case 1:
        ok_update_c(1, 11);
        break;
    case 2:
        ok_update_c(2, 10);
        break;
    case 3:
        ok_update_c(3, 9);
        break;
    case 4:
        ok_update_c(4, 8);
        break;
    case 5:
        ok_update_c(5, 7);
        break;
    case 6:
        ok_update_c(6, 6);
        break;
    case 7:
        ok_update_c(7, 5);
        break;
    case 8:
        ok_update_c(8, 4);
        break;
    case 9:
        ok_update_c(9, 3);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok13(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 13);
        break;
    case 1:
        ok_update_c(1, 12);
        break;
    case 2:
        ok_update_c(2, 11);
        break;
    case 3:
        ok_update_c(3, 10);
        break;
    case 4:
        ok_update_c(4, 9);
        break;
    case 5:
        ok_update_c(5, 8);
        break;
    case 6:
        ok_update_c(6, 7);
        break;
    case 7:
        ok_update_c(7, 6);
        break;
    case 8:
        ok_update_c(8, 5);
        break;
    case 9:
        ok_update_c(9, 4);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok14(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 14);
        break;
    case 1:
        ok_update_c(1, 13);
        break;
    case 2:
        ok_update_c(2, 12);
        break;
    case 3:
        ok_update_c(3, 11);
        break;
    case 4:
        ok_update_c(4, 10);
        break;
    case 5:
        ok_update_c(5, 9);
        break;
    case 6:
        ok_update_c(6, 8);
        break;
    case 7:
        ok_update_c(7, 7);
        break;
    case 8:
        ok_update_c(8, 6);
        break;
    case 9:
        ok_update_c(9, 5);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok15(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 15);
        break;
    case 1:
        ok_update_c(1, 14);
        break;
    case 2:
        ok_update_c(2, 13);
        break;
    case 3:
        ok_update_c(3, 12);
        break;
    case 4:
        ok_update_c(4, 11);
        break;
    case 5:
        ok_update_c(5, 10);
        break;
    case 6:
        ok_update_c(6, 9);
        break;
    case 7:
        ok_update_c(7, 8);
        break;
    case 8:
        ok_update_c(8, 7);
        break;
    case 9:
        ok_update_c(9, 6);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok16(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 16);
        break;
    case 1:
        ok_update_c(1, 15);
        break;
    case 2:
        ok_update_c(2, 14);
        break;
    case 3:
        ok_update_c(3, 13);
        break;
    case 4:
        ok_update_c(4, 12);
        break;
    case 5:
        ok_update_c(5, 11);
        break;
    case 6:
        ok_update_c(6, 10);
        break;
    case 7:
        ok_update_c(7, 9);
        break;
    case 8:
        ok_update_c(8, 8);
        break;
    case 9:
        ok_update_c(9, 7);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok17(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 17);
        break;
    case 1:
        ok_update_c(1, 16);
        break;
    case 2:
        ok_update_c(2, 15);
        break;
    case 3:
        ok_update_c(3, 14);
        break;
    case 4:
        ok_update_c(4, 13);
        break;
    case 5:
        ok_update_c(5, 12);
        break;
    case 6:
        ok_update_c(6, 11);
        break;
    case 7:
        ok_update_c(7, 10);
        break;
    case 8:
        ok_update_c(8, 9);
        break;
    case 9:
        ok_update_c(9, 8);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok18(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 18);
        break;
    case 1:
        ok_update_c(1, 17);
        break;
    case 2:
        ok_update_c(2, 16);
        break;
    case 3:
        ok_update_c(3, 15);
        break;
    case 4:
        ok_update_c(4, 14);
        break;
    case 5:
        ok_update_c(5, 13);
        break;
    case 6:
        ok_update_c(6, 12);
        break;
    case 7:
        ok_update_c(7, 11);
        break;
    case 8:
        ok_update_c(8, 10);
        break;
    case 9:
        ok_update_c(9, 9);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok19(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 19);
        break;
    case 1:
        ok_update_c(1, 18);
        break;
    case 2:
        ok_update_c(2, 17);
        break;
    case 3:
        ok_update_c(3, 16);
        break;
    case 4:
        ok_update_c(4, 15);
        break;
    case 5:
        ok_update_c(5, 14);
        break;
    case 6:
        ok_update_c(6, 13);
        break;
    case 7:
        ok_update_c(7, 12);
        break;
    case 8:
        ok_update_c(8, 11);
        break;
    case 9:
        ok_update_c(9, 10);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok20(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 20);
        break;
    case 1:
        ok_update_c(1, 19);
        break;
    case 2:
        ok_update_c(2, 18);
        break;
    case 3:
        ok_update_c(3, 17);
        break;
    case 4:
        ok_update_c(4, 16);
        break;
    case 5:
        ok_update_c(5, 15);
        break;
    case 6:
        ok_update_c(6, 14);
        break;
    case 7:
        ok_update_c(7, 13);
        break;
    case 8:
        ok_update_c(8, 12);
        break;
    case 9:
        ok_update_c(9, 11);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok21(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 21);
        break;
    case 1:
        ok_update_c(1, 20);
        break;
    case 2:
        ok_update_c(2, 19);
        break;
    case 3:
        ok_update_c(3, 18);
        break;
    case 4:
        ok_update_c(4, 17);
        break;
    case 5:
        ok_update_c(5, 16);
        break;
    case 6:
        ok_update_c(6, 15);
        break;
    case 7:
        ok_update_c(7, 14);
        break;
    case 8:
        ok_update_c(8, 13);
        break;
    case 9:
        ok_update_c(9, 12);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok22(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 22);
        break;
    case 1:
        ok_update_c(1, 21);
        break;
    case 2:
        ok_update_c(2, 20);
        break;
    case 3:
        ok_update_c(3, 19);
        break;
    case 4:
        ok_update_c(4, 18);
        break;
    case 5:
        ok_update_c(5, 17);
        break;
    case 6:
        ok_update_c(6, 16);
        break;
    case 7:
        ok_update_c(7, 15);
        break;
    case 8:
        ok_update_c(8, 14);
        break;
    case 9:
        ok_update_c(9, 13);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok23(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 23);
        break;
    case 1:
        ok_update_c(1, 22);
        break;
    case 2:
        ok_update_c(2, 21);
        break;
    case 3:
        ok_update_c(3, 20);
        break;
    case 4:
        ok_update_c(4, 19);
        break;
    case 5:
        ok_update_c(5, 18);
        break;
    case 6:
        ok_update_c(6, 17);
        break;
    case 7:
        ok_update_c(7, 16);
        break;
    case 8:
        ok_update_c(8, 15);
        break;
    case 9:
        ok_update_c(9, 14);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok24(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 24);
        break;
    case 1:
        ok_update_c(1, 23);
        break;
    case 2:
        ok_update_c(2, 22);
        break;
    case 3:
        ok_update_c(3, 21);
        break;
    case 4:
        ok_update_c(4, 20);
        break;
    case 5:
        ok_update_c(5, 19);
        break;
    case 6:
        ok_update_c(6, 18);
        break;
    case 7:
        ok_update_c(7, 17);
        break;
    case 8:
        ok_update_c(8, 16);
        break;
    case 9:
        ok_update_c(9, 15);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok25(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 25);
        break;
    case 1:
        ok_update_c(1, 24);
        break;
    case 2:
        ok_update_c(2, 23);
        break;
    case 3:
        ok_update_c(3, 22);
        break;
    case 4:
        ok_update_c(4, 21);
        break;
    case 5:
        ok_update_c(5, 20);
        break;
    case 6:
        ok_update_c(6, 19);
        break;
    case 7:
        ok_update_c(7, 18);
        break;
    case 8:
        ok_update_c(8, 17);
        break;
    case 9:
        ok_update_c(9, 16);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok26(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 26);
        break;
    case 1:
        ok_update_c(1, 25);
        break;
    case 2:
        ok_update_c(2, 24);
        break;
    case 3:
        ok_update_c(3, 23);
        break;
    case 4:
        ok_update_c(4, 22);
        break;
    case 5:
        ok_update_c(5, 21);
        break;
    case 6:
        ok_update_c(6, 20);
        break;
    case 7:
        ok_update_c(7, 19);
        break;
    case 8:
        ok_update_c(8, 18);
        break;
    case 9:
        ok_update_c(9, 17);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok27(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 27);
        break;
    case 1:
        ok_update_c(1, 26);
        break;
    case 2:
        ok_update_c(2, 25);
        break;
    case 3:
        ok_update_c(3, 24);
        break;
    case 4:
        ok_update_c(4, 23);
        break;
    case 5:
        ok_update_c(5, 22);
        break;
    case 6:
        ok_update_c(6, 21);
        break;
    case 7:
        ok_update_c(7, 20);
        break;
    case 8:
        ok_update_c(8, 19);
        break;
    case 9:
        ok_update_c(9, 18);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok28(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 28);
        break;
    case 1:
        ok_update_c(1, 27);
        break;
    case 2:
        ok_update_c(2, 26);
        break;
    case 3:
        ok_update_c(3, 25);
        break;
    case 4:
        ok_update_c(4, 24);
        break;
    case 5:
        ok_update_c(5, 23);
        break;
    case 6:
        ok_update_c(6, 22);
        break;
    case 7:
        ok_update_c(7, 21);
        break;
    case 8:
        ok_update_c(8, 20);
        break;
    case 9:
        ok_update_c(9, 19);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok29(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 29);
        break;
    case 1:
        ok_update_c(1, 28);
        break;
    case 2:
        ok_update_c(2, 27);
        break;
    case 3:
        ok_update_c(3, 26);
        break;
    case 4:
        ok_update_c(4, 25);
        break;
    case 5:
        ok_update_c(5, 24);
        break;
    case 6:
        ok_update_c(6, 23);
        break;
    case 7:
        ok_update_c(7, 22);
        break;
    case 8:
        ok_update_c(8, 21);
        break;
    case 9:
        ok_update_c(9, 20);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok30(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 30);
        break;
    case 1:
        ok_update_c(1, 29);
        break;
    case 2:
        ok_update_c(2, 28);
        break;
    case 3:
        ok_update_c(3, 27);
        break;
    case 4:
        ok_update_c(4, 26);
        break;
    case 5:
        ok_update_c(5, 25);
        break;
    case 6:
        ok_update_c(6, 24);
        break;
    case 7:
        ok_update_c(7, 23);
        break;
    case 8:
        ok_update_c(8, 22);
        break;
    case 9:
        ok_update_c(9, 21);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok31(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 31);
        break;
    case 1:
        ok_update_c(1, 30);
        break;
    case 2:
        ok_update_c(2, 29);
        break;
    case 3:
        ok_update_c(3, 28);
        break;
    case 4:
        ok_update_c(4, 27);
        break;
    case 5:
        ok_update_c(5, 26);
        break;
    case 6:
        ok_update_c(6, 25);
        break;
    case 7:
        ok_update_c(7, 24);
        break;
    case 8:
        ok_update_c(8, 23);
        break;
    case 9:
        ok_update_c(9, 22);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok32(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 32);
        break;
    case 1:
        ok_update_c(1, 31);
        break;
    case 2:
        ok_update_c(2, 30);
        break;
    case 3:
        ok_update_c(3, 29);
        break;
    case 4:
        ok_update_c(4, 28);
        break;
    case 5:
        ok_update_c(5, 27);
        break;
    case 6:
        ok_update_c(6, 26);
        break;
    case 7:
        ok_update_c(7, 25);
        break;
    case 8:
        ok_update_c(8, 24);
        break;
    case 9:
        ok_update_c(9, 23);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok33(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 33);
        break;
    case 1:
        ok_update_c(1, 32);
        break;
    case 2:
        ok_update_c(2, 31);
        break;
    case 3:
        ok_update_c(3, 30);
        break;
    case 4:
        ok_update_c(4, 29);
        break;
    case 5:
        ok_update_c(5, 28);
        break;
    case 6:
        ok_update_c(6, 27);
        break;
    case 7:
        ok_update_c(7, 26);
        break;
    case 8:
        ok_update_c(8, 25);
        break;
    case 9:
        ok_update_c(9, 24);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok34(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 34);
        break;
    case 1:
        ok_update_c(1, 33);
        break;
    case 2:
        ok_update_c(2, 32);
        break;
    case 3:
        ok_update_c(3, 31);
        break;
    case 4:
        ok_update_c(4, 30);
        break;
    case 5:
        ok_update_c(5, 29);
        break;
    case 6:
        ok_update_c(6, 28);
        break;
    case 7:
        ok_update_c(7, 27);
        break;
    case 8:
        ok_update_c(8, 26);
        break;
    case 9:
        ok_update_c(9, 25);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok35(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 35);
        break;
    case 1:
        ok_update_c(1, 34);
        break;
    case 2:
        ok_update_c(2, 33);
        break;
    case 3:
        ok_update_c(3, 32);
        break;
    case 4:
        ok_update_c(4, 31);
        break;
    case 5:
        ok_update_c(5, 30);
        break;
    case 6:
        ok_update_c(6, 29);
        break;
    case 7:
        ok_update_c(7, 28);
        break;
    case 8:
        ok_update_c(8, 27);
        break;
    case 9:
        ok_update_c(9, 26);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok36(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 36);
        break;
    case 1:
        ok_update_c(1, 35);
        break;
    case 2:
        ok_update_c(2, 34);
        break;
    case 3:
        ok_update_c(3, 33);
        break;
    case 4:
        ok_update_c(4, 32);
        break;
    case 5:
        ok_update_c(5, 31);
        break;
    case 6:
        ok_update_c(6, 30);
        break;
    case 7:
        ok_update_c(7, 29);
        break;
    case 8:
        ok_update_c(8, 28);
        break;
    case 9:
        ok_update_c(9, 27);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok37(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 37);
        break;
    case 1:
        ok_update_c(1, 36);
        break;
    case 2:
        ok_update_c(2, 35);
        break;
    case 3:
        ok_update_c(3, 34);
        break;
    case 4:
        ok_update_c(4, 33);
        break;
    case 5:
        ok_update_c(5, 32);
        break;
    case 6:
        ok_update_c(6, 31);
        break;
    case 7:
        ok_update_c(7, 30);
        break;
    case 8:
        ok_update_c(8, 29);
        break;
    case 9:
        ok_update_c(9, 28);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok38(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 38);
        break;
    case 1:
        ok_update_c(1, 37);
        break;
    case 2:
        ok_update_c(2, 36);
        break;
    case 3:
        ok_update_c(3, 35);
        break;
    case 4:
        ok_update_c(4, 34);
        break;
    case 5:
        ok_update_c(5, 33);
        break;
    case 6:
        ok_update_c(6, 32);
        break;
    case 7:
        ok_update_c(7, 31);
        break;
    case 8:
        ok_update_c(8, 30);
        break;
    case 9:
        ok_update_c(9, 29);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok39(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 39);
        break;
    case 1:
        ok_update_c(1, 38);
        break;
    case 2:
        ok_update_c(2, 37);
        break;
    case 3:
        ok_update_c(3, 36);
        break;
    case 4:
        ok_update_c(4, 35);
        break;
    case 5:
        ok_update_c(5, 34);
        break;
    case 6:
        ok_update_c(6, 33);
        break;
    case 7:
        ok_update_c(7, 32);
        break;
    case 8:
        ok_update_c(8, 31);
        break;
    case 9:
        ok_update_c(9, 30);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok40(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 40);
        break;
    case 1:
        ok_update_c(1, 39);
        break;
    case 2:
        ok_update_c(2, 38);
        break;
    case 3:
        ok_update_c(3, 37);
        break;
    case 4:
        ok_update_c(4, 36);
        break;
    case 5:
        ok_update_c(5, 35);
        break;
    case 6:
        ok_update_c(6, 34);
        break;
    case 7:
        ok_update_c(7, 33);
        break;
    case 8:
        ok_update_c(8, 32);
        break;
    case 9:
        ok_update_c(9, 31);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok41(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 41);
        break;
    case 1:
        ok_update_c(1, 40);
        break;
    case 2:
        ok_update_c(2, 39);
        break;
    case 3:
        ok_update_c(3, 38);
        break;
    case 4:
        ok_update_c(4, 37);
        break;
    case 5:
        ok_update_c(5, 36);
        break;
    case 6:
        ok_update_c(6, 35);
        break;
    case 7:
        ok_update_c(7, 34);
        break;
    case 8:
        ok_update_c(8, 33);
        break;
    case 9:
        ok_update_c(9, 32);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok42(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 42);
        break;
    case 1:
        ok_update_c(1, 41);
        break;
    case 2:
        ok_update_c(2, 40);
        break;
    case 3:
        ok_update_c(3, 39);
        break;
    case 4:
        ok_update_c(4, 38);
        break;
    case 5:
        ok_update_c(5, 37);
        break;
    case 6:
        ok_update_c(6, 36);
        break;
    case 7:
        ok_update_c(7, 35);
        break;
    case 8:
        ok_update_c(8, 34);
        break;
    case 9:
        ok_update_c(9, 33);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok43(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 43);
        break;
    case 1:
        ok_update_c(1, 42);
        break;
    case 2:
        ok_update_c(2, 41);
        break;
    case 3:
        ok_update_c(3, 40);
        break;
    case 4:
        ok_update_c(4, 39);
        break;
    case 5:
        ok_update_c(5, 38);
        break;
    case 6:
        ok_update_c(6, 37);
        break;
    case 7:
        ok_update_c(7, 36);
        break;
    case 8:
        ok_update_c(8, 35);
        break;
    case 9:
        ok_update_c(9, 34);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok44(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 44);
        break;
    case 1:
        ok_update_c(1, 43);
        break;
    case 2:
        ok_update_c(2, 42);
        break;
    case 3:
        ok_update_c(3, 41);
        break;
    case 4:
        ok_update_c(4, 40);
        break;
    case 5:
        ok_update_c(5, 39);
        break;
    case 6:
        ok_update_c(6, 38);
        break;
    case 7:
        ok_update_c(7, 37);
        break;
    case 8:
        ok_update_c(8, 36);
        break;
    case 9:
        ok_update_c(9, 35);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok45(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 45);
        break;
    case 1:
        ok_update_c(1, 44);
        break;
    case 2:
        ok_update_c(2, 43);
        break;
    case 3:
        ok_update_c(3, 42);
        break;
    case 4:
        ok_update_c(4, 41);
        break;
    case 5:
        ok_update_c(5, 40);
        break;
    case 6:
        ok_update_c(6, 39);
        break;
    case 7:
        ok_update_c(7, 38);
        break;
    case 8:
        ok_update_c(8, 37);
        break;
    case 9:
        ok_update_c(9, 36);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok46(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 46);
        break;
    case 1:
        ok_update_c(1, 45);
        break;
    case 2:
        ok_update_c(2, 44);
        break;
    case 3:
        ok_update_c(3, 43);
        break;
    case 4:
        ok_update_c(4, 42);
        break;
    case 5:
        ok_update_c(5, 41);
        break;
    case 6:
        ok_update_c(6, 40);
        break;
    case 7:
        ok_update_c(7, 39);
        break;
    case 8:
        ok_update_c(8, 38);
        break;
    case 9:
        ok_update_c(9, 37);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok47(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 47);
        break;
    case 1:
        ok_update_c(1, 46);
        break;
    case 2:
        ok_update_c(2, 45);
        break;
    case 3:
        ok_update_c(3, 44);
        break;
    case 4:
        ok_update_c(4, 43);
        break;
    case 5:
        ok_update_c(5, 42);
        break;
    case 6:
        ok_update_c(6, 41);
        break;
    case 7:
        ok_update_c(7, 40);
        break;
    case 8:
        ok_update_c(8, 39);
        break;
    case 9:
        ok_update_c(9, 38);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok48(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 48);
        break;
    case 1:
        ok_update_c(1, 47);
        break;
    case 2:
        ok_update_c(2, 46);
        break;
    case 3:
        ok_update_c(3, 45);
        break;
    case 4:
        ok_update_c(4, 44);
        break;
    case 5:
        ok_update_c(5, 43);
        break;
    case 6:
        ok_update_c(6, 42);
        break;
    case 7:
        ok_update_c(7, 41);
        break;
    case 8:
        ok_update_c(8, 40);
        break;
    case 9:
        ok_update_c(9, 39);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok49(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 49);
        break;
    case 1:
        ok_update_c(1, 48);
        break;
    case 2:
        ok_update_c(2, 47);
        break;
    case 3:
        ok_update_c(3, 46);
        break;
    case 4:
        ok_update_c(4, 45);
        break;
    case 5:
        ok_update_c(5, 44);
        break;
    case 6:
        ok_update_c(6, 43);
        break;
    case 7:
        ok_update_c(7, 42);
        break;
    case 8:
        ok_update_c(8, 41);
        break;
    case 9:
        ok_update_c(9, 40);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok50(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 50);
        break;
    case 1:
        ok_update_c(1, 49);
        break;
    case 2:
        ok_update_c(2, 48);
        break;
    case 3:
        ok_update_c(3, 47);
        break;
    case 4:
        ok_update_c(4, 46);
        break;
    case 5:
        ok_update_c(5, 45);
        break;
    case 6:
        ok_update_c(6, 44);
        break;
    case 7:
        ok_update_c(7, 43);
        break;
    case 8:
        ok_update_c(8, 42);
        break;
    case 9:
        ok_update_c(9, 41);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok51(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 51);
        break;
    case 1:
        ok_update_c(1, 50);
        break;
    case 2:
        ok_update_c(2, 49);
        break;
    case 3:
        ok_update_c(3, 48);
        break;
    case 4:
        ok_update_c(4, 47);
        break;
    case 5:
        ok_update_c(5, 46);
        break;
    case 6:
        ok_update_c(6, 45);
        break;
    case 7:
        ok_update_c(7, 44);
        break;
    case 8:
        ok_update_c(8, 43);
        break;
    case 9:
        ok_update_c(9, 42);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok52(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 52);
        break;
    case 1:
        ok_update_c(1, 51);
        break;
    case 2:
        ok_update_c(2, 50);
        break;
    case 3:
        ok_update_c(3, 49);
        break;
    case 4:
        ok_update_c(4, 48);
        break;
    case 5:
        ok_update_c(5, 47);
        break;
    case 6:
        ok_update_c(6, 46);
        break;
    case 7:
        ok_update_c(7, 45);
        break;
    case 8:
        ok_update_c(8, 44);
        break;
    case 9:
        ok_update_c(9, 43);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok53(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 53);
        break;
    case 1:
        ok_update_c(1, 52);
        break;
    case 2:
        ok_update_c(2, 51);
        break;
    case 3:
        ok_update_c(3, 50);
        break;
    case 4:
        ok_update_c(4, 49);
        break;
    case 5:
        ok_update_c(5, 48);
        break;
    case 6:
        ok_update_c(6, 47);
        break;
    case 7:
        ok_update_c(7, 46);
        break;
    case 8:
        ok_update_c(8, 45);
        break;
    case 9:
        ok_update_c(9, 44);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok54(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 54);
        break;
    case 1:
        ok_update_c(1, 53);
        break;
    case 2:
        ok_update_c(2, 52);
        break;
    case 3:
        ok_update_c(3, 51);
        break;
    case 4:
        ok_update_c(4, 50);
        break;
    case 5:
        ok_update_c(5, 49);
        break;
    case 6:
        ok_update_c(6, 48);
        break;
    case 7:
        ok_update_c(7, 47);
        break;
    case 8:
        ok_update_c(8, 46);
        break;
    case 9:
        ok_update_c(9, 45);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok55(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 55);
        break;
    case 1:
        ok_update_c(1, 54);
        break;
    case 2:
        ok_update_c(2, 53);
        break;
    case 3:
        ok_update_c(3, 52);
        break;
    case 4:
        ok_update_c(4, 51);
        break;
    case 5:
        ok_update_c(5, 50);
        break;
    case 6:
        ok_update_c(6, 49);
        break;
    case 7:
        ok_update_c(7, 48);
        break;
    case 8:
        ok_update_c(8, 47);
        break;
    case 9:
        ok_update_c(9, 46);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok56(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 56);
        break;
    case 1:
        ok_update_c(1, 55);
        break;
    case 2:
        ok_update_c(2, 54);
        break;
    case 3:
        ok_update_c(3, 53);
        break;
    case 4:
        ok_update_c(4, 52);
        break;
    case 5:
        ok_update_c(5, 51);
        break;
    case 6:
        ok_update_c(6, 50);
        break;
    case 7:
        ok_update_c(7, 49);
        break;
    case 8:
        ok_update_c(8, 48);
        break;
    case 9:
        ok_update_c(9, 47);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok57(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 57);
        break;
    case 1:
        ok_update_c(1, 56);
        break;
    case 2:
        ok_update_c(2, 55);
        break;
    case 3:
        ok_update_c(3, 54);
        break;
    case 4:
        ok_update_c(4, 53);
        break;
    case 5:
        ok_update_c(5, 52);
        break;
    case 6:
        ok_update_c(6, 51);
        break;
    case 7:
        ok_update_c(7, 50);
        break;
    case 8:
        ok_update_c(8, 49);
        break;
    case 9:
        ok_update_c(9, 48);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok58(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 58);
        break;
    case 1:
        ok_update_c(1, 57);
        break;
    case 2:
        ok_update_c(2, 56);
        break;
    case 3:
        ok_update_c(3, 55);
        break;
    case 4:
        ok_update_c(4, 54);
        break;
    case 5:
        ok_update_c(5, 53);
        break;
    case 6:
        ok_update_c(6, 52);
        break;
    case 7:
        ok_update_c(7, 51);
        break;
    case 8:
        ok_update_c(8, 50);
        break;
    case 9:
        ok_update_c(9, 49);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok59(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 59);
        break;
    case 1:
        ok_update_c(1, 58);
        break;
    case 2:
        ok_update_c(2, 57);
        break;
    case 3:
        ok_update_c(3, 56);
        break;
    case 4:
        ok_update_c(4, 55);
        break;
    case 5:
        ok_update_c(5, 54);
        break;
    case 6:
        ok_update_c(6, 53);
        break;
    case 7:
        ok_update_c(7, 52);
        break;
    case 8:
        ok_update_c(8, 51);
        break;
    case 9:
        ok_update_c(9, 50);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok60(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 60);
        break;
    case 1:
        ok_update_c(1, 59);
        break;
    case 2:
        ok_update_c(2, 58);
        break;
    case 3:
        ok_update_c(3, 57);
        break;
    case 4:
        ok_update_c(4, 56);
        break;
    case 5:
        ok_update_c(5, 55);
        break;
    case 6:
        ok_update_c(6, 54);
        break;
    case 7:
        ok_update_c(7, 53);
        break;
    case 8:
        ok_update_c(8, 52);
        break;
    case 9:
        ok_update_c(9, 51);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok61(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 61);
        break;
    case 1:
        ok_update_c(1, 60);
        break;
    case 2:
        ok_update_c(2, 59);
        break;
    case 3:
        ok_update_c(3, 58);
        break;
    case 4:
        ok_update_c(4, 57);
        break;
    case 5:
        ok_update_c(5, 56);
        break;
    case 6:
        ok_update_c(6, 55);
        break;
    case 7:
        ok_update_c(7, 54);
        break;
    case 8:
        ok_update_c(8, 53);
        break;
    case 9:
        ok_update_c(9, 52);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok62(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 62);
        break;
    case 1:
        ok_update_c(1, 61);
        break;
    case 2:
        ok_update_c(2, 60);
        break;
    case 3:
        ok_update_c(3, 59);
        break;
    case 4:
        ok_update_c(4, 58);
        break;
    case 5:
        ok_update_c(5, 57);
        break;
    case 6:
        ok_update_c(6, 56);
        break;
    case 7:
        ok_update_c(7, 55);
        break;
    case 8:
        ok_update_c(8, 54);
        break;
    case 9:
        ok_update_c(9, 53);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok63(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 63);
        break;
    case 1:
        ok_update_c(1, 62);
        break;
    case 2:
        ok_update_c(2, 61);
        break;
    case 3:
        ok_update_c(3, 60);
        break;
    case 4:
        ok_update_c(4, 59);
        break;
    case 5:
        ok_update_c(5, 58);
        break;
    case 6:
        ok_update_c(6, 57);
        break;
    case 7:
        ok_update_c(7, 56);
        break;
    case 8:
        ok_update_c(8, 55);
        break;
    case 9:
        ok_update_c(9, 54);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok64(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 64);
        break;
    case 1:
        ok_update_c(1, 63);
        break;
    case 2:
        ok_update_c(2, 62);
        break;
    case 3:
        ok_update_c(3, 61);
        break;
    case 4:
        ok_update_c(4, 60);
        break;
    case 5:
        ok_update_c(5, 59);
        break;
    case 6:
        ok_update_c(6, 58);
        break;
    case 7:
        ok_update_c(7, 57);
        break;
    case 8:
        ok_update_c(8, 56);
        break;
    case 9:
        ok_update_c(9, 55);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok65(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 65);
        break;
    case 1:
        ok_update_c(1, 64);
        break;
    case 2:
        ok_update_c(2, 63);
        break;
    case 3:
        ok_update_c(3, 62);
        break;
    case 4:
        ok_update_c(4, 61);
        break;
    case 5:
        ok_update_c(5, 60);
        break;
    case 6:
        ok_update_c(6, 59);
        break;
    case 7:
        ok_update_c(7, 58);
        break;
    case 8:
        ok_update_c(8, 57);
        break;
    case 9:
        ok_update_c(9, 56);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok66(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 66);
        break;
    case 1:
        ok_update_c(1, 65);
        break;
    case 2:
        ok_update_c(2, 64);
        break;
    case 3:
        ok_update_c(3, 63);
        break;
    case 4:
        ok_update_c(4, 62);
        break;
    case 5:
        ok_update_c(5, 61);
        break;
    case 6:
        ok_update_c(6, 60);
        break;
    case 7:
        ok_update_c(7, 59);
        break;
    case 8:
        ok_update_c(8, 58);
        break;
    case 9:
        ok_update_c(9, 57);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok67(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 67);
        break;
    case 1:
        ok_update_c(1, 66);
        break;
    case 2:
        ok_update_c(2, 65);
        break;
    case 3:
        ok_update_c(3, 64);
        break;
    case 4:
        ok_update_c(4, 63);
        break;
    case 5:
        ok_update_c(5, 62);
        break;
    case 6:
        ok_update_c(6, 61);
        break;
    case 7:
        ok_update_c(7, 60);
        break;
    case 8:
        ok_update_c(8, 59);
        break;
    case 9:
        ok_update_c(9, 58);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok68(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 68);
        break;
    case 1:
        ok_update_c(1, 67);
        break;
    case 2:
        ok_update_c(2, 66);
        break;
    case 3:
        ok_update_c(3, 65);
        break;
    case 4:
        ok_update_c(4, 64);
        break;
    case 5:
        ok_update_c(5, 63);
        break;
    case 6:
        ok_update_c(6, 62);
        break;
    case 7:
        ok_update_c(7, 61);
        break;
    case 8:
        ok_update_c(8, 60);
        break;
    case 9:
        ok_update_c(9, 59);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok69(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 69);
        break;
    case 1:
        ok_update_c(1, 68);
        break;
    case 2:
        ok_update_c(2, 67);
        break;
    case 3:
        ok_update_c(3, 66);
        break;
    case 4:
        ok_update_c(4, 65);
        break;
    case 5:
        ok_update_c(5, 64);
        break;
    case 6:
        ok_update_c(6, 63);
        break;
    case 7:
        ok_update_c(7, 62);
        break;
    case 8:
        ok_update_c(8, 61);
        break;
    case 9:
        ok_update_c(9, 60);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok70(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 70);
        break;
    case 1:
        ok_update_c(1, 69);
        break;
    case 2:
        ok_update_c(2, 68);
        break;
    case 3:
        ok_update_c(3, 67);
        break;
    case 4:
        ok_update_c(4, 66);
        break;
    case 5:
        ok_update_c(5, 65);
        break;
    case 6:
        ok_update_c(6, 64);
        break;
    case 7:
        ok_update_c(7, 63);
        break;
    case 8:
        ok_update_c(8, 62);
        break;
    case 9:
        ok_update_c(9, 61);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok71(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 71);
        break;
    case 1:
        ok_update_c(1, 70);
        break;
    case 2:
        ok_update_c(2, 69);
        break;
    case 3:
        ok_update_c(3, 68);
        break;
    case 4:
        ok_update_c(4, 67);
        break;
    case 5:
        ok_update_c(5, 66);
        break;
    case 6:
        ok_update_c(6, 65);
        break;
    case 7:
        ok_update_c(7, 64);
        break;
    case 8:
        ok_update_c(8, 63);
        break;
    case 9:
        ok_update_c(9, 62);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok72(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 72);
        break;
    case 1:
        ok_update_c(1, 71);
        break;
    case 2:
        ok_update_c(2, 70);
        break;
    case 3:
        ok_update_c(3, 69);
        break;
    case 4:
        ok_update_c(4, 68);
        break;
    case 5:
        ok_update_c(5, 67);
        break;
    case 6:
        ok_update_c(6, 66);
        break;
    case 7:
        ok_update_c(7, 65);
        break;
    case 8:
        ok_update_c(8, 64);
        break;
    case 9:
        ok_update_c(9, 63);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok73(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 73);
        break;
    case 1:
        ok_update_c(1, 72);
        break;
    case 2:
        ok_update_c(2, 71);
        break;
    case 3:
        ok_update_c(3, 70);
        break;
    case 4:
        ok_update_c(4, 69);
        break;
    case 5:
        ok_update_c(5, 68);
        break;
    case 6:
        ok_update_c(6, 67);
        break;
    case 7:
        ok_update_c(7, 66);
        break;
    case 8:
        ok_update_c(8, 65);
        break;
    case 9:
        ok_update_c(9, 64);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok74(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 74);
        break;
    case 1:
        ok_update_c(1, 73);
        break;
    case 2:
        ok_update_c(2, 72);
        break;
    case 3:
        ok_update_c(3, 71);
        break;
    case 4:
        ok_update_c(4, 70);
        break;
    case 5:
        ok_update_c(5, 69);
        break;
    case 6:
        ok_update_c(6, 68);
        break;
    case 7:
        ok_update_c(7, 67);
        break;
    case 8:
        ok_update_c(8, 66);
        break;
    case 9:
        ok_update_c(9, 65);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok75(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 75);
        break;
    case 1:
        ok_update_c(1, 74);
        break;
    case 2:
        ok_update_c(2, 73);
        break;
    case 3:
        ok_update_c(3, 72);
        break;
    case 4:
        ok_update_c(4, 71);
        break;
    case 5:
        ok_update_c(5, 70);
        break;
    case 6:
        ok_update_c(6, 69);
        break;
    case 7:
        ok_update_c(7, 68);
        break;
    case 8:
        ok_update_c(8, 67);
        break;
    case 9:
        ok_update_c(9, 66);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok76(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 76);
        break;
    case 1:
        ok_update_c(1, 75);
        break;
    case 2:
        ok_update_c(2, 74);
        break;
    case 3:
        ok_update_c(3, 73);
        break;
    case 4:
        ok_update_c(4, 72);
        break;
    case 5:
        ok_update_c(5, 71);
        break;
    case 6:
        ok_update_c(6, 70);
        break;
    case 7:
        ok_update_c(7, 69);
        break;
    case 8:
        ok_update_c(8, 68);
        break;
    case 9:
        ok_update_c(9, 67);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok77(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 77);
        break;
    case 1:
        ok_update_c(1, 76);
        break;
    case 2:
        ok_update_c(2, 75);
        break;
    case 3:
        ok_update_c(3, 74);
        break;
    case 4:
        ok_update_c(4, 73);
        break;
    case 5:
        ok_update_c(5, 72);
        break;
    case 6:
        ok_update_c(6, 71);
        break;
    case 7:
        ok_update_c(7, 70);
        break;
    case 8:
        ok_update_c(8, 69);
        break;
    case 9:
        ok_update_c(9, 68);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok78(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 78);
        break;
    case 1:
        ok_update_c(1, 77);
        break;
    case 2:
        ok_update_c(2, 76);
        break;
    case 3:
        ok_update_c(3, 75);
        break;
    case 4:
        ok_update_c(4, 74);
        break;
    case 5:
        ok_update_c(5, 73);
        break;
    case 6:
        ok_update_c(6, 72);
        break;
    case 7:
        ok_update_c(7, 71);
        break;
    case 8:
        ok_update_c(8, 70);
        break;
    case 9:
        ok_update_c(9, 69);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok79(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 79);
        break;
    case 1:
        ok_update_c(1, 78);
        break;
    case 2:
        ok_update_c(2, 77);
        break;
    case 3:
        ok_update_c(3, 76);
        break;
    case 4:
        ok_update_c(4, 75);
        break;
    case 5:
        ok_update_c(5, 74);
        break;
    case 6:
        ok_update_c(6, 73);
        break;
    case 7:
        ok_update_c(7, 72);
        break;
    case 8:
        ok_update_c(8, 71);
        break;
    case 9:
        ok_update_c(9, 70);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok80(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 80);
        break;
    case 1:
        ok_update_c(1, 79);
        break;
    case 2:
        ok_update_c(2, 78);
        break;
    case 3:
        ok_update_c(3, 77);
        break;
    case 4:
        ok_update_c(4, 76);
        break;
    case 5:
        ok_update_c(5, 75);
        break;
    case 6:
        ok_update_c(6, 74);
        break;
    case 7:
        ok_update_c(7, 73);
        break;
    case 8:
        ok_update_c(8, 72);
        break;
    case 9:
        ok_update_c(9, 71);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok81(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 81);
        break;
    case 1:
        ok_update_c(1, 80);
        break;
    case 2:
        ok_update_c(2, 79);
        break;
    case 3:
        ok_update_c(3, 78);
        break;
    case 4:
        ok_update_c(4, 77);
        break;
    case 5:
        ok_update_c(5, 76);
        break;
    case 6:
        ok_update_c(6, 75);
        break;
    case 7:
        ok_update_c(7, 74);
        break;
    case 8:
        ok_update_c(8, 73);
        break;
    case 9:
        ok_update_c(9, 72);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok82(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 82);
        break;
    case 1:
        ok_update_c(1, 81);
        break;
    case 2:
        ok_update_c(2, 80);
        break;
    case 3:
        ok_update_c(3, 79);
        break;
    case 4:
        ok_update_c(4, 78);
        break;
    case 5:
        ok_update_c(5, 77);
        break;
    case 6:
        ok_update_c(6, 76);
        break;
    case 7:
        ok_update_c(7, 75);
        break;
    case 8:
        ok_update_c(8, 74);
        break;
    case 9:
        ok_update_c(9, 73);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok83(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 83);
        break;
    case 1:
        ok_update_c(1, 82);
        break;
    case 2:
        ok_update_c(2, 81);
        break;
    case 3:
        ok_update_c(3, 80);
        break;
    case 4:
        ok_update_c(4, 79);
        break;
    case 5:
        ok_update_c(5, 78);
        break;
    case 6:
        ok_update_c(6, 77);
        break;
    case 7:
        ok_update_c(7, 76);
        break;
    case 8:
        ok_update_c(8, 75);
        break;
    case 9:
        ok_update_c(9, 74);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok84(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 84);
        break;
    case 1:
        ok_update_c(1, 83);
        break;
    case 2:
        ok_update_c(2, 82);
        break;
    case 3:
        ok_update_c(3, 81);
        break;
    case 4:
        ok_update_c(4, 80);
        break;
    case 5:
        ok_update_c(5, 79);
        break;
    case 6:
        ok_update_c(6, 78);
        break;
    case 7:
        ok_update_c(7, 77);
        break;
    case 8:
        ok_update_c(8, 76);
        break;
    case 9:
        ok_update_c(9, 75);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok85(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 85);
        break;
    case 1:
        ok_update_c(1, 84);
        break;
    case 2:
        ok_update_c(2, 83);
        break;
    case 3:
        ok_update_c(3, 82);
        break;
    case 4:
        ok_update_c(4, 81);
        break;
    case 5:
        ok_update_c(5, 80);
        break;
    case 6:
        ok_update_c(6, 79);
        break;
    case 7:
        ok_update_c(7, 78);
        break;
    case 8:
        ok_update_c(8, 77);
        break;
    case 9:
        ok_update_c(9, 76);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok86(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 86);
        break;
    case 1:
        ok_update_c(1, 85);
        break;
    case 2:
        ok_update_c(2, 84);
        break;
    case 3:
        ok_update_c(3, 83);
        break;
    case 4:
        ok_update_c(4, 82);
        break;
    case 5:
        ok_update_c(5, 81);
        break;
    case 6:
        ok_update_c(6, 80);
        break;
    case 7:
        ok_update_c(7, 79);
        break;
    case 8:
        ok_update_c(8, 78);
        break;
    case 9:
        ok_update_c(9, 77);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok87(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 87);
        break;
    case 1:
        ok_update_c(1, 86);
        break;
    case 2:
        ok_update_c(2, 85);
        break;
    case 3:
        ok_update_c(3, 84);
        break;
    case 4:
        ok_update_c(4, 83);
        break;
    case 5:
        ok_update_c(5, 82);
        break;
    case 6:
        ok_update_c(6, 81);
        break;
    case 7:
        ok_update_c(7, 80);
        break;
    case 8:
        ok_update_c(8, 79);
        break;
    case 9:
        ok_update_c(9, 78);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok88(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 88);
        break;
    case 1:
        ok_update_c(1, 87);
        break;
    case 2:
        ok_update_c(2, 86);
        break;
    case 3:
        ok_update_c(3, 85);
        break;
    case 4:
        ok_update_c(4, 84);
        break;
    case 5:
        ok_update_c(5, 83);
        break;
    case 6:
        ok_update_c(6, 82);
        break;
    case 7:
        ok_update_c(7, 81);
        break;
    case 8:
        ok_update_c(8, 80);
        break;
    case 9:
        ok_update_c(9, 79);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok89(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 89);
        break;
    case 1:
        ok_update_c(1, 88);
        break;
    case 2:
        ok_update_c(2, 87);
        break;
    case 3:
        ok_update_c(3, 86);
        break;
    case 4:
        ok_update_c(4, 85);
        break;
    case 5:
        ok_update_c(5, 84);
        break;
    case 6:
        ok_update_c(6, 83);
        break;
    case 7:
        ok_update_c(7, 82);
        break;
    case 8:
        ok_update_c(8, 81);
        break;
    case 9:
        ok_update_c(9, 80);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok90(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 90);
        break;
    case 1:
        ok_update_c(1, 89);
        break;
    case 2:
        ok_update_c(2, 88);
        break;
    case 3:
        ok_update_c(3, 87);
        break;
    case 4:
        ok_update_c(4, 86);
        break;
    case 5:
        ok_update_c(5, 85);
        break;
    case 6:
        ok_update_c(6, 84);
        break;
    case 7:
        ok_update_c(7, 83);
        break;
    case 8:
        ok_update_c(8, 82);
        break;
    case 9:
        ok_update_c(9, 81);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok91(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 91);
        break;
    case 1:
        ok_update_c(1, 90);
        break;
    case 2:
        ok_update_c(2, 89);
        break;
    case 3:
        ok_update_c(3, 88);
        break;
    case 4:
        ok_update_c(4, 87);
        break;
    case 5:
        ok_update_c(5, 86);
        break;
    case 6:
        ok_update_c(6, 85);
        break;
    case 7:
        ok_update_c(7, 84);
        break;
    case 8:
        ok_update_c(8, 83);
        break;
    case 9:
        ok_update_c(9, 82);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok92(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 92);
        break;
    case 1:
        ok_update_c(1, 91);
        break;
    case 2:
        ok_update_c(2, 90);
        break;
    case 3:
        ok_update_c(3, 89);
        break;
    case 4:
        ok_update_c(4, 88);
        break;
    case 5:
        ok_update_c(5, 87);
        break;
    case 6:
        ok_update_c(6, 86);
        break;
    case 7:
        ok_update_c(7, 85);
        break;
    case 8:
        ok_update_c(8, 84);
        break;
    case 9:
        ok_update_c(9, 83);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok93(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 93);
        break;
    case 1:
        ok_update_c(1, 92);
        break;
    case 2:
        ok_update_c(2, 91);
        break;
    case 3:
        ok_update_c(3, 90);
        break;
    case 4:
        ok_update_c(4, 89);
        break;
    case 5:
        ok_update_c(5, 88);
        break;
    case 6:
        ok_update_c(6, 87);
        break;
    case 7:
        ok_update_c(7, 86);
        break;
    case 8:
        ok_update_c(8, 85);
        break;
    case 9:
        ok_update_c(9, 84);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok94(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 94);
        break;
    case 1:
        ok_update_c(1, 93);
        break;
    case 2:
        ok_update_c(2, 92);
        break;
    case 3:
        ok_update_c(3, 91);
        break;
    case 4:
        ok_update_c(4, 90);
        break;
    case 5:
        ok_update_c(5, 89);
        break;
    case 6:
        ok_update_c(6, 88);
        break;
    case 7:
        ok_update_c(7, 87);
        break;
    case 8:
        ok_update_c(8, 86);
        break;
    case 9:
        ok_update_c(9, 85);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok95(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 95);
        break;
    case 1:
        ok_update_c(1, 94);
        break;
    case 2:
        ok_update_c(2, 93);
        break;
    case 3:
        ok_update_c(3, 92);
        break;
    case 4:
        ok_update_c(4, 91);
        break;
    case 5:
        ok_update_c(5, 90);
        break;
    case 6:
        ok_update_c(6, 89);
        break;
    case 7:
        ok_update_c(7, 88);
        break;
    case 8:
        ok_update_c(8, 87);
        break;
    case 9:
        ok_update_c(9, 86);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok96(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 96);
        break;
    case 1:
        ok_update_c(1, 95);
        break;
    case 2:
        ok_update_c(2, 94);
        break;
    case 3:
        ok_update_c(3, 93);
        break;
    case 4:
        ok_update_c(4, 92);
        break;
    case 5:
        ok_update_c(5, 91);
        break;
    case 6:
        ok_update_c(6, 90);
        break;
    case 7:
        ok_update_c(7, 89);
        break;
    case 8:
        ok_update_c(8, 88);
        break;
    case 9:
        ok_update_c(9, 87);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok97(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 97);
        break;
    case 1:
        ok_update_c(1, 96);
        break;
    case 2:
        ok_update_c(2, 95);
        break;
    case 3:
        ok_update_c(3, 94);
        break;
    case 4:
        ok_update_c(4, 93);
        break;
    case 5:
        ok_update_c(5, 92);
        break;
    case 6:
        ok_update_c(6, 91);
        break;
    case 7:
        ok_update_c(7, 90);
        break;
    case 8:
        ok_update_c(8, 89);
        break;
    case 9:
        ok_update_c(9, 88);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok98(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_update_c(0, 98);
        break;
    case 1:
        ok_update_c(1, 97);
        break;
    case 2:
        ok_update_c(2, 96);
        break;
    case 3:
        ok_update_c(3, 95);
        break;
    case 4:
        ok_update_c(4, 94);
        break;
    case 5:
        ok_update_c(5, 93);
        break;
    case 6:
        ok_update_c(6, 92);
        break;
    case 7:
        ok_update_c(7, 91);
        break;
    case 8:
        ok_update_c(8, 90);
        break;
    case 9:
        ok_update_c(9, 89);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok99(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
         RammerLikeCellModel<t_hidden_size> *__restrict__ models,
         RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_hasw_not_update_c(0, 99);
        break;
    case 1:
        ok_update_c(1, 98);
        break;
    case 2:
        ok_update_c(2, 97);
        break;
    case 3:
        ok_update_c(3, 96);
        break;
    case 4:
        ok_update_c(4, 95);
        break;
    case 5:
        ok_update_c(5, 94);
        break;
    case 6:
        ok_update_c(6, 93);
        break;
    case 7:
        ok_update_c(7, 92);
        break;
    case 8:
        ok_update_c(8, 91);
        break;
    case 9:
        ok_update_c(9, 90);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok100(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
          RammerLikeCellModel<t_hidden_size> *__restrict__ models,
          RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_not_update_c(1, 99);
        break;
    case 1:
        ok_update_c(2, 98);
        break;
    case 2:
        ok_update_c(3, 97);
        break;
    case 3:
        ok_update_c(4, 96);
        break;
    case 4:
        ok_update_c(5, 95);
        break;
    case 5:
        ok_update_c(6, 94);
        break;
    case 6:
        ok_update_c(7, 93);
        break;
    case 7:
        ok_update_c(8, 92);
        break;
    case 8:
        ok_update_c(9, 91);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok101(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
          RammerLikeCellModel<t_hidden_size> *__restrict__ models,
          RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_not_update_c(2, 99);
        break;
    case 1:
        ok_update_c(3, 98);
        break;
    case 2:
        ok_update_c(4, 97);
        break;
    case 3:
        ok_update_c(5, 96);
        break;
    case 4:
        ok_update_c(6, 95);
        break;
    case 5:
        ok_update_c(7, 94);
        break;
    case 6:
        ok_update_c(8, 93);
        break;
    case 7:
        ok_update_c(9, 92);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok102(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
          RammerLikeCellModel<t_hidden_size> *__restrict__ models,
          RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_not_update_c(3, 99);
        break;
    case 1:
        ok_update_c(4, 98);
        break;
    case 2:
        ok_update_c(5, 97);
        break;
    case 3:
        ok_update_c(6, 96);
        break;
    case 4:
        ok_update_c(7, 95);
        break;
    case 5:
        ok_update_c(8, 94);
        break;
    case 6:
        ok_update_c(9, 93);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok103(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
          RammerLikeCellModel<t_hidden_size> *__restrict__ models,
          RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_not_update_c(4, 99);
        break;
    case 1:
        ok_update_c(5, 98);
        break;
    case 2:
        ok_update_c(6, 97);
        break;
    case 3:
        ok_update_c(7, 96);
        break;
    case 4:
        ok_update_c(8, 95);
        break;
    case 5:
        ok_update_c(9, 94);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok104(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
          RammerLikeCellModel<t_hidden_size> *__restrict__ models,
          RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_not_update_c(5, 99);
        break;
    case 1:
        ok_update_c(6, 98);
        break;
    case 2:
        ok_update_c(7, 97);
        break;
    case 3:
        ok_update_c(8, 96);
        break;
    case 4:
        ok_update_c(9, 95);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok105(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
          RammerLikeCellModel<t_hidden_size> *__restrict__ models,
          RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_not_update_c(6, 99);
        break;
    case 1:
        ok_update_c(7, 98);
        break;
    case 2:
        ok_update_c(8, 97);
        break;
    case 3:
        ok_update_c(9, 96);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok106(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
          RammerLikeCellModel<t_hidden_size> *__restrict__ models,
          RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_not_update_c(7, 99);
        break;
    case 1:
        ok_update_c(8, 98);
        break;
    case 2:
        ok_update_c(9, 97);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok107(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
          RammerLikeCellModel<t_hidden_size> *__restrict__ models,
          RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    switch (blockIdx.x >> 3) {
    case 0:
        ok_not_update_c(8, 99);
        break;
    case 1:
        ok_update_c(9, 98);
        break;
    }
}

template <unsigned int t_hidden_size, unsigned int t_num_layer>
__global__ void __launch_bounds__(128, 1)
    ok108(RammerLikeCellInput<t_hidden_size> *__restrict__ inputs,
          RammerLikeCellModel<t_hidden_size> *__restrict__ models,
          RammerLikeCellOutput *__restrict__ outputs) {
    __shared__ float nndense_output[4][32];
    ok_not_update_c(9, 99);
}

template __global__ void
    ok_1<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok0<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                 RammerLikeCellModel<256> *__restrict__ models,
                 RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok1<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                 RammerLikeCellModel<256> *__restrict__ models,
                 RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok2<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                 RammerLikeCellModel<256> *__restrict__ models,
                 RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok3<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                 RammerLikeCellModel<256> *__restrict__ models,
                 RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok4<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                 RammerLikeCellModel<256> *__restrict__ models,
                 RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok5<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                 RammerLikeCellModel<256> *__restrict__ models,
                 RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok6<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                 RammerLikeCellModel<256> *__restrict__ models,
                 RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok7<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                 RammerLikeCellModel<256> *__restrict__ models,
                 RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok8<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                 RammerLikeCellModel<256> *__restrict__ models,
                 RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok9<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                 RammerLikeCellModel<256> *__restrict__ models,
                 RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok10<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok11<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok12<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok13<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok14<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok15<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok16<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok17<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok18<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok19<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok20<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok21<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok22<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok23<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok24<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok25<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok26<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok27<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok28<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok29<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok30<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok31<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok32<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok33<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok34<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok35<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok36<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok37<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok38<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok39<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok40<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok41<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok42<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok43<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok44<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok45<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok46<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok47<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok48<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok49<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok50<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok51<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok52<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok53<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok54<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok55<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok56<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok57<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok58<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok59<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok60<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok61<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok62<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok63<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok64<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok65<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok66<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok67<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok68<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok69<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok70<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok71<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok72<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok73<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok74<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok75<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok76<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok77<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok78<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok79<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok80<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok81<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok82<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok83<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok84<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok85<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok86<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok87<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok88<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok89<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok90<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok91<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok92<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok93<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok94<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok95<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok96<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok97<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok98<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok99<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                  RammerLikeCellModel<256> *__restrict__ models,
                  RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok100<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                   RammerLikeCellModel<256> *__restrict__ models,
                   RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok101<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                   RammerLikeCellModel<256> *__restrict__ models,
                   RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok102<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                   RammerLikeCellModel<256> *__restrict__ models,
                   RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok103<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                   RammerLikeCellModel<256> *__restrict__ models,
                   RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok104<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                   RammerLikeCellModel<256> *__restrict__ models,
                   RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok105<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                   RammerLikeCellModel<256> *__restrict__ models,
                   RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok106<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                   RammerLikeCellModel<256> *__restrict__ models,
                   RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok107<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                   RammerLikeCellModel<256> *__restrict__ models,
                   RammerLikeCellOutput *__restrict__ outputs);
template __global__ void
    ok108<256, 10>(RammerLikeCellInput<256> *__restrict__ inputs,
                   RammerLikeCellModel<256> *__restrict__ models,
                   RammerLikeCellOutput *__restrict__ outputs);