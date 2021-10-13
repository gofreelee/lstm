; ModuleID = './adjust_loop_number_256.cu'
source_filename = "./adjust_loop_number_256.cu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.dim3 = type { i32, i32, i32 }
%struct.CUstream_st = type opaque

; Function Attrs: norecurse uwtable
define dso_local void @_Z63__device_stub__Dot_float_float_float_cuda_Dot_8157_block_kernelPfS_S_iiPc(float* %input0, float* %input1, float* %output0, i32 %thread_id, i32 %block_id, i8* %shared_buffer) #0 {
entry:
  %input0.addr = alloca float*, align 8
  %input1.addr = alloca float*, align 8
  %output0.addr = alloca float*, align 8
  %thread_id.addr = alloca i32, align 4
  %block_id.addr = alloca i32, align 4
  %shared_buffer.addr = alloca i8*, align 8
  %grid_dim = alloca %struct.dim3, align 8
  %block_dim = alloca %struct.dim3, align 8
  %shmem_size = alloca i64, align 8
  %stream = alloca i8*, align 8
  store float* %input0, float** %input0.addr, align 8, !tbaa !3
  store float* %input1, float** %input1.addr, align 8, !tbaa !3
  store float* %output0, float** %output0.addr, align 8, !tbaa !3
  store i32 %thread_id, i32* %thread_id.addr, align 4, !tbaa !7
  store i32 %block_id, i32* %block_id.addr, align 4, !tbaa !7
  store i8* %shared_buffer, i8** %shared_buffer.addr, align 8, !tbaa !3
  %kernel_args3 = alloca [6 x i8*], align 16
  %kernel_args3.sub = getelementptr inbounds [6 x i8*], [6 x i8*]* %kernel_args3, i64 0, i64 0
  %0 = bitcast [6 x i8*]* %kernel_args3 to float***
  store float** %input0.addr, float*** %0, align 16
  %1 = getelementptr inbounds [6 x i8*], [6 x i8*]* %kernel_args3, i64 0, i64 1
  %2 = bitcast i8** %1 to float***
  store float** %input1.addr, float*** %2, align 8
  %3 = getelementptr inbounds [6 x i8*], [6 x i8*]* %kernel_args3, i64 0, i64 2
  %4 = bitcast i8** %3 to float***
  store float** %output0.addr, float*** %4, align 16
  %5 = getelementptr inbounds [6 x i8*], [6 x i8*]* %kernel_args3, i64 0, i64 3
  %6 = bitcast i8** %5 to i32**
  store i32* %thread_id.addr, i32** %6, align 8
  %7 = getelementptr inbounds [6 x i8*], [6 x i8*]* %kernel_args3, i64 0, i64 4
  %8 = bitcast i8** %7 to i32**
  store i32* %block_id.addr, i32** %8, align 16
  %9 = getelementptr inbounds [6 x i8*], [6 x i8*]* %kernel_args3, i64 0, i64 5
  %10 = bitcast i8** %9 to i8***
  store i8** %shared_buffer.addr, i8*** %10, align 8
  %11 = call i32 @__cudaPopCallConfiguration(%struct.dim3* nonnull %grid_dim, %struct.dim3* nonnull %block_dim, i64* nonnull %shmem_size, i8** nonnull %stream)
  %12 = load i64, i64* %shmem_size, align 8
  %13 = bitcast i8** %stream to %struct.CUstream_st**
  %14 = load %struct.CUstream_st*, %struct.CUstream_st** %13, align 8
  %grid_dim.coerce.sroa.0.0..sroa_cast = bitcast %struct.dim3* %grid_dim to i64*
  %grid_dim.coerce.sroa.0.0.copyload = load i64, i64* %grid_dim.coerce.sroa.0.0..sroa_cast, align 8
  %grid_dim.coerce.sroa.2.0..sroa_idx2 = getelementptr inbounds %struct.dim3, %struct.dim3* %grid_dim, i64 0, i32 2
  %grid_dim.coerce.sroa.2.0.copyload = load i32, i32* %grid_dim.coerce.sroa.2.0..sroa_idx2, align 8
  %block_dim.coerce.sroa.0.0..sroa_cast = bitcast %struct.dim3* %block_dim to i64*
  %block_dim.coerce.sroa.0.0.copyload = load i64, i64* %block_dim.coerce.sroa.0.0..sroa_cast, align 8
  %block_dim.coerce.sroa.2.0..sroa_idx1 = getelementptr inbounds %struct.dim3, %struct.dim3* %block_dim, i64 0, i32 2
  %block_dim.coerce.sroa.2.0.copyload = load i32, i32* %block_dim.coerce.sroa.2.0..sroa_idx1, align 8
  %call = call i32 @cudaLaunchKernel(i8* bitcast (void (float*, float*, float*, i32, i32, i8*)* @_Z63__device_stub__Dot_float_float_float_cuda_Dot_8157_block_kernelPfS_S_iiPc to i8*), i64 %grid_dim.coerce.sroa.0.0.copyload, i32 %grid_dim.coerce.sroa.2.0.copyload, i64 %block_dim.coerce.sroa.0.0.copyload, i32 %block_dim.coerce.sroa.2.0.copyload, i8** nonnull %kernel_args3.sub, i64 %12, %struct.CUstream_st* %14)
  ret void
}

declare dso_local i32 @__cudaPopCallConfiguration(%struct.dim3*, %struct.dim3*, i64*, i8**) local_unnamed_addr

declare dso_local i32 @cudaLaunchKernel(i8*, i64, i32, i64, i32, i8**, i64, %struct.CUstream_st*) local_unnamed_addr

attributes #0 = { norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!"clang version 11.1.0"}
!3 = !{!4, !4, i64 0}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !5, i64 0}
