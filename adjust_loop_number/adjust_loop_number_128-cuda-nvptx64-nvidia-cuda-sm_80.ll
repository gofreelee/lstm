; ModuleID = './adjust_loop_number_128.cu'
source_filename = "./adjust_loop_number_128.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.cudaFuncAttributes = type { i64, i64, i64, i32, i32, i32, i32, i32, i32, i32 }

; Function Attrs: nounwind
define weak dso_local i32 @cudaMalloc(i8** %p, i64 %s) local_unnamed_addr #0 {
entry:
  ret i32 999
}

; Function Attrs: nounwind
define weak dso_local i32 @cudaFuncGetAttributes(%struct.cudaFuncAttributes* %p, i8* %c) local_unnamed_addr #0 {
entry:
  ret i32 999
}

; Function Attrs: nounwind
define weak dso_local i32 @cudaDeviceGetAttribute(i32* %value, i32 %attr, i32 %device) local_unnamed_addr #0 {
entry:
  ret i32 999
}

; Function Attrs: nounwind
define weak dso_local i32 @cudaGetDevice(i32* %device) local_unnamed_addr #0 {
entry:
  ret i32 999
}

; Function Attrs: nounwind
define weak dso_local i32 @cudaOccupancyMaxActiveBlocksPerMultiprocessor(i32* %numBlocks, i8* %func, i32 %blockSize, i64 %dynamicSmemSize) local_unnamed_addr #0 {
entry:
  ret i32 999
}

; Function Attrs: nounwind
define weak dso_local i32 @cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(i32* %numBlocks, i8* %func, i32 %blockSize, i64 %dynamicSmemSize, i32 %flags) local_unnamed_addr #0 {
entry:
  ret i32 999
}

; Function Attrs: convergent norecurse nounwind
define dso_local void @_Z48Dot_float_float_float_cuda_Dot_8157_block_kernelPfS_S_iiPc(float* nocapture readonly %input0, float* nocapture readonly %input1, float* nocapture %output0, i32 %thread_id, i32 %block_id, i8* nocapture readnone %shared_buffer) local_unnamed_addr #1 {
entry:
  %cmp = icmp sgt i32 %thread_id, 127
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5, !range !6
  %shr = lshr i32 %0, 5
  %and = and i32 %0, 31
  %1 = shl i32 %block_id, 5
  %div = and i32 %1, 1073741792
  %add = or i32 %and, %div
  %cmp3 = icmp ult i32 %add, 256
  br i1 %cmp3, label %if.then4, label %return

if.then4:                                         ; preds = %if.end
  %mul5 = shl nuw nsw i32 %shr, 6
  %mul7 = add nuw nsw i32 %mul5, 64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %cmp14 = icmp eq i32 %shr, 0
  %idxprom16 = zext i32 %add to i64
  br i1 %cmp14, label %if.then15, label %if.end18

for.body:                                         ; preds = %if.then4, %for.body
  %i.054 = phi i32 [ %mul5, %if.then4 ], [ %inc, %for.body ]
  %val.053 = phi float [ 0.000000e+00, %if.then4 ], [ %4, %for.body ]
  %idxprom = zext i32 %i.054 to i64
  %arrayidx = getelementptr inbounds float, float* %input0, i64 %idxprom
  %2 = load float, float* %arrayidx, align 4, !tbaa !7
  %mul9 = shl nsw i32 %i.054, 8
  %add10 = add nuw nsw i32 %mul9, %add
  %idxprom11 = zext i32 %add10 to i64
  %arrayidx12 = getelementptr inbounds float, float* %input1, i64 %idxprom11
  %3 = load float, float* %arrayidx12, align 4, !tbaa !7
  %4 = tail call float @llvm.fma.f32(float %2, float %3, float %val.053) #5
  %inc = add nuw nsw i32 %i.054, 1
  %cmp8 = icmp ult i32 %inc, %mul7
  br i1 %cmp8, label %for.body, label %for.cond.cleanup

if.then15:                                        ; preds = %for.cond.cleanup
  %arrayidx17 = getelementptr inbounds float, float* %output0, i64 %idxprom16
  store float 0.000000e+00, float* %arrayidx17, align 4, !tbaa !7
  br label %if.end18

if.end18:                                         ; preds = %for.cond.cleanup, %if.then15
  tail call void @llvm.nvvm.barrier0()
  %add.ptr = getelementptr inbounds float, float* %output0, i64 %idxprom16
  %5 = atomicrmw fadd float* %add.ptr, float %4 seq_cst
  br label %return

return:                                           ; preds = %if.end, %if.end18, %entry
  ret void
}

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

; Function Attrs: nounwind readnone speculatable willreturn
declare float @llvm.fma.f32(float, float, float) #4

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+ptx70,+sm_80" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+ptx70,+sm_80" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind readnone speculatable willreturn }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!nvvm.annotations = !{!3}
!llvm.ident = !{!4, !5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{void (float*, float*, float*, i32, i32, i8*)* @_Z48Dot_float_float_float_cuda_Dot_8157_block_kernelPfS_S_iiPc, !"kernel", i32 1}
!4 = !{!"clang version 11.1.0"}
!5 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!6 = !{i32 0, i32 1024}
!7 = !{!8, !8, i64 0}
!8 = !{!"float", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}