	.text
	.file	"adjust_loop_number_128.cu"
	.globl	_Z63__device_stub__Dot_float_float_float_cuda_Dot_8157_block_kernelPfS_S_iiPc # -- Begin function _Z63__device_stub__Dot_float_float_float_cuda_Dot_8157_block_kernelPfS_S_iiPc
	.p2align	4, 0x90
	.type	_Z63__device_stub__Dot_float_float_float_cuda_Dot_8157_block_kernelPfS_S_iiPc,@function
_Z63__device_stub__Dot_float_float_float_cuda_Dot_8157_block_kernelPfS_S_iiPc: # @_Z63__device_stub__Dot_float_float_float_cuda_Dot_8157_block_kernelPfS_S_iiPc
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$152, %rsp
	.cfi_def_cfa_offset 160
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movl	%ecx, 12(%rsp)
	movl	%r8d, 8(%rsp)
	movq	%r9, 64(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__cudaPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	96(%rsp), %r9
	movl	$_Z63__device_stub__Dot_float_float_float_cuda_Dot_8157_block_kernelPfS_S_iiPc, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	cudaLaunchKernel
	addq	$168, %rsp
	.cfi_adjust_cfa_offset -168
	retq
.Lfunc_end0:
	.size	_Z63__device_stub__Dot_float_float_float_cuda_Dot_8157_block_kernelPfS_S_iiPc, .Lfunc_end0-_Z63__device_stub__Dot_float_float_float_cuda_Dot_8157_block_kernelPfS_S_iiPc
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 11.1.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z63__device_stub__Dot_float_float_float_cuda_Dot_8157_block_kernelPfS_S_iiPc
