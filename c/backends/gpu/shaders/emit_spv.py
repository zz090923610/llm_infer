#!/usr/bin/env python3
"""Emit SPIR-V compute shaders matching the GLSL in this directory.

Used when glslangValidator is not installed. compile.sh prefers glslang.
"""
from __future__ import annotations

import os
import struct
import sys

# Opcodes
OpCapability = 17
OpExtInstImport = 11
OpMemoryModel = 14
OpEntryPoint = 15
OpExecutionMode = 16
OpName = 5
OpDecorate = 71
OpMemberDecorate = 72
OpTypeVoid = 19
OpTypeBool = 20
OpTypeInt = 21
OpTypeFloat = 22
OpTypeVector = 23
OpTypeArray = 28
OpTypeRuntimeArray = 29
OpTypeStruct = 30
OpTypePointer = 32
OpTypeFunction = 33
OpConstant = 43
OpConstantComposite = 44
OpExtInst = 12
OpFunction = 54
OpFunctionEnd = 56
OpVariable = 59
OpLoad = 61
OpStore = 62
OpAccessChain = 65
OpVectorExtractDynamic = 77
OpCompositeExtract = 81
OpIAdd = 128
OpFAdd = 129
OpISub = 130
OpFSub = 131
OpIMul = 132
OpFMul = 133
OpUDiv = 134
OpFDiv = 136
OpUMod = 137
OpFNegate = 127
OpBitcast = 124
OpConvertSToF = 111
OpConvertUToF = 112
OpLogicalAnd = 167
OpLogicalNot = 168
OpSelect = 169
OpIEqual = 170
OpINotEqual = 171
OpUGreaterThan = 172
OpUGreaterThanEqual = 174
OpULessThan = 176
OpSLessThan = 177
OpULessThanEqual = 178
OpSLessThanEqual = 179
OpFOrdGreaterThan = 186
OpShiftRightLogical = 194
OpControlBarrier = 224
OpPhi = 245
OpLoopMerge = 246
OpSelectionMerge = 247
OpLabel = 248
OpBranch = 249
OpBranchConditional = 250
OpReturn = 253
OpTypeImage = 25
OpImageFetch = 95

CapShader = 1
CapSampledBuffer = 46
DimBuffer = 5
ImgUnknown = 0
StorageUniformConstant = 0
AddrLogical = 0
MemGLSL450 = 1
ExecGLCompute = 5
ModeLocalSize = 17
FnNone = 0
StorageUniform = 2
StorageInput = 1
StoragePushConstant = 9
StorageWorkgroup = 4
StorageFunction = 7
DecBufferBlock = 3
DecBlock = 2
DecArrayStride = 6
DecBuiltin = 11
DecBinding = 33
DecDescriptorSet = 34
DecOffset = 35
BuiltinWorkgroupId = 26
BuiltinLocalInvocationId = 27
BuiltinGlobalInvocationId = 28
BuiltinNumWorkgroups = 24
ScopeWorkgroup = 2
MemSemAcquireReleaseWorkgroup = 0x108  # AcquireRelease | WorkgroupMemory
GLSL_FMax = 40
GLSL_Exp = 27
GLSL_InverseSqrt = 32
GLSL_UnpackHalf2x16 = 59
OpBitwiseAnd = 199
OpBitwiseOr = 197
OpShiftLeftLogical = 196
OpShiftRightArithmetic = 195


def _str_words(s: str) -> list[int]:
    data = s.encode("utf-8") + b"\x00"
    while len(data) % 4:
        data += b"\x00"
    return list(struct.unpack("<" + "I" * (len(data) // 4), data))


class Spv:
    def __init__(self):
        self.next_id = 1
        self.caps: list[list[int]] = []
        self.imports: list[list[int]] = []
        self.memory: list[list[int]] = []
        self.entry: list[list[int]] = []
        self.modes: list[list[int]] = []
        self.debug: list[list[int]] = []
        self.ann: list[list[int]] = []
        self.types: list[list[int]] = []
        self.fn: list[list[int]] = []
        self.ids: dict[str, int] = {}

    def id(self, name: str | None = None) -> int:
        i = self.next_id
        self.next_id += 1
        if name:
            self.ids[name] = i
        return i

    def emit(self, sec: list[list[int]], op: int, *ops: int):
        words = list(ops)
        sec.append([((len(words) + 1) << 16) | op, *words])

    def inst(self, op: int, *ops: int) -> None:
        self.emit(self.fn, op, *ops)

    def result(self, op: int, ty: int, *ops: int) -> int:
        rid = self.id()
        self.inst(op, ty, rid, *ops)
        return rid

    def finish(self) -> bytes:
        body: list[int] = []
        for sec in (
            self.caps,
            self.imports,
            self.memory,
            self.entry,
            self.modes,
            self.debug,
            self.ann,
            self.types,
            self.fn,
        ):
            for inst in sec:
                body.extend(inst)
        header = [0x07230203, 0x00010000, 0, self.next_id, 0]
        words = header + body
        return struct.pack("<" + "I" * len(words), *words)


class Comp:
    """One compute shader with unified push constants, SSBOs, optional shared[64]."""

    def __init__(self, n_ssbo: int, shared_n: int = 0, use_global: bool = False, uint_bind=(),
                 texel0: bool = False):
        self.s = Spv()
        self.n_ssbo = n_ssbo
        self.shared_n = shared_n
        self.use_global = use_global
        self.uint_bind = set(uint_bind)
        self.texel0 = texel0
        self._build_preamble()

    def _build_preamble(self):
        s = self.s
        s.emit(s.caps, OpCapability, CapShader)
        if self.texel0:
            s.emit(s.caps, OpCapability, CapSampledBuffer)
        self.glsl = s.id("glsl")
        s.emit(s.imports, OpExtInstImport, self.glsl, *_str_words("GLSL.std.450"))
        s.emit(s.memory, OpMemoryModel, AddrLogical, MemGLSL450)

        self.void = s.id()
        s.emit(s.types, OpTypeVoid, self.void)
        self.fn_ty = s.id()
        s.emit(s.types, OpTypeFunction, self.fn_ty, self.void)
        self.bool = s.id()
        s.emit(s.types, OpTypeBool, self.bool)
        self.i32 = s.id()
        s.emit(s.types, OpTypeInt, self.i32, 32, 1)
        self.u32 = s.id()
        s.emit(s.types, OpTypeInt, self.u32, 32, 0)
        self.f32 = s.id()
        s.emit(s.types, OpTypeFloat, self.f32, 32)
        self.v2f = s.id()
        s.emit(s.types, OpTypeVector, self.v2f, self.f32, 2)
        self.uvec3 = s.id()
        s.emit(s.types, OpTypeVector, self.uvec3, self.u32, 3)

        def const_u(v: int) -> int:
            cid = s.id()
            s.emit(s.types, OpConstant, self.u32, cid, v & 0xFFFFFFFF)
            return cid

        def const_i(v: int) -> int:
            cid = s.id()
            s.emit(s.types, OpConstant, self.i32, cid, v & 0xFFFFFFFF)
            return cid

        def const_f(v: float) -> int:
            cid = s.id()
            bits = struct.unpack("<I", struct.pack("<f", v))[0]
            s.emit(s.types, OpConstant, self.f32, cid, bits)
            return cid

        self.cu = {k: const_u(k) for k in (
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 23, 24, 31, 32, 64, 112, 255, 256, 0x3FF, 0x8000, 0xFFFF,
        )}
        self.ci0 = const_i(0)
        self.f0 = const_f(0.0)
        self.f1 = const_f(1.0)
        self.f_bigneg = const_f(-1.0e30)
        self.u_scope = const_u(ScopeWorkgroup)
        self.u_sem = const_u(MemSemAcquireReleaseWorkgroup)

        # runtime arrays
        self.rt_f = s.id()
        s.emit(s.types, OpTypeRuntimeArray, self.rt_f, self.f32)
        s.emit(s.ann, OpDecorate, self.rt_f, DecArrayStride, 4)
        self.rt_i = s.id()
        s.emit(s.types, OpTypeRuntimeArray, self.rt_i, self.i32)
        s.emit(s.ann, OpDecorate, self.rt_i, DecArrayStride, 4)
        self.st_f = s.id()
        s.emit(s.types, OpTypeStruct, self.st_f, self.rt_f)
        s.emit(s.ann, OpDecorate, self.st_f, DecBufferBlock)
        s.emit(s.ann, OpMemberDecorate, self.st_f, 0, DecOffset, 0)
        self.st_i = s.id()
        s.emit(s.types, OpTypeStruct, self.st_i, self.rt_i)
        s.emit(s.ann, OpDecorate, self.st_i, DecBufferBlock)
        s.emit(s.ann, OpMemberDecorate, self.st_i, 0, DecOffset, 0)
        self.ptr_uni_f = s.id()
        s.emit(s.types, OpTypePointer, self.ptr_uni_f, StorageUniform, self.st_f)
        self.ptr_uni_i = s.id()
        s.emit(s.types, OpTypePointer, self.ptr_uni_i, StorageUniform, self.st_i)
        self.ptr_el_f = s.id()
        s.emit(s.types, OpTypePointer, self.ptr_el_f, StorageUniform, self.f32)
        self.ptr_el_i = s.id()
        s.emit(s.types, OpTypePointer, self.ptr_el_i, StorageUniform, self.i32)
        self.rt_u = s.id()
        s.emit(s.types, OpTypeRuntimeArray, self.rt_u, self.u32)
        s.emit(s.ann, OpDecorate, self.rt_u, DecArrayStride, 4)
        self.st_u = s.id()
        s.emit(s.types, OpTypeStruct, self.st_u, self.rt_u)
        s.emit(s.ann, OpDecorate, self.st_u, DecBufferBlock)
        s.emit(s.ann, OpMemberDecorate, self.st_u, 0, DecOffset, 0)
        self.ptr_uni_u = s.id()
        s.emit(s.types, OpTypePointer, self.ptr_uni_u, StorageUniform, self.st_u)
        self.ptr_el_u = s.id()
        s.emit(s.types, OpTypePointer, self.ptr_el_u, StorageUniform, self.u32)

        self.v4f = None
        self.img_ty = None
        self.texel_var = None
        self.texel_img = None
        self.ssbo = []
        start = 0
        if self.texel0:
            self.v4f = s.id()
            s.emit(s.types, OpTypeVector, self.v4f, self.f32, 4)
            self.img_ty = s.id()
            s.emit(s.types, OpTypeImage, self.img_ty, self.f32, DimBuffer, 0, 0, 0, 1, ImgUnknown)
            self.ptr_img = s.id()
            s.emit(s.types, OpTypePointer, self.ptr_img, StorageUniformConstant, self.img_ty)
            self.texel_var = s.id()
            s.emit(s.types, OpVariable, self.ptr_img, self.texel_var, StorageUniformConstant)
            s.emit(s.ann, OpDecorate, self.texel_var, DecDescriptorSet, 0)
            s.emit(s.ann, OpDecorate, self.texel_var, DecBinding, 0)
            self.ssbo.append(None)
            start = 1
        for b in range(start, self.n_ssbo):
            vid = s.id()
            pty = self.ptr_uni_u if b in self.uint_bind else self.ptr_uni_f
            s.emit(s.types, OpVariable, pty, vid, StorageUniform)
            s.emit(s.ann, OpDecorate, vid, DecDescriptorSet, 0)
            s.emit(s.ann, OpDecorate, vid, DecBinding, b)
            self.ssbo.append(vid)

        # push constants: 8 int + 4 float
        members = [self.i32] * 8 + [self.f32] * 4
        self.pc_ty = s.id()
        s.emit(s.types, OpTypeStruct, self.pc_ty, *members)
        s.emit(s.ann, OpDecorate, self.pc_ty, DecBlock)
        for i in range(12):
            s.emit(s.ann, OpMemberDecorate, self.pc_ty, i, DecOffset, i * 4)
        self.ptr_pc = s.id()
        s.emit(s.types, OpTypePointer, self.ptr_pc, StoragePushConstant, self.pc_ty)
        self.ptr_pc_i = s.id()
        s.emit(s.types, OpTypePointer, self.ptr_pc_i, StoragePushConstant, self.i32)
        self.ptr_pc_f = s.id()
        s.emit(s.types, OpTypePointer, self.ptr_pc_f, StoragePushConstant, self.f32)
        self.pc = s.id()
        s.emit(s.types, OpVariable, self.ptr_pc, self.pc, StoragePushConstant)

        self.ptr_in_v3 = s.id()
        s.emit(s.types, OpTypePointer, self.ptr_in_v3, StorageInput, self.uvec3)
        self.wgid = s.id()
        s.emit(s.types, OpVariable, self.ptr_in_v3, self.wgid, StorageInput)
        s.emit(s.ann, OpDecorate, self.wgid, DecBuiltin, BuiltinWorkgroupId)
        self.lidv = s.id()
        s.emit(s.types, OpVariable, self.ptr_in_v3, self.lidv, StorageInput)
        s.emit(s.ann, OpDecorate, self.lidv, DecBuiltin, BuiltinLocalInvocationId)
        iface = [self.wgid, self.lidv]
        if self.use_global:
            self.gidv = s.id()
            s.emit(s.types, OpVariable, self.ptr_in_v3, self.gidv, StorageInput)
            s.emit(s.ann, OpDecorate, self.gidv, DecBuiltin, BuiltinGlobalInvocationId)
            self.nwg = s.id()
            s.emit(s.types, OpVariable, self.ptr_in_v3, self.nwg, StorageInput)
            s.emit(s.ann, OpDecorate, self.nwg, DecBuiltin, BuiltinNumWorkgroups)
            iface.extend([self.gidv, self.nwg])
        else:
            self.gidv = None
            self.nwg = None

        if self.shared_n:
            arr_len = const_u(self.shared_n)
            self.sh_ty = s.id()
            s.emit(s.types, OpTypeArray, self.sh_ty, self.f32, arr_len)
            s.emit(s.ann, OpDecorate, self.sh_ty, DecArrayStride, 4)
            self.ptr_sh_arr = s.id()
            s.emit(s.types, OpTypePointer, self.ptr_sh_arr, StorageWorkgroup, self.sh_ty)
            self.ptr_sh_f = s.id()
            s.emit(s.types, OpTypePointer, self.ptr_sh_f, StorageWorkgroup, self.f32)
            self.shared = []
            for _ in range(2):  # qsh + red (unused extra is harmless if only 1 used)
                vid = s.id()
                s.emit(s.types, OpVariable, self.ptr_sh_arr, vid, StorageWorkgroup)
                self.shared.append(vid)
        else:
            self.shared = []

        self.cur = None
        self.main = s.id("main")
        name = "main"
        s.emit(
            s.entry,
            OpEntryPoint,
            ExecGLCompute,
            self.main,
            *_str_words(name),
            *iface,
        )
        s.emit(s.modes, OpExecutionMode, self.main, ModeLocalSize, 64, 1, 1)

        self.ssbo_i = []

    def begin_main(self):
        s = self.s
        s.inst(OpFunction, self.void, self.main, FnNone, self.fn_ty)
        self.entry_lab = s.id()
        s.inst(OpLabel, self.entry_lab)
        wg = s.result(OpLoad, self.uvec3, self.wgid)
        lid = s.result(OpLoad, self.uvec3, self.lidv)
        self.wg_x = s.result(OpCompositeExtract, self.u32, wg, 0)
        self.wg_y = s.result(OpCompositeExtract, self.u32, wg, 1)
        self.lid = s.result(OpCompositeExtract, self.u32, lid, 0)
        if self.texel0:
            self.texel_img = s.result(OpLoad, self.img_ty, self.texel_var)
        self.cur = self.entry_lab
        if self.use_global:
            gid = s.result(OpLoad, self.uvec3, self.gidv)
            nwg = s.result(OpLoad, self.uvec3, self.nwg)
            self.gid_x = s.result(OpCompositeExtract, self.u32, gid, 0)
            self.gid_y = s.result(OpCompositeExtract, self.u32, gid, 1)
            self.nwg_x = s.result(OpCompositeExtract, self.u32, nwg, 0)

    def end_main(self):
        self.s.inst(OpReturn)
        self.s.inst(OpFunctionEnd)

    def pc_i(self, idx: int) -> int:
        ptr = self.s.result(OpAccessChain, self.ptr_pc_i, self.pc, self.cu[idx])
        return self.s.result(OpLoad, self.i32, ptr)

    def pc_u(self, idx: int) -> int:
        return self.s.result(OpBitcast, self.u32, self.pc_i(idx))

    def pc_f(self, idx: int) -> int:
        ptr = self.s.result(OpAccessChain, self.ptr_pc_f, self.pc, self.cu[8 + idx])
        return self.s.result(OpLoad, self.f32, ptr)

    def load_f(self, binding: int, idx: int) -> int:
        if binding == 0 and self.texel0:
            coord = self.s.result(OpBitcast, self.i32, idx)
            v4 = self.s.result(OpImageFetch, self.v4f, self.texel_img, coord)
            return self.s.result(OpCompositeExtract, self.f32, v4, 0)
        ptr = self.s.result(OpAccessChain, self.ptr_el_f, self.ssbo[binding], self.cu[0], idx)
        return self.s.result(OpLoad, self.f32, ptr)

    def load_v4(self, binding: int, idx: int) -> int:
        coord = self.s.result(OpBitcast, self.i32, idx)
        return self.s.result(OpImageFetch, self.v4f, self.texel_img, coord)

    def ext_f(self, vec: int, i: int) -> int:
        return self.s.result(OpCompositeExtract, self.f32, vec, i)

    def load_f16(self, binding: int, idx: int) -> int:
        """Packed f16 in the float SSBO. Software decode — Adreno rejects UnpackHalf2x16."""
        pack = self.shr(idx, self.cu[1])
        bits = self.bitcast_u(self.load_f(binding, pack))
        odd = self.s.result(OpINotEqual, self.bool, self.band(idx, self.cu[1]), self.cu[0])
        h = self.select(odd, self.shr(bits, self.cu[16]), self.band(bits, self.cu[0xFFFF]), ty=self.u32)
        sign = self.shl(self.band(h, self.cu[0x8000]), self.cu[16])
        exp = self.band(self.shr(h, self.cu[10]), self.cu[31])
        man = self.band(h, self.cu[0x3FF])
        norm = self.bor(
            sign,
            self.bor(self.shl(self.uadd(exp, self.cu[112]), self.cu[23]), self.shl(man, self.cu[13])),
        )
        z = self.s.result(OpIEqual, self.bool, exp, self.cu[0])
        return self.s.result(OpBitcast, self.f32, self.select(z, sign, norm, ty=self.u32))

    def band(self, a, b):
        return self.s.result(OpBitwiseAnd, self.u32, a, b)

    def bor(self, a, b):
        return self.s.result(OpBitwiseOr, self.u32, a, b)

    def shl(self, a, b):
        return self.s.result(OpShiftLeftLogical, self.u32, a, b)

    def store_f(self, binding: int, idx: int, val: int) -> None:
        ptr = self.s.result(OpAccessChain, self.ptr_el_f, self.ssbo[binding], self.cu[0], idx)
        self.s.inst(OpStore, ptr, val)

    def load_i(self, binding: int, idx: int) -> int:
        bits = self.load_f(binding, idx)
        return self.s.result(OpBitcast, self.i32, bits)

    def sh_load(self, which: int, idx: int) -> int:
        ptr = self.s.result(OpAccessChain, self.ptr_sh_f, self.shared[which], idx)
        return self.s.result(OpLoad, self.f32, ptr)

    def sh_store(self, which: int, idx: int, val: int) -> None:
        ptr = self.s.result(OpAccessChain, self.ptr_sh_f, self.shared[which], idx)
        self.s.inst(OpStore, ptr, val)

    def barrier(self) -> None:
        self.s.inst(OpControlBarrier, self.u_scope, self.u_scope, self.u_sem)

    def uadd(self, a, b):
        return self.s.result(OpIAdd, self.u32, a, b)

    def umul(self, a, b):
        return self.s.result(OpIMul, self.u32, a, b)

    def usub(self, a, b):
        return self.s.result(OpISub, self.u32, a, b)

    def udiv(self, a, b):
        return self.s.result(OpUDiv, self.u32, a, b)

    def umod(self, a, b):
        return self.s.result(OpUMod, self.u32, a, b)

    def shr(self, a, b):
        return self.s.result(OpShiftRightLogical, self.u32, a, b)

    def ult(self, a, b):
        return self.s.result(OpULessThan, self.bool, a, b)

    def ule(self, a, b):
        return self.s.result(OpULessThanEqual, self.bool, a, b)

    def slt(self, a, b):
        return self.s.result(OpSLessThan, self.bool, a, b)

    def sle(self, a, b):
        return self.s.result(OpSLessThanEqual, self.bool, a, b)

    def ueq(self, a, b):
        return self.s.result(OpIEqual, self.bool, a, b)

    def land(self, a, b):
        return self.s.result(OpLogicalAnd, self.bool, a, b)

    def lnot(self, a):
        return self.s.result(OpLogicalNot, self.bool, a)

    def fadd(self, a, b):
        return self.s.result(OpFAdd, self.f32, a, b)

    def fsub(self, a, b):
        return self.s.result(OpFSub, self.f32, a, b)

    def fmul(self, a, b):
        return self.s.result(OpFMul, self.f32, a, b)

    def fdiv(self, a, b):
        return self.s.result(OpFDiv, self.f32, a, b)

    def fneg(self, a):
        return self.s.result(OpFNegate, self.f32, a)

    def fgt(self, a, b):
        return self.s.result(OpFOrdGreaterThan, self.bool, a, b)

    def select(self, c, t, f, ty=None):
        return self.s.result(OpSelect, ty or self.f32, c, t, f)

    def bitcast_u(self, i):
        return self.s.result(OpBitcast, self.u32, i)

    def bitcast_i(self, u):
        return self.s.result(OpBitcast, self.i32, u)

    def bitcast_f(self, u):
        return self.s.result(OpBitcast, self.f32, u)

    def iadd(self, a, b):
        return self.s.result(OpIAdd, self.i32, a, b)

    def utof(self, u):
        return self.s.result(OpConvertUToF, self.f32, u)

    def itof(self, i):
        return self.s.result(OpConvertSToF, self.f32, i)

    def ishl(self, a, b):
        return self.s.result(OpShiftLeftLogical, self.i32, a, b)

    def sshr(self, a, b):
        return self.s.result(OpShiftRightArithmetic, self.i32, a, b)

    def load_q8_w(self, binding: int, block_base: int, lane: int) -> int:
        """36-byte block at float index block_base: float scale + 32 int8s. lane 0..31."""
        d = self.load_f(binding, block_base)
        word_i = self.uadd(block_base, self.uadd(self.cu[1], self.shr(lane, self.cu[2])))
        word = self.bitcast_u(self.load_f(binding, word_i))
        sh = self.shl(self.band(lane, self.cu[3]), self.cu[3])
        b = self.band(self.shr(word, sh), self.cu[255])
        q = self.sshr(self.ishl(self.bitcast_i(b), self.cu[24]), self.cu[24])
        return self.fmul(self.itof(q), d)

    def fmax(self, a, b):
        return self.s.result(OpExtInst, self.f32, self.glsl, GLSL_FMax, a, b)

    def exp(self, a):
        return self.s.result(OpExtInst, self.f32, self.glsl, GLSL_Exp, a)

    def rsqrt(self, a):
        return self.s.result(OpExtInst, self.f32, self.glsl, GLSL_InverseSqrt, a)

    def global_1d(self) -> int:
        stride = self.umul(self.nwg_x, self.cu[64])
        return self.uadd(self.umul(self.gid_y, stride), self.gid_x)

    def if_(self, cond: int, then_fn, else_fn=None):
        s = self.s
        then_l = s.id()
        else_l = s.id() if else_fn else None
        merge_l = s.id()
        s.inst(OpSelectionMerge, merge_l, 0)
        s.inst(OpBranchConditional, cond, then_l, else_l or merge_l)
        s.inst(OpLabel, then_l)
        self.cur = then_l
        then_fn()
        s.inst(OpBranch, merge_l)
        if else_fn:
            s.inst(OpLabel, else_l)
            self.cur = else_l
            else_fn()
            s.inst(OpBranch, merge_l)
        s.inst(OpLabel, merge_l)
        self.cur = merge_l

    def for_u(self, init: int, end: int, step: int, body, carried=None):
        """for (u = init; u < end; u += step). carried: {name: (ty, init_id)} -> finals."""
        s = self.s
        carried = carried or {}
        pre = s.id()  # we are already in a block; just branch
        loop = s.id()
        body_l = s.id()
        cont = s.id()
        merge = s.id()
        s.inst(OpBranch, loop)

        s.inst(OpLabel, loop)
        u_phi = s.id()
        s.inst(OpPhi, self.u32, u_phi, init, self.entry_lab if False else 0)
        # patch phi predecessors after we know ids — emit properly now
        # SPIR-V OpPhi: type, result, (val, block)*
        # We already emitted a dummy; rebuild: can't patch easily. Emit correctly:

        raise RuntimeError("replaced below")


def _for_u(c: Comp, init: int, end: int, step_const: int, body, carried=None, pred=None):
    """pred is the current block id. We track current block via c.cur."""
    s = c.s
    carried = dict(carried or {})
    loop = s.id()
    body_l = s.id()
    cont = s.id()
    merge = s.id()
    pre = c.cur
    s.inst(OpBranch, loop)

    s.inst(OpLabel, loop)
    u_phi = s.id()
    for name, (ty, init_v) in list(carried.items()):
        cid = s.id()
        carried[name] = (ty, init_v, cid)
    s.fn.append(None)
    u_phi_idx = len(s.fn) - 1
    carried_idx = {}
    for name, (ty, init_v, cid) in carried.items():
        s.fn.append(None)
        carried_idx[name] = len(s.fn) - 1

    # LoopMerge must be immediately followed by the terminating branch.
    cmp = s.result(OpULessThan, c.bool, u_phi, end)
    s.inst(OpLoopMerge, merge, cont, 0)
    s.inst(OpBranchConditional, cmp, body_l, merge)

    s.inst(OpLabel, body_l)
    c.cur = body_l
    finals = body(u_phi, {n: carried[n][2] for n in carried}) or {}
    s.inst(OpBranch, cont)

    s.inst(OpLabel, cont)
    u_next = c.uadd(u_phi, c.cu[step_const])
    s.inst(OpBranch, loop)

    # fill phis
    def phi_inst(ty, rid, v0, b0, v1, b1):
        words = [ty, rid, v0, b0, v1, b1]
        return [((len(words) + 1) << 16) | OpPhi, *words]

    s.fn[u_phi_idx] = phi_inst(c.u32, u_phi, init, pre, u_next, cont)
    for name, (ty, init_v, cid) in carried.items():
        s.fn[carried_idx[name]] = phi_inst(ty, cid, init_v, pre, finals[name], cont)

    s.inst(OpLabel, merge)
    c.cur = merge
    out = {n: carried[n][2] for n in carried}
    out["_i"] = u_phi
    return out


# Patch Comp.for_u
Comp.for_u = lambda self, init, end, step, body, carried=None: _for_u(
    self, init, end, step, body, carried
)


def reduce_add(c: Comp, lid: int, sh: int = 0):
    def body(sval, car):
        def then():
            v = c.sh_load(sh, lid)
            other = c.sh_load(sh, c.uadd(lid, sval))
            c.sh_store(sh, lid, c.fadd(v, other))

        c.if_(c.ult(lid, sval), then)
        c.barrier()
        return {}

    # s = 32; s > 0; s >>= 1  — use for with end 0 doesn't work (unsigned <).
    # Unroll reduction: 32,16,8,4,2,1
    for off in (32, 16, 8, 4, 2, 1):
        o = c.cu[off] if off in c.cu else None
        if o is None:
            # 16,8,4,2,1 not in cu except 1,32,64. Add them.
            pass
    return


def ensure_cu(c: Comp):
    s = c.s
    for k in (8, 16):
        if k not in c.cu:
            cid = s.id()
            s.emit(s.types, OpConstant, c.u32, cid, k)
            c.cu[k] = cid


def reduce_unrolled(c: Comp, lid: int, sh: int, op="add"):
    ensure_cu(c)
    for off in (32, 16, 8, 4, 2, 1):
        o = c.cu[off]

        def then(off=off, o=o):
            v = c.sh_load(sh, lid)
            other = c.sh_load(sh, c.uadd(lid, o))
            if op == "add":
                c.sh_store(sh, lid, c.fadd(v, other))
            else:
                c.sh_store(sh, lid, c.fmax(v, other))

        c.if_(c.ult(lid, o), then)
        c.barrier()


def store_y(c: Comp, acc: int, idx: int, v: int) -> None:
    v = c.select(c.ueq(acc, c.cu[0]), v, c.fadd(c.load_f(2, idx), v))
    c.store_f(2, idx, v)


def shader_linear() -> bytes:
    """Each WG computes 8 consecutive outputs for one token; threads reduce over n_in."""
    c = Comp(3, shared_n=64, texel0=True)
    ensure_cu(c)
    c.begin_main()
    c.cur = c.entry_lab
    n_tok, n_out, n_in = c.pc_u(0), c.pc_u(1), c.pc_u(2)
    acc = c.pc_u(3)
    row0 = c.umul(c.wg_x, c.cu[8])
    t, lid = c.wg_y, c.lid

    def body():
        xbase = c.umul(t, n_in)

        def loop_body(j, car):
            xv = c.load_f(1, c.uadd(xbase, j))
            out = {}
            for r in range(8):
                i = c.uadd(row0, c.cu[r])
                inb = c.ult(i, n_out)
                widx = c.select(inb, c.uadd(c.umul(i, n_in), j), c.cu[0], ty=c.u32)
                w = c.load_f(0, widx)
                out[f"a{r}"] = c.fadd(car[f"a{r}"], c.select(inb, c.fmul(w, xv), c.f0))
            return out

        carried = {f"a{r}": (c.f32, c.f0) for r in range(8)}
        fin = c.for_u(lid, n_in, 64, loop_body, carried)

        for r in range(8):
            c.sh_store(0, lid, fin[f"a{r}"])
            c.barrier()
            reduce_unrolled(c, lid, 0, "add")
            i = c.uadd(row0, c.cu[r])

            def wr(i=i):
                store_y(c, acc, c.uadd(c.umul(t, n_out), i), c.sh_load(0, c.cu[0]))

            c.if_(c.land(c.ueq(lid, c.cu[0]), c.ult(i, n_out)), wr)
            c.barrier()

    c.if_(c.ult(t, n_tok), body)
    c.end_main()
    return c.s.finish()


def shader_linear_q8() -> bytes:
    """Q8_0 GEMV: 32-wide blocks, 8 output rows, 64 threads (two blocks/iter)."""
    c = Comp(3, shared_n=64)
    ensure_cu(c)
    c.begin_main()
    c.cur = c.entry_lab
    n_tok, n_out, n_in = c.pc_u(0), c.pc_u(1), c.pc_u(2)
    row0 = c.umul(c.wg_x, c.cu[8])
    t, lid = c.wg_y, c.lid
    n_blocks = c.shr(n_in, c.cu[5])
    group = c.shr(lid, c.cu[5])
    lane = c.band(lid, c.cu[31])

    def body():
        xbase = c.umul(t, n_in)

        def loop_body(blk, car):
            xv = c.load_f(1, c.uadd(xbase, c.uadd(c.umul(blk, c.cu[32]), lane)))
            out = {}
            for r in range(8):
                i = c.uadd(row0, c.cu[r])
                inb = c.ult(i, n_out)
                base = c.umul(c.uadd(c.umul(i, n_blocks), blk), c.cu[9])
                base = c.select(inb, base, c.cu[0], ty=c.u32)
                w = c.load_q8_w(0, base, lane)
                out[f"a{r}"] = c.fadd(car[f"a{r}"], c.select(inb, c.fmul(w, xv), c.f0))
            return out

        carried = {f"a{r}": (c.f32, c.f0) for r in range(8)}
        fin = c.for_u(group, n_blocks, 2, loop_body, carried)

        for r in range(8):
            c.sh_store(0, lid, fin[f"a{r}"])
            c.barrier()
            reduce_unrolled(c, lid, 0, "add")
            i = c.uadd(row0, c.cu[r])

            def wr(i=i):
                c.store_f(2, c.uadd(c.umul(t, n_out), i), c.sh_load(0, c.cu[0]))

            c.if_(c.land(c.ueq(lid, c.cu[0]), c.ult(i, n_out)), wr)
            c.barrier()

    c.if_(c.ult(t, n_tok), body)
    c.end_main()
    return c.s.finish()


def shader_linear_gemm() -> bytes:
    """4 tokens per WG column. Each W row is loaded once and reused across the tile."""
    tm = 4
    c = Comp(3, shared_n=64, texel0=True)
    ensure_cu(c)
    c.begin_main()
    c.cur = c.entry_lab
    n_tok, n_out, n_in = c.pc_u(0), c.pc_u(1), c.pc_u(2)
    acc = c.pc_u(3)
    row0 = c.umul(c.wg_x, c.cu[8])
    t0 = c.umul(c.wg_y, c.cu[tm])
    lid = c.lid

    for r in range(8):
        i = c.uadd(row0, c.cu[r])
        inb = c.ult(i, n_out)

        def loop_body(j, car, i=i, inb=inb):
            widx = c.select(inb, c.uadd(c.umul(i, n_in), j), c.cu[0], ty=c.u32)
            w = c.select(inb, c.load_f(0, widx), c.f0)
            out = {}
            for t in range(tm):
                tt = c.uadd(t0, c.cu[t])
                tv = c.ult(tt, n_tok)
                xv = c.select(tv, c.load_f(1, c.uadd(c.umul(tt, n_in), j)), c.f0)
                out[f"a{t}"] = c.fadd(car[f"a{t}"], c.select(c.land(inb, tv), c.fmul(w, xv), c.f0))
            return out

        fin = c.for_u(lid, n_in, 64, loop_body, {f"a{t}": (c.f32, c.f0) for t in range(tm)})
        for t in range(tm):
            tt = c.uadd(t0, c.cu[t])
            c.sh_store(0, lid, fin[f"a{t}"])
            c.barrier()
            reduce_unrolled(c, lid, 0, "add")

            def wr(tt=tt, i=i):
                store_y(c, acc, c.uadd(c.umul(tt, n_out), i), c.sh_load(0, c.cu[0]))

            c.if_(c.land(c.ueq(lid, c.cu[0]), c.land(c.ult(tt, n_tok), inb)), wr)
            c.barrier()

    c.end_main()
    return c.s.finish()


def shader_argmax() -> bytes:
    """One workgroup: argmax over n floats. Out[0] bits are the uint index."""
    c = Comp(2, shared_n=64)
    ensure_cu(c)
    c.begin_main()
    c.cur = c.entry_lab
    n, lid = c.pc_u(0), c.lid

    def scan(i, car):
        v = c.load_f(0, i)
        better = c.fgt(v, car["m"])
        return {"m": c.select(better, v, car["m"]), "idx": c.select(better, i, car["idx"], ty=c.u32)}

    fin = c.for_u(lid, n, 64, scan, {"m": (c.f32, c.f_bigneg), "idx": (c.u32, c.cu[0])})
    c.sh_store(0, lid, fin["m"])
    c.sh_store(1, lid, c.bitcast_f(fin["idx"]))
    c.barrier()

    for off in (32, 16, 8, 4, 2, 1):
        o = c.cu[off]

        def then(o=o):
            mv = c.sh_load(0, lid)
            ov = c.sh_load(0, c.uadd(lid, o))
            mi = c.sh_load(1, lid)
            oi = c.sh_load(1, c.uadd(lid, o))
            better = c.fgt(ov, mv)
            c.sh_store(0, lid, c.select(better, ov, mv))
            c.sh_store(1, lid, c.select(better, oi, mi))

        c.if_(c.ult(lid, o), then)
        c.barrier()

    def wr():
        c.store_f(1, c.cu[0], c.sh_load(1, c.cu[0]))

    c.if_(c.ueq(lid, c.cu[0]), wr)
    c.end_main()
    return c.s.finish()


def shader_linear_f16() -> bytes:
    """f16 RGBA16 texel GEMV: each thread dots 4 K-values; n_in must be a multiple of 4."""
    c = Comp(3, shared_n=64, texel0=True)
    ensure_cu(c)
    c.begin_main()
    c.cur = c.entry_lab
    n_tok, n_out, n_in = c.pc_u(0), c.pc_u(1), c.pc_u(2)
    acc = c.pc_u(3)
    row0 = c.umul(c.wg_x, c.cu[8])
    t, lid = c.wg_y, c.lid
    j0 = c.umul(lid, c.cu[4])

    def body():
        xbase = c.umul(t, n_in)

        def loop_body(j, car):
            xv0 = c.load_f(1, c.uadd(xbase, j))
            xv1 = c.load_f(1, c.uadd(xbase, c.uadd(j, c.cu[1])))
            xv2 = c.load_f(1, c.uadd(xbase, c.uadd(j, c.cu[2])))
            xv3 = c.load_f(1, c.uadd(xbase, c.uadd(j, c.cu[3])))
            out = {}
            for r in range(8):
                i = c.uadd(row0, c.cu[r])
                inb = c.ult(i, n_out)
                tidx = c.shr(c.select(inb, c.uadd(c.umul(i, n_in), j), c.cu[0], ty=c.u32), c.cu[2])
                w4 = c.load_v4(0, tidx)
                dot = c.fadd(
                    c.fadd(c.fmul(c.ext_f(w4, 0), xv0), c.fmul(c.ext_f(w4, 1), xv1)),
                    c.fadd(c.fmul(c.ext_f(w4, 2), xv2), c.fmul(c.ext_f(w4, 3), xv3)),
                )
                out[f"a{r}"] = c.fadd(car[f"a{r}"], c.select(inb, dot, c.f0))
            return out

        carried = {f"a{r}": (c.f32, c.f0) for r in range(8)}
        fin = c.for_u(j0, n_in, 256, loop_body, carried)

        for r in range(8):
            c.sh_store(0, lid, fin[f"a{r}"])
            c.barrier()
            reduce_unrolled(c, lid, 0, "add")
            i = c.uadd(row0, c.cu[r])

            def wr(i=i):
                store_y(c, acc, c.uadd(c.umul(t, n_out), i), c.sh_load(0, c.cu[0]))

            c.if_(c.land(c.ueq(lid, c.cu[0]), c.ult(i, n_out)), wr)
            c.barrier()

    c.if_(c.ult(t, n_tok), body)
    c.end_main()
    return c.s.finish()


def shader_rmsnorm() -> bytes:
    c = Comp(3, shared_n=64)
    ensure_cu(c)
    c.begin_main()
    c.cur = c.entry_lab
    n, n_tok = c.pc_u(0), c.pc_u(1)
    eps = c.pc_f(0)
    t, lid = c.wg_x, c.lid

    def body():
        base = c.umul(t, n)

        def ss_body(i, car):
            v = c.load_f(0, c.uadd(base, i))
            return {"ss": c.fadd(car["ss"], c.fmul(v, v))}

        fin = c.for_u(lid, n, 64, ss_body, {"ss": (c.f32, c.f0)})
        c.sh_store(0, lid, fin["ss"])
        c.barrier()
        reduce_unrolled(c, lid, 0, "add")
        tot = c.sh_load(0, c.cu[0])
        inv = c.rsqrt(c.fadd(c.fdiv(tot, c.utof(n)), eps))
        c.barrier()

        def scale_body(i, car):
            v = c.load_f(0, c.uadd(base, i))
            w = c.load_f(1, i)
            c.store_f(2, c.uadd(base, i), c.fmul(c.fmul(v, inv), w))
            return {}

        c.for_u(lid, n, 64, scale_body)

    c.if_(c.ult(t, n_tok), body)
    c.end_main()
    return c.s.finish()


def shader_silu() -> bytes:
    c = Comp(2, use_global=True)
    c.begin_main()
    c.cur = c.entry_lab
    n = c.pc_u(0)
    i = c.global_1d()

    def body():
        x = c.load_f(0, i)
        y = c.fdiv(x, c.fadd(c.f1, c.exp(c.fneg(x))))
        c.store_f(1, i, y)

    c.if_(c.ult(i, n), body)
    c.end_main()
    return c.s.finish()


def shader_silu_mul() -> bytes:
    c = Comp(3, use_global=True)
    c.begin_main()
    c.cur = c.entry_lab
    n = c.pc_u(0)
    i = c.global_1d()

    def body():
        x = c.load_f(0, i)
        g = c.load_f(1, i)
        s = c.fdiv(x, c.fadd(c.f1, c.exp(c.fneg(x))))
        c.store_f(2, i, c.fmul(s, g))

    c.if_(c.ult(i, n), body)
    c.end_main()
    return c.s.finish()


def shader_vec(op: str) -> bytes:
    c = Comp(3, use_global=True)
    c.begin_main()
    c.cur = c.entry_lab
    n = c.pc_u(0)
    i = c.global_1d()

    def body():
        a = c.load_f(0, i)
        b = c.load_f(1, i)
        y = c.fadd(a, b) if op == "add" else c.fmul(a, b)
        c.store_f(2, i, y)

    c.if_(c.ult(i, n), body)
    c.end_main()
    return c.s.finish()


def shader_softmax() -> bytes:
    c = Comp(1, shared_n=64)
    ensure_cu(c)
    c.begin_main()
    c.cur = c.entry_lab
    n, n_rows = c.pc_u(0), c.pc_u(1)
    row, lid = c.wg_x, c.lid

    def body():
        base = c.umul(row, n)

        def mx_body(i, car):
            v = c.load_f(0, c.uadd(base, i))
            return {"m": c.fmax(car["m"], v)}

        fin = c.for_u(lid, n, 64, mx_body, {"m": (c.f32, c.f_bigneg)})
        c.sh_store(0, lid, fin["m"])
        c.barrier()
        reduce_unrolled(c, lid, 0, "max")
        m = c.sh_load(0, c.cu[0])
        c.barrier()

        def exp_body(i, car):
            idx = c.uadd(base, i)
            e = c.exp(c.fsub(c.load_f(0, idx), m))
            c.store_f(0, idx, e)
            return {"s": c.fadd(car["s"], e)}

        fin2 = c.for_u(lid, n, 64, exp_body, {"s": (c.f32, c.f0)})
        c.sh_store(0, lid, fin2["s"])
        c.barrier()
        reduce_unrolled(c, lid, 0, "add")
        tot = c.sh_load(0, c.cu[0])
        inv = c.select(c.fgt(tot, c.f0), c.fdiv(c.f1, tot), c.f0)
        c.barrier()

        def norm_body(i, car):
            idx = c.uadd(base, i)
            c.store_f(0, idx, c.fmul(c.load_f(0, idx), inv))
            return {}

        c.for_u(lid, n, 64, norm_body)

    c.if_(c.ult(row, n_rows), body)
    c.end_main()
    return c.s.finish()


def shader_rope() -> bytes:
    c = Comp(4, use_global=True)
    c.begin_main()
    c.cur = c.entry_lab
    B, n_head, S, hd = c.pc_u(0), c.pc_u(1), c.pc_u(2), c.pc_u(3)
    pairs = c.udiv(hd, c.cu[2])
    total = c.umul(c.umul(c.umul(B, n_head), S), pairs)
    idx = c.global_1d()

    def body():
        p = c.umod(idx, pairs)
        tmp = c.udiv(idx, pairs)
        s = c.umod(tmp, S)
        tmp = c.udiv(tmp, S)
        h = c.umod(tmp, n_head)
        b = c.udiv(tmp, n_head)
        pos = c.load_i(3, c.uadd(c.umul(b, S), s))
        row = c.uadd(c.umul(c.uadd(c.umul(c.uadd(c.umul(b, n_head), h), S), s), hd), c.umul(p, c.cu[2]))
        cidx = c.uadd(c.umul(c.bitcast_u(pos), hd), c.umul(p, c.cu[2]))
        x0 = c.load_f(0, row)
        x1 = c.load_f(0, c.uadd(row, c.cu[1]))
        cv = c.load_f(1, cidx)
        sv = c.load_f(2, cidx)
        c.store_f(0, row, c.fsub(c.fmul(x0, cv), c.fmul(x1, sv)))
        c.store_f(0, c.uadd(row, c.cu[1]), c.fadd(c.fmul(x0, sv), c.fmul(x1, cv)))

    c.if_(c.ult(idx, total), body)
    c.end_main()
    return c.s.finish()


def shader_pack(merge: bool) -> bytes:
    c = Comp(2, use_global=True)
    c.begin_main()
    c.cur = c.entry_lab
    B, S, H, d = c.pc_u(0), c.pc_u(1), c.pc_u(2), c.pc_u(3)
    total = c.umul(c.umul(c.umul(B, S), H), d)
    idx = c.global_1d()

    def body():
        di = c.umod(idx, d)
        tmp = c.udiv(idx, d)
        h = c.umod(tmp, H)
        tmp = c.udiv(tmp, H)
        s = c.umod(tmp, S)
        b = c.udiv(tmp, S)
        src = c.uadd(c.umul(c.uadd(c.umul(c.uadd(c.umul(b, S), s), H), h), d), di)
        dst = c.uadd(c.umul(c.uadd(c.umul(c.uadd(c.umul(b, H), h), S), s), d), di)
        if merge:
            c.store_f(1, src, c.load_f(0, dst))
        else:
            c.store_f(1, dst, c.load_f(0, src))

    c.if_(c.ult(idx, total), body)
    c.end_main()
    return c.s.finish()


def shader_cache_store() -> bytes:
    c = Comp(6, use_global=True)
    c.begin_main()
    c.cur = c.entry_lab
    B, S, H, d = c.pc_u(0), c.pc_u(1), c.pc_u(2), c.pc_u(3)
    batch_stride, head_stride = c.pc_u(4), c.pc_u(5)
    total = c.umul(c.umul(c.umul(B, H), S), d)
    idx = c.global_1d()

    def body():
        di = c.umod(idx, d)
        tmp = c.udiv(idx, d)
        t = c.umod(tmp, S)
        tmp = c.udiv(tmp, S)
        h = c.umod(tmp, H)
        b = c.udiv(tmp, H)
        n = c.load_i(4, b)
        t_i = c.bitcast_i(t)

        def do_copy():
            s0 = c.load_i(5, b)
            src = c.uadd(c.umul(c.uadd(c.umul(c.uadd(c.umul(b, H), h), S), t), d), di)
            dst = c.uadd(
                c.uadd(
                    c.uadd(c.umul(b, batch_stride), c.umul(h, head_stride)),
                    c.umul(c.bitcast_u(c.iadd(s0, t_i)), d),
                ),
                di,
            )
            c.store_f(0, dst, c.load_f(2, src))
            c.store_f(1, dst, c.load_f(3, src))

        c.if_(c.slt(t_i, n), do_copy)

    c.if_(c.ult(idx, total), body)
    c.end_main()
    return c.s.finish()


def shader_attn() -> bytes:
    c = Comp(7, shared_n=64, use_global=True)
    ensure_cu(c)
    c.begin_main()
    c.cur = c.entry_lab
    B, n_head, S, n_kv = c.pc_u(0), c.pc_u(1), c.pc_u(2), c.pc_u(3)
    d = c.pc_u(4)
    max_k = c.pc_i(5)
    batch_stride, head_stride = c.pc_u(6), c.pc_u(7)
    scale = c.pc_f(0)
    lid = c.lid
    wg = c.uadd(c.umul(c.wg_y, c.nwg_x), c.wg_x)
    hs = c.umul(n_head, S)
    b = c.udiv(wg, hs)
    rem = c.umod(wg, hs)
    h = c.udiv(rem, S)
    sidx = c.umod(rem, S)

    def in_range():
        qpos = c.load_i(4, c.uadd(c.umul(b, S), sidx))
        vl = c.load_i(5, b)
        q_valid = c.slt(c.bitcast_i(sidx), vl)
        n_rep = c.udiv(n_head, n_kv)
        kv_h = c.udiv(h, n_rep)
        qbase = c.umul(c.uadd(c.umul(c.uadd(c.umul(b, n_head), h), S), sidx), d)
        qv = c.select(c.ult(lid, d), c.load_f(1, c.uadd(qbase, lid)), c.f0)
        c.sh_store(0, lid, qv)
        c.barrier()

        def invalid():
            def z():
                c.store_f(0, c.uadd(qbase, lid), c.f0)

            c.if_(c.ult(lid, d), z)

        def valid():
            klen = c.load_i(6, b)
            kbase0 = c.uadd(c.umul(b, batch_stride), c.umul(kv_h, head_stride))

            def kloop(kp_u, car):
                kp_i = c.bitcast_i(kp_u)
                vis = c.land(c.sle(kp_i, qpos), c.slt(kp_i, klen))
                kbase = c.uadd(kbase0, c.umul(kp_u, d))
                prod = c.select(
                    c.ult(lid, d),
                    c.fmul(c.sh_load(0, lid), c.load_f(2, c.uadd(kbase, lid))),
                    c.f0,
                )
                c.sh_store(1, lid, prod)
                c.barrier()
                reduce_unrolled(c, lid, 1, "add")
                score = c.select(vis, c.fmul(c.sh_load(1, c.cu[0]), scale), c.f_bigneg)
                m = car["m"]
                lsum = car["l"]
                acc = car["acc"]
                m_new = c.fmax(m, score)
                e = c.exp(c.fsub(score, m_new))
                alpha = c.exp(c.fsub(m, m_new))
                v = c.select(c.ult(lid, d), c.load_f(3, c.uadd(kbase, lid)), c.f0)
                acc2 = c.fadd(c.fmul(acc, alpha), c.fmul(e, v))
                l2 = c.fadd(c.fmul(lsum, alpha), e)
                c.barrier()
                return {"m": m_new, "l": l2, "acc": acc2}

            max_ku = c.bitcast_u(max_k)
            fin = c.for_u(
                c.cu[0],
                max_ku,
                1,
                kloop,
                {"m": (c.f32, c.f_bigneg), "l": (c.f32, c.f0), "acc": (c.f32, c.f0)},
            )

            def wr():
                y = c.select(c.fgt(fin["l"], c.f0), c.fdiv(fin["acc"], fin["l"]), c.f0)
                c.store_f(0, c.uadd(qbase, lid), y)

            c.if_(c.ult(lid, d), wr)

        c.if_(q_valid, valid, invalid)

    c.if_(c.ult(b, B), in_range)
    c.end_main()
    return c.s.finish()


def shader_gather(f16: bool = False) -> bytes:
    c = Comp(3, use_global=True)
    c.begin_main()
    c.cur = c.entry_lab
    n_rows, d = c.pc_u(0), c.pc_u(1)
    idx = c.global_1d()
    total = c.umul(n_rows, d)

    def body():
        j = c.umod(idx, d)
        t = c.udiv(idx, d)
        tid = c.bitcast_u(c.load_i(1, t))
        src = c.uadd(c.umul(tid, d), j)
        c.store_f(2, idx, c.load_f16(0, src) if f16 else c.load_f(0, src))

    c.if_(c.ult(idx, total), body)
    c.end_main()
    return c.s.finish()


SHADERS = [
    ("linear", shader_linear),
    ("rmsnorm", shader_rmsnorm),
    ("silu", shader_silu),
    ("vec_add", lambda: shader_vec("add")),
    ("vec_mul", lambda: shader_vec("mul")),
    ("softmax", shader_softmax),
    ("rope", shader_rope),
    ("pack", lambda: shader_pack(False)),
    ("merge", lambda: shader_pack(True)),
    ("cache_store", shader_cache_store),
    ("attn", shader_attn),
    ("gather", shader_gather),
    ("linear_f16", shader_linear_f16),
    ("gather_f16", lambda: shader_gather(True)),
    ("linear_gemm", shader_linear_gemm),
    ("argmax", shader_argmax),
    ("linear_q8", shader_linear_q8),
    ("silu_mul", shader_silu_mul),
]


def embed_header(blobs: dict[str, bytes], path: str) -> None:
    names = [n for n, _ in SHADERS]
    lines = [
        "/* Generated by emit_spv.py / compile.sh. Do not edit. */",
        "#ifndef LLM_SHADERS_SPV_H",
        "#define LLM_SHADERS_SPV_H",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "enum {",
    ]
    for i, n in enumerate(names):
        lines.append(f"    PIPE_{n.upper()} = {i},")
    lines.append("    PIPE_COUNT")
    lines.append("};")
    lines.append("")
    for n in names:
        words = struct.unpack("<" + "I" * (len(blobs[n]) // 4), blobs[n])
        lines.append(f"static const uint32_t spv_{n}[] = {{")
        row = []
        for w in words:
            row.append(f"0x{w:08x}")
            if len(row) == 8:
                lines.append("    " + ", ".join(row) + ",")
                row = []
        if row:
            lines.append("    " + ", ".join(row) + ",")
        lines.append("};")
        lines.append("")
    lines.append("typedef struct { const uint32_t *words; size_t n_words; } SpvBlob;")
    lines.append("static const SpvBlob spv_blobs[PIPE_COUNT] = {")
    for n in names:
        lines.append(f"    {{spv_{n}, sizeof(spv_{n}) / sizeof(spv_{n}[0])}},")
    lines.append("};")
    lines.append("")
    lines.append("#endif")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    blobs = {}
    for name, fn in SHADERS:
        data = fn()
        assert data[:4] == b"\x03\x02\x23\x07", name
        blobs[name] = data
        print(f"  {name}: {len(data)} bytes, {len(data)//4} words")
    embed_header(blobs, os.path.join(here, "shaders_spv.h"))
    print("wrote shaders_spv.h")


if __name__ == "__main__":
    main()
