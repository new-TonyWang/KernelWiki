/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "ir.h"
#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/IR/Instructions.h"


using namespace mlir;
namespace py = pybind11;

struct AscendNPUIROpBuilder : public TritonOpBuilder {};

namespace {
struct CoreAndPipes {
  hivm::TCoreTypeAttr core;
  hivm::PipeAttr producer;
  hivm::PipeAttr consumer;
};

CoreAndPipes GetCoreAndPipes(MLIRContext *ctx, llvm::StringRef opName,
                             llvm::StringRef sender) {
  // Step1: Decide pipes
  hivm::PipeAttr producer;
  hivm::PipeAttr consumer = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_MTE2);

  if (sender == "cube") {
    producer = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_FIX);
  } else {
    producer = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_MTE3);
  }

  // Step 2: Decide core type
  hivm::TCoreTypeAttr core;
  if (sender == "cube") {
    if (opName == "sync_block_set")
      core = hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::CUBE);
    else
      core = hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR);
  } else {
    if (opName == "sync_block_set")
      core = hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR);
    else
      core = hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::CUBE);
  }

  return {core, producer, consumer};
}
} // namespace

void init_ascend_ir(py::module &&m) {
  py::enum_<hivm::AddressSpace>(m, "AddressSpace", py::module_local())
      .value("L1", hivm::AddressSpace::L1)
      .value("UB", hivm::AddressSpace::UB)
      .value("L0A", hivm::AddressSpace::L0A)
      .value("L0B", hivm::AddressSpace::L0B)
      .value("L0C", hivm::AddressSpace::L0C)
      .export_values();

  m.def("load_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    registry.insert<mlir::hivm::HIVMDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  py::class_<AscendNPUIROpBuilder, TritonOpBuilder>(
      m, "ascendnpu_ir_builder", py::module_local(), py::dynamic_attr())
      .def(py::init<MLIRContext *>())
      .def("create_get_sub_vec_id",
           [](AscendNPUIROpBuilder &self) -> Value {
             return self.create<hivm::GetSubBlockIdxOp>();
           })
      .def("sync_block_set",
           [](AscendNPUIROpBuilder &self, std::string &sender, int id) -> void {
             auto *ctx = self.getBuilder().getContext();
             auto [coreAttr, prodPipe, consPipe] =
                 GetCoreAndPipes(ctx, "sync_block_set", sender);
             mlir::IndexType indexType = mlir::IndexType::get(ctx);
             mlir::Attribute indexAttribute =
                 mlir::IntegerAttr::get(indexType, static_cast<int64_t>(id));
             self.create<hivm::SyncBlockSetOp>(coreAttr, prodPipe, consPipe,
                                               indexAttribute);
           })
      .def("sync_block_wait",
           [](AscendNPUIROpBuilder &self, std::string &sender, int id) -> void {
             auto *ctx = self.getBuilder().getContext();
             auto [coreAttr, prodPipe, consPipe] =
                 GetCoreAndPipes(ctx, "sync_block_wait", sender);
             mlir::IndexType indexType = mlir::IndexType::get(ctx);
             mlir::Attribute indexAttribute =
                 mlir::IntegerAttr::get(indexType, static_cast<int64_t>(id));
             self.create<hivm::SyncBlockWaitOp>(coreAttr, prodPipe, consPipe,
                                                indexAttribute);
           })
      .def("get_target_attribute",
           [](AscendNPUIROpBuilder &self,
              hivm::AddressSpace &addressSpace) -> Attribute {
             return hivm::AddressSpaceAttr::get(self.getBuilder().getContext(),
                                                addressSpace);
           })
      .def("create_copy_buffer",
           [](AscendNPUIROpBuilder &self, Value src, Value dst) {
             self.create<hivm::CopyOp>(mlir::TypeRange{}, src, dst);
           })
      .def("create_copy_tensor",
           [](AscendNPUIROpBuilder &self, Value src, Value dst) {
             return self
                 .create<hivm::CopyOp>(mlir::TypeRange{dst.getType()}, src, dst)
                 .getResult(0);
           })
      .def("create_fixpipe",
           [](AscendNPUIROpBuilder &self, Value src, Value dst, bool enable_nz2nd) -> void {
             if (!dyn_cast<RankedTensorType>(src.getType())) {
               llvm_unreachable("src is not of RankedTensorType");
             }
             if (!dyn_cast<MemRefType>(dst.getType())) {
               llvm_unreachable("dst is not of MemRefType");
             }
             auto *ctx = self.getBuilder().getContext();
             auto op = self.create<hivm::FixpipeOp>(mlir::TypeRange{},
                                          mlir::ValueRange{src, dst});
             if (enable_nz2nd) {
               op->setAttr("enable_nz2nd", mlir::UnitAttr::get(ctx));
             }
           });
}
