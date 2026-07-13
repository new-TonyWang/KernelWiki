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
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/IR/Instructions.h"

using namespace mlir;
namespace py = pybind11;

struct AscendNPUIROpBuilder : public TritonOpBuilder {};

namespace {

hivm::TCoreTypeAttr GetCore(MLIRContext *ctx, llvm::StringRef opName,
                            llvm::StringRef sender) {
  // Decide core type
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

  return core;
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

  py::enum_<hivm::PIPE>(m, "PIPE", py::module_local())
      .value("PIPE_S", hivm::PIPE::PIPE_S)
      .value("PIPE_V", hivm::PIPE::PIPE_V)
      .value("PIPE_M", hivm::PIPE::PIPE_M)
      .value("PIPE_MTE1", hivm::PIPE::PIPE_MTE1)
      .value("PIPE_MTE2", hivm::PIPE::PIPE_MTE2)
      .value("PIPE_MTE3", hivm::PIPE::PIPE_MTE3)
      .value("PIPE_ALL", hivm::PIPE::PIPE_ALL)
      .value("PIPE_FIX", hivm::PIPE::PIPE_FIX)
      .export_values();

  m.def("load_dialects", [](MLIRContext &context) {
    // Allow unregistered dialects so we can parse HACC attributes without registering the dialect
    context.allowUnregisteredDialects();
    
    DialectRegistry registry;
    registry.insert<mlir::hivm::HIVMDialect, scope::ScopeDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  py::class_<AscendNPUIROpBuilder, TritonOpBuilder>(
      m, "ascendnpu_ir_builder", py::module_local(), py::dynamic_attr())
      .def(py::init<MLIRContext *>())
      .def("create_get_sub_vec_id",
           [](AscendNPUIROpBuilder &self) -> Value {
             auto subBlockIdxOp = self.create<hivm::GetSubBlockIdxOp>();
             auto moduleOp = subBlockIdxOp->getParentOfType<ModuleOp>();
             auto *ctx = self.getBuilder().getContext();
             // If user explicitly uses sub.block idx, add attribute to module.
             // NPU compiler will parse this attribute and disable auto tile and bind subblock pass.
             moduleOp->setAttr("hivm.disable_auto_tile_and_bind_subblock", mlir::UnitAttr::get(ctx));
             return subBlockIdxOp;
           })
      .def("sync_block_set",
           [](AscendNPUIROpBuilder &self, std::string &sender,
              std::string &receiver, int id, hivm::PIPE senderPipe,
              hivm::PIPE receiverPipe) -> void {
             auto *ctx = self.getBuilder().getContext();
             hivm::TCoreTypeAttr coreAttr =
                 GetCore(ctx, "sync_block_set", sender);
             hivm::PipeAttr prodPipe = hivm::PipeAttr::get(ctx, senderPipe);
             hivm::PipeAttr consPipe = hivm::PipeAttr::get(ctx, receiverPipe);
             mlir::IndexType indexType = mlir::IndexType::get(ctx);
             mlir::Attribute indexAttribute =
                 mlir::IntegerAttr::get(indexType, static_cast<int64_t>(id));
             self.create<hivm::SyncBlockSetOp>(coreAttr, prodPipe, consPipe,
                                               indexAttribute);
           })
      .def("sync_block_wait",
           [](AscendNPUIROpBuilder &self, std::string &sender,
              std::string &receiver, int id, hivm::PIPE senderPipe,
              hivm::PIPE receiverPipe) -> void {
             auto *ctx = self.getBuilder().getContext();
             hivm::TCoreTypeAttr coreAttr =
                 GetCore(ctx, "sync_block_wait", sender);
             hivm::PipeAttr prodPipe = hivm::PipeAttr::get(ctx, senderPipe);
             hivm::PipeAttr consPipe = hivm::PipeAttr::get(ctx, receiverPipe);
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
      .def("get_function_kind_device_attr",
           [](AscendNPUIROpBuilder &self) -> Attribute {
             // Parse attribute from string to avoid interface initialization issues
             auto *ctx = self.getBuilder().getContext();
             return mlir::parseAttribute("#hacc.function_kind<DEVICE>", ctx);
           })
      .def("get_func_core_type_aic_attr",
           [](AscendNPUIROpBuilder &self) -> Attribute {
             return hivm::TFuncCoreTypeAttr::get(self.getBuilder().getContext(),
                                                hivm::TFuncCoreType::AIC);
           })
      .def("get_func_core_type_aiv_attr",
           [](AscendNPUIROpBuilder &self) -> Attribute {
             return hivm::TFuncCoreTypeAttr::get(self.getBuilder().getContext(),
                                                hivm::TFuncCoreType::AIV);
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
           })
      .def("create_scope_op",
           [](AscendNPUIROpBuilder &self, py::dict &scopeAttrs) -> OpState {
             llvm::SmallVector<NamedAttribute> attrs;
             for (auto item : scopeAttrs) {
               std::string key = py::cast<std::string>(item.first);
               Attribute value = py::cast<Attribute>(item.second);
               attrs.push_back(NamedAttribute(
                   self.getBuilder().getStringAttr(key), value));
             }
             auto scopeOp = self.create<scope::ScopeOp>();
             for (const auto &attr : attrs) {
               scopeOp->setAttr(attr.getName(), attr.getValue());
             }
             Block *entryBlock = self.getBuilder().createBlock(&scopeOp.getRegion());
             return OpState(scopeOp);
           })
      .def("scope_return",
           [](AscendNPUIROpBuilder &self) -> OpState {
             return self.create<scope::ReturnOp>();
           });
}
