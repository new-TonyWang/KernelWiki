
#include "../op_kernel/mte_micro_bench_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  MteMicroBenchTilingData *tiling = context->GetTilingData<MteMicroBenchTilingData>();
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int64_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  }
  const gert::RuntimeAttrs *attrs = context->GetAttrs();
  int64_t mode = attrs && attrs->GetInt(0) ? *attrs->GetInt(0) : 0;
  int64_t loops = attrs && attrs->GetInt(1) ? *attrs->GetInt(1) : 1024;
  int64_t bytes_per_loop = attrs && attrs->GetInt(2) ? *attrs->GetInt(2) : 32768;
  tiling->size = static_cast<uint32_t>(data_sz);
  tiling->mode = static_cast<uint32_t>(mode);
  tiling->loops = static_cast<uint32_t>(loops);
  tiling->elemsPerLoop = static_cast<uint32_t>(bytes_per_loop / 2);
  if (tiling->elemsPerLoop == 0 || tiling->elemsPerLoop > 16 * 1024) tiling->elemsPerLoop = 16 * 1024;
  context->SetBlockDim(1);
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class MteMicroBench : public OpDef {
public:
    explicit MteMicroBench(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("mode").Int();
        this->Attr("loops").Int();
        this->Attr("bytes_per_loop").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(MteMicroBench);
}
