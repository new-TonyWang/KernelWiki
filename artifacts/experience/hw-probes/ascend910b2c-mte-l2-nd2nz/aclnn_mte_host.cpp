#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include "aclnn_mte_micro_bench.h"
#define CHECK_ACL(x) do { aclError _e=(x); if(_e!=ACL_SUCCESS){fprintf(stderr,"ACL fail %s:%d code=%d\n",__FILE__,__LINE__,(int)_e); return 2;} } while(0)
#define CHECK_NN(x) do { aclnnStatus _e=(x); if(_e!=0){fprintf(stderr,"ACLNN fail %s:%d code=%d\n",__FILE__,__LINE__,(int)_e); return 3;} } while(0)
static int RunOnce(aclTensor* tx, aclTensor* ty, aclrtStream stream, int mode, int loops, int bytesPerLoop, void** wsOut=nullptr, uint64_t* wsSizeOut=nullptr) {
  uint64_t wsSize=0; aclOpExecutor* executor=nullptr;
  aclnnStatus st=aclnnMteMicroBenchGetWorkspaceSize(tx, mode, loops, bytesPerLoop, ty, &wsSize, &executor);
  if(st!=0){fprintf(stderr,"GetWorkspace failed mode=%d code=%d\n",mode,(int)st); return 3;}
  void* ws=nullptr; if(wsSize>0){ aclError e=aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST); if(e!=ACL_SUCCESS) return 2; }
  st=aclnnMteMicroBench(ws, wsSize, executor, stream);
  if(st!=0){fprintf(stderr,"Run failed mode=%d code=%d\n",mode,(int)st); return 3;}
  if(wsOut){*wsOut=ws; *wsSizeOut=wsSize;} else if(ws) aclrtFree(ws);
  return 0;
}
int main(int argc, char** argv){
  int mode = argc>1 ? atoi(argv[1]) : 99;
  int loops = argc>2 ? atoi(argv[2]) : 1;
  int64_t nelems = argc>3 ? atoll(argv[3]) : (8LL*1024*1024);
  int bytesPerLoop = argc>4 ? atoi(argv[4]) : 32768;
  int dev = argc>5 ? atoi(argv[5]) : 3;
  int repeats = argc>6 ? atoi(argv[6]) : 1;
  int warmups = argc>7 ? atoi(argv[7]) : 1;
  CHECK_ACL(aclInit(nullptr)); CHECK_ACL(aclrtSetDevice(dev));
  aclrtStream stream=nullptr; CHECK_ACL(aclrtCreateStream(&stream));
  size_t nbytes=(size_t)nelems*2; void *dx=nullptr,*dy=nullptr;
  CHECK_ACL(aclrtMalloc(&dx,nbytes,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK_ACL(aclrtMalloc(&dy,nbytes,ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<uint16_t> hx(nelems>1024?1024:nelems,0x3c00); CHECK_ACL(aclrtMemcpy(dx,hx.size()*2,hx.data(),hx.size()*2,ACL_MEMCPY_HOST_TO_DEVICE));
  int64_t dims[1]={nelems}; int64_t strides[1]={1};
  aclTensor* tx=aclCreateTensor(dims,1,ACL_FLOAT16,strides,0,ACL_FORMAT_ND,dims,1,dx);
  aclTensor* ty=aclCreateTensor(dims,1,ACL_FLOAT16,strides,0,ACL_FORMAT_ND,dims,1,dy);
  if(!tx||!ty){fprintf(stderr,"aclCreateTensor failed\n"); return 4;}
  int probeMode=mode;
  if(mode==30 || mode==31){
    // single-cube warm mode0 first, then time a two-cube probe on the same allocation.
    int rc=RunOnce(tx,ty,stream,0,loops,bytesPerLoop); if(rc) return rc; CHECK_ACL(aclrtSynchronizeStream(stream));
    probeMode=(mode==30)?20:21;
    warmups=0;
  }
  for(int i=0;i<warmups;i++){ int rc=RunOnce(tx,ty,stream,probeMode,loops,bytesPerLoop); if(rc) return rc; }
  CHECK_ACL(aclrtSynchronizeStream(stream));
  aclrtEvent ev0=nullptr, ev1=nullptr; CHECK_ACL(aclrtCreateEvent(&ev0)); CHECK_ACL(aclrtCreateEvent(&ev1));
  CHECK_ACL(aclrtRecordEvent(ev0, stream));
  for(int i=0;i<repeats;i++){ int rc=RunOnce(tx,ty,stream,probeMode,loops,bytesPerLoop); if(rc) return rc; }
  CHECK_ACL(aclrtRecordEvent(ev1, stream)); CHECK_ACL(aclrtSynchronizeStream(stream));
  float ms=0.0f; CHECK_ACL(aclrtEventElapsedTime(&ms, ev0, ev1));
  printf("OK mode=%d probeMode=%d loops=%d nelems=%ld bytesPerLoop=%d repeats=%d total_ms=%.6f avg_us=%.6f\n", mode, probeMode, loops, (long)nelems, bytesPerLoop, repeats, ms, ms*1000.0f/repeats);
  aclrtDestroyEvent(ev0); aclrtDestroyEvent(ev1); aclDestroyTensor(tx); aclDestroyTensor(ty); aclrtFree(dx); aclrtFree(dy); aclrtDestroyStream(stream); aclrtResetDevice(dev); aclFinalize();
  return 0;
}
