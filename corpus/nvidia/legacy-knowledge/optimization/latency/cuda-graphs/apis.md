# Cuda Graphs -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cudaGraphAddKernelNode` | Runtime API | Creates a kernel execution node and adds it to a graph |
| `cudaGraphCreate` | Runtime API | Creates a graph |
| `cudaGraphExecUpdate` | Runtime API | Updates an instantiated graph with changes from the source graph |
| `cudaGraphInstantiate` | Runtime API | Creates an executable graph from a graph |
| `cudaGraphInstantiateWithFlags` | Runtime API | Creates an executable graph from a graph with flags |
| `cudaGraphInstantiateWithParams` | Runtime API | Creates an executable graph from a graph with params (upload stream, error info) |
| `cudaGraphLaunch` | Runtime API | Launches an executable graph in a stream |
| `cudaGraphUpload` | Runtime API | Uploads an executable graph to a device |
| `cudaStreamBeginCapture` | Runtime API | Begins graph capture on a stream |
| `cudaStreamBeginCaptureToGraph` | Runtime API | Begins graph capture on a stream to an existing graph |
| `cudaStreamEndCapture` | Runtime API | Ends capture on a stream, returning the captured graph |

## Related APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cudaDeviceGetGraphMemAttribute` | Runtime API | Query async allocation attributes related to graphs |
| `cudaDeviceGraphMemTrim` | Runtime API | Free unused cached graph memory back to OS |
| `cudaDeviceSetGraphMemAttribute` | Runtime API | Set async allocation attributes related to graphs |
| `cudaFuncGetParamInfo` | Runtime API | Returns the offset and size of a kernel parameter in device-side parameter layout |
| `cudaGetCurrentGraphExec` | Runtime API | (__device__) Get the currently running device graph id |
| `cudaGraphAddChildGraphNode` | Runtime API | Creates a child graph node and adds it to a graph |
| `cudaGraphAddDependencies` | Runtime API | Adds dependency edges to a graph |
| `cudaGraphAddEmptyNode` | Runtime API | Creates an empty node and adds it to a graph |
| `cudaGraphAddEventRecordNode` | Runtime API | Creates an event record node and adds it to a graph |
| `cudaGraphAddEventWaitNode` | Runtime API | Creates an event wait node and adds it to a graph |
| `cudaGraphAddExternalSemaphoresSignalNode` | Runtime API | Creates an external semaphore signal node |
| `cudaGraphAddExternalSemaphoresWaitNode` | Runtime API | Creates an external semaphore wait node |
| `cudaGraphAddHostNode` | Runtime API | Creates a host execution node and adds it to a graph |
| `cudaGraphAddMemAllocNode` | Runtime API | Creates a memory allocation node and adds it to a graph |
| `cudaGraphAddMemFreeNode` | Runtime API | Creates a memory free node and adds it to a graph |
| `cudaGraphAddMemcpyNode` | Runtime API | Creates a memcpy node and adds it to a graph |
| `cudaGraphAddMemcpyNode1D` | Runtime API | Creates a 1D memcpy node and adds it to a graph |
| `cudaGraphAddMemsetNode` | Runtime API | Creates a memset node and adds it to a graph |
| `cudaGraphAddNode` | Runtime API | Creates a generic typed node and adds it to a graph |
| `cudaGraphClone` | Runtime API | Clones a graph |
| `cudaGraphConditionalHandleCreate` | Runtime API | Create a conditional handle for graph conditional nodes |
| `cudaGraphConditionalHandleCreate_v2` | Runtime API | Create a conditional handle with extended params |
| `cudaGraphDestroy` | Runtime API | Destroys a graph |
| `cudaGraphDestroyNode` | Runtime API | Remove a node from the graph and destroy it |
| `cudaGraphExecChildGraphNodeSetParams` | Runtime API | Updates child graph node params in executable graph |
| `cudaGraphExecDestroy` | Runtime API | Destroys an executable graph |
| `cudaGraphExecEventRecordNodeSetEvent` | Runtime API | Sets event for event record node in executable graph |
| `cudaGraphExecEventWaitNodeSetEvent` | Runtime API | Sets event for event wait node in executable graph |
| `cudaGraphExecGetFlags` | Runtime API | Returns the flags that were passed to instantiation |
| `cudaGraphExecHostNodeSetParams` | Runtime API | Sets host node params in an executable graph |
| `cudaGraphExecKernelNodeSetParams` | Runtime API | Sets the params of a kernel node in an executable graph |
| `cudaGraphExecMemcpyNodeSetParams` | Runtime API | Sets memcpy node params in an executable graph |
| `cudaGraphExecMemsetNodeSetParams` | Runtime API | Sets memset node params in an executable graph |
| `cudaGraphExecNodeSetParams` | Runtime API | Update params of any node type in an executable graph |
| `cudaGraphGetEdges` | Runtime API | Returns a graph's dependency edges |
| `cudaGraphGetNodes` | Runtime API | Returns a graph's nodes |
| `cudaGraphGetRootNodes` | Runtime API | Returns a graph's root nodes |
| `cudaGraphInstantiateFlags` | Runtime API | Graph instantiate flags (DeviceLaunch, UseNodePriority) |
| `cudaGraphKernelNodeCopyAttributes` | Runtime API | Copy attributes from one kernel node to another |
| `cudaGraphKernelNodeGetAttribute` | Runtime API | Queries node attribute (e.g., access policy window) |
| `cudaGraphKernelNodeGetParams` | Runtime API | Gets a kernel node's parameters |
| `cudaGraphKernelNodeSetAttribute` | Runtime API | Sets node attribute (e.g., access policy window) |
| `cudaGraphKernelNodeSetEnabled` | Runtime API | Enables or disables a kernel node |
| `cudaGraphKernelNodeSetGridDim` | Runtime API | Updates the grid dimensions of a kernel node |
| `cudaGraphKernelNodeSetParam` | Runtime API | (__device__) Sets a single parameter of a kernel node from device code |
| `cudaGraphKernelNodeSetParams` | Runtime API | Sets a kernel node's parameters |
| `cudaGraphKernelNodeUpdatesApply` | Runtime API | (__device__) Applies kernel node parameter updates from device code |
| `cudaGraphNodeFindInClone` | Runtime API | Finds a node in a cloned graph corresponding to original |
| `cudaGraphNodeGetDependencies` | Runtime API | Returns a node's dependencies |
| `cudaGraphNodeGetDependentNodes` | Runtime API | Returns a node's dependent nodes |
| `cudaGraphNodeGetEnabled` | Runtime API | Queries whether a node is enabled |
| `cudaGraphNodeGetParams` | Runtime API | Returns params for a node of any type |
| `cudaGraphNodeGetType` | Runtime API | Returns a node's type |
| `cudaGraphNodeSetEnabled` | Runtime API | Enables/disables a node |
| `cudaGraphNodeSetParams` | Runtime API | Update params for any node type |
| `cudaGraphRemoveDependencies` | Runtime API | Removes dependency edges from a graph |
| `cudaGraphSetConditional` | Runtime API | Set conditional handle value |
| `cudaStreamCaptureMode` | Runtime API | Graph capture mode (Global, ThreadLocal, Relaxed) |
| `cudaStreamGetCaptureInfo` | Runtime API | Query a stream's capture state (id, graph, dependencies) |
| `cudaStreamIsCapturing` | Runtime API | Returns a stream's capture status |
| `cudaStreamUpdateCaptureDependencies` | Runtime API | Update capture dependencies of a stream |
| `cudaThreadExchangeStreamCaptureMode` | Runtime API | Swaps the stream capture mode of the calling thread |
