diff --git a/src/ExpansionData.hpp b/src/ExpansionData.hpp
index ce0a77b..9d41d47 100644
--- a/src/ExpansionData.hpp
+++ b/src/ExpansionData.hpp
@@ -12,7 +12,8 @@ struct ExpansionData
     gfl::VectorView<NodeInfo> childrenInfo{};
     gfl::VectorView<NodeInfo> parentInfo{};
 
-    gfl::ArrayView<Node> tmpView;
+    gfl::ArrayView<Node> tmpViewNodes;
+    gfl::ArrayView<NodeInfo> tmpViewInfo;
     gfl::VectorView<Node> tmpNodes{};
     gfl::VectorView<NodeInfo> tmpInfo{};
 
diff --git a/src_gpu/ExpansionEngineGpu.cuh b/src_gpu/ExpansionEngineGpu.cuh
index 050669c..9db0926 100644
--- a/src_gpu/ExpansionEngineGpu.cuh
+++ b/src_gpu/ExpansionEngineGpu.cuh
@@ -50,7 +50,27 @@ class ExpansionEngineGpu : public ExpansionEngine<Model,Node>
     }
 
     void swapParentsAndChildren()
-    { expData.swapParentsAndChildren(); }
+    {
+        using namespace gfl;
+        auto & parents = expData.parents;
+        auto & children = expData.children;
+        auto & childrenInfo = expData.childrenInfo;
+        auto & parentInfo = expData.parentInfo;
+        auto & tmpNodes = expData.tmpNodes;
+        auto & tmpInfo = expData.tmpInfo;
+
+        parents.resizeTo(childrenInfo.size());
+        i32 blockSize = 128;
+        i32 gridSize = ceil<i32>(childrenInfo.size(),blockSize) ;
+        copyByInfoKernel<<<gridSize,blockSize>>>(&parents,&children, &childrenInfo);
+        CHECK_LAST_CUDA_ERROR();
+        resizeToKernel<<<1,1>>>(&children,0);
+        resizeToKernel<<<1,1>>>(&childrenInfo,0);
+        resizeToKernel<<<1,1>>>(&parentInfo,0);
+        resizeToKernel<<<1,1>>>(&tmpNodes,0);
+        resizeToKernel<<<1,1>>>(&tmpInfo,0);
+        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
+    }
 
     void expandParents(
         Model const * const model,
@@ -105,12 +125,12 @@ class ExpansionEngineGpu : public ExpansionEngine<Model,Node>
         CHECK_LAST_CUDA_ERROR();
         resizeToKernel<<<1,1>>>(&childrenInfo,&nFlagged);
         CHECK_LAST_CUDA_ERROR();
-        resizeToKernel<<<1,1>>>(&tmpNodes,&nFlagged);
-        CHECK_LAST_CUDA_ERROR();
-        copyByInfoKernel<<<gridSize,blockSize>>>(&tmpNodes,&children,&childrenInfo);
-        CHECK_LAST_CUDA_ERROR();
-        swapKernel<<<1,1>>>(&tmpNodes, &children);
-        CHECK_LAST_CUDA_ERROR();
+        // resizeToKernel<<<1,1>>>(&tmpNodes,&nFlagged);
+        // CHECK_LAST_CUDA_ERROR();
+        // copyByInfoKernel<<<gridSize,blockSize>>>(&tmpNodes,&children,&childrenInfo);
+        // CHECK_LAST_CUDA_ERROR();
+        // swapKernel<<<1,1>>>(&tmpNodes, &children);
+        // CHECK_LAST_CUDA_ERROR();
 
         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
     }
@@ -136,12 +156,12 @@ class ExpansionEngineGpu : public ExpansionEngine<Model,Node>
         CHECK_LAST_CUDA_ERROR();
         swapKernel<<<1,1>>>(&childrenInfo, &tmpInfo);
         CHECK_LAST_CUDA_ERROR();
-        resizeToKernel<<<1,1>>>(&childrenInfo,children.sizePtr());
-        CHECK_LAST_CUDA_ERROR();
-        copyByInfoKernel<<<gridSize,blockSize>>>(&tmpNodes,&children, &childrenInfo);
-        CHECK_LAST_CUDA_ERROR();
-        swapKernel<<<1,1>>>(&tmpNodes, &children);
-        CHECK_LAST_CUDA_ERROR();
+        // resizeToKernel<<<1,1>>>(&childrenInfo,children.sizePtr());
+        // CHECK_LAST_CUDA_ERROR();
+        // copyByInfoKernel<<<gridSize,blockSize>>>(&tmpNodes,&children, &childrenInfo);
+        // CHECK_LAST_CUDA_ERROR();
+        // swapKernel<<<1,1>>>(&tmpNodes, &children);
+        // CHECK_LAST_CUDA_ERROR();
 
         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
     }
@@ -196,8 +216,6 @@ class ExpansionEngineGpu : public ExpansionEngine<Model,Node>
         CHECK_LAST_CUDA_ERROR();
         swapKernel<<<1,1>>>(&tmpInfo, &parentsInfo);
         CHECK_LAST_CUDA_ERROR();
-        resizeToKernel<<<1,1>>>(&parentsInfo, &nFlagged);
-        CHECK_LAST_CUDA_ERROR();
         resizeByKernel<<<1,1>>>(&cutset, &nFlagged);
         CHECK_LAST_CUDA_ERROR();
 
@@ -223,49 +241,89 @@ class ExpansionEngineGpu : public ExpansionEngine<Model,Node>
         auto & children = expData.children;
         auto & tmpNodes = expData.tmpNodes;
         auto & childrenInfo = expData.childrenInfo;
-        auto & childrenPrefix = expData.tmpView;
+        auto & tmpInfo = expData.tmpInfo;        // now holds NodeInfo, not Nodes
+        auto & childrenSuffix = expData.tmpViewInfo;
 
         // Init
-        tmpNodes.resizeTo(children.size());
-        initChildrenPrefix(width_,&children,&childrenPrefix);
+        initChildrenSuffix(width_, &childrenInfo, &childrenSuffix);
 
-        // Merge the last children - (width - 1) nodes
+        // Merge
         i32 const blockSize       = 32;
         i32 const nodesPerThread  = 32;
         i32 const reductionFactor = blockSize * nodesPerThread;  // 1024
-        i32 const nNodes          = childrenPrefix.size(); // real item count
-        i32 const nBlocks1        = ceil<i32>(nNodes,   reductionFactor);  // gridDim for pass1, real count for pass2
-        i32 const nBlocks2        = ceil<i32>(nBlocks1, reductionFactor);  // gridDim for pass2, real count for pass3
-        // pass1: childrenPrefix → tmpNodes, count = nNodes
-        reductionKernel<Model,Node><<<nBlocks1, blockSize>>>(&childrenPrefix,&tmpNodes,nNodes);
+        i32 const nNodes          = childrenSuffix.size();
+        i32 const nBlocks1        = ceil<i32>(nNodes,   reductionFactor);
+        i32 const nBlocks2        = ceil<i32>(nBlocks1, reductionFactor);
+
+        // pass1: childrenPrefix → tmpInfo, count = nNodes
+        reductionKernel<Model,Node><<<nBlocks1, blockSize>>>(&children, &childrenSuffix, &tmpInfo, nNodes);
         CHECK_LAST_CUDA_ERROR();
-        // pass2: tmpNodes → childrenPrefix, count = nBlocks1
-        reductionKernel<Model,Node><<<nBlocks2, blockSize>>>(&tmpNodes,&childrenPrefix,nBlocks1);
+        // pass2: tmpInfo → childrenPrefix, count = nBlocks1
+        reductionKernel<Model,Node><<<nBlocks2, blockSize>>>(&children, &tmpInfo, &childrenSuffix, nBlocks1);
         CHECK_LAST_CUDA_ERROR();
-        // pass3: childrenPrefix → childrenPrefix[0], count = nBlocks2
-        reductionSeqKernel<Model,Node><<<1,1>>>(&childrenPrefix, nBlocks2);
+        // pass3: childrenPrefix[0] = merged result, count = nBlocks2
+        reductionSeqKernel<Model,Node><<<1,1>>>(&children, &childrenSuffix, nBlocks2);
         CHECK_LAST_CUDA_ERROR();
-        resizeToKernel<<<1,1>>>(&children,width_);
+        resizeToKernel<<<1,1>>>(&children, width_);
         CHECK_LAST_CUDA_ERROR();
-        resizeToKernel<<<1,1>>>(&childrenInfo,width_);
+        resizeToKernel<<<1,1>>>(&childrenInfo, width_);
         CHECK_LAST_CUDA_ERROR();
 
         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
     }
 
+    // void mergeChildren()
+    // {
+    //     using namespace gfl;
+    //
+    //     auto const & parents = expData.parents;
+    //     auto & children = expData.children;
+    //     auto & tmpNodes = expData.tmpNodes;
+    //     auto & childrenInfo = expData.childrenInfo;
+    //     auto & childrenPrefix = expData.tmpView;
+    //
+    //     // Init
+    //     tmpNodes.resizeTo(children.size());
+    //     initChildrenPrefix(width_,&children,&childrenPrefix);
+    //
+    //     // Merge the last children - (width - 1) nodes
+    //     i32 const blockSize       = 32;
+    //     i32 const nodesPerThread  = 32;
+    //     i32 const reductionFactor = blockSize * nodesPerThread;  // 1024
+    //     i32 const nNodes          = childrenPrefix.size(); // real item count
+    //     i32 const nBlocks1        = ceil<i32>(nNodes,   reductionFactor);  // gridDim for pass1, real count for pass2
+    //     i32 const nBlocks2        = ceil<i32>(nBlocks1, reductionFactor);  // gridDim for pass2, real count for pass3
+    //     // pass1: childrenPrefix → tmpNodes, count = nNodes
+    //     reductionKernel<Model,Node><<<nBlocks1, blockSize>>>(&childrenPrefix,&tmpNodes,nNodes);
+    //     CHECK_LAST_CUDA_ERROR();
+    //     // pass2: tmpNodes → childrenPrefix, count = nBlocks1
+    //     reductionKernel<Model,Node><<<nBlocks2, blockSize>>>(&tmpNodes,&childrenPrefix,nBlocks1);
+    //     CHECK_LAST_CUDA_ERROR();
+    //     // pass3: childrenPrefix → childrenPrefix[0], count = nBlocks2
+    //     reductionSeqKernel<Model,Node><<<1,1>>>(&childrenPrefix, nBlocks2);
+    //     CHECK_LAST_CUDA_ERROR();
+    //     resizeToKernel<<<1,1>>>(&children,width_);
+    //     CHECK_LAST_CUDA_ERROR();
+    //     resizeToKernel<<<1,1>>>(&childrenInfo,width_);
+    //     CHECK_LAST_CUDA_ERROR();
+    //
+    //     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
+    // }
+
     void calcOutLabels(
-        Model const * const model,
-        gfl::VectorView<Node> * const nodes,
-        gfl::i64 const primal,
-        gfl::i64 const dual)
+    Model const * const model,
+    gfl::VectorView<Node>     * const nodes,
+    gfl::VectorView<NodeInfo> * const nodesInfo,
+    gfl::i64 const primal,
+    gfl::i64 const dual)
     {
         using namespace gfl;
 
-        if (not nodes->empty())
+        if (not nodesInfo->empty())
         {
             i32 const blockSize = 128;
-            i32 const gridSize = ceil<i32>(nodes->size(),blockSize);
-            calcOutLabelsKernel<<<gridSize,blockSize>>>(model,nodes,primal,dual,DDRelaxed);
+            i32 const gridSize = ceil<i32>(nodesInfo->size(), blockSize);
+            calcOutLabelsKernel<<<gridSize,blockSize>>>(model, nodes, nodesInfo, primal, dual, DDRelaxed);
             CHECK_LAST_CUDA_ERROR();
 
             CHECK_CUDA_ERROR(cudaDeviceSynchronize());
@@ -274,19 +332,54 @@ class ExpansionEngineGpu : public ExpansionEngine<Model,Node>
 
     void checkForTarget(
         Model const * const model,
-        gfl::VectorView<Node> * const nodes)
+        gfl::VectorView<Node>     * const nodes,
+        gfl::VectorView<NodeInfo> * const nodesInfo)
     {
         using namespace gfl;
 
         targetFound = false;
-        if (not nodes->empty())
+        if (not nodesInfo->empty())
         {
-            checkForTargetKernel<<<1,1>>>(&targetFound,model,nodes);
+            checkForTargetKernel<<<1,1>>>(&targetFound, model, nodes, nodesInfo);
             CHECK_LAST_CUDA_ERROR();
         }
         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
     }
 
+    // void calcOutLabels(
+    //     Model const * const model,
+    //     gfl::VectorView<Node> * const nodes,
+    //     gfl::i64 const primal,
+    //     gfl::i64 const dual)
+    // {
+    //     using namespace gfl;
+    //
+    //     if (not nodes->empty())
+    //     {
+    //         i32 const blockSize = 128;
+    //         i32 const gridSize = ceil<i32>(nodes->size(),blockSize);
+    //         calcOutLabelsKernel<<<gridSize,blockSize>>>(model,nodes,primal,dual,DDRelaxed);
+    //         CHECK_LAST_CUDA_ERROR();
+    //
+    //         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
+    //     }
+    // }
+    //
+    // void checkForTarget(
+    //     Model const * const model,
+    //     gfl::VectorView<Node> * const nodes)
+    // {
+    //     using namespace gfl;
+    //
+    //     targetFound = false;
+    //     if (not nodes->empty())
+    //     {
+    //         checkForTargetKernel<<<1,1>>>(&targetFound,model,nodes);
+    //         CHECK_LAST_CUDA_ERROR();
+    //     }
+    //     CHECK_CUDA_ERROR(cudaDeviceSynchronize());
+    // }
+
     void keepOnlyBestChild()
     {
         using namespace gfl;
@@ -295,9 +388,6 @@ class ExpansionEngineGpu : public ExpansionEngine<Model,Node>
         auto & tmpNodes = expData.tmpNodes;
         auto & childrenInfo = expData.childrenInfo;
         auto & tmpInfo = expData.tmpInfo;
-        i32 const ApproximatedFlag = 1;
-        i32 const ExactFlag = 0;
-
 
         if (not children.empty())
         {
@@ -305,12 +395,6 @@ class ExpansionEngineGpu : public ExpansionEngine<Model,Node>
             i32 const gridSize = ceil<i32>(children.size(), blockSize);
             setScoreGKernel<Model><<<gridSize, blockSize>>>(&children, &childrenInfo);
             CHECK_LAST_CUDA_ERROR();
-            setFlagKernel<<<gridSize, blockSize>>>(ExactFlag,&childrenInfo);
-            CHECK_LAST_CUDA_ERROR();
-            setApproximatedFlagKernel<<<gridSize, blockSize>>>(ApproximatedFlag,&children, &childrenInfo);
-            CHECK_LAST_CUDA_ERROR();
-            resizeToKernel<<<1,1>>>(&tmpInfo,childrenInfo.sizePtr());
-            CHECK_LAST_CUDA_ERROR();
             sortKernel<NodeInfo::ScoreFlagDecomposer><<<1,1>>>(&childrenInfo,&tmpInfo,&cubAuxMem);
             CHECK_LAST_CUDA_ERROR();
             swapKernel<<<1,1>>>(&tmpInfo, &childrenInfo);
@@ -364,6 +448,8 @@ class ExpansionEngineGpu : public ExpansionEngine<Model,Node>
     {
         using namespace gfl;
         auto & children = expData.children;
+        auto & childrenInfo = expData.childrenInfo;
+
         expandParents(model,primal,brachFactor);
         //printKernel<<<1,1>>>(0,&children);
         //cudaDeviceSynchronize();
@@ -378,9 +464,9 @@ class ExpansionEngineGpu : public ExpansionEngine<Model,Node>
                 mergeChildren();
             }
 
-            calcOutLabels(model,&children,primal,dual);
+            calcOutLabels(model,&children,&childrenInfo,primal,dual);
 
-            checkForTarget(model,&children);
+            checkForTarget(model,&children,&childrenInfo);
         }
         //printKernel<<<1,1>>>(1,&children);
         CHECK_CUDA_ERROR(cudaDeviceSynchronize());
diff --git a/src_gpu/ExpansionKernels.cuh b/src_gpu/ExpansionKernels.cuh
index beba479..6d52c8f 100644
--- a/src_gpu/ExpansionKernels.cuh
+++ b/src_gpu/ExpansionKernels.cuh
@@ -433,15 +433,15 @@ void resizeByKernel(
 { cutset->addSegment(*count);}
 
 template<typename Node>
-void initChildrenPrefix(
+void initChildrenSuffix(
     gfl::i32 const width,
     gfl::ArrayView<Node> const * const children,
-    gfl::ArrayView<Node> * const childrenPrefix)
+    gfl::ArrayView<NodeInfo> * const childrenSuffix)
 {
     using namespace gfl;
 
     i32 const prefixSize = width - 1;
-    *childrenPrefix = children->slice(prefixSize, children->size());
+    *childrenSuffix = children->slice(prefixSize, children->size());
 }
 
 
@@ -457,47 +457,54 @@ void mergeNodeWith(Node & main, Node const & toMerge)
 template<typename Model, typename Node>
 GFL_GLOBAL
 __global__ void reductionKernel(
-    gfl::ArrayView<Node> const * const inBuffer,
-    gfl::ArrayView<Node> const * const outBuffer,
+    gfl::ArrayView<Node>     const * const nodes,
+    gfl::ArrayView<NodeInfo> const * const inBuffer,
+    gfl::ArrayView<NodeInfo> const * const outBuffer,
     gfl::i32 const count)
 {
     using namespace gfl;
 
     assert(blockDim.x == 32);
-    __shared__ Node tmpNodes[32];
-    //assert(blockDim.x * sizeof(Node) <= getSharedMemSize());
+    __shared__ NodeInfo tmpInfos[32];
 
     auto [begin, end] = calcSlice<int>(blockIdx.x, gridDim.x, count);
     i32 const nodesOfBlock = min<i32>(blockDim.x, end - begin);
     if (threadIdx.x < nodesOfBlock)
     {
-        Node fNode_r = inBuffer->at(begin + threadIdx.x);
+        NodeInfo fInfo_r = inBuffer->at(begin + threadIdx.x);
+        Node fNode_r = nodes->at(fInfo_r.idx);
         fNode_r.approximated(true);
         for (i32 i = begin + threadIdx.x + blockDim.x; i < end; i += blockDim.x)
         {
-            Node const & iNode = inBuffer->at(i);
-            mergeNodeWith<Model>( fNode_r, iNode);
+            NodeInfo const & iInfo = inBuffer->at(i);
+            Node     const & iNode = nodes->at(iInfo.idx);
+            mergeNodeWith<Model>(fNode_r, iNode);
         }
-        tmpNodes[threadIdx.x] = fNode_r;
+        nodes->at(fInfo_r.idx) = fNode_r; // write merged node back
+        tmpInfos[threadIdx.x] = fInfo_r;
     }
     __syncthreads();
 
     if (threadIdx.x == 0 and nodesOfBlock > 0)
     {
-        Node fNode_r = tmpNodes[0];
+        NodeInfo fInfo_r = tmpInfos[0];
+        Node fNode_r = nodes->at(fInfo_r.idx);
         for (int i = 1; i < nodesOfBlock; ++i)
         {
-            Node const & iNode = tmpNodes[i];
-            mergeNodeWith<Model>( fNode_r, iNode);
+            NodeInfo const & iInfo = tmpInfos[i];
+            Node     const & iNode = nodes->at(iInfo.idx);
+            mergeNodeWith<Model>(fNode_r, iNode);
         }
-        outBuffer->at(blockIdx.x) = fNode_r;
+        nodes->at(fInfo_r.idx) = fNode_r;
+        outBuffer->at(blockIdx.x) = fInfo_r; // output the NodeInfo of the winner
     }
 }
 
 template<typename Model, typename Node>
 GFL_GLOBAL
 void reductionSeqKernel(
-    gfl::ArrayView<Node> const * const inBuffer,
+    gfl::ArrayView<Node>     const * const nodes,
+    gfl::ArrayView<NodeInfo> const * const inBuffer,
     gfl::i32 const count)
 {
     using namespace gfl;
@@ -506,20 +513,107 @@ void reductionSeqKernel(
     assert(gridDim.x == 1);
     assert(count > 0);
 
-    Node result = inBuffer->at(0);
+    NodeInfo fInfo_r = inBuffer->at(0);
+    Node result = nodes->at(fInfo_r.idx);
     result.approximated(true);
     for (i32 i = 1; i < count; ++i)
     {
-        mergeNodeWith<Model>(result, inBuffer->at(i));
+        NodeInfo const & iInfo = inBuffer->at(i);
+        Node     const & iNode = nodes->at(iInfo.idx);
+        mergeNodeWith<Model>(result, iNode);
     }
-    inBuffer->at(0) = result;
+    nodes->at(fInfo_r.idx) = result;
+    inBuffer->at(0) = fInfo_r; // first NodeInfo slot holds the result's info
 }
 
+
+// template<typename Model, typename Node>
+// GFL_GLOBAL
+// __global__ void reductionKernel(
+//     gfl::ArrayView<Node> const * const inBuffer,
+//     gfl::ArrayView<Node> const * const outBuffer,
+//     gfl::i32 const count)
+// {
+//     using namespace gfl;
+//
+//     assert(blockDim.x == 32);
+//     __shared__ Node tmpNodes[32];
+//     //assert(blockDim.x * sizeof(Node) <= getSharedMemSize());
+//
+//     auto [begin, end] = calcSlice<int>(blockIdx.x, gridDim.x, count);
+//     i32 const nodesOfBlock = min<i32>(blockDim.x, end - begin);
+//     if (threadIdx.x < nodesOfBlock)
+//     {
+//         Node fNode_r = inBuffer->at(begin + threadIdx.x);
+//         fNode_r.approximated(true);
+//         for (i32 i = begin + threadIdx.x + blockDim.x; i < end; i += blockDim.x)
+//         {
+//             Node const & iNode = inBuffer->at(i);
+//             mergeNodeWith<Model>( fNode_r, iNode);
+//         }
+//         tmpNodes[threadIdx.x] = fNode_r;
+//     }
+//     __syncthreads();
+//
+//     if (threadIdx.x == 0 and nodesOfBlock > 0)
+//     {
+//         Node fNode_r = tmpNodes[0];
+//         for (int i = 1; i < nodesOfBlock; ++i)
+//         {
+//             Node const & iNode = tmpNodes[i];
+//             mergeNodeWith<Model>( fNode_r, iNode);
+//         }
+//         outBuffer->at(blockIdx.x) = fNode_r;
+//     }
+// }
+//
+// template<typename Model, typename Node>
+// GFL_GLOBAL
+// void reductionSeqKernel(
+//     gfl::ArrayView<Node> const * const inBuffer,
+//     gfl::i32 const count)
+// {
+//     using namespace gfl;
+//
+//     assert(blockDim.x == 1);
+//     assert(gridDim.x == 1);
+//     assert(count > 0);
+//
+//     Node result = inBuffer->at(0);
+//     result.approximated(true);
+//     for (i32 i = 1; i < count; ++i)
+//     {
+//         mergeNodeWith<Model>(result, inBuffer->at(i));
+//     }
+//     inBuffer->at(0) = result;
+// }
+
 template<typename Model, typename Node>
 GFL_GLOBAL
 void calcOutLabelsKernel(
         Model const * const model,
-        gfl::ArrayView<Node> const * const nodes,
+        gfl::ArrayView<Node>     const * const nodes,
+        gfl::ArrayView<NodeInfo> const * const nodesInfo,
+        gfl::f64 const primal,
+        gfl::f64 const dual,
+        DDContext const ddCtx)
+{
+    using namespace gfl;
+
+    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x, nodesInfo->size());
+    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
+    {
+        NodeInfo const & info = nodesInfo->at(i);
+        Node & node = nodes->at(info.idx);
+        node.labels(model->lgf(node.state(), primal, dual, ddCtx));
+    }
+}
+
+template<typename Model, typename Node>
+GFL_GLOBAL
+void calcOutLabelsKernel(
+        Model const * const model,
+        gfl::ArrayView<Node>     const * const nodes,
         gfl::f64 const primal,
         gfl::f64 const dual,
         DDContext const ddCtx)
@@ -534,6 +628,25 @@ void calcOutLabelsKernel(
     }
 }
 
+// template<typename Model, typename Node>
+// GFL_GLOBAL
+// void calcOutLabelsKernel(
+//         Model const * const model,
+//         gfl::ArrayView<Node> const * const nodes,
+//         gfl::f64 const primal,
+//         gfl::f64 const dual,
+//         DDContext const ddCtx)
+// {
+//     using namespace gfl;
+//
+//     auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x, nodes->size());
+//     for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
+//     {
+//         Node & node = nodes->at(i);
+//         node.labels(model->lgf(node.state(), primal, dual, ddCtx));
+//     }
+// }
+
 template<typename Model, typename Node>
 GFL_GLOBAL
 void setHKernel(
@@ -559,7 +672,8 @@ GFL_GLOBAL
 void checkForTargetKernel(
     bool * const found,
     Model const * const model,
-    gfl::VectorView<Node> const * nodes)
+    gfl::VectorView<Node> const * nodes,
+    gfl::ArrayView<NodeInfo> const * nodesInfo)
 {
     assert(nodes != nullptr);
     assert(not nodes->empty());
@@ -568,7 +682,8 @@ void checkForTargetKernel(
 
     using namespace gfl;
 
-    *found = nodes->at(0).isTarget(model);
+    auto const & info = nodesInfo->at(0);
+    *found = nodes->at(info.idx).isTarget(model);
 }
 
 
