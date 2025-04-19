### Reference
- [ring attention + flash attention：超长上下文之路](https://zhuanlan.zhihu.com/p/683714620)
- [Attention的分块计算: 从flash-attention到ring-attention](https://zhuanlan.zhihu.com/p/686240618)
- [手把手实现Ring Attention学习版](https://zhuanlan.zhihu.com/p/684715644)
- [LLM(31)：序列并行的典型方案与实现细节](https://zhuanlan.zhihu.com/p/14665512019)
- [由Ring-Attention性能问题引发的计算通信overlap分析](https://zhuanlan.zhihu.com/p/706805407)
- [我爱DeepSpeed-Ulysses：重新审视大模型序列并行技术](http://zhuanlan.zhihu.com/p/703669087)

**FA in Ascend NPU**
- [torch_npu.npu_fusion_attention](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/apiref/apilist/ptaoplist_000142.html)
- [昇腾NPU的踩坑之路](https://zhuanlan.zhihu.com/p/25147199560)

**online softmax**
- [手撕online softmax](https://zhuanlan.zhihu.com/p/5078640012)

### Attention

$$
\mathrm{attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$