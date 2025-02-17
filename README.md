# 从零开始理解大语言模型

<div align="center">
    <img src="assets/cover.png" alt="logo" width="80%"  style="padding-bottom: 20px"/>
</div>

申明：本教程的所有内容(文字，图片，代码等)可以用于非盈利目的个人使用和分享。但如果用于盈利目的，包括但不限于卖课，公众号，视频号等需要经由作者的批准。谢谢理解。[\[知乎链接\]](https://zhuanlan.zhihu.com/p/19275166926)

## 前言
进入工业界两年，一回头发现和在学校相比，时间似乎溜走地更悄无声息了。没有论文来总结自己每个阶段的思考和成果，似乎我的价值只存在于这六七人小团队的梦呓呢喃中，一旦离开了屈指可数的小圈子，自己这两年的所有足迹便被一个响指抹平了。本着知识就应该被分享的开源精神和一点无法被公司满足的小虚荣心，我决定写一个系列分享一下自己这两年从事大语言模型的一些理解和思考。从23年7月开始到24年底，我在公司主要做了两个和大语言模型相关的项目。其一是从23年7月开始为期半年的和中国移动合作的端侧小模型项目，在这项目中我们算法团队四个人从零开始参考[GPT2](https://github.com/openai/gpt-2)和[ChatGLM](https://github.com/THUDM/ChatGLM-6B)，训练了一个0.34B的中文语言模型（出于端侧芯片算力和我们自身训练资源和时间的考量，我们在项目要求时限内仅能训练GPT2-medium参数量的小模型），并在自建的家庭对话垂域数据上进行微调，最后通过ONNX部署在移动端的安卓智慧屏，这个项目参展了2023年中国移动全球合作伙伴大会，到24年初我们又更新了一版1B的模型进一步优化了聊天效果。第二个项目是24年5月开始的对公司内某个图文多模态大模型进行算力优化的项目，我们参考了一些开源的论文，通过对网络结构和推理逻辑的调整在量化模型的基础上进一步提升了30%的推理速度。在这两个项目中，虽然训练规模有限，但我也算是完整地了解并实践了大语言模型和图文多模态大模型的网络结构设计和训练流程，抛开那些无法分享的公司项目细节，我打算整理一份比较基础的，从零开始实现大语言模型的教程，让新入门的同学们可以更快的了解必要的知识，少走弯路。当然同时也可以作为一个记录我思考的笔记本，供该领域的从业者们参考和交流，也请大家检验下我自己的认知和理解是否存在偏差。

## 涵盖范围
该系列的目的是让读者可以在基础的pytorch上，不依赖任何其他现成的外部库，从零开始理解并实现一个大语言模型的所有组成部分，以及训练微调代码，因此读者仅需python，pytorch，numpy和最基础深度学习背景知识即可。后续也考虑拓展到图文多模态大语言模型。考虑到国内外网上已经有大量现成的大语言模型教程和代码，本系列除了单纯的梳理知识点外，也记录了自己在实践中的思考和做项目时遇到的具体业界问题，还有一些最新热点模型的知识点，比如deepseek的网络结构和正常的大语言模型有什么区别，为什么生成式语言模型一定要因果注意力（causal attention），pytorch的动态长度推理怎么转换为需要静态张量形状的ONNX推理格式，如何简单有效地加速首轮问答响应时间，RMSNorm归一化层在只支持FP16计算的NPU芯片上怎么解决值域越界，tokenizer分词器词表裁剪等。

## 章节链接
本系列目前计划将内容分为如下章节：

1. [大语言模型结构概览](chapter1/README.md)
2. [注意力模块与KV Cache](chapter2/README.md)
3. [DeepSeekV3的注意力优化](chapter3/README.md)
3. 旋转位置编码 (待更新)
4. 前馈网络 (待更新)
5. 归一化层 (待更新)
6. tokenizer分词器 (待更新)
7. 文本预训练 (待更新)
8. 对话数据微调 (待更新)
9. LoRA高效微调 (待更新)
10. 视觉Transformer网络 (？待定)
11. 图文多模态网络 (？待定)
12. 张量并行多卡部署 (？待定)

具体章节名称和数量在实际撰写时可能进行调整。

## 引用链接

```
@misc{tang2025all,
title = {All you need to know about LLM: a LLM tutorial},
author = {Tang, Kaihua},
year = {2025},
note = {\url{https://github.com/KaihuaTang/All-you-need-to-know-about-LLM}},
}
```
