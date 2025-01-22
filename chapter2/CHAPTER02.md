# 02注意力模块

申明：本教程的所有内容(文字，图片，代码等)可以用于非盈利目的个人使用和分享。但如果用于盈利目的，包括但不限于卖课，公众号，视频号等需要经由作者的批准。谢谢理解。

[\[主目录链接\]](https://github.com/KaihuaTang/All-you-need-to-know-about-LLM)

## 前言
作为大语言模型中核心的核心，我将注意力模块排在了其他模块之前放在最前面讲解。我们在本章里会从其原理，结构，各种优化版本讲到目前主流开源大语言模型的具体代码。但本章节仅限于对注意力结构本身原理的阐述，并不会太涉及优化，比如目前主流的[FlashAttention-v1/v2/v3](https://github.com/Dao-AILab/flash-attention)或者一些[线性注意力架构](https://arxiv.org/abs/2401.04658)，这些要么就是基于硬件做的数学等价优化，要么就是完全改变了传统注意力计算形式尚没有被主流认可。不过我相信只要打好基础，做到能逐行代码地理解目前主流大语言模型的所有细节，后续当读者们再看这些进阶知识时，也会更容易理解他们的原理。

## 一. 注意力原理

## 二. 自注意力结构

## 三. 注意力优化

## 四. 大语言模型中的注意力

### 1. Qwen2的注意力代码详解

### 2. LLaMA3的注意力代码详解

## 五. 扩展知识

## 引用链接

```
@misc{tang2025all,
title = {All you need to know about LLM: a LLM tutorial},
author = {Tang, Kaihua},
year = {2025},
note = {\url{https://github.com/KaihuaTang/All-you-need-to-know-about-LLM}},
}
```