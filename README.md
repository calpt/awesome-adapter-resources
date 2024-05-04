# Awesome Adapter Resources

![](https://img.shields.io/badge/Resources-67-blue)

This repository collects important tools and papers related to adapter methods for recent large pre-trained neural networks.

_Adapters_ (aka _Parameter-Efficient Transfer Learning (PETL)_ or _Parameter-Efficient Fine-Tuning (PEFT)_ methods) include various parameter-efficient approaches of adapting large pre-trained models to new tasks.

## Content

- [Why Adapters?](#why-adapters)
- [Frameworks and Tools](#frameworks-and-tools)
- [Surveys](#surveys)
- [Natural Language Processing](#natural-language-processing)
  - [Methods](#methods)
  - [Composition Methods](#composition-methods)
  - [Analysis and Evaluation](#analysis-and-evaluation)
  - [Applications](#applications)
  - [Serving](#serving)
- [Computer Vision](#computer-vision)
  - [Methods](#methods-1)
- [Audio Processing](#audio-processing)
  - [Applications](#applications-1)
- [Multi-Modal](#multi-modal)
  - [Methods](#methods-2)
- [Contributing](#contributing)

## Why Adapters?

Large pre-trained (Transformer-based) models have become the foundation of various ML domains in recent years.
While the most prevalent method of adapting these models to new tasks involves costly full fine-tuning of all model parameters, a series of parameter-efficient and lightweight alternatives, _adapters_, have been established in recent time.

Using adapters provides multiple benefits. They are ...
- ... **parameter-efficient**, i.e. they only update a very small subset (e.g. under 1%) of a model's parameters.
- ... **modular**, i.e. the updated parameters can be extracted and shared independently of the base model parameters
- ... easy to **share** and easy to **deploy at scale** due to their small file sizes. E.g. requiring only ~3MB per task instead of ~500MB for sharing a full model.
- ... often **composable**, i.e. can be stacked, fused or mixed to leverage their combined knowledge.
- ... often **on-par** in terms of **performance** with full fine-tuning.

## Frameworks and Tools

- **AdapterHub: A Framework for Adapting Transformers**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/adapter-hub/adapter-transformers?color=yellow&logo=github) 

  Conference on Empirical Methods in Natural Language Processing

  _Jonas Pfeiffer, Andreas Rücklé, Clifton A. Poth, Aishwarya Kamath, Ivan Vulic, Sebastian Ruder, Kyunghyun Cho, Iryna Gurevych_ (2020)

  <details>
    <summary>TLDR</summary>
    AdaptersHub is proposed, a framework that allows dynamic “stiching-in” of pre-trained adapters for different tasks and languages that enables scalable and easy access to sharing of task-specific models, particularly in low-resource scenarios.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2007.07779)&nbsp; [[Code]](https://github.com/adapter-hub/adapter-transformers)&nbsp; [[Website]](https://adapterhub.ml)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/063f8b1ecf2394ca776ac61869734de9c1953808)

- **Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/adapter-hub/adapters?color=yellow&logo=github) 

  Conference on Empirical Methods in Natural Language Processing

  _Clifton A. Poth, Hannah Sterz, Indraneil Paul, Sukannya Purkayastha, Leon Arne Engländer, Timo Imhof, Ivan Vuli'c, Sebastian Ruder, Iryna Gurevych, Jonas Pfeiffer_ (2023)

  <details>
    <summary>TLDR</summary>
    Adapters, an open-source library that unifies parameter-efficient and modular transfer learning in large language models and allows researchers and practitioners to leverage adapter modularity through composition blocks, enabling the design of complex adapter setups, is introduced.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2311.11077.pdf)&nbsp; [[Code]](https://github.com/adapter-hub/adapters)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/e1f4b94479bfcb735a1a0add178a2337def07c9b)

- **OpenDelta**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/thunlp/OpenDelta?color=yellow&logo=github) 

   

  

  [[Code]](https://github.com/thunlp/OpenDelta)&nbsp; [[Website]](https://opendelta.readthedocs.io/)

- **PEFT: State-of-the-art Parameter-Efficient Fine-Tuning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/huggingface/peft?color=yellow&logo=github) 

   

  

  [[Code]](https://github.com/huggingface/peft)

- **LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/AGI-Edgerunners/LLM-Adapters?color=yellow&logo=github) 

  Conference on Empirical Methods in Natural Language Processing

  _Zhiqiang Hu, Yihuai Lan, Lei Wang, Wanyu Xu, Ee-Peng Lim, R. Lee, Lidong Bing, Soujanya Poria_ (2023)

  <details>
    <summary>TLDR</summary>
    LLM-Adapters is presented, an easy-to-use framework that integrates various adapters into LLMs and can execute these adapter-based PEFT methods of LLMs for different tasks, demonstrating that using adapter- based PEFT in smaller-scale LLMs with few extra trainable parameters yields comparable, and in some cases superior, performance to powerful LLMs in zero-shot inference on both reasoning tasks.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2304.01933.pdf)&nbsp; [[Code]](https://github.com/AGI-Edgerunners/LLM-Adapters)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/bdb68c5e2369633b20e733774ac66eb4600c34d1)

- **Alpaca-LoRA**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/tloen/alpaca-lora?color=yellow&logo=github) 

   

  

  [[Code]](https://github.com/tloen/alpaca-lora)


## Surveys

- **Modular Deep Learning**&nbsp; 

  arXiv.org

  _Jonas Pfeiffer, Sebastian Ruder, Ivan Vulic, E. Ponti_ (2023)

  <details>
    <summary>TLDR</summary>
    A survey of modular architectures is offered, providing a unified view over several threads of research that evolved independently in the scientific literature, and various additional purposes of modularity are explored, including scaling language models, causal inference, programme induction, and planning in reinforcement learning.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2302.11529.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/1f346f74e8eabececa4896d734ab9b261f30830d)

- **Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning**&nbsp; 

  arXiv.org

  _Vladislav Lialin, Vijeta Deshpande, Anna Rumshisky_ (2023)

  <details>
    <summary>TLDR</summary>
    A taxonomy that covers a broad range of methods and present a detailed method comparison with a specific focus on real-life efficiency and fine-tuning multibillion-scale language models is provided.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2303.15647.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/6007263dd3d14373be5f84fb6ccb0be3f7fce903)

- **PEFT-Ref: A Modular Reference Architecture and Typology for Parameter-Efficient Finetuning Techniques**&nbsp; 

  arXiv.org

  _Mohammed Sabry, Anya Belz_ (2023)

  <details>
    <summary>TLDR</summary>
    A reference architecture is presented which standardises aspects shared by different PEFT techniques, while isolating differences to specific locations and interactions with the standard components, supporting not only direct comparison of different techniques and their efficiency and task performance, but also systematic exploration of reusability and composability of the different types of finetuned modules.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2304.12410.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/2afd51e83e87acf02c0044b34c6d4984e814900e)

- **Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey**&nbsp; 

  arXiv.org

  _Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, Sai Qian Zhang_ (2024)

  <details>
    <summary>TLDR</summary>
    This survey presents comprehensive studies of various PEFT algorithms, examining their performance and computational overhead, and overview of applications developed using different PEFT algorithms and discusses common techniques employed to mitigate computation costs for PEFT.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2403.14608.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/916b4926cda574dc3f9486bb9994b6f2788dd800)


## Natural Language Processing

### Methods

- **Parameter-Efficient Transfer Learning for NLP**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/google-research/adapter-bert?color=yellow&logo=github) ![](https://img.shields.io/badge/-Bottleneck%20adapter-blue)

  International Conference on Machine Learning

  _N. Houlsby, A. Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, S. Gelly_ (2019)

  <details>
    <summary>TLDR</summary>
    To demonstrate adapter's effectiveness, the recently proposed BERT Transformer model is transferred to 26 diverse text classification tasks, including the GLUE benchmark, and adapter attain near state-of-the-art performance, whilst adding only a few parameters per task.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/1902.00751.pdf)&nbsp; [[Code]](https://github.com/google-research/adapter-bert)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/29ddc1f43f28af7c846515e32cc167bc66886d0c)

- **K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/K-Adapter?color=yellow&logo=github) ![](https://img.shields.io/badge/-K--Adapter-blue)

  Findings

  _Ruize Wang, Duyu Tang, Nan Duan, Zhongyu Wei, Xuanjing Huang, Jianshu Ji, Guihong Cao, Daxin Jiang, Ming Zhou_ (2020)

  <details>
    <summary>TLDR</summary>
    K-Adapter is proposed, which remains the original parameters of the pre-trained model fixed and supports continual knowledge infusion and captures richer factual and commonsense knowledge than RoBERTa.
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.findings-acl.121.pdf)&nbsp; [[Code]](https://github.com/microsoft/K-Adapter)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/4f03e69963b9649950ba29ae864a0de8c14f1f86)

- **Parameter-Efficient Transfer Learning with Diff Pruning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/dguo98/DiffPruning?color=yellow&logo=github) ![](https://img.shields.io/badge/-Diff%20pruning-blue)

  Annual Meeting of the Association for Computational Linguistics

  _Demi Guo, Alexander M. Rush, Yoon Kim_ (2020)

  <details>
    <summary>TLDR</summary>
    Diff pruning can match the performance of finetuned baselines on the GLUE benchmark while only modifying 0.5% of the pretrained model’s parameters per task and scales favorably in comparison to popular pruning approaches.
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.acl-long.378.pdf)&nbsp; [[Code]](https://github.com/dguo98/DiffPruning)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/d22e4cc3a501c17881b9478621f29760e429e76e)

- **Prefix-Tuning: Optimizing Continuous Prompts for Generation**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/XiangLi1999/PrefixTuning?color=yellow&logo=github) ![](https://img.shields.io/badge/-Prefix--Tuning-blue)

  Annual Meeting of the Association for Computational Linguistics

  _Xiang Lisa Li, Percy Liang_ (2021)

  <details>
    <summary>TLDR</summary>
    Prefix-tuning is proposed, a lightweight alternative to fine- Tuning for natural language generation tasks, which keeps language model parameters frozen and instead optimizes a sequence of continuous task-specific vectors, which is called the prefix.
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.acl-long.353.pdf)&nbsp; [[Code]](https://github.com/XiangLi1999/PrefixTuning)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/53d8b356551a2361020a948f64454a6d599af69f)

- **The Power of Scale for Parameter-Efficient Prompt Tuning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/google-research/prompt-tuning?color=yellow&logo=github) ![](https://img.shields.io/badge/-Prompt%20Tuning-blue)

  Conference on Empirical Methods in Natural Language Processing

  _Brian Lester, Rami Al-Rfou, Noah Constant_ (2021)

  <details>
    <summary>TLDR</summary>
    This work explores “prompt tuning,” a simple yet effective mechanism for learning “soft prompts” to condition frozen language models to perform specific downstream tasks and shows that conditioning a frozen model with soft prompts confers benefits in robustness to domain transfer and enables efficient “Prompt ensembling.”
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.emnlp-main.243.pdf)&nbsp; [[Code]](https://github.com/google-research/prompt-tuning)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/ffdbd7f0b03b85747b001b4734d5ee31b5229aa4)

- **Compacter: Efficient Low-Rank Hypercomplex Adapter Layers**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/rabeehk/compacter?color=yellow&logo=github) ![](https://img.shields.io/badge/-Compacter-blue) ![](https://img.shields.io/badge/-Compacter++-blue) ![](https://img.shields.io/badge/-PHM--Adapter-blue)

  Neural Information Processing Systems

  _Joe Davison_ (2021)

  <details>
    <summary>TLDR</summary>
    Compacter is proposed, a method for fine-tuning large-scale language models with a better trade-off between task performance and the number of trainable parameters than prior work, and accomplishes this by building on top of ideas from adapters, low-rank optimization, and parameterized hypercomplex multiplication layers.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2106.04647.pdf)&nbsp; [[Code]](https://github.com/rabeehk/compacter)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/656ed155c2d345c19d9bff4b50f2ae00db8407cc)

- **LoRA: Low-Rank Adaptation of Large Language Models**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/LoRA?color=yellow&logo=github) ![](https://img.shields.io/badge/-LoRA-blue)

  International Conference on Learning Representations

  _J. E. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Weizhu Chen_ (2021)

  <details>
    <summary>TLDR</summary>
    Low-Rank Adaptation, or LoRA, is proposed, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2106.09685.pdf)&nbsp; [[Code]](https://github.com/microsoft/LoRA)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/a8ca46b171467ceb2d7652fbfb67fe701ad86092)

- **Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/rabeehk/hyperformer?color=yellow&logo=github) ![](https://img.shields.io/badge/-HyperFormer-blue)

  Annual Meeting of the Association for Computational Linguistics

  _Rabeeh Karimi Mahabadi, Sebastian Ruder, Mostafa Dehghani, J. Henderson_ (2021)

  <details>
    <summary>TLDR</summary>
    This paper shows that one can learn adapter parameters for all layers and tasks by generating them using shared hypernetworks, which condition on task, adapter position, and layer id in a transformer model.
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.acl-long.47.pdf)&nbsp; [[Code]](https://github.com/rabeehk/hyperformer)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/bb3425318de7eed5641cda147d61c9a057b9d054)

- **MAD-G: Multilingual Adapter Generation for Efficient Cross-Lingual Transfer**&nbsp; ![](https://img.shields.io/badge/-MAD--G-blue)

  Conference on Empirical Methods in Natural Language Processing

  _Alan Ansell, E. Ponti, Jonas Pfeiffer, Sebastian Ruder, Goran Glavas, Ivan Vulic, A. Korhonen_ (2021)

  <details>
    <summary>TLDR</summary>
    MAD-G is proposed, which contextually generates language adapters from language representations based on typological features and remains competitive with more expensive methods for language-speciﬁc adapter training across the board.
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.findings-emnlp.410.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/6adc9c231d874ea358554b8680a6aaba4bd6c963)

- **BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/benzakenelad/BitFit?color=yellow&logo=github) ![](https://img.shields.io/badge/-BitFit-blue)

  Annual Meeting of the Association for Computational Linguistics

  _Elad Ben-Zaken, Shauli Ravfogel, Yoav Goldberg_ (2021)

  <details>
    <summary>TLDR</summary>
    BitFit is introduced, a sparse-finetuning method where only the bias-terms of the model (or a subset of them) are being modified, which shows that with small-to-medium training data, applying BitFit on pre-trained BERT models is competitive with (and sometimes better than) fine-tuning the entire model.
  </details>

  [[Paper PDF]](https://aclanthology.org/2022.acl-short.1.pdf)&nbsp; [[Code]](https://github.com/benzakenelad/BitFit)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/339b2b711fb5b228d097b03ebc3e62a521779235)

- **Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/r-three/t-few?color=yellow&logo=github) ![](https://img.shields.io/badge/-T--Few-blue) ![](https://img.shields.io/badge/-(IA)^3-blue)

  Neural Information Processing Systems

  _Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, Colin Raffel_ (2022)

  <details>
    <summary>TLDR</summary>
    This paper rigorously compares few-shot ICL and PEFT and demonstrates that the latter offers better accuracy as well as dramatically lower computational costs, and introduces a new PEFT method called (IA)$^3$ that scales activations by learned vectors, attaining stronger performance while only introducing a relatively tiny amount of new parameters.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2205.05638.pdf)&nbsp; [[Code]](https://github.com/r-three/t-few)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/7cdaa08890895e1ad92afb5fad429690ad7b1dac)

- **AutoPEFT: Automatic Configuration Search for Parameter-Efficient Fine-Tuning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/cambridgeltl/autopeft?color=yellow&logo=github) ![](https://img.shields.io/badge/-AutoPEFT-blue)

  arXiv.org

  _Han Zhou, Xingchen Wan, Ivan Vulic, A. Korhonen_ (2023)

  <details>
    <summary>TLDR</summary>
    Inspired by advances in neural architecture search, AutoPEFT is proposed for automatic PEFT configuration selection and it is shown that AutoPEFT-discovered configurations significantly outperform existing PEFT methods and are on par or better than FFT without incurring substantial training efficiency costs.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2301.12132.pdf)&nbsp; [[Code]](https://github.com/cambridgeltl/autopeft)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/b9d77cd9be54a228f811b1ac6212a7041792f217)

- **Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/QingruZhang/AdaLoRA?color=yellow&logo=github) ![](https://img.shields.io/badge/-AdaLoRA-blue)

  _Qingru Zhang, Minshuo Chen, Alexander W. Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, Tuo Zhao_ (2023)

  <details>
    <summary>TLDR</summary>
    The proposed AdaLoRA adaptively allocates the parameter budget among weight matrices according to their importance score, which allows us to effectively prune the singular values of unimportant updates, which is essentially to reduce their parameter budget but circumvent intensive exact SVD computations.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2303.10512.pdf)&nbsp; [[Code]](https://github.com/QingruZhang/AdaLoRA)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/b612fc6af23cccf2133c2ea40597453ab40dc2c3)

- **QLoRA: Efficient Finetuning of Quantized LLMs**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/artidoro/qlora?color=yellow&logo=github) ![](https://img.shields.io/badge/-QLoRA-blue)

  Neural Information Processing Systems

  _Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer_ (2023)

  <details>
    <summary>TLDR</summary>
    QLoRA finetuning on a small high-quality dataset leads to state-of-the-art results, even when using smaller models than the previous SoTA, and current chatbot benchmarks are not trustworthy to accurately evaluate the performance levels of chatbots.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2305.14314.pdf)&nbsp; [[Code]](https://github.com/artidoro/qlora)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/32ac52069e562d4f900afee70bdca63f53461481)

- **Composable Sparse Fine-Tuning for Cross-Lingual Transfer**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/cambridgeltl/composable-sft?color=yellow&logo=github) ![](https://img.shields.io/badge/-LT--SFT-blue)

  Annual Meeting of the Association for Computational Linguistics

  _Alan Ansell, E. Ponti, A. Korhonen, Ivan Vulic_ (2021)

  <details>
    <summary>TLDR</summary>
    This work introduces a new fine-tuning method that outperforms adapters in zero-shot cross-lingual transfer by a large margin in a series of multilingual benchmarks, including Universal Dependencies, MasakhaNER, and AmericasNLI.
  </details>

  [[Paper PDF]](https://aclanthology.org/2022.acl-long.125.pdf)&nbsp; [[Code]](https://github.com/cambridgeltl/composable-sft)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/fc58779940abb92166b73f47867763a07368c739)

- **VeRA: Vector-based Random Matrix Adaptation**&nbsp; ![](https://img.shields.io/badge/-VeRA-blue)

  arXiv.org

  _Dawid Jan Kopiczko, Tijmen Blankevoort, Yuki Markus Asano_ (2023)

  <details>
    <summary>TLDR</summary>
    Vector-based Random Matrix Adaptation (VeRA) is presented, which significantly reduces the number of trainable parameters compared to LoRA, yet maintains the same performance by using a single pair of low-rank matrices shared across all layers and learning small scaling vectors instead.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2310.11454.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/0d7f24578340aae6df610ed95aaa276b9c3ddcd3)

- **DoRA: Weight-Decomposed Low-Rank Adaptation**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/NVlabs/DoRA?color=yellow&logo=github) ![](https://img.shields.io/badge/-DoRA-blue) ![](https://img.shields.io/badge/-LoRA-blue)

  arXiv.org

  _Shih-yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen_ (2024)

  <details>
    <summary>TLDR</summary>
    Weight-Decomposed LowRank Adaptation (DoRA) is proposed, which decomposes the pre-trained weight into two components, magnitude and direction, for fine-tuning, specifically employing LoRA for directional updates to efficiently minimize the number of trainable parameters.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2402.09353.pdf)&nbsp; [[Code]](https://github.com/NVlabs/DoRA)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/da053e2a4ba1b244940c8f2cad5dcdf0d730f85f)

- **ReFT: Representation Finetuning for Language Models**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/stanfordnlp/pyreft?color=yellow&logo=github) ![](https://img.shields.io/badge/-ReFT-blue) ![](https://img.shields.io/badge/-LoReFT-blue) ![](https://img.shields.io/badge/-interventions-blue)

  _Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Daniel Jurafsky, Christopher D. Manning, Christopher Potts_ (2024)

  <details>
    <summary>TLDR</summary>
    A strong instance of the ReFT family is defined, Low-rank Linear Subspace ReFT (LoReFT), which is a drop-in replacement for existing PEFTs and learns interventions that are 10x-50x more parameter-efficient than prior state-of-the-art PEFTs.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2404.03592.pdf)&nbsp; [[Code]](https://github.com/stanfordnlp/pyreft)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/1bda8efbbf4abae6c8c1da97d6137396807b1e09)

### Composition Methods

- **MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/adapter-hub/adapter-transformers?color=yellow&logo=github) ![](https://img.shields.io/badge/-MAD--X-blue) ![](https://img.shields.io/badge/-Invertible%20adapter-blue)

  Conference on Empirical Methods in Natural Language Processing

  _Jonas Pfeiffer, Ivan Vulic, Iryna Gurevych, Sebastian Ruder_ (2020)

  <details>
    <summary>TLDR</summary>
    MAD-X is proposed, an adapter-based framework that enables high portability and parameter-efficient transfer to arbitrary tasks and languages by learning modular language and task representations and introduces a novel invertible adapter architecture and a strong baseline method for adapting a pretrained multilingual model to a new language.
  </details>

  [[Paper PDF]](https://aclanthology.org/2020.emnlp-main.617.pdf)&nbsp; [[Code]](https://github.com/adapter-hub/adapter-transformers)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/26299d5fdc5137291dc6a091573b3d18aba1d1c2)

- **AdapterFusion: Non-Destructive Task Composition for Transfer Learning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/adapter-hub/adapter-transformers?color=yellow&logo=github) ![](https://img.shields.io/badge/-AdapterFusion-blue)

  Conference of the European Chapter of the Association for Computational Linguistics

  _Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, Iryna Gurevych_ (2020)

  <details>
    <summary>TLDR</summary>
    This work proposes AdapterFusion, a new two stage learning algorithm that leverages knowledge from multiple tasks by separating the two stages, i.e., knowledge extraction and knowledge composition, so that the classifier can effectively exploit the representations learned frommultiple tasks in a non-destructive manner.
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.eacl-main.39.pdf)&nbsp; [[Code]](https://github.com/adapter-hub/adapter-transformers)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/98ef0db84e62aef969629264c9de1f4d0013f3b9)

- **Towards a Unified View of Parameter-Efficient Transfer Learning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/jxhe/unify-parameter-efficient-tuning?color=yellow&logo=github) ![](https://img.shields.io/badge/-Mix--and--Match%20adapters-blue) ![](https://img.shields.io/badge/-Parallel%20adapters-blue)

  International Conference on Learning Representations

  _Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig_ (2021)

  <details>
    <summary>TLDR</summary>
    This paper re-frame state-of-the-art parameter-efficient transfer learning methods as modifications to specific hidden states in pre-trained models, and defines a set of design dimensions along which different methods vary, achieving comparable results to fine-tuning all parameters on all four tasks.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2110.04366.pdf)&nbsp; [[Code]](https://github.com/jxhe/unify-parameter-efficient-tuning)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/43a87867fe6bf4eb920f97fc753be4b727308923)

- **AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/AdaMix?color=yellow&logo=github) ![](https://img.shields.io/badge/-AdaMix-blue) ![](https://img.shields.io/badge/-MoE-blue) ![](https://img.shields.io/badge/-Model%20merging-blue)

  Conference on Empirical Methods in Natural Language Processing

  _Yaqing Wang, Subhabrata Mukherjee, Xiaodong Liu, Jing Gao, Jianfeng Gao_ (2022)

  <details>
    <summary>TLDR</summary>
    AdaMix is proposed as a general PEFT method that tunes a mixture of adaptation modules – given the underlyingPEFT method of choice – introduced in each Transformer layer while keeping most of the PLM weights frozen, and outperforms SOTA parameter-efficient fine-tuning and full model fine- Tuning for both NLU and NLG tasks.
  </details>

  [[Paper PDF]](https://aclanthology.org/2022.emnlp-main.388.pdf)&nbsp; [[Code]](https://github.com/microsoft/AdaMix)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/eb4d54651c4f610749caf2bf401af3ce28ddc439)

- **Composing Parameter-Efficient Modules with Arithmetic Operations**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/hkust-nlp/PEM_composition?color=yellow&logo=github) ![](https://img.shields.io/badge/-Model%20merging-blue)

  Neural Information Processing Systems

  _Jinghan Zhang, Shiqi Chen, Junteng Liu, Junxian He_ (2023)

  <details>
    <summary>TLDR</summary>
    This paper proposes to compose parameter-efficient modules through linear arithmetic operations in the weight space, thereby integrating different module capabilities and extends this approach to detoxify Alpaca-LoRA, the latest instruction-tuned large language model based on LLaMA.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2306.14870.pdf)&nbsp; [[Code]](https://github.com/hkust-nlp/PEM_composition)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/7f1a473834eea608980e4e04cce21be18d65b9b6)

- **AdapterSoup: Weight Averaging to Improve Generalization of Pretrained Language Models**&nbsp; ![](https://img.shields.io/badge/-Model%20merging-blue)

  Findings

  _Alexandra Chronopoulou, Matthew E. Peters, Alexander M. Fraser, Jesse Dodge_ (2023)

  <details>
    <summary>TLDR</summary>
    This paper introduces AdapterSoup, an approach that performs weight-space averaging of adapters trained on different domains, and explores various approaches for choosing which adapters to combine, such as text clustering and semantic similarity.
  </details>

  [[Paper PDF]](https://aclanthology.org/2023.findings-eacl.153.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/629bc57782bb4326a3eb5f89314e350729c5f417)

- **Combining Modular Skills in Multitask Learning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/McGill-NLP/polytropon?color=yellow&logo=github) ![](https://img.shields.io/badge/-Polytropon-blue)

  arXiv.org

  _E. Ponti, Alessandro Sordoni, Siva Reddy_ (2022)

  <details>
    <summary>TLDR</summary>
    It is found that the modular design of a network significantly increases sample efficiency in reinforcement learning and few-shot generalisation in supervised learning, compared to baselines with fully shared, task-specific, or conditionally generated parameters where knowledge is entangled across tasks.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2202.13914.pdf)&nbsp; [[Code]](https://github.com/McGill-NLP/polytropon)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/0d56e3d69c9b3112d77187f96fabcfbdf5303971)

- **Multi-Head Adapter Routing for Data-Efficient Fine-Tuning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/mttl?color=yellow&logo=github) ![](https://img.shields.io/badge/-Multi--Head%20routing-blue)

  arXiv.org

  _Lucas Caccia, E. Ponti, Lu Liu, Matheus Pereira, Nicolas Le Roux, Alessandro Sordoni_ (2022)

  <details>
    <summary>TLDR</summary>
    This paper investigates to what extent the ability to control which adapters are active for each task leads to sample-efﬁcient generalization and proposes less expressive variants where the authors perform weighted averaging of the adapters before few-shot adaptation (Poly - µ ) instead of learning a routing function.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2211.03831.pdf)&nbsp; [[Code]](https://github.com/microsoft/mttl)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/9201e46b2fe90c636d57b076020051953456473c)

- **Pushing Mixture of Experts to the Limit: Extremely Parameter Efficient MoE for Instruction Tuning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/for-ai/parameter-efficient-moe?color=yellow&logo=github) ![](https://img.shields.io/badge/-MoE-blue) ![](https://img.shields.io/badge/-Instruction%20tuning-blue) ![](https://img.shields.io/badge/-MoV-blue) ![](https://img.shields.io/badge/-MoLoRA-blue)

  arXiv.org

  _Ted Zadouri, A. Ustun, Arash Ahmadian, Beyza Ermics, Acyr Locatelli, Sara Hooker_ (2023)

  <details>
    <summary>TLDR</summary>
    This paper proposes extremely parameter-efficient MoE by uniquely combining MoE architecture with lightweight experts and is on par with full fine-tuning by only updating the lightweight experts -- less than 1% of an 11B parameters model.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2309.05444.pdf)&nbsp; [[Code]](https://github.com/for-ai/parameter-efficient-moe)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/5aae7d84f8eaa55f3386cee41d94769e7ab01e9d)

### Analysis and Evaluation

- **Common Sense or World Knowledge? Investigating Adapter-Based Knowledge Injection into Pretrained Transformers**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/wluper/retrograph?color=yellow&logo=github) ![](https://img.shields.io/badge/-ConceptNet-blue)

  Workshop on Knowledge Extraction and Integration for Deep Learning Architectures; Deep Learning Inside Out

  _Anne Lauscher, Olga Majewska, Leonardo F. R. Ribeiro, Iryna Gurevych, N. Rozanov, Goran Glavavs_ (2020)

  <details>
    <summary>TLDR</summary>
    A deeper analysis reveals that the adapter-based models substantially outperform BERT on inference tasks that require the type of conceptual knowledge explicitly present in ConceptNet and its corresponding Open Mind Common Sense corpus.
  </details>

  [[Paper PDF]](https://aclanthology.org/2020.deelio-1.5.pdf)&nbsp; [[Code]](https://github.com/wluper/retrograph)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/8b8c29c0cbb6cbae26b930840396596dd5806f33)

- **On the Effectiveness of Adapter-based Tuning for Pretrained Language Model Adaptation**&nbsp; 

  Annual Meeting of the Association for Computational Linguistics

  _Ruidan He, Linlin Liu, Hai Ye, Qingyu Tan, Bosheng Ding, Liying Cheng, Jia-Wei Low, Lidong Bing, Luo Si_ (2021)

  <details>
    <summary>TLDR</summary>
    It is demonstrated that 1) adapter-based tuning outperforms fine-tuning on low-resource and cross-lingual tasks; 2) it is more robust to overfitting and less sensitive to changes in learning rates.
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.acl-long.172.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/448af0627240e46df757e7b9c640ee30507c18e9)

- **Robust Transfer Learning with Pretrained Language Models through Adapters**&nbsp; 

  Annual Meeting of the Association for Computational Linguistics

  _Wenjuan Han, Bo Pang, Y. Wu_ (2021)

  <details>
    <summary>TLDR</summary>
    This work inserts small bottleneck layers (i.e., adapter) within each layer of a pretrained model, then fix the pretrained layers and train the adapter layers on the downstream task data, leading to improved stability and adversarial robustness in transfer learning to various downstream tasks.
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.acl-short.108.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/cdcffb2f1678d7252bfd9b902d3cd676a5217005)

- **AdapterDrop: On the Efficiency of Adapters in Transformers**&nbsp; ![](https://img.shields.io/badge/-AdapterDrop-blue) ![](https://img.shields.io/badge/-Parallel%20inference-blue)

  Conference on Empirical Methods in Natural Language Processing

  _Andreas Rücklé, Gregor Geigle, Max Glockner, Tilman Beck, Jonas Pfeiffer, Nils Reimers, Iryna Gurevych_ (2020)

  <details>
    <summary>TLDR</summary>
    This paper proposes AdapterDrop, removing adapters from lower transformer layers during training and inference, which incorporates concepts from all three directions and can dynamically reduce the computational overhead when performing inference over multiple tasks simultaneously, with minimal decrease in task performances.
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.emnlp-main.626.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/bdeec55f95fd6b73e3e4635459b14c7248543efb)

- **What to Pre-Train on? Efficient Intermediate Task Selection**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/adapter-hub/efficient-task-transfer?color=yellow&logo=github) ![](https://img.shields.io/badge/-Intermediate%20task%20transfer-blue)

  Conference on Empirical Methods in Natural Language Processing

  _Clifton A. Poth, Jonas Pfeiffer, Andreas Ruckl'e, Iryna Gurevych_ (2021)

  <details>
    <summary>TLDR</summary>
    This work provides a comprehensive comparison of different methods for efficiently identifying beneficial tasks for intermediate transfer learning, focusing on parameter and computationally efficient adapter settings, highlight different data-availability scenarios, and provide expense estimates for each method.
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.emnlp-main.827.pdf)&nbsp; [[Code]](https://github.com/adapter-hub/efficient-task-transfer)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/7b99c51d562e33309a46601c846abbe72a65c6a4)

- **Orthogonal Language and Task Adapters in Zero-Shot Cross-Lingual Transfer**&nbsp; ![](https://img.shields.io/badge/-Orthogonal%20adapters-blue)

  arXiv.org

  _M. Vidoni, Ivan Vulic, Goran Glavas_ (2020)

  <details>
    <summary>TLDR</summary>
    This work proposes orthogonal language and task adapters (dubbed orthoadapters) for cross-lingual transfer that are trained to encode language- and task-specific information that is complementary to the knowledge already stored in the pretrained transformer's parameters.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2012.06460.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/79165e99d67d2b4f5841464ad8eaf9e30205b62a)

- **P-Tuning: Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/THUDM/P-tuning-v2?color=yellow&logo=github) ![](https://img.shields.io/badge/-P--Tuning%20v2-blue) ![](https://img.shields.io/badge/-Prefix--Tuning-blue)

  Annual Meeting of the Association for Computational Linguistics

  _Xiao Liu, Kaixuan Ji, Yicheng Fu, W. Tam, Zhengxiao Du, Zhilin Yang, Jie Tang_ (2022)

  <details>
    <summary>TLDR</summary>
    The method P-Tuning v2 is an implementation of Deep Prompt Tuning (CITATION) optimized and adapted for NLU and can serve as an alternative to finetuning and a strong baseline for future research.
  </details>

  [[Paper PDF]](https://aclanthology.org/2022.acl-short.8.pdf)&nbsp; [[Code]](https://github.com/THUDM/P-tuning-v2)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/ec936b808e0fab9281c050ad4010cddec92c8cbe)

- **Delta Tuning: A Comprehensive Study of Parameter Efficient Methods for Pre-trained Language Models**&nbsp; 

  arXiv.org

  _Ning Ding, Yujia Qin, Guang Yang, Fu Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, Jing Yi, Weilin Zhao, Xiaozhi Wang, Zhiyuan Liu, Haitao Zheng, Jianfei Chen, Yang Liu, Jie Tang, Juan Li, Maosong Sun_ (2022)

  <details>
    <summary>TLDR</summary>
    The theoretical principles underlying the effectiveness of delta tuning are discussed and frameworks to interpret delta tuning from the perspective of optimization and optimal control are proposed, where results on over 100 NLP tasks demonstrate a comprehensive performance comparison of different approaches.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2203.06904.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/8c62277dada489904a63de4dd87336c27c68fb5e)

- **UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/morningmoni/unipelt?color=yellow&logo=github) ![](https://img.shields.io/badge/-UniPELT-blue)

  Annual Meeting of the Association for Computational Linguistics

  _Yuning Mao, Lambert Mathias, Rui Hou, Amjad Almahairi, Hao Ma, Jiawei Han, Wen-tau Yih, Madian Khabsa_ (2021)

  <details>
    <summary>TLDR</summary>
    A unified framework, UniPELT, is proposed, which incorporates different PELT methods as submodules and learns to activate the ones that best suit the current data or task setup via gating mechanism, indicating that a mixture of multiple P ELT methods may be inherently more effective than single methods.
  </details>

  [[Paper PDF]](https://aclanthology.org/2022.acl-long.433.pdf)&nbsp; [[Code]](https://github.com/morningmoni/unipelt)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/ad471be93216ddbf8544721d50ee5aed14f07cae)

### Applications

- **Simple, Scalable Adaptation for Neural Machine Translation**&nbsp; ![](https://img.shields.io/badge/-Bottleneck%20adapter-blue) ![](https://img.shields.io/badge/-Machine%20Translation-blue)

  Conference on Empirical Methods in Natural Language Processing

  _Ankur Bapna, N. Arivazhagan, Orhan Firat_ (2019)

  <details>
    <summary>TLDR</summary>
    The proposed approach consists of injecting tiny task specific adapter layers into a pre-trained model, which adapt the model to multiple individual tasks simultaneously, paving the way towards universal machine translation.
  </details>

  [[Paper PDF]](https://aclanthology.org/D19-1165.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/48530f3d6425f2f150f07ccdd61ba951951a0a7d)

- **Monolingual Adapters for Zero-Shot Neural Machine Translation**&nbsp; ![](https://img.shields.io/badge/-Bottleneck%20adapter-blue) ![](https://img.shields.io/badge/-Machine%20Translation-blue)

  Conference on Empirical Methods in Natural Language Processing

  _Jerin Philip, Alexandre Berard, Matthias Gallé, L. Besacier_ (2020)

  <details>
    <summary>TLDR</summary>
    A novel adapter layer formalism for adapting multilingual models is proposed, which is more parameter-efficient than existing adapter layers while obtaining as good or better performance.
  </details>

  [[Paper PDF]](https://aclanthology.org/2020.emnlp-main.361.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/8b31fef217004560b8c2517c0f6fdc1c3cf55112)

- **UDapter: Language Adaptation for Truly Universal Dependency Parsing**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/ahmetustun/udapter?color=yellow&logo=github) ![](https://img.shields.io/badge/-UDapter-blue) ![](https://img.shields.io/badge/-Dependency%20Parsing-blue)

  Conference on Empirical Methods in Natural Language Processing

  _A. Ustun, Arianna Bisazza, G. Bouma, Gertjan van Noord_ (2020)

  <details>
    <summary>TLDR</summary>
    A novel multilingual task adaptation approach based on recent work in parameter-efficient transfer learning, which allows for an easy but effective integration of existing linguistic typology features into the parsing network, and consistently outperforms strong monolingual and multilingual baselines on both high-resource and low-resource languages.
  </details>

  [[Paper PDF]](https://aclanthology.org/2020.emnlp-main.180.pdf)&nbsp; [[Code]](https://github.com/ahmetustun/udapter)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/3b233bdb697cc43effa1eb6d2868ff14efbbab7a)

- **Single-dataset Experts for Multi-dataset Question Answering**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/princeton-nlp/made?color=yellow&logo=github) ![](https://img.shields.io/badge/-Bottleneck%20adapter-blue) ![](https://img.shields.io/badge/-MADE-blue)

  Conference on Empirical Methods in Natural Language Processing

  _Dan Friedman, Ben Dodge, Danqi Chen_ (2021)

  <details>
    <summary>TLDR</summary>
    This work trains a collection of lightweight, dataset-specific adapter modules that share an underlying Transformer model, and finds that these Multi-Adapter Dataset Experts (MADE) outperform all the authors' baselines in terms of in-distribution accuracy, and simple methods based on parameter-averaging lead to better zero-shot generalization and few-shot transfer performance.
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.emnlp-main.495.pdf)&nbsp; [[Code]](https://github.com/princeton-nlp/made)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/67dc4ba4542d5895862c8b5af5023f659c14542c)

- **UNKs Everywhere: Adapting Multilingual Language Models to New Scripts**&nbsp; ![](https://img.shields.io/badge/-Bottleneck%20adapter-blue) ![](https://img.shields.io/badge/-MAD--X-blue)

  Conference on Empirical Methods in Natural Language Processing

  _Jonas Pfeiffer, Ivan Vulic, Iryna Gurevych, Sebastian Ruder_ (2020)

  <details>
    <summary>TLDR</summary>
    This work proposes a series of novel data-efficient methods that enable quick and effective adaptation of pretrained multilingual models to such low-resource languages and unseen scripts and demonstrates that they can yield improvements for low- resource languages written in scripts covered by the pretrained model.
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.emnlp-main.800.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/13bcfb944779165983aaef22cec8a3bbd3e98e62)

- **Multilingual Domain Adaptation for NMT: Decoupling Language and Domain Information with Adapters**&nbsp; ![](https://img.shields.io/badge/-Bottleneck%20adapter-blue) ![](https://img.shields.io/badge/-Machine%20Translation-blue)

  Conference on Machine Translation

  _Asa Cooper Stickland, Alexandre Berard, Vassilina Nikoulina_ (2021)

  <details>
    <summary>TLDR</summary>
    This work study the compositionality of language and domain adapters in the context of Machine Translation, and aims to study parameter-efficient adaptation to multiple domains and languages simultaneously and cross-lingual transfer in domains where parallel data is unavailable for certain language pairs.
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.wmt-1.64.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/51d62830c1112ea7443398990b850a988ed7c86c)

- **Multilingual Unsupervised Neural Machine Translation with Denoising Adapters**&nbsp; ![](https://img.shields.io/badge/-Denoising%20adapter-blue) ![](https://img.shields.io/badge/-Machine%20Translation-blue)

  Conference on Empirical Methods in Natural Language Processing

  _A. Ustun, Alexandre Berard, L. Besacier, Matthias Gallé_ (2021)

  <details>
    <summary>TLDR</summary>
    This paper proposes to use _denoising adapters_, adapter layers with a denoising objective, on top of pre-trained mBART-50, and shows that the resulting translations are on-par with back-translating as measured by BLEU, and furthermore it allows adding unseen languages incrementally.
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.emnlp-main.533.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/99b12d0df2b93e800207a5e4618a353912f3dff8)

- **Efficient Test Time Adapter Ensembling for Low-resource Language Varieties**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/cindyxinyiwang/emea?color=yellow&logo=github) ![](https://img.shields.io/badge/-EMEA-blue)

  Conference on Empirical Methods in Natural Language Processing

  _Xinyi Wang, Yulia Tsvetkov, Sebastian Ruder, Graham Neubig_ (2021)

  <details>
    <summary>TLDR</summary>
    This paper finds that ensembling multiple existing language adapters makes the fine-tuned model significantly more robust to other language varieties not included in these adapters, and proposes EMA, a method that optimizes the ensemble weights of the pretrained language adapters for each test sentence by minimizing the entropy of its predictions.
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.findings-emnlp.63.pdf)&nbsp; [[Code]](https://github.com/cindyxinyiwang/emea)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/7b5b15279e5a52439614f886b79fa33f4b88bfb2)

- **LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/zrrskywalker/llama-adapter?color=yellow&logo=github) ![](https://img.shields.io/badge/-Prefix--Tuning-blue) ![](https://img.shields.io/badge/-Instruction%20tuning-blue)

  arXiv.org

  _Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, Y. Qiao_ (2023)

  <details>
    <summary>TLDR</summary>
    A zero-initialized attention mechanism with zero gating is proposed, which adaptively injects the new instructional cues into LLaMA, while effectively preserves its pre-trained knowledge on traditional vision and language tasks, demonstrating the superior generalization capacity of the approach.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2303.16199.pdf)&nbsp; [[Code]](https://github.com/zrrskywalker/llama-adapter)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/a757999ed260d7bc45484dc6b4456bf33fe6f679)

### Serving

- **Punica: Multi-Tenant LoRA Serving**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/punica-ai/punica?color=yellow&logo=github) ![](https://img.shields.io/badge/-Punica-blue) ![](https://img.shields.io/badge/-LoRA-blue)

  arXiv.org

  _Lequn Chen, Zihao Ye, Yongji Wu, Danyang Zhuo, Luis Ceze, Arvind Krishnamurthy University of Washington, Duke University_ (2023)

  <details>
    <summary>TLDR</summary>
    Punica is a system to serve multiple LoRA models in a shared GPU cluster that contains a new CUDA kernel design that allows batching of GPU operations for differentLoRA models, significantly enhancing GPU efficiency in terms of both memory and computation.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2310.18547.pdf)&nbsp; [[Code]](https://github.com/punica-ai/punica)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/c2314751d367b34239a537fe27e2bd51a8b84528)

- **S-LoRA: Serving Thousands of Concurrent LoRA Adapters**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/S-LoRA/S-LoRA?color=yellow&logo=github) ![](https://img.shields.io/badge/-S--LoRA-blue) ![](https://img.shields.io/badge/-LoRA-blue)

  arXiv.org

  _Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu, Lianmin Zheng, Kurt Keutzer, Joseph E. Gonzalez, Ion Stoica_ (2023)

  <details>
    <summary>TLDR</summary>
    S-LoRA enables scalable serving of many task-specific fine-tuned models and offers the potential for large-scale customized fine- Tuning services.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2311.03285.pdf)&nbsp; [[Code]](https://github.com/S-LoRA/S-LoRA)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/9bb8c4325c609caeade9c3ed7036d2b9953e278c)


## Computer Vision

### Methods

- **Learning multiple visual domains with residual adapters**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/srebuffi/residual_adapters?color=yellow&logo=github) ![](https://img.shields.io/badge/-Bottleneck%20adapter-blue)

  Neural Information Processing Systems

  _Sylvestre-Alvise Rebuffi, Hakan Bilen, A. Vedaldi_ (2017)

  <details>
    <summary>TLDR</summary>
    This paper develops a tunable deep network architecture that, by means of adapter residual modules, can be steered on the fly to diverse visual domains and introduces the Visual Decathlon Challenge, a benchmark that evaluates the ability of representations to capture simultaneously ten very differentVisual domains and measures their ability to recognize well uniformly.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/1705.08045.pdf)&nbsp; [[Code]](https://github.com/srebuffi/residual_adapters)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/d89ee98810039d2061ed42ee8026da49c503d16b)

- **Efficient Parametrization of Multi-domain Deep Neural Networks**&nbsp; ![](https://img.shields.io/badge/-Bottleneck%20adapter-blue)

  2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition

  _Sylvestre-Alvise Rebuffi, Hakan Bilen, A. Vedaldi_ (2018)

  <details>
    <summary>TLDR</summary>
    This paper proposes to consider universal parametric families of neural networks, which still contain specialized problem-specific models, but differing only by a small number of parameters, and shows that these universal parametrization are very effective for transfer learning, where they outperform traditional fine-tuning techniques.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/1803.10082.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/4081de7e0f94e7e0d7b645c298d7768698d05774)

- **Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets**&nbsp; ![](https://img.shields.io/badge/-Conv--Adapter-blue)

  arXiv.org

  _Hao Chen, R. Tao, Han Zhang, Yidong Wang, Weirong Ye, Jindong Wang, Guosheng Hu, M. Savvides_ (2022)

  <details>
    <summary>TLDR</summary>
    Conv-Adapter, a PET module designed for ConvNets, is light-weight, domain-transferable, and architecture-agnostic with generalized performance on different tasks with comparable or surpasses the performance of full fine-tuning on 23 classification tasks of various domains.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2208.07463.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/3049c05e77a5761aa051b812a91d445ac3b31256)

- **AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/ShoufaChen/AdaptFormer?color=yellow&logo=github) ![](https://img.shields.io/badge/-Bottleneck%20adapter-blue) ![](https://img.shields.io/badge/-Image%20classification-blue) ![](https://img.shields.io/badge/-Video%20classification-blue)

  Neural Information Processing Systems

  _Shoufa Chen, Chongjian Ge, Zhan Tong, Jiangliu Wang, Yibing Song, Jue Wang, Ping Luo_ (2022)

  <details>
    <summary>TLDR</summary>
    AdaptFormer introduces lightweight modules that only add less than 2% extra parameters to a ViT, while it is able to increase the ViT's transferability without updating its original pre-trained parameters, significantly outperforming the existing 100\% fully fine-tuned models on action recognition benchmarks.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2205.13535.pdf)&nbsp; [[Code]](https://github.com/ShoufaChen/AdaptFormer)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/2fe2f849b94cf08b559226bc9d78adcaef5ef186)


## Audio Processing

### Applications

- **Lightweight Adapter Tuning for Multilingual Speech Translation**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/formiel/fairseq?color=yellow&logo=github) ![](https://img.shields.io/badge/-Speech%20translation-blue)

  Annual Meeting of the Association for Computational Linguistics

  _Hang Le, J. Pino, Changhan Wang, Jiatao Gu, D. Schwab, L. Besacier_ (2021)

  <details>
    <summary>TLDR</summary>
    This paper proposes a comprehensive analysis of adapters for multilingual speech translation (ST) and shows that adapters can be used to efficiently specialize ST to specific language pairs with a low extra cost in terms of parameters.
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.acl-short.103.pdf)&nbsp; [[Code]](https://github.com/formiel/fairseq/blob/master/examples/speech_to_text/docs/adapters.md)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/eacb5dc57a167aeda3b23c28abfc2b51095f1b7c)

- **Efficient Adapter Transfer of Self-Supervised Speech Models for Automatic Speech Recognition**&nbsp; ![](https://img.shields.io/badge/-ASR-blue)

  IEEE International Conference on Acoustics, Speech, and Signal Processing

  _Bethan Thomas, Samuel Kessler, S. Karout_ (2022)

  <details>
    <summary>TLDR</summary>
    Adapters are applied to wav2vec 2.0 to reduce the number of parameters required for downstream ASR tasks, and increase scalability of the model to multiple tasks or languages, and support the theory that higher pre-trained layers encode more phonemic information.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2202.03218.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/f6919b54a4f06367947f0cf58cda54cdd08cd5f2)


## Multi-Modal

### Methods

- **VL-ADAPTER: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/ylsung/VL_adapter?color=yellow&logo=github) ![](https://img.shields.io/badge/-VL--Adapter-blue)

  Computer Vision and Pattern Recognition

  _Yi-Lin Sung, Jaemin Cho, Mohit Bansal_ (2021)

  <details>
    <summary>TLDR</summary>
    The results demonstrate that training the adapter with the weight-sharing technique can match the performance of fine-tuning the entire model, and enhance the efficiency and performance of adapters by sharing their weights to attain knowledge across tasks.
  </details>

  [[Paper PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Sung_VL-Adapter_Parameter-Efficient_Transfer_Learning_for_Vision-and-Language_Tasks_CVPR_2022_paper.pdf)&nbsp; [[Code]](https://github.com/ylsung/VL_adapter)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/55a19318cc93714802c7ac59e07651789749b20c)

- **LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/ylsung/ladder-side-tuning?color=yellow&logo=github) ![](https://img.shields.io/badge/-Ladder%20Side--Tuning-blue)

  Neural Information Processing Systems

  _Yi-Lin Sung, Jaemin Cho, Mohit Bansal_ (2022)

  <details>
    <summary>TLDR</summary>
    LST has significantly lower memory requirements than previous methods, because it does not require backpropagation through the backbone network, but instead only through the side network and ladder connections, and achieves higher accuracy than Adapter and LoRA in a low-memory regime.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2206.06522.pdf)&nbsp; [[Code]](https://github.com/ylsung/ladder-side-tuning)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/960d40497717ad22a7ebb84db238fa2415fc89cc)

- **Conditional Adapters: Parameter-efficient Transfer Learning with Fast Inference**&nbsp; ![](https://img.shields.io/badge/-CODA-blue)

  Neural Information Processing Systems

  _Tao Lei, Junwen Bai, Siddhartha Brahma, J. Ainslie, Kenton Lee, Yanqi Zhou, Nan Du, Vincent Zhao, Yuexin Wu, Bo Li, Yu Zhang, Ming-Wei Chang_ (2023)

  <details>
    <summary>TLDR</summary>
    This work proposes Conditional Adapter (CoDA), a parameter-efficient transfer learning method that also improves inference efficiency and achieves a 2x to 8x inference speed-up compared to the state-of-the-art Adapter approaches with moderate to no accuracy loss and the same parameter efficiency.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2304.04947.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/148644bf4ccef7e022b965304e8b3178be8af0fa)

- **VL-PET: Vision-and-Language Parameter-Efficient Tuning via Granularity Control**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/HenryHZY/VL-PET?color=yellow&logo=github) ![](https://img.shields.io/badge/-VL--PET-blue)

  IEEE International Conference on Computer Vision

  _Zi-Yuan Hu, Yanyang Li, M. Lyu, Liwei Wang_ (2023)

  <details>
    <summary>TLDR</summary>
    A Vision-and-Language Parameter-Efficient Tuning (VL-PET) framework to impose effective control over modular modifications via a novel granularity-controlled mechanism and a variety of model-agnostic VL-PET modules can be instantiated from this framework for better efficiency and effective-ness trade-offs is proposed.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2308.09804.pdf)&nbsp; [[Code]](https://github.com/HenryHZY/VL-PET)&nbsp; [[Website]](https://henryhzy.github.io/VL-PET)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/80a791f644defb54f4eb24f99df31e6f995be3aa)



## Contributing

Contributions of new awesome adapter-related resources are very welcome!
Before contributing, make sure to read this repository's [contributing guide](https://github.com/calpt/awesome-adapter-resources/blob/main/CONTRIBUTING.md).

## Acknowledgments

Paper metadata is partially retrieved via [Semantic Scholar's API](https://www.semanticscholar.org/product/api).
Paper TLDRs are provided by [Semantic Scholar's TLDR feature](https://www.semanticscholar.org/product/tldr).
