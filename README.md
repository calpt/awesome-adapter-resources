# Awesome Adapter Resources

![](https://img.shields.io/badge/Resources-49-blue)

This repository collects important tools and papers related to adapter methods for recent large pre-trained neural networks.

_Adapters_ (aka _Parameter-Efficient Transfer Learning (PETL)_ or _Parameter-Efficient Fine-Tuning (PEFT)_ methods) include various parameter-efficient approaches of adapting large pre-trained models to new tasks.

## Content

- [Why Adapters?](#why-adapters)
- [Frameworks and Tools](#frameworks-and-tools)
- [Surveys](#surveys)
- [Natural Language Processing](#natural-language-processing)
  - [Methods](#methods)
  - [Analysis and Evaluation](#analysis-and-evaluation)
  - [Applications](#applications)
- [Computer Vision](#computer-vision)
  - [Methods](#methods)
- [Audio Processing](#audio-processing)
  - [Applications](#applications)
- [Multi-Modal](#multi-modal)
  - [Methods](#methods)
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

- **OpenDelta**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/thunlp/OpenDelta?color=yellow&logo=github) 

   

  

  [[Code]](https://github.com/thunlp/OpenDelta)&nbsp; [[Website]](https://opendelta.readthedocs.io/)

- **PEFT: State-of-the-art Parameter-Efficient Fine-Tuning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/huggingface/peft?color=yellow&logo=github) 

   

  

  [[Code]](https://github.com/huggingface/peft)

- **LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/AGI-Edgerunners/LLM-Adapters?color=yellow&logo=github) 

  arXiv.org

  _Zhiqiang Hu, Yihuai Lan, Lei Wang, Wanyu Xu, Ee-Peng Lim, R. Lee, Lidong Bing, Soujanya Poria_ (2023)

  <details>
    <summary>TLDR</summary>
    LLM-Adapters is presented, an easy-to-use framework that integrates various adapters into LLMs and can execute these adapter-based PEFT methods of LLMs for different tasks, and provides a promising framework for fine-tuning large LLMs on downstream tasks.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2304.01933.pdf)&nbsp; [[Code]](https://github.com/AGI-Edgerunners/LLM-Adapters)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/32d3b01a83eeb996052eb6d03a7667a30c5a9969)


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

- **Towards a Unified View of Parameter-Efficient Transfer Learning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/jxhe/unify-parameter-efficient-tuning?color=yellow&logo=github) ![](https://img.shields.io/badge/-Mix--and--Match%20adapters-blue) ![](https://img.shields.io/badge/-Parallel%20adapters-blue)

  International Conference on Learning Representations

  _Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig_ (2021)

  <details>
    <summary>TLDR</summary>
    This paper re-frame state-of-the-art parameter-efficient transfer learning methods as modifications to specific hidden states in pre-trained models, and defines a set of design dimensions along which different methods vary, achieving comparable results to fine-tuning all parameters on all four tasks.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2110.04366.pdf)&nbsp; [[Code]](https://github.com/jxhe/unify-parameter-efficient-tuning)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/43a87867fe6bf4eb920f97fc753be4b727308923)

- **Compacter: Efficient Low-Rank Hypercomplex Adapter Layers**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/rabeehk/compacter?color=yellow&logo=github) ![](https://img.shields.io/badge/-Compacter-blue) ![](https://img.shields.io/badge/-Compacter++-blue) ![](https://img.shields.io/badge/-PHM--Adapter-blue)

  Neural Information Processing Systems

  _Rabeeh Karimi Mahabadi, James Henderson, Sebastian Ruder_ (2021)

  <details>
    <summary>TLDR</summary>
    Compacter is proposed, a method for fine-tuning large-scale language models with a better trade-off between task performance and the number of trainable parameters than prior work, and accomplishes this by building on top of ideas from adapters, low-rank optimization, and parameterized hypercomplex multiplication layers.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2106.04647.pdf)&nbsp; [[Code]](https://github.com/rabeehk/compacter)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/b19cba7bfe318c69d5e62f8322cb5d75228452f4)

- **LoRA: Low-Rank Adaptation of Large Language Models**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/LoRA?color=yellow&logo=github) ![](https://img.shields.io/badge/-LoRA-blue)

  International Conference on Learning Representations

  _Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Weizhu Chen_ (2021)

  <details>
    <summary>TLDR</summary>
    Low-Rank Adaptation, or LoRA, is proposed, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2106.09685.pdf)&nbsp; [[Code]](https://github.com/microsoft/LoRA)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/a8ca46b171467ceb2d7652fbfb67fe701ad86092)

- **Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/rabeehk/hyperformer?color=yellow&logo=github) ![](https://img.shields.io/badge/-HyperFormer-blue)

  Annual Meeting of the Association for Computational Linguistics

  _Rabeeh Karimi Mahabadi, Sebastian Ruder, M. Dehghani, J. Henderson_ (2021)

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
    MAD-G is proposed, which contextually generates language adapters from language representations based on typological features and remains competitive with more expensive methods for language-speciﬁc adapter training across the board, particularly on the NER task in low-resource African languages.
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

  arXiv.org

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
    AutoPEFT, a novel framework to traverse this configuration space: it automatically configures multiple PEFT modules via high-dimensional Bayesian optimisation, and shows the resource scalability and task transferability of AutoPEFT-found configurations, outperforming existing PEFT methods on average on the standard GLUE benchmark.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2301.12132.pdf)&nbsp; [[Code]](https://github.com/cambridgeltl/autopeft)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/d3fea42e76b093e78e61073fefe0cfa63b543d60)

- **Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/QingruZhang/AdaLoRA?color=yellow&logo=github) ![](https://img.shields.io/badge/-AdaLoRA-blue)

  arXiv.org

  _Qingru Zhang, Minshuo Chen, Alexander W. Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, Tuo Zhao_ (2023)

  <details>
    <summary>TLDR</summary>
    The proposed AdaLoRA adaptively allocates the parameter budget among weight matrices according to their importance score, which allows us to effectively prune the singular values of unimportant updates, which is essentially to reduce their parameter budget but circumvent intensive exact SVD computations.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2303.10512.pdf)&nbsp; [[Code]](https://github.com/QingruZhang/AdaLoRA)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/5ef82a8c8aa50f99285f2143b57ca4e82da1af80)

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

- **AdapterDrop: On the Efficiency of Adapters in Transformers**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/google-research/adapter-bert?color=yellow&logo=github) ![](https://img.shields.io/badge/-AdapterDrop-blue) ![](https://img.shields.io/badge/-Parallel%20inference-blue)

  Conference on Empirical Methods in Natural Language Processing

  _Andreas Rücklé, Gregor Geigle, Max Glockner, Tilman Beck, Jonas Pfeiffer, Nils Reimers, Iryna Gurevych_ (2020)

  <details>
    <summary>TLDR</summary>
    This paper proposes AdapterDrop, removing adapters from lower transformer layers during training and inference, which incorporates concepts from all three directions and can dynamically reduce the computational overhead when performing inference over multiple tasks simultaneously, with minimal decrease in task performances.
  </details>

  [[Paper PDF]](https://aclanthology.org/2021.emnlp-main.626.pdf)&nbsp; [[Code]](https://github.com/google-research/adapter-bert)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/bdeec55f95fd6b73e3e4635459b14c7248543efb)

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
    Though initially proposed as an efficient method to steer large models, some of the fascinating evidence discovered along with delta tuning could help further reveal the mechanisms of PLMs and even deep neural networks.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2203.06904.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/8c62277dada489904a63de4dd87336c27c68fb5e)

- **UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/morningmoni/unipelt?color=yellow&logo=github) ![](https://img.shields.io/badge/-UniPELT-blue)

  Annual Meeting of the Association for Computational Linguistics

  _Yuning Mao, Lambert Mathias, Rui Hou, Amjad Almahairi, Hao Ma, Jiawei Han, Wen-tau Yih, Madian Khabsa_ (2021)

  <details>
    <summary>TLDR</summary>
    A unified framework, UniPELT, is proposed, which incorporates different PELT methods as submodules and learns to activate the ones that best suit the current data or task setup via gating mechanism, indicating that a mixture of multiple P ELT methods may be inherently more effective than single methods.
  </details>

  [[Paper PDF]](https://aclanthology.org/2022.acl-long.433.pdf)&nbsp; [[Code]](https://github.com/morningmoni/unipelt)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/34027ecbb3fa651c4ea3980911cb813317769dc0)

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

  _Jerin Philip, Alexandre Bérard, Matthias Gallé, L. Besacier_ (2020)

  

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
    A zero-init attention mechanism with zero gating is proposed, which adaptively injects the new instructional cues into LLaMA, while effectively preserves its pre-trained knowledge.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2303.16199.pdf)&nbsp; [[Code]](https://github.com/zrrskywalker/llama-adapter)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/b259d853b71a2d03cefa844bb9343b8e3ed816b1)


## Computer Vision

### Methods

- **Learning multiple visual domains with residual adapters**&nbsp; ![GitHub Repo stars](https://img.shields.io/github/stars/srebuffi/residual_adapters?color=yellow&logo=github) ![](https://img.shields.io/badge/-Bottleneck%20adapter-blue)

  NIPS

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

  arXiv.org

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

  arXiv.org

  _Yi-Lin Sung, Jaemin Cho, Mohit Bansal_ (2022)

  <details>
    <summary>TLDR</summary>
    LST has significantly lower memory requirements than previous methods, because it does not require backpropagation through the backbone network, but instead only through the side network and ladder connections, and achieves higher accuracy than Adapter and LoRA in a low-memory regime.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2206.06522.pdf)&nbsp; [[Code]](https://github.com/ylsung/ladder-side-tuning)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/960d40497717ad22a7ebb84db238fa2415fc89cc)

- **Conditional Adapters: Parameter-efficient Transfer Learning with Fast Inference**&nbsp; ![](https://img.shields.io/badge/-CODA-blue)

  arXiv.org

  _Tao Lei, Junwen Bai, Siddhartha Brahma, J. Ainslie, Kenton Lee, Yanqi Zhou, Nan Du, Vincent Zhao, Yuexin Wu, Bo Li, Yu Zhang, Ming-Wei Chang_ (2023)

  <details>
    <summary>TLDR</summary>
    This work proposes Conditional Adapter (CoDA), a parameter-efficient transfer learning method that also improves inference efficiency and achieves a 2x to 8x inference speed-up compared to the state-of-the-art Adapter approach with moderate to no accuracy loss and the same parameter efficiency.
  </details>

  [[Paper PDF]](https://arxiv.org/pdf/2304.04947.pdf)&nbsp; [[Semantic Scholar]](https://www.semanticscholar.org/paper/148644bf4ccef7e022b965304e8b3178be8af0fa)



## Contributing

Contributions of new awesome adapter-related resources are very welcome!
Before contributing, make sure to read this repository's [contributing guide](https://github.com/calpt/awesome-adapter-resources/blob/main/CONTRIBUTING.md).

## Acknowledgments

Paper metadata is partially retrieved via [Semantic Scholar's API](https://www.semanticscholar.org/product/api).
Paper TLDRs are provided by [Semantic Scholar's TLDR feature](https://www.semanticscholar.org/product/tldr).
