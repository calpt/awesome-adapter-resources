# Awesome Adapter Resources

![](https://img.shields.io/badge/Resources-{0}-blue)

This repository collects important tools and papers related to adapter methods for recent large pre-trained neural networks.

_Adapters_ (aka _Parameter-Efficient Transfer Learning (PETL)_ or _Parameter-Efficient Fine-Tuning (PEFT)_ methods) include various parameter-efficient approaches of adapting large pre-trained models to new tasks.

## Content

- [Why Adapters?](#why-adapters)
{1}
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

{2}

## Contributing

Contributions of new awesome adapter-related resources are very welcome!
Before contributing, make sure to read this repository's [contributing guide](https://github.com/calpt/awesome-adapter-resources/blob/main/CONTRIBUTING.md).

## Acknowledgments

Paper metadata is partially retrieved via [Semantic Scholar's API](https://www.semanticscholar.org/product/api).
Paper TLDRs are provided by [Semantic Scholar's TLDR feature](https://www.semanticscholar.org/product/tldr).
