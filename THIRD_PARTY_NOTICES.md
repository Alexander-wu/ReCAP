# Third-party notices

Original ReCAP additions use Apache-2.0. The following bundled components retain their upstream terms. Modifications include portable local asset paths, launcher integration, dataset adaptation and the ReCAP experiment functions; original attribution headers are retained.

| Component | Bundled use | License text |
|---|---|---|
| [iVideoGPT](https://github.com/thuml/iVideoGPT), 2024 THUML @ Tsinghua University | Tokenizer, processor and training implementation | [MIT](third_party/iVideoGPT-LICENSE) |
| [RLVR-World](https://github.com/thuml/RLVR-World), 2025 THUML @ Tsinghua University | Action-conditioned world-model training and inference base | [MIT](third_party/RLVR-World-LICENSE) |
| [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch), Phil Wang | Finite scalar quantization implementation embedded by upstream | [MIT](third_party/FSQ-LICENSE) |
| [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity), Richard Zhang et al. | Stripped LPIPS training utility | [BSD-2-Clause](third_party/PerceptualSimilarity-LICENSE) |
| [Paella](https://github.com/dome272/Paella), Dominic Rampas | Discriminator implementation embedded by upstream | [MIT](third_party/Paella-LICENSE) |
| [Diffusers](https://github.com/huggingface/diffusers), Hugging Face | Adapted VAE/model components | [Apache-2.0](third_party/diffusers-LICENSE) |
| [taming-transformers](https://github.com/CompVis/taming-transformers), CompVis | VQ/LPIPS utility ancestry embedded by upstream | [MIT](third_party/taming-transformers-LICENSE) |

`SOURCE_PROVENANCE.json` records hashes of the historical experiment source imported into the candidate. It is an import audit, not the current release file manifest. The precise upstream commit of every historical local file was not recorded; no commit identity is fabricated. `SOURCE_MANIFEST.json` is regenerated for the current source archive.

Starting checkpoints are [thuml/rt1-compressive-tokenizer](https://huggingface.co/thuml/rt1-compressive-tokenizer) and [thuml/rt1-world-model-multi-step-rlvr](https://huggingface.co/thuml/rt1-world-model-multi-step-rlvr). Their model cards identify MIT licensing. Derived model exports carry separate cards describing adaptation and pairing; they do not claim to be the original RT-1 checkpoints.

Dataset images and illustrative GIFs retain source dataset attribution and terms; no ownership or blanket Apache license is asserted over them. See [data sources](docs/DATA.md) and [case provenance](docs/CASES.md). LPIPS/VGG metric weights and raw datasets are downloaded from their publishers and are not bundled as ReCAP model weights.
