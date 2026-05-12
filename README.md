# CwlLlmSwift

Swift implementations of [`llm.c`](https://github.com/karpathy/llm.c), plus a SwiftUI test harness for training, inference, and cross-implementation validation.

This branch contains the phase-1 engines only: `llm.c`, Basic Swift, Fast Swift, Multithreaded Swift, Direct AMX, and Metal.

Parts of this repository vendor and modify code by Andrej Karpathy from [`llm.c`](https://github.com/karpathy/llm.c) and James Thompson from [llm.metal](https://github.com/regrettable-username/llm.metal) both under the MIT License.

## Getting started

Running this project requires input files:

* tiny_shakespeare_train.bin (training data)
* tiny_shakespeare_val.bin (validation data)
* gpt2_tokenizer.bin (tokenizer)
* gpt2_124M.bin (checkpoint)

From the following git repository:

    https://huggingface.co/datasets/karpathy/llmc-starter-pack

Follow the instructions there to clone or download the required files.

> NOTE: you will need to install `git-xet` to correctly download that repository
>
>     brew install git-xet
>     git xet install

In the test harness, you'll be asked to select these files to create a training dataset.

If you get an error in the app that the training dataset isn't UInt16 formatted, it probably means `git-xet` wasn't installed when you cloned the llmc-starter-pack repo.