# AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage

[![🤗 Benchmark (HuggingFace)](https://img.shields.io/badge/Dataset-HuggingFace-FFD21E.svg?logo=huggingface&logoColor=yellow)](https://huggingface.co/datasets/ai9stars/ReproduceBench) [![📑 Paper (arXiv:2505.20662)](https://img.shields.io/badge/arXiv-2505.20662-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2505.20662)


This is the official repo of AutoReproduce and ReproduceBench.

## Overview
![main](autorp.png)

## AutoReproduce
We are currently organizing the code.
### Quick Start
```
export OPENAI_API_KEY="<OPENAI_API_KEY>"
export BASE_URL="<BASE_URL>" #If necessary

python reproduce.py #Default setting
```

For reproduce certain paper, the paper content should be downloaded (We are currently organizing the use of [Mineru](https://huggingface.co/spaces/opendatalab/MinerU) for automation.).

If the data cannot be obtained directly, please download the data in advance and modify the instruction to specify the path. 
```
python reproduce.py --paper-path xxx --dataloader-path xxx
```

## ReproduceBench
### Download Datasets
All the datasets and human-curated reference code could be available at [ReproduceBench](https://huggingface.co/datasets/ai9stars/ReproduceBench).
```
pip install -U huggingface_hub
cd AutoReproduce
huggingface-cli download --repo-type dataset --resume-download ai9stars/ReproduceBench --local-dir ReproduceBench
```
### Evaluation
All the evaluation code are under ```evaluation```. The current code is not well-structured. We are currently working on organizing it.
```
# First summarize the key points of the paper.
python evaluation/summarize_points.py
# Then run the following files to calculate align-score.
python evaluation/eval_high.py
python evaluation/eval_low.py
python evaluation/eval_mixed.py
```

## Contact

For any questions, you can contact [2429527z@gmail.com](mailto:2429527z@gmail.com).


## Citation
If you find this work useful, consider giving this repository a star ⭐️ and citing 📝 our paper as follows:
```
@misc{zhao2025autoreproduceautomaticaiexperiment,
      title={AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage}, 
      author={Xuanle Zhao and Zilin Sang and Yuxuan Li and Qi Shi and Shuo Wang and Duzhen Zhang and Xu Han and Zhiyuan Liu and Maosong Sun},
      year={2025},
      eprint={2505.20662},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.20662}, 
}
```

## Acknowledgement
The code is based on the [Agent Laboratory](https://github.com/SamuelSchmidgall/AgentLaboratory). Thanks for these great works and open sourcing!
