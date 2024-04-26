# <div align="center">Converting Representational Counterfactuals to Natural Language</div>
<div align="center">Matan Avitan, Ryan Cotterell, Yoav Goldberg, Shauli Ravfogel
<br><br>

[Please see full details in our pre-print on arxiv](https://arxiv.org/abs/2402.11355)
</div> 
<div align="center">
<<a href="https://huggingface.co/MatanAvitan/gtr__nq__64_bios__correct"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-xl.svg" alt="BiasBios Inversion Model on HF"></a>
</div>


# TL;DR
Interventions targeting the representation space of language models (LMs) have emerged as effective means to influence model behavior. 
Such methods are employed, for example, to eliminate or alter the encoding of demographic information such as gender within the model's representations, and, in so doing, create a counterfactual representation. 
However, because the intervention operates within the representation space, understanding precisely what aspects of the text it modifies poses a challenge. 
In this paper, we give a method to convert representation counterfactuals into string counterfactuals.
We demonstrate that this approach enables us to analyze the linguistic alterations corresponding to a given representation space intervention and to interpret the features utilized for encoding a specific concept. Moreover, the resulting counterfactuals can be used to mitigate bias in classification through data augmentation.

<p align="center">
<object data="figures/Fig1.pdf" width="650" type="application/pdf">
</p>


