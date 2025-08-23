# MIHBench&DAB: MIHBench: Benchmarking and Mitigating Multi-Image Hallucinations in Multimodal Large Language Models


This is the official repo for Dynamic Attention Balancing (DAB), a lightweight, training-free approach for reducing multi-image hallucinations in Multimodal Large Language Models (MLLMs) during decoding, without introducing additional inference overhead.

<div style='display:flex; gap: 0.25rem; '>
<a href='https://arxiv.org/abs/2508.00726'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
</div>

## 🔍 Method Overview
![DAB](assets/method.png){: width="50%" }
We introduce **Dynamic Attention Balancing (DAB)**, a simple and training-free method that adaptively redistributes attention across multiple image inputs during decoding.
The core of DAB involves the following steps:
1. **Attention Ratio Calculation**  
   The attention ratio for each image \( k \) is calculated as:
```math
\text{ratio}_k = \frac{\sum_{i=1}^{N_I} a_{l,h}^{i,j}}{N_I \times N_X}
```
   where:
   - \( a_{l,h}^{i,j} \) represents the attention weight from the \( i \)-th image token to the \( j \)-th text token in the \( l \)-th layer and \( h \)-th head.
   - \( N_I \) is the number of image tokens in the \( k \)-th image.
   - \( N_X \) is the number of text tokens.
2. **Average Attention Ratio**  
   The average attention ratio \( \text{avg\_ratio} \) across all images is computed as:
```math
\text{avg\_ratio} = \frac{1}{N_I} \sum_{k=1}^{N_I} \text{ratio}_k
```
3. **Attention Shift Calculation**  
   The attention shift for each image is calculated as:
```math
\Delta a_{l,h}^{k,j} = \text{avg\_ratio} - \text{ratio}_k
```
   where \( \Delta a_{l,h}^{k,j} \) represents the adjustment made to the attention weight for the \( k \)-th image's tokens.
4. **Adjusted Attention Weight**  
   Finally, the adjusted attention weight is:
```math
a_{l,h}^{\text{adjusted}} = a_{l,h}^{i,j} + \alpha \times \Delta a_{l,h}^{k,j}
```
   where \( \alpha \) is a balancing coefficient that controls the intensity of the adjustment.
By ensuring a more balanced attention allocation across all input images, DAB effectively mitigates over-reliance on individual image tokens, reducing multi-image hallucinations and improving semantic integration in multi-image reasoning tasks.




