# 主结果复现（tdrive，n=3，backend=siliconflow，model=siliconflow:Qwen/Qwen2.5-72B-Instruct）

| Method | Scale | Coverage | PSS | paper Coverage | paper PSS |
|---|---|---|---|---|---|
| RN | large | 5.321 ± 0.012 | 0.701 ± 0.098 | 4.974 | 0.594 |
| TVPG | large | 5.375 ± 0.041 | 0.738 ± 0.107 | 5.067 | 0.590 |
| TCPG | large | 5.333 ± 0.029 | 0.776 ± 0.097 | 5.043 | 0.602 |
| MSA | large | 5.354 ± 0.038 | 0.694 ± 0.044 | 5.131 | 0.598 |
| MSAGI | large | 5.378 ± 0.043 | 0.754 ± 0.151 | 5.128 | 0.590 |
| GraphDP | large | 5.285 ± 0.015 | 0.717 ± 0.103 | 5.214 | 0.628 |
| GPT-4.1-mini | large | 5.266 ± 0.073 | 0.636 ± 0.043 | 4.936 | 0.691 |
| MAPUS | large | 5.349 ± 0.004 | 0.791 ± 0.026 | 5.251 | 0.898 |
