# Multi-Method-Jailbreak-Attacks <img src="Figures/大模型.svg" alt="icon" width="35" height="35" />



> 本项目是“网络信息与安全”课程的实验课题，旨在系统化评估大语言模型的越狱攻击与防御。

## 项目简介

随着大语言模型（LLM）在自然语言处理、智能问答、辅助决策等领域的广泛应用，其安全性与可靠性问题日益凸显。本项目以 **LLaVA-1.5-7B**、**ChatGLM3-6B**、**Qwen-7B** 及 **GPT 系列** 为测试对象，复现并评估了三类代表性越狱攻击方法：  
1. **基于专业知识的提示工程**  
2. **黑盒自动化 PAIR 算法**  
3. **白盒最优化 GCG 算法**  

我们在统一实验环境下，对比了各方法的 **攻击成功率**、**平均查询次数** 及 **计算耗时**，并提出了针对不同防御强度模型的优化策略与后续研究方向。

---

## 方法概览

| 方法                   | 类型    | 核心思路                                                     | 要求           |
| ---------------------- | ------- | ------------------------------------------------------------ | -------------- |
| 提示工程（Prompting）  | 黑盒    | 设计专家级提示模板，引导模型生成敏感内容                     | 无需模型内部信息 |
| PAIR 算法              | 黑盒    | 自动化迭代优化提示，通过“提示→响应→判定→改进”循环直至成功   | 无需梯度信息   |
| GCG 算法               | 白盒    | 将可注入后缀视作可微参数，贪心替换梯度贡献最大的 token        | 需访问梯度     |

---

## 实验环境

- **操作系统**：Ubuntu 22.04  
- **开发语言**：Python 3.10  
- **关键依赖**：
  - `torch` ≥ 2.1  
  - `transformers` ≥ 4.30  
  - `bitsandbytes`  
  - `tqdm`  
- **硬件**：NVIDIA H100 80GB 或等效 GPU  

---

## 快速开始

1. 克隆仓库并进入目录：
   ```bash
   git clone https://github.com/Little-Fridge/Multi-Method-Jailbreak-Attacks.git
   cd Multi-Method-Jailbreak-Attacks
   ```
   
2. 创建环境
  ```bash
  conda create -n exp python=3.10
  conda activate exp
  ```
3. 下载项目依赖
 ```bash
 pip install -r requirements.txt
```
4.在huggingface上下载相关模型权重，并拥有GPT系列相关的api密钥

5.测试demo——提示工程
```bash
python prompt_knowledge.py \
  -i data/prompts.jsonl \
  -o knowledge_results.jsonl \
  -m your model path
```
6.测试demo——PAIR
7.测试demo——GCG

