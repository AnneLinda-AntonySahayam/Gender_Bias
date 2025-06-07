# Gender_Bias
# Gender Bias Evaluation and Debiasing in Large Language Models

This repository contains the datasets, evaluation scripts, and results used for the thesis **"Gender Bias Evaluation and Mitigation in Multilingual Large Language Models"**. The goal of the project is to assess and reduce gender bias in occupational predictions made by large language models (LLMs), specifically **LLaMA 2 (7B)** and **LLaMA 3.2 (1B)** across **English** and **Italian** prompts.

---

## 🔍 Project Overview

- 📌 **Bias Types Evaluated**:  
  - **Explicit Bias** – Direct prompts (e.g., _"What is the gender of a nurse?"_)
  - **Implicit Bias** – Conversational prompts (e.g., _"Tell me about your friend who is a doctor."_)

- 🧪 **Debiasing Strategy**:  
  - Zero-shot prompt-based debiasing using instructional instructions
  - Multiple abstraction levels: `None`, `Low`, `Medium`, `High`

- 🌍 **Languages Covered**:  
  - English (US-based occupations with gender ratios from BLS)
  - Italian (translated and adapted version created under supervision)

---

## 📁 Directory Structure

