# AutoEdgeML

AutoEdgeML is a lightweight, production-ready toolkit that converts trained **scikit-learn models and pipelines** into **C header format** for efficient deployment on edge devices, mobile applications and low-resource environments.
  
**Whether your model is a simple estimator or a multi-step pipeline, AutoEdgeML handles it automatically.**
---

## Features

- Convert **scikit-learn models** to **C header format** 
- Full support for **sklearn Pipelines** (scalers, preprocessors) after intelligent model-type detection (regression / binary / multiclass)
- Works with scikit-learn models like:
    * Ridge, LinearRegression, LogisticRegression, ElasticNet, Lasso
    * Along with Scalers like Standard Scaler and MinMax Scaler
    * **Any model inside a Pipeline**
---

## Installation

```bash
git clone https://github.com/shubhamsingh47/AutoEdge-ML-to-C.git

cd AutoEdgeML
```


## Generated Filename Format

Clean automatic C header filename generation  

```bash
{File_base_name}__{model_type}__{scaler_name}__{timestamp}.h
```

## Architecture
```bash

                           ┌───────────────────────────┐
                           │        User Input         │
                           │      (eg. model.pkl)      │
                           └──────────────┬────────────┘
                                          │
                                          ▼
                           ┌───────────────────────────┐
                           │       Model Loader        │
                           │  - load pickle/model      │
                           │  - unwrap pipelines       │
                           └──────────────┬─────────────┘
                                          │
                                          ▼
                     ┌──────────────────────────────────────────┐
                     │      Model Introspection Engine          │
                     │  - detect estimator type                 │
                     │  - detect task (reg./binary/multiclass)  │
                     │  - extract scaler (if any)               │
                     └────────────────────┬─────────────────────┘
                                          │
                                          ▼
                         ┌───────────────────────────┐
                         │       Converter Layer      │
                         │  - Linear Converter        │
                         │  - Logistic Converter      │
                         │  - Scaler Converter        │
                         │  - Pipeline Converter      │
                         └──────────────┬─────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │      C header Export Engine │
                         │  - builds clean file name   │
                         │  - exports .h header file   │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                           ┌────────────────────────────────┐
                           │         Output '.h' file       │
                           │{base}_{type}_{scaler}_{time}.h │
                           └────────────────────────────────┘

```