## 数据集

这些数据集体量大，标签通常只是“Laughter”，你需要通过**能量（Energy）\**或\**持续时间**进行二次筛选才能得到“开怀大笑”。

| 数据集名称                             | 描述与特点                                                   | 处理建议                                                     |                                                     |
| :------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | --------------------------------------------------- |
| **AudioSet** (Google)                  | 世界上最大的音频事件数据集，包含大量 `Laughter`, `Giggle`, `Snicker` 等标签。 | 使用 `Laughter` 类，并结合音量/能量阈值筛选。还可以寻找子集 `Baby laughter` (通常很真诚) 或 `Belly laugh` (捧腹大笑)。 | https://huggingface.co/datasets/agkphysics/AudioSet |
| **Switchboard / ICSI Meeting Corpus**  | 经典的电话对话和会议录音数据集。包含大量社交笑声（Social Laughter）。 | 大部分是礼貌性笑声，但也有若干“爆笑”片段。通常会在转录中标注 `[Laughter]`。 |                                                     |
| **Audioset-Laughter (Gillick et al.)** | 从AudioSet中提取并在Interspeech 2021论文中清洗过的纯净笑声子集。 | 很好的预训练基底，可以在此基础上微调你的“开怀大笑”分类器。   |                                                     |

## TTS

|               |      |                                                        |
| ------------- | ---- | ------------------------------------------------------ |
| F5-TTS        | 上交 | https://github.com/SWivid/F5-TTS                       |
| Seed-TTS      | 字节 | https://bytedancespeech.github.io/seedtts_tech_report/ |
| **CosyVoice** | 阿里 | https://github.com/FunAudioLLM/CosyVoice               |
| ChatTTS       |      | https://github.com/2noise/ChatTTS                      |

## ASR

|            |      |                                                              |
| ---------- | ---- | ------------------------------------------------------------ |
| SenseVoice | 阿里 | https://github.com/FunAudioLLM/SenseVoice https://arxiv.org/pdf/2407.04051 |
|            |      |                                                              |
|            |      |                                                              |

https://huggingface.co/spaces/hf-audio/open_asr_leaderboard

## SenseVoice

### 总览图（训练/推理流程）

```mermaid
flowchart TB
  %% Inputs
  subgraph Inputs
    A_speech["speech: fbank/feat (B, T, 80)"]
    A_text["text: tokens (B, L)"]
  end

  %% Frontend
  subgraph Frontend
    F1["(train only) SpecAugment?"]
    F2["Normalize? (e.g., CMVN)"]
  end

  A_speech --> F1 --> F2

  %% Prefix Queries (4帧)
  subgraph PrefixQueries["4帧前缀（条件注入）"]
    PQ1["语言: text[:,0] -> lid_int_dict -> Embedding -> 1帧"]
    PQ2["事件/情绪: 固定token [1,2] -> Embedding -> 2帧"]
    PQ3["文本规范: text[:,3] 或 推理: text_norm/use_itn -> Embedding -> 1帧"]

    note1["最终顺序: [语言, 事件1, 事件2, 文本规范] 在声学帧前"]
  end

  F2 --> C_cat["Concat: 4帧前缀 + speech; lengths += 4"]
  PQ1 --> C_cat
  PQ2 --> C_cat
  PQ3 --> C_cat

  %% Encoder
  subgraph Encoder["SenseVoiceEncoderSmall (d_model=output_size)"]
    E0["Scale by sqrt(d_model) + Sinusoidal PosEnc"]
    E1["encoders0: 1层 (in: input_size -> out: d_model)"]
    E2["encoders: (num_blocks-1) 层 (d_model)"]
    E3["after_norm (LayerNorm)"]
    E4["tp_encoders: tp_blocks 层 (可选)"]
    E5["tp_norm (LayerNorm)"]
  end

  C_cat --> E0 --> E1 --> E2 --> E3 --> E4 --> E5

  %% Heads & Losses
  subgraph CTC_Head["CTC 头"]
    H1["Linear (ctc_lo) + (log)softmax"]
  end

  E5 --> H1

  %% Training losses
  subgraph Training["训练目标"]
    L_rich["Rich CE: 前4帧 vs text[:,:4]（用 ctc_lo 输出）"]
    L_ctc["CTC: 从第5帧起 vs text[:,4:]"]
  end

  H1 --> L_rich
  H1 --> L_ctc

  %% Inference
  subgraph Inference["推理"]
    I1["Greedy: argmax → unique_consecutive → tokenizer.decode"]
    I2["(可选) ban_emo_unk 屏蔽未知情绪token"]
    I3["(可选) 强制对齐生成时间戳 ctc_forced_align（从第5帧/第5个token起）"]
  end

  H1 --> I2 --> I1
  H1 --> I3
```

### 编码器与注意力细节（SANM 层）

```mermaid
flowchart LR
  subgraph EncoderLayerSANM
    X_in["x (B, T, d_in)"] --> N1["LayerNorm (pre-norm)"]
    N1 --> ATTN["MultiHeadedAttentionSANM"]
    ATTN --> D1["Dropout"]
    D1 --> RES1["Residual (+)"]
    RES1 --> N2["LayerNorm (pre-norm)"]
    N2 --> FFN["Positionwise FFN (Linear→Act→Dropout→Linear)"]
    FFN --> D2["Dropout"]
    D2 --> RES2["Residual (+)"]
    RES2 --> X_out["x' (B, T, d_out)"]
  end

  subgraph MultiHeadedAttentionSANM
    QKV["Linear(in_feat → 3 * d_model) → split Q,K,V → 头拆分"] --> SCL["Q * K^T / sqrt(d_k)"]
    SCL --> MSK["Mask (padding/块掩码)"]
    MSK --> SM["Softmax → Dropout"]
    SM --> CTX["Context = Attn * V → Linear(d_model)"]
    VBR["V (B,T,d) → Depthwise Conv1d (FSMN记忆) + 残差 → Dropout"] --> MEM["FSMN Memory"]
    CTX --> ADD["Add: Context + FSMN Memory"]
    MEM --> ADD
  end
```

- 关键说明
  - 前缀4帧顺序固定为：语言、事件1、事件2、文本规范；CTC 只对第5帧起的内容做识别损失。
  - Rich CE 用 CTC 头的线性层输出对前4帧进行分类监督，稳定“条件注入”学习。
  - SANM 将多头注意力与深度可分离卷积记忆（FSMN）相加，兼顾局部与长程依赖，友好流式。

https://www.gradio.app/guides/real-time-speech-recognition