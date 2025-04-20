
# 赛博禅师 (Cyber Zen Master)

一个基于多种大型语言模型的哲理问答思维链多步推理工作流。

## 架構概述

本项目现已升级为支援多种LLM提供商的抽象架构，包括：

- DeepSeek（原生支持）
- OpenAI（GPT-3.5/GPT-4）
- X.AI (Grok)
- 外部API整合

## 使用方法

### 安装依赖

使用pip安装所需依赖包：

```bash
pip install -r requirements.txt
```

或手动安装各个依赖：

```bash
pip install openai>=1.6.0 requests>=2.28.0 python-dotenv>=1.0.0
```

### 配置API密钥

可通过以下方式之一配置API密钥：

1. **环境变量**（推荐）：

```bash
# DeepSeek
export DEEPSEEK_API_KEY=your_deepseek_api_key

# OpenAI
export OPENAI_API_KEY=your_openai_api_key

# X.AI
export XAI_API_KEY=your_xai_api_key
```

2. **.env文件**：

在项目根目录或用户主目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key
XAI_API_KEY=your_xai_api_key
```

3. **配置文件**：

在用户主目录创建 `.llm_config.json` 文件：

```json
{
  "deepseek": {
    "api_key": "your_deepseek_api_key",
    "base_url": "https://api.deepseek.com/v1"
  },
  "openai": {
    "api_key": "your_openai_api_key",
    "base_url": "https://api.openai.com/v1"
  },
  "xai": {
    "api_key": "your_xai_api_key",
    "base_url": "https://api.x.ai/v1"
  },
  "default_provider": "deepseek"
}
```

优先级：环境变量 > .env文件 > 配置文件

### 运行主程序

在主程序`main.py`的入口的`topics`变量中，以列表形式填入你想问的一个或多个问题，并使用如下命令运行：

```bash
python main.py
```

### 示例程序

本项目提供了多个示例程序展示如何使用不同的LLM提供商：

- 通用示例：`python examples/llm_example.py`
- X.AI Grok示例：`python examples/xai_example.py`

## 输出位置

最终结果和中间各步骤推理过程，会存入`output`文件夹中。每个问题的最终输出会保存在以问题命名的文件夹下的`stage3/self_expression.txt`文件中。

## LLM提供商对比

| 提供商 | 模型 | 特点 | 适用场景 |
|-------|------|------|---------|
| DeepSeek | deepseek-chat | 中文理解能力强 | 中文哲理问答，文学创作 |
| OpenAI | gpt-3.5-turbo, gpt-4 | 通用能力强 | 广泛的问答和创作需求 |
| X.AI | grok-3-beta | 最新信息，代码能力强 | 技术问题，代码生成 |

## 自定义提供商

本架构支持通过`ExternalAPIProvider`集成任何提供REST API的服务。详细使用方法请参考`external_api_provider_example.md`文件。

## 致谢

提示词来自[李继刚老师](https://web.okjike.com/u/752D3103-1107-43A0-BA49-20EC29D09E36)的[汉语新解](https://web.okjike.com/originalPost/66e263c2610bbfc39f1a4031)系列Lisp提示词。项目地址：[链接](https://github.com/lijigang/write-prompt/tree/main)。

架构设计文档请参考`llm_architecture_proposal.md`。