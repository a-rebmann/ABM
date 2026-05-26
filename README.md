# Agent Behavior Mining: Implementation Repository

This repository contains the implementation of the Agent Behavior Mining (ABM) scenario described in the paper **"Agent Behavior Mining: Generative AI Agent Governance in Business Processes"**.

## Overview

This implementation demonstrates how to render generative AI agent behavior observable and accountable through process mining techniques. The repository provides a complete multi-agent Order-to-Cash (O2C) system where specialized agents work together to manage a coffee shop, with full traceability of their decision-making processes.

## Core Concepts

For theoretical background on Agent Behavior Mining, the invisible autonomy risk, and the event data model, please refer to the research paper. This repository focuses on the practical implementation.

**The Multi-Agent Coffee Shop System includes:**
- **Order Agent** - Takes and processes customer orders
- **Inventory Agent** - Manages stock levels and availability
- **Barista Agent** - Handles order preparation and quality
- **Customer Service Agent** - Manages customer satisfaction and issue resolution

## Features

- **Multi-Agent System**: Complete implementation using LangGraph and LangChain
- **Automatic Trace Collection**: Captures all agent activities 
- **Event Log Generation**: Converts agent traces to standardized process logs
- **XES and CSV Export**: Event logs compatible with any process mining tool
- **Interactive Interface**: Jupyter notebook-based user interface
- **Three Exercises**: Guided tutorials covering standard operations, exception handling, and system extension
- **Process Mining Ready**: Usable for analysis with ProM, pm4py, or commercial tools


## Requirements

### Prerequisites
- [Python](https://www.python.org/downloads/) >= 3.13
- (Recommended) [poetry](https://python-poetry.org/) for managing packages and virtual environment - [Installation guide](https://python-poetry.org/docs/#installing-with-pipx)
  - Alternative: Use pip with the provided `requirements.txt`
- (Recommended) [poetry-jupyter-plugin](https://pypi.org/project/poetry-jupyter-plugin/) to install the virtual environment as a Jupyter kernel:
  ```bash
  poetry self add poetry-jupyter-plugin
  ```
  - Alternative: Set up the Jupyter kernel manually
- API key for an [LLM provider supported by LangChain](https://python.langchain.com/docs/integrations/chat/#featured-providers) (e.g., OpenAI, Anthropic, etc.)

### Installation

1. **Install the project dependencies:**
   ```bash
   poetry install
   ```

2. **Install Jupyter kernel via poetry:**
   ```bash
   poetry jupyter install
   ```

3. **Activate the virtual environment:**
   ```bash
   poetry shell
   ```
   Alternative: Prefix all subsequent commands with `poetry run`

4. **Install LangChain integration for your LLM provider:**
   
   Identify the package name from the [documentation](https://github.com/langchain-ai/langgraph/blob/a10a66cbd151c92f89d6476fb70e5e405ce50b98/docs/docs/snippets/chat_model_tabs.md), then install it:
   ```bash
   pip install "langchain[PROVIDER]<1.0.0"
   ```
   
   Examples:
   - OpenAI: `pip install "langchain[openai]<1.0.0"`
   - Anthropic: `pip install "langchain[anthropic]<1.0.0"`

5. **Configure LLM access:**
   
   Open `src/coffee_shop.py` and configure your LLM credentials starting at line 20, following the [LangChain documentation](https://github.com/langchain-ai/langgraph/blob/a10a66cbd151c92f89d6476fb70e5e405ce50b98/docs/docs/snippets/chat_model_tabs.md).

6. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

7. **Open the first notebook:**
   
   Navigate to `1_Standard_agentic_coffee_shop.ipynb` and start exploring!

## Quick Start

### Running the Coffee Shop

1. Open `1_Standard_agentic_coffee_shop.ipynb`
2. Follow the step-by-step instructions in the notebook
3. Interact with the multi-agent system through the chat interface
4. Generate traces of agent behavior automatically

### Generating Event Logs

After interacting with the coffee shop:

```python
from src import TraceProcessor

processor = TraceProcessor()

# Export as CSV 
processor.process_all_traces()

# Export as XES 
processor.process_all_traces(export_as_xes=True)

# Export multiple formats
processor.process_all_traces(export_as_json=True, export_as_xes=True)
```

**Export Formats:**
- **CSV**: Standard format for spreadsheet and data analysis tools
- **JSON**: Structured format for programmatic processing
- **XES**: IEEE standard format for process mining tools (ProM, pm4py, etc.)

The XES format includes standard XES extensions (concept, time, org, identity) plus a custom AI extension for agent-specific attributes, with full compatibility with major process mining tools.

Generated files are saved in the `generated_event_log/` folder.

### Analyzing with Process Mining Tools

The generated files can be imported into:
- **Open Source**: [ProM](http://www.promtools.org/), [pm4py](https://pm4py.fit.fraunhofer.de/)
- **Commercial**: Any process mining platform supporting CSV import

## Repository Structure

```
ABM/
├── src/
│   ├── coffee_shop.py              # Main coffee shop application
│   ├── styles.py                    # UI styling
│   ├── agents/                      # Agent implementations
│   │   ├── order_agent.py          # Order processing
│   │   ├── inventory_agent.py      # Stock management
│   │   ├── barista_agent.py        # Order preparation
│   │   ├── customer_service_agent.py # Customer support
│   │   └── shared_components.py    # Shared utilities
│   └── trace_processing/            # Event log generation
│       ├── log_generator.py        # Trace to event conversion
│       └── trace_processor.py      # Batch processing
├── 1_Standard_agentic_coffee_shop.ipynb    # Exercise 1: First order
├── 2_Exceptions_agentic_coffee_shop.ipynb  # Exercise 2: Error handling
├── 3_Extending_agentic_coffee_shop.ipynb   # Exercise 3: System extension
├── assets/                          # Supplementary materials
│   ├── Case Study Agent Event Log.csv      # Event log from the case study
│   ├── sample_log_agentic_coffee_shop.csv  # Sample event log
│   ├── Survey_instrument.pdf               # Survey instrument used in the study
│   ├── Exercise 1.png                      # Exercise 1 illustration
│   ├── Exercise 2.png                      # Exercise 2 illustration
│   ├── Exercise 3.png                      # Exercise 3 illustration
│   └── dashboard/                          # Process mining analysis dashboard
│       ├── Case Study-Analysis Dashboard.json  # Dashboard definition
│       └── README.md                           # Dashboard usage notes
├── pyproject.toml                   # Poetry configuration
├── requirements.txt                 # Pip requirements
└── README.md                        # This file
```

## Exercises

The repository includes three self-contained Jupyter notebook exercises:

### Exercise 1: Standard Operations
**File**: `1_Standard_agentic_coffee_shop.ipynb`

The basics:
- Set up and initialize the coffee shop
- Place successful orders
- Generate your first event log
- Analyze agent behavior with process mining

### Exercise 2: Exception Handling
**File**: `2_Exceptions_agentic_coffee_shop.ipynb`

Explore edge cases:
- Handle out-of-stock scenarios
- Deal with customer complaints
- Analyze behavioral variants
- Compare exception handling patterns

### Exercise 3: System Extension
**File**: `3_Extending_agentic_coffee_shop.ipynb`

Experiment with agent customization:
- Modify agent instructions
- Add or remove agent tools
- Monitor behavioral changes
- Validate improvements with process mining

## Agent Architecture

### Order Agent
**Role**: Takes and processes customer orders

**Tools**:
- `process_order()` - Parse customer orders
- `calculate_total()` - Calculate pricing with discount capabilities
- Handoff tools to inventory and customer service agents

**Responsibilities**:
- Welcome customers and take orders
- Validate menu items and quantities
- Calculate totals and apply discounts
- Transfer to inventory for availability checks

### Inventory Agent
**Role**: Manages stock levels and availability

**Tools**:
- `check_inventory()` - Verify item availability for orders
- `update_stock()` - Decrease inventory after confirmed orders
- `get_alternatives()` - Find substitute items for out-of-stock products
- Handoff tools to barista and customer service agents

**Responsibilities**:
- Check item availability against current stock
- Update inventory after order confirmation
- Suggest alternatives for unavailable items
- Transfer to barista when items are available
- Escalate to customer service for stock issues

### Barista Agent
**Role**: Handles order preparation and quality

**Tools**:
- `prepare_order()` - Simulate order preparation with realistic error handling
- `remake_order_item()` - Handle preparation errors and remakes
- `estimate_prep_time()` - Provide accurate timing estimates
- Handoff tool to customer service for issues

**Responsibilities**:
- Prepare drinks and food items
- Handle preparation errors (20% failure rate simulation)
- Provide preparation time estimates
- Quality control and remake capabilities

### Customer Service Agent
**Role**: Manages customer satisfaction and issue resolution

**Tools**:
- `offer_refund()` - Process full refunds when necessary
- `offer_partial_refund()` - Process partial refunds when appropriate
- Handoff tools to all other agents

**Responsibilities**:
- Handle customer complaints with empathy
- Offer appropriate compensation (remakes, refunds, discounts)
- Suggest alternatives with customer service touch
- Coordinate with other agents for resolution

## Troubleshooting

### Common Issues

**Problem**: `ModuleNotFoundError: No module named 'langchain_openai'`
- **Solution**: Install the LangChain integration for your provider:
  ```bash
  pip install "langchain[openai]<1.0.0"
  ```

**Problem**: LLM authentication errors
- **Solution**: Verify your API key is correctly configured in `src/coffee_shop.py` (line 20+) and that your API key environment variable is set correctly.

**Problem**: No trace files found
- **Solution**: Make sure you've interacted with the coffee shop (step 3) before generating logs (step 4). Each message creates a trace.

**Problem**: Jupyter kernel not found
- **Solution**: Run `poetry jupyter install` to register the kernel, then restart Jupyter.

## License
This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSES/Apache-2.0.txt) file for details.