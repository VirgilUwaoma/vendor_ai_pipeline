# Vendor Cost Optimization AI Pipeline  

An AI-powered pipeline for analyzing vendor costs, combining structured data processing with LLM-driven insights to generate actionable recommendations. Built with Python, LangChain, and OpenAI GPT-4.  

## Features  
- **Multi-stage analysis**: Vendor identification → service discovery → categorization → cost optimization.  
- **AI-augmented insights**: LLM-powered prompts with web search (GoogleSerperAPI) for real-time vendor data.  
- **Actionable outputs**: Recommends optimization, consolidation, or termination based on cost, redundancy, and strategic fit.  
- **Structured workflows**: Modular LangChain pipeline with enforced output formats for automation.  

## Prerequisites  
- Python 3.12  
- OpenAI API key (for GPT-4)  
- Google Serper API key (optional, for web-augmented search)  

## Setup  
1. **Clone the repository**:  
   - git clone https://github.com/your-username/vendor_ai_pipeline.git
   - cd vendor_ai_pipeline

2. **Install dependencies**:
   pip install -r requirements.txt
   
4. **Configure API keys**: Create a .env file in the root directory and add your keys: 
   OPENAI_API_KEY=your_openai_key
   SERPER_API_KEY=your_serper_key
   
### Prepare input data:
Place vendor data in a CSV file with columns: `Vendor`, `Amount`  in the same directory as the script.
In my case the file is names `vendors.csv`

### Run the pipeline:
Execute the python script: `python vendor_ai.py`
