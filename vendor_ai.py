import csv
import json
import datetime
import time
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv()

# Initialize components
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
search = GoogleSerperAPIWrapper()

# Chain 1: Generate a search query from vendor name - using explicit constructor
query_template = "Generate a search query to find what services {vendor_name} provides."
query_prompt = PromptTemplate(
    template=query_template,
    input_variables=["vendor_name"]
)
query_chain = query_prompt | llm | (lambda x: x.content)

# Chain 2: Summarize search results into one sentence - using explicit constructor
summary_template = """
Analyze the search results about a vendor and infer the core services they provide. 
Focus on identifying patterns, key offerings, and the primary value proposition—not just summarizing the text. 
Avoid mentioning the vendor's name. Be concise and descriptive in one sentence.  

Search Results:  
{search_results}  

Description of Services:"""
summary_prompt = PromptTemplate(
    template=summary_template,
    input_variables=["vendor_name", "search_results"]
)
summary_chain = summary_prompt | llm | (lambda x: x.content)

# Chain 3: Classify vendor using description and name
departments = ["Engineering", "Facilities", "G&A", "Legal", "M&A", "Marketing",
              "SaaS", "Product", "Professional Services", "Sales", "Support", "Finance"]
classification_template = """
Objective: 
Analyze the following vendor service description and any other relevant context to determine the most appropriate departmental category. 
Focus on the core function of the service, not the vendor's name.  

Service Description: {service_description}  

Available Categories: {departments}  

Guidelines:
- Choose the category that best aligns with the primary purpose of the service.  
- If the service spans multiple departments, pick the one it most directly supports.  
- Return only the category name, without explanations or extra text.  

Answer:"""
classification_prompt = PromptTemplate(
    template=classification_template,
    input_variables=["vendor_name", "service_description"],
    partial_variables={"departments": ", ".join(departments)}
)
classification_chain = classification_prompt | llm | (lambda x: x.content)

analyze_pipeline = (
    RunnablePassthrough.assign(search_query=query_chain)
    | RunnablePassthrough.assign(search_results=lambda x: search.run(x["search_query"]))
    | RunnablePassthrough.assign(service_description=summary_chain)
    | RunnablePassthrough.assign(category=classification_chain) 
)

def save_to_csv(data: list[dict]) -> str:
    """Save processed vendor data to a timestamped CSV file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results_{timestamp}.csv"
    
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Vendor", "Amount", "Description", "Category", "Action"])
        writer.writeheader()
        writer.writerows(data)
    return f"Data saved to {output_file}"

def analyze_vendors(input_file: str) -> list[dict]:
    """Process vendors from input file and return analyzed data"""
    with open(input_file, "r") as f:
        vendors = list(csv.DictReader(f))
    
    processed_data = []
    for vendor in vendors:
        result = analyze_pipeline.invoke({"vendor_name": vendor["Vendor"]})
        processed_data.append({
            "Vendor": vendor["Vendor"],
            "Amount": float(vendor["Amount"].replace(',', '').replace('$', '')),
            "Description": result["service_description"],
            "Category": result["category"]
        })
        print(f"classifiying vendor: {vendor['Vendor']} - {result["category"]} - {result["service_description"]}")
    return processed_data

def recommend_actions(processed_data: list[dict]) -> list[dict]:
    """
    Uses LLM to analyze all vendors collectively and recommend actions 
    (optimize, consolidate, terminate) for each vendor, considering portfolio context.
    Adds an 'Action' field to each vendor record.
    """
    # First create a summary of all vendors for context    
    vendor_summary = "\n".join(
        f"{v['Vendor']} (${v['Amount']}, {v['Category']}): {v['Description']}"
        for v in processed_data
    )
    
    # Define the recommendation prompt template
    recommendation_template = """
    Objective:  
    Analyze the vendor portfolio holistically and recommend the best action for 
    {vendor_name} based on strategic alignment, cost efficiency, and redundancy.  

    Input Data:  
    - Full Vendor Portfolio: {vendor_summary}  
    - Target Vendor: {vendor_name}  
    - Spend: ${amount}  
    - Category: {category}  
    - Service Description: {description}  

    Evaluation Criteria:  
    1. Duplicates/Overlaps: Does another vendor provide the same or similar service?  
    2. Spend Efficiency: Is the cost justified relative to value?  
    3. Portfolio Fit: Does this vendor align with long-term needs?  

    Possible Actions:  
    - `optimize` (Renegotiate terms, improve efficiency)  
    - `consolidate: [Vendor Name]` (Merge with a specific vendor; must specify)  
    - `terminate` (Discontinue due to irelevant to business operations, redundancy or low value)  

    Rules:  
    - Return ONLY the action in the exact format:  
    - `optimize`  
    - `consolidate: [Vendor Name]`  
    - `terminate`  
    - Never add explanations or deviations.  

    Example Outputs:  
    - "consolidate: AWS"  
    - "terminate"  
    - "optimize"  

Recommendation for {vendor_name}: """
    
    recommendation_prompt = PromptTemplate.from_template(recommendation_template)
    recommendation_chain = recommendation_prompt | llm | (lambda x: x.content.strip().lower())
    
    # Get recommendations for each vendor with full context
    for vendor in processed_data:
        recommendation = recommendation_chain.invoke({
            "vendor_summary": vendor_summary,
            "vendor_name": vendor["Vendor"],
            "amount": vendor["Amount"],
            "category": vendor["Category"],
            "description": vendor["Description"]
        })
        vendor["Action"] = recommendation
        print(f"recommending action for vendor: {vendor['Vendor']} - {recommendation}")
        
    return processed_data

def identify_top_opportunities(processed_data_with_actions: list[dict]) -> dict:
    """
    Analyzes the vendor portfolio, selects and returns the top 3 cost streamlining opportunities
    with explanations. Uses LLM to evaluate and rank opportunities across the entire portfolio.
    """

    analysis_template = """
    Analyze the vendor dataset holistically and identify the TOP 3 cost-saving opportunities with the highest potential impact. 
    Evaluate each opportunity using these criteria in order of priority:

    1. Potential savings magnitude (prioritize highest $ impact)
    2. Service/category redundancy (overlap with other vendors)
    3. Strategic alignment (low-value or non-core services)

    Input Data:
    {vendor_actions}

    Required Output Format:
    - Strictly CSV format with EXACTLY these columns: "Vendor name","Recommended action","Explanation"
    - Each explanation must include:
    Specific $ impact potential (estimate if exact unavailable)
    Clear redundancy/overlap evidence (if applicable)
    Strategic rationale

    Rules:
    - Output ONLY valid CSV data - no headers, titles, or explanations
    - Never use markdown code blocks or quotes
    - Limit the output to only the TOP 3
    - If fewer than 3 opportunities exist, leave remaining rows empty

    Example Output:
    "Acme Corp","consolidate: XYZ Inc","$250K potential savings, duplicate CRM tools"
    "Beta LLC","terminate","$180K savings, non-core legal service"
    "Gamma Inc","optimize","$90K via contract renegotiation"
"""

    # Format vendor actions for the prompt
    vendor_actions = "\n".join(
        f"{v['Vendor']} (${v['Amount']}): {v['Action']} | {v['Description']}"
        for v in processed_data_with_actions
    )

    # Create and run the analysis chain
    analysis_prompt = PromptTemplate.from_template(analysis_template)
    analysis_chain = analysis_prompt | llm | (lambda x: x.content)
    
    print("identifying opportunities")
    opportunities = analysis_chain.invoke({"vendor_actions": vendor_actions})
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"opportunities_{timestamp}.txt"

    # Save to file
    with open(filename, 'w') as file:
        file.write(opportunities)

    return opportunities

process_vendors_tool = RunnablePassthrough.assign(
    processed_data=lambda x: analyze_vendors(x["input_file"])
)

add_actions_tool = RunnablePassthrough.assign(
    processed_data_with_actions=lambda x: recommend_actions(x["processed_data"])
)

identify_opportunities_tool = RunnablePassthrough.assign(
    top_opportunities=lambda x: identify_top_opportunities(x["processed_data_with_actions"])
)

save_results_tool = RunnablePassthrough(lambda x: save_to_csv(x["processed_data"]))

full_pipeline =  (
    RunnablePassthrough()
    | process_vendors_tool
    | add_actions_tool
    | identify_opportunities_tool
    | save_results_tool
    )

results = full_pipeline.invoke({"input_file": "vendors.csv"})
