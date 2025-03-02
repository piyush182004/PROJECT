import ollama

def generate_investment_analysis(class_name, user_data):
    """Use LLM to analyze business investment opportunities and return structured data."""
    
    prompt = f"""
    Based on an image classified as {class_name}, the following land details were provided:

    {user_data}

    Generate a business investment analysis including:
    1. Best business opportunities near the land.
    2. Expected investment range in INR (as a number only).
    3. ROI estimation over 5 years (percentage only).
    4. Feasibility of different businesses in this area.
    5. Proper price in INR for the land an investor should invest (as a number only).
    6. Expected economic growth in the area over the next 5 years (percentage only).
    """

    response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    text_response = response["message"]["content"]
    
    # Extract numbers for investment, ROI, and growth using simple parsing (assuming LLM returns clear numeric values)
    investment_range = extract_number(text_response, "investment range")
    roi_estimate = extract_number(text_response, "ROI estimation")
    land_price = extract_number(text_response, "price for the land")
    growth_estimate = extract_number(text_response, "economic growth")
    
    return {
        "analysis_text": text_response,
        "investment_range": investment_range,
        "roi_estimate": roi_estimate,
        "land_price": land_price,
        "growth_estimate": growth_estimate
    }

def extract_number(text, keyword):
    """Extract the first numeric value found after a keyword."""
    import re
    match = re.search(rf"{keyword}[^\d]*(\d+\.?\d*)", text, re.IGNORECASE)
    return float(match.group(1)) if match else None
