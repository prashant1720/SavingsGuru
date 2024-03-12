from flask import Flask, render_template, request, jsonify
import plotly.graph_objs as go
from openai import OpenAI
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd
from flask import send_file

app = Flask(__name__)
nltk.download('vader_lexicon')
#  portfolio analysis,
# market trend monitoring, and 
#personalized news recommendations tailored to individual investors'portfolio

sectors = ['Financial Services', 'Technology and Innovation', 'Healthcare', 'Manufacturing', 'Energy', 'Agriculture', 'Retail and Consumer Goods', 'Education', 'Entertainment and Media', 'Transportation']
companies_data = [
    {"name": "Qualcomm Inc", "key_product": "Semiconductors, Wireless Technology"},
    {"name": "Tyson Foods", "key_product": "Meat and Poultry Products"},
    {"name": "AstraZeneca PLC", "key_product": "Pharmaceutical Drugs"},
    {"name": "Google", "key_product": "Search Engine"}
]
sid = SentimentIntensityAnalyzer()



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze',)
def analyze_portfolio():
  return render_template('Portfolio.html')

@app.route('/openai')
def openai():
    key_products = [(company['name'], company['key_product']) for company in companies_data]
    return render_template('OpenAi.html', key_products=key_products)

openai.api_key = 'mykey'

@app.route('/chatgpt', methods=['POST','GET'])
def chatgpt():
   if request.method == 'POST' or request.method == 'GET':
        company = request.args.get('company')
        product = request.args.get('product')

        if company:
            text = f"Generate the latest news articles related to {company}"
        elif product:
            text = f"Generate the latest news articles related to {product}"
        else:
            text = "Generate the latest news articles related to semiconductors"
        # text = "Generate the latest news articles related to semicondutor"
        openai = OpenAI(
            api_key="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"
        )
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
        
        output_raw = completion.choices[0].message.content
        output = output_raw.split("\n")
        score=analyze_nltk(output)
        
        print(output)
        print(score)
        return render_template('OpenAi.html',output=output,score=score)
   else:
        return render_template('OpenAi.html',output="",score="")


def analyze_nltk(output):
    sid = SentimentIntensityAnalyzer()
    
    # Join the list of strings into a single string
    text = ' '.join(output)
    
    # Analyze sentiment using VADER
    sentiment_scores = sid.polarity_scores(text)
    
    # Return the sentiment analysis result
    return sentiment_scores

@app.route('/market')
def market():
    # Data for the pie chart
    data = {
        'labels': ['Financial Services', 'Technology and Innovation', 'Healthcare', 'Manufacturing', 'Energy', 'Agriculture', 'Retail and Consumer Goods','Education','Entertainment and Media','Transportation'],
        'values': [8, 7, 18, 12, 7, 2, 11, 8, 3, 9] , # Example data (percentage distribution)
        'size_layout': 'responsive',  # Size layout option
         
    }

    if request.method == 'POST' or request.method == 'GET':
        sectors = [
            'Financial Services',
            'Technology and Innovation',
            'Healthcare',
            'Manufacturing',
            'Energy',
            'Agriculture',
            'Retail and Consumer Goods',
            'Education',
            'Entertainment and Media',
            'Transportation'
        ]


        openai = OpenAI(
            api_key="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"
        )
        growth_data = {}
        for sector in sectors:
            completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": f"Generate the latest growth information for {sector}.",
                },
            ],
        )
            output_raw = completion.choices[0].message.content
            growth_data[sector] = output_raw
            print(growth_data)

        
    return render_template('result.html', data=data,sectors=sectors, growth_data=growth_data)

# @app.route('/detgraph')
# def detgraph():
#     # Load data from CSV
#     df = pd.read_csv('static/det.csv')
#     df=df.head(10)
#     print(df)
    
#     # Convert 'Record Date' column to datetime format
#     df['Record Date'] = pd.to_datetime(df['Record Date'])
    
#     # Create line chart
#     plt.figure(figsize=(10, 6))
#     for column in df.columns[1:]:
#         plt.plot(df['Record Date'], df[column], label=column)
#     plt.xlabel('Record Date')
#     plt.ylabel('Debt')
#     plt.title('Debt Over Time')
#     plt.legend()
#     plt.grid(True)
    
#     # Save the plot to a temporary file
#     plot_path = 'static/line_chart.png'
#     plt.savefig(plot_path)
    
#     # Close the plot to free memory
#     plt.close()
    
#     # Pass the plot path to the template
#     return render_template('result.html', plot_path=plot_path)

@app.route('/show_chart')
def show_chart():
    return send_file('line_chart.html')



@app.route('/growth', methods=['POST','GET'])
def get_sector_growth():

    if request.method == 'POST' or request.method == 'GET':
        sectors = [
            'Financial Services',
            'Technology and Innovation',
            'Healthcare',
            'Manufacturing',
            'Energy',
            'Agriculture',
            'Retail and Consumer Goods',
            'Education',
            'Entertainment and Media',
            'Transportation'
        ]


        openai = OpenAI(
            api_key="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"
        )
        growth_data = {}
        for sector in sectors:
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": f"Generate the latest growth information for {sector}",
                    },
                ],
            )
            output_raw = completion.choices[0].message.content
            growth_data[sector] = output_raw
            print(growth_data)
        
        return render_template('result.html', sectors=sectors, growth_data=growth_data)
    else:
        return render_template('result.html', sectors=[], growth_data={})

    

if __name__ == '__main__':
    app.run(debug=True)
