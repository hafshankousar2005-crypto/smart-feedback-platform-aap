# SMART FEEDBACK INTELLIGENCE PLATFORM
# ================================================================================
# SECTION 1:IMPORTS (All basic packages that come with Anaconda)
# ================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
import os
from collections import Counter
from textblob import TextBlob
import random

warnings.filterwarnings('ignore')

print("=" * 60)
print(" SMART FEEDBACK INTELLIGENCE PLATFORM")
print("=" * 60)

# ================================================================================
# SECTION 2: DATA GENERATION
# ================================================================================

print("\n Generating realistic feedback data...")

# Customer data
customers = [
    {"id": "CUST1001", "name": "John Smith", "segment": "Premium", "tenure": 24},
    {"id": "CUST1002", "name": "Sarah Johnson", "segment": "Regular", "tenure": 12},
    {"id": "CUST1003", "name": "Mike Chen", "segment": "Premium", "tenure": 36},
    {"id": "CUST1004", "name": "Emma Wilson", "segment": "New", "tenure": 2},
    {"id": "CUST1005", "name": "David Brown", "segment": "Regular", "tenure": 18},
    {"id": "CUST1006", "name": "Lisa Anderson", "segment": "Premium", "tenure": 48},
    {"id": "CUST1007", "name": "James Taylor", "segment": "New", "tenure": 1},
    {"id": "CUST1008", "name": "Maria Garcia", "segment": "Regular", "tenure": 8},
    {"id": "CUST1009", "name": "Robert Lee", "segment": "Premium", "tenure": 60},
    {"id": "CUST1010", "name": "Patricia White", "segment": "Regular", "tenure": 15}
]

# Products
products = [
    {"id": "PROD101", "name": "SmartPhone X12", "category": "Electronics", "price": 699},
    {"id": "PROD102", "name": "Laptop Pro", "category": "Electronics", "price": 1299},
    {"id": "PROD103", "name": "Wireless Earbuds", "category": "Electronics", "price": 149},
    {"id": "PROD104", "name": "Coffee Maker", "category": "Home", "price": 89},
    {"id": "PROD105", "name": "Running Shoes", "category": "Sports", "price": 129},
    {"id": "PROD106", "name": "Desk Chair", "category": "Furniture", "price": 199},
    {"id": "PROD107", "name": "LED Lamp", "category": "Home", "price": 45},
    {"id": "PROD108", "name": "Power Bank", "category": "Electronics", "price": 39}
]

# Locations
locations = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia"]

# Channels
channels = ["Mobile App", "Website", "In-Store", "Social Media", "Email"]

# Categories
categories = ["Product Quality", "Delivery", "Customer Service", "Website Experience", 
              "Price", "Returns", "Packaging", "Product Features"]

# Generate 200 feedback records
data = []
for i in range(200):
    customer = random.choice(customers)
    product = random.choice(products)
    location = random.choice(locations)
    channel = random.choice(channels)
    category = random.choice(categories)
    
    # Random date within last 3 months
    days_ago = random.randint(0, 90)
    hours = random.randint(0, 23)
    minutes = random.randint(0, 59)
    timestamp = datetime.now() - timedelta(days=days_ago, hours=hours, minutes=minutes)
    
    # Generate rating and sentiment
    rating = random.choices([5,4,3,2,1], weights=[40,25,15,12,8])[0]
    
    # Generate feedback text based on rating
    if rating >= 4:
        sentiment = "Positive"
        texts = [
            f"Great {product['name']}! Really love the quality.",
            f"Excellent service! Very happy with my purchase.",
            f"The {product['name']} exceeded my expectations.",
            f"Fast delivery and perfect packaging. Thank you!",
            f"Amazing product! Will definitely buy again.",
            f"Best {product['category']} product I've ever bought.",
            f"Customer service was very helpful and friendly.",
            f"Great value for money. Highly recommended!"
        ]
    elif rating == 3:
        sentiment = "Neutral"
        texts = [
            f"The {product['name']} is okay, nothing special.",
            f"Average product. Does the job.",
            f"Delivery was on time. Product works fine.",
            f"It's decent for the price.",
            f"Not bad, but could be better."
        ]
    else:
        sentiment = "Negative"
        texts = [
            f"Very disappointed with {product['name']}. Poor quality.",
            f"Terrible customer service. No one responded.",
            f"The product stopped working after a few days.",
            f"Delivery was very late. Package was damaged.",
            f"Not worth the money. Cheap quality.",
            f"Website kept crashing during checkout.",
            f"Received wrong item. Return process is complicated."
        ]
    
    feedback_text = random.choice(texts)
    
    # Calculate other metrics
    order_value = product['price'] * random.uniform(0.8, 1.2)
    returned = random.choices([True, False], weights=[15, 85])[0] if rating <= 2 else False
    response_time = random.randint(5, 120) if rating >= 4 else random.randint(30, 1440)
    resolved = random.choices([True, False], weights=[90, 10])[0] if rating >= 3 else random.choices([True, False], weights=[70, 30])[0]
    
    data.append({
        'feedback_id': f"FBK{str(i+1).zfill(4)}",
        'timestamp': timestamp,
        'customer_id': customer['id'],
        'customer_name': customer['name'],
        'customer_segment': customer['segment'],
        'customer_tenure': customer['tenure'],
        'product_name': product['name'],
        'product_category': product['category'],
        'feedback_text': feedback_text,
        'rating': rating,
        'sentiment': sentiment,
        'category': category,
        'channel': channel,
        'location': location,
        'order_value': round(order_value, 2),
        'returned': returned,
        'response_time_mins': response_time,
        'resolved': resolved
    })

df = pd.DataFrame(data)
print(f" Generated {len(df)} feedback records")

# ================================================================================
# SECTION 3: SENTIMENT ANALYSIS (Using TextBlob)
# ================================================================================

print("\n Running sentiment analysis...")

def analyze_sentiment_deep(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Word counts
    words = text.lower().split()
    positive_words = sum(1 for w in words if w in ['great', 'excellent', 'amazing', 'love', 'best', 'good', 'happy'])
    negative_words = sum(1 for w in words if w in ['bad', 'worst', 'terrible', 'poor', 'disappointed', 'late', 'damaged'])
    
    # Composite score (0-100)
    composite = (polarity + 1) * 50
    composite = max(0, min(100, composite))
    
    return {
        'polarity': polarity,
        'subjectivity': subjectivity,
        'composite_score': composite,
        'positive_words': positive_words,
        'negative_words': negative_words
    }

# Apply analysis
sentiment_results = df['feedback_text'].apply(analyze_sentiment_deep)
sentiment_df = pd.DataFrame(sentiment_results.tolist())
df = pd.concat([df, sentiment_df], axis=1)

print(" Sentiment analysis complete")

# ================================================================================
# SECTION 4: BUSINESS INTELLIGENCE & METRICS
# ================================================================================

print("\n Calculating business metrics...")

# KPIs
kpis = {
    'Total Feedback': len(df),
    'Average Rating': round(df['rating'].mean(), 2),
    'Positive Rate': f"{(df['sentiment'] == 'Positive').mean() * 100:.1f}%",
    'Negative Rate': f"{(df['sentiment'] == 'Negative').mean() * 100:.1f}%",
    'Return Rate': f"{df['returned'].mean() * 100:.1f}%",
    'Resolution Rate': f"{df['resolved'].mean() * 100:.1f}%",
    'Avg Response Time': f"{df['response_time_mins'].mean():.0f} mins",
    'Avg Order Value': f"${df['order_value'].mean():.2f}"
}

print("\n KEY METRICS:")
for key, value in kpis.items():
    print(f"  {key}: {value}")

# Category performance
print("\n CATEGORY PERFORMANCE:")
category_perf = df.groupby('category').agg({
    'rating': 'mean',
    'feedback_id': 'count'
}).round(2).sort_values('rating', ascending=False)
print(category_perf)

# Channel performance
print("\n CHANNEL PERFORMANCE:")
channel_perf = df.groupby('channel').agg({
    'rating': 'mean',
    'feedback_id': 'count'
}).round(2).sort_values('rating', ascending=False)
print(channel_perf)

# Customer segment analysis
print("\n CUSTOMER SEGMENT ANALYSIS:")
segment_perf = df.groupby('customer_segment').agg({
    'rating': 'mean',
    'customer_id': 'count',
    'order_value': 'mean'
}).round(2)
print(segment_perf)

# ================================================================================
# SECTION 5: INSIGHTS GENERATION
# ================================================================================

print("\n KEY INSIGHTS:")

insights = []

# Sentiment trend
sentiment_by_day = df.groupby(df['timestamp'].dt.date)['rating'].mean()
if len(sentiment_by_day) > 1:
    trend = sentiment_by_day.iloc[-1] - sentiment_by_day.iloc[0]
    if trend > 0.2:
        insights.append(" Customer satisfaction is improving!")
    elif trend < -0.2:
        insights.append(" Customer satisfaction is declining - take action!")

# Best and worst categories
best_category = category_perf.index[0]
worst_category = category_perf.index[-1]
insights.append(f" Best performing category: {best_category}")
insights.append(f" Needs improvement: {worst_category}")

# Channel insights
best_channel = channel_perf.index[0]
worst_channel = channel_perf.index[-1]
insights.append(f" Best channel: {best_channel}")
insights.append(f" Worst channel: {worst_channel}")

# Return rate insight
if df['returned'].mean() > 0.15:
    insights.append(" High return rate - investigate quality issues")

# Response time insight
avg_response = df['response_time_mins'].mean()
if avg_response > 120:
    insights.append(f" Response time too high ({avg_response:.0f} mins)")

for i, insight in enumerate(insights, 1):
    print(f"  {i}. {insight}")

# ================================================================================
# SECTION 6: RECOMMENDATIONS
# ================================================================================

print("\n RECOMMENDATIONS:")

recommendations = []

# Based on negative feedback
negative_count = (df['sentiment'] == 'Negative').sum()
if negative_count > 0:
    top_negative_cats = df[df['sentiment'] == 'Negative']['category'].value_counts().head(2)
    recommendations.append(f"Priority 1: Improve {top_negative_cats.index[0]}")
    if len(top_negative_cats) > 1:
        recommendations.append(f"Priority 2: Address {top_negative_cats.index[1]} issues")

# Based on returns
return_products = df[df['returned']]['product_name'].value_counts().head(2)
if len(return_products) > 0:
    recommendations.append(f"Review quality of: {', '.join(return_products.index)}")

# General recommendations
recommendations.append("Implement customer feedback loop")
recommendations.append("Monitor sentiment trends weekly")
recommendations.append("Train staff on top complaint areas")
recommendations.append("Consider loyalty program for Premium customers")

for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")

# ================================================================================
# SECTION 7: VISUALIZATIONS
# ================================================================================

print("\n Generating visualizations...")

# Create a dashboard of charts
fig = plt.figure(figsize=(20, 12))

# 1. Rating Distribution
ax1 = plt.subplot(2, 3, 1)
rating_counts = df['rating'].value_counts().sort_index()
bars1 = ax1.bar(rating_counts.index, rating_counts.values, color='skyblue')
ax1.set_title('Rating Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Rating')
ax1.set_ylabel('Count')
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', 
             ha='center', va='bottom', fontweight='bold')

# 2. Sentiment Distribution
ax2 = plt.subplot(2, 3, 2)
sentiment_counts = df['sentiment'].value_counts()
colors = ['green' if x=='Positive' else 'red' if x=='Negative' else 'gray' 
          for x in sentiment_counts.index]
bars2 = ax2.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
ax2.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Sentiment')
ax2.set_ylabel('Count')
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', 
             ha='center', va='bottom', fontweight='bold')

# 3. Category Performance
ax3 = plt.subplot(2, 3, 3)
category_avg = df.groupby('category')['rating'].mean().sort_values()
bars3 = ax3.barh(category_avg.index, category_avg.values, color='lightcoral')
ax3.set_title('Average Rating by Category', fontsize=14, fontweight='bold')
ax3.set_xlabel('Average Rating')
for bar in bars3:
    width = bar.get_width()
    ax3.text(width + 0.05, bar.get_y() + bar.get_height()/2., f'{width:.2f}', 
             ha='left', va='center', fontweight='bold')

# 4. Channel Analysis
ax4 = plt.subplot(2, 3, 4)
channel_counts = df['channel'].value_counts()
ax4.pie(channel_counts.values, labels=channel_counts.index, autopct='%1.1f%%'),
     colors=(['#ff9999','#66b3ff','#99ff99','#ffcc99','#ff99cc'])
ax4.set_title('Feedback by Channel', fontsize=14, fontweight='bold')

# 5. Customer Segment Analysis
ax5 = plt.subplot(2, 3, 5)
segment_avg = df.groupby('customer_segment')['rating'].mean()
bars5 = ax5.bar(segment_avg.index, segment_avg.values, color=['gold', 'silver', 'brown'])
ax5.set_title('Rating by Customer Segment', fontsize=14, fontweight='bold')
ax5.set_xlabel('Segment')
ax5.set_ylabel('Avg Rating')
ax5.set_ylim(0, 5)
for i, (idx, val) in enumerate(segment_avg.items()):
    ax5.text(i, val + 0.1, f'{val:.2f}', ha='center', fontweight='bold')

# 6. Time Trend
ax6 = plt.subplot(2, 3, 6)
daily_avg = df.groupby(df['timestamp'].dt.date)['rating'].mean()
ax6.plot(range(len(daily_avg)), daily_avg.values, marker='o', linestyle='-', color='purple', linewidth=2)
ax6.set_title('Rating Trend Over Time', fontsize=14, fontweight='bold')
ax6.set_xlabel('Days (Most Recent First)')
ax6.set_ylabel('Avg Rating')
ax6.grid(True, alpha=0.3)

plt.suptitle('SMART FEEDBACK ANALYZER - COMPLETE DASHBOARD', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ================================================================================
# SECTION 8: EXPORT RESULTS
# ================================================================================

print("\n Exporting results...")

# Save data to CSV
df.to_csv('feedback_analysis_complete.csv', index=False)
print(" Data saved to 'feedback_analysis_complete.csv'")

# Create summary report
report = {
    'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_records': len(df),
    'kpi_summary': kpis,
    'top_insights': insights[:5],
    'key_recommendations': recommendations[:5]
}

with open('analysis_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(" Report saved to 'analysis_report.json'")

# ================================================================================
# SECTION 9: SUMMARY
# ================================================================================

print("\n" + "=" * 60)
print(" ANALYSIS COMPLETE!")
print("=" * 60)
print("\n Files created:")
print("    feedback_analysis_complete.csv - Complete dataset")
print("    analysis_report.json - Summary report")
print("    Charts displayed above")
print("\n Key Findings:")
print(f"   Overall Rating: {kpis['Average Rating']}/5.0")
print(f"   Positive Feedback: {kpis['Positive Rate']}")
print(f"   Top Category: {category_perf.index[0]}")
print(f"   Best Channel: {channel_perf.index[0]}")
print("=" * 60)