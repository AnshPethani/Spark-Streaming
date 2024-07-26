from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
import os

# Function to scrape data from a single URL
def scrape_url_page(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for any HTTP error
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all product meta information containers
        product_meta_info_containers = soup.find_all('div', class_='product-productMetaInfo')
        
        all_listings = []
        
        # Iterate over each product meta information container
        for container in product_meta_info_containers:
            # Extract title
            title_elem = container.find('h4', class_='product-product')
            title = title_elem.text.strip() if title_elem else None
            
            # Extract price
            price_elem = container.find('span', class_='product-discountedPrice')
            price = price_elem.text.strip() if price_elem else None

            listing_data = {
                'title': title,
                'price': price,
            }

            all_listings.append(listing_data)

        return all_listings
    except Exception as e:
        print(f"\nError scraping data from {url}: {str(e)}\n")
        return None


# UDF (User Defined Function) for parallelized scraping
scrape_udf = udf(scrape_url_page, StringType())

# Function to create bar chart analysis of prices
def create_price_bar_chart(all_data):
    # Combine data from all batches into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Convert price column to numeric
    combined_df['price'] = combined_df['price'].apply(lambda x: np.mean([float(val.replace('$', '')) for val in x.split(' to ')]))
    
    # Create price ranges
    price_ranges = [0, 10, 20, 30, 40, 50, 100, 150, 200, 250]
    price_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-100', '100-150', '150-200', '200-250']

    # Group prices into ranges
    combined_df['price_range'] = pd.cut(combined_df['price'], bins=price_ranges, labels=price_labels, right=False)

    # Count occurrences of each price range
    price_counts = combined_df['price_range'].value_counts().sort_index()

    # Plot bar chart
    plt.figure(figsize=(16, 12))
    price_counts.plot(kind='bar', color='blue')
    plt.xlabel('Price Range ($)', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.title('Price Analysis', fontsize=24)
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    # Save plot to bytes object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()

    plt.close()

    return plot_data

# Function to create plot of price vs. item
def create_price_item_plot(all_data):
    # Combine data from all batches into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Convert price column to numeric
    combined_df['price'] = combined_df['price'].apply(lambda x: np.mean([float(val.replace('$', '')) for val in x.split(' to ')]))
    
    # Plot price vs. item
    plt.figure(figsize=(16, 12))
    plt.plot(combined_df['price'], marker='o', linestyle='-')
    plt.xlabel('Item Index', fontsize=16)
    plt.ylabel('Price ($)', fontsize=16)
    plt.title('Price vs. Item', fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    # Save plot to bytes object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()

    plt.close()

    return plot_data


if __name__ == '__main__':
    # Initialize Spark session
    spark = SparkSession.builder.appName("WebScrapingWithPySpark").getOrCreate()

    # Number of cycles
    cycles = 2

    # Keep track of all scraped data and bar chart data
    all_scraped_data = []
    all_bar_chart_data = []
    all_price_item_plot_data = []

    for cycle in range(1, cycles + 1):
        # Scrape data for one page in each cycle
        page = cycle
        category = 'shirts'
        url = {f'https://www.myntra.com/{category}?rawQuery={category}&p={page}'}
        scraped_data = scrape_url_page(url)
        page = page + 1

        if scraped_data:
            # Convert scraped data into DataFrame
            df = pd.DataFrame(scraped_data)

            # Display message
            print("\nData scraped\n")

            # Display the DataFrame
            print(f"\nData Scrapped: \n{df}\n")

            # Display message
            print("\nData received by Spark\n")

            # Append DataFrame to list
            all_scraped_data.append(df)

            # Create and append bar chart data
            bar_chart_data = create_price_bar_chart(all_scraped_data)
            all_bar_chart_data.append(bar_chart_data)

            # Create and append price-item plot data
            price_item_plot_data = create_price_item_plot(all_scraped_data)
            all_price_item_plot_data.append(price_item_plot_data)

            # Display bar chart and price-item plot side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Display bar chart
            img_bar = BytesIO(base64.b64decode(bar_chart_data))
            ax1.imshow(plt.imread(img_bar))
            ax1.axis('off')

            # Display price-item plot
            img_price_item = BytesIO(base64.b64decode(price_item_plot_data))
            ax2.imshow(plt.imread(img_price_item))
            ax2.axis('off')

            # Add a delay to allow the plot to be displayed before the program terminates
            plt.pause(2)

        # Check if it's not the last cycle
        if cycle < cycles:
            plt.close()
            # Delay before starting the next cycle
            print("\nWaiting for 5 seconds before the next cycle...\n")
            time.sleep(5)
    
    plt.show()

    # Stop Spark session
    spark.stop()
