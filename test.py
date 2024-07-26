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
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for any HTTP error
        soup = BeautifulSoup(response.content, 'html.parser')
        # Modify this line to target the correct element containing the listings
        data = soup.find_all('li', class_='s-item')
        all_listings = []
        for listing in data:
            title_elem = listing.find('div', class_='s-item__title').find('span', role='heading')
            title = title_elem.text.strip() if title_elem else None
            price_elem = listing.find('span', class_='s-item__price')
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

    # Close the figure
    plt.close()

    return plot_data


if __name__ == '__main__':
    # Initialize Spark session
    spark = SparkSession.builder.appName("WebScrapingWithPySpark").getOrCreate()

    # Number of cycles
    cycles = 4

    # Keep track of all scraped data and bar chart data
    all_scraped_data = []
    all_bar_chart_data = []

    for cycle in range(1, cycles + 1):
        # Scrape data for one page in each cycle
        page = cycle
        category = 'food'
        url = (f'https://www.ebay.com/sch/i.html?_nkw={category}&_pgn={page}')
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

            # Display bar chart popup
            print("\nShowing bar graph\n")
            plt.figure()
            img = BytesIO(base64.b64decode(bar_chart_data))
            plt.imshow(plt.imread(img))
            plt.axis('off')
            # Add a delay to allow the plot to be displayed before the program terminates
            plt.pause(5)

        # Check if it's not the last cycle
        if cycle < cycles:
            plt.close()
            # Delay before starting the next cycle
            print("\nWaiting for 2 seconds before the next cycle...\n")
            time.sleep(2)
    
    plt.show()

    # Stop Spark session
    spark.stop()
