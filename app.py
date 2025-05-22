# app.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Function to scrape job data
def scrape_karkidi_jobs(keyword="data science", pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        job_blocks = soup.find_all("div", class_="ads-details")
        for job in job_blocks:
            try:
                title = job.find("h4").get_text(strip=True)
                company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
                location = job.find("p").get_text(strip=True)
                experience = job.find("p", class_="emp-exp").get_text(strip=True)
                jobs_list.append({
                    "Title": title,
                    "Company": company,
                    "Location": location,
                    "Experience": experience
                })
            except Exception as e:
                pass

        time.sleep(1)  # be kind to servers

    return pd.DataFrame(jobs_list)

# Streamlit UI
st.title("Karkidi Job Scraper")
keyword = st.text_input("Enter job keyword", "data science")
pages = st.slider("Select number of pages to scrape", 1, 5, 1)

if st.button("Scrape Jobs"):
    df = scrape_karkidi_jobs(keyword, pages)
    st.success(f"Scraped {len(df)} jobs.")
    st.dataframe(df)

    # Optional: Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="jobs.csv", mime="text/csv")
