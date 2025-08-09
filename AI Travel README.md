# CIS-9660-Summer-2025

# 🌍 AI Travel Itinerary Planner

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cis-9660-will-apps.streamlit.app)

A Streamlit web application that creates personalized **multi-day travel itineraries** using:

- Google Places API (Text Search) to fetch top attractions in a given city 🗺️  
- OpenRouter’s LLM (GPT‑3.5) completion API to generate a ready-to-use itinerary including **timings**, **meals**, and **local tips**

---

## 🚀 Project Overview

Planning a day-by-day travel route can be complex for travelers. This app automates the process:

1. **Fetch notable attractions** for a destination city using the Google Places Text Search API.  
2. **Use OpenRouter GPT** (e.g. `openai/gpt-3.5-turbo` etc.) to craft a **detailed itinerary** for the specified number of days—incorporating meals, schedule slots, and helpful tips and warnings.

The goal is to minimize planning time and provide travelers with a personalized travel itinerary in seconds—no manual Google-searching required.

---

## 🔧 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Will-Peters/CIS-9660-Summer-2025.git
cd CIS-9660-Summer-2025
