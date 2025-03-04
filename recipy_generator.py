import streamlit as st
import requests
import mysql.connector
from pyzbar.pyzbar import decode
from PIL import Image
import io
import datetime

# Database Connection
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Pakistan@123',
    database='smart_fridge',
    autocommit=True
)
cursor = conn.cursor()

# Function to get food details from OpenFoodFacts API
def get_food_details(barcode_or_name):
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode_or_name}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "product" in data:
            product = data["product"]
            return {
                "item_name": product.get("product_name", "Unknown"),
                "calories": product.get("nutriments", {}).get("energy-kcal_100g", "N/A"),
                "ingredients": product.get("ingredients_text", "N/A")
            }
    return None

# Function to insert food into the database
def insert_food(item_name, expiry_date):
    quantity = 1  # Default quantity
    cursor.execute(
        "INSERT INTO food_inventory (item_name, quantity, expiry_date) VALUES (%s, %s, %s)",
        (item_name, quantity, expiry_date)
    )

# Function to get expiring food items
def get_expiring_food():
    today = datetime.date.today()
    cursor.execute("SELECT item_name, expiry_date FROM food_inventory WHERE expiry_date <= %s", (today,))
    return [{"item_name": item[0], "expiry_date": str(item[1])} for item in cursor.fetchall()]

# Function to suggest recipes
def suggest_recipes():
    cursor.execute("SELECT item_name FROM food_inventory")
    available_food = [item[0] for item in cursor.fetchall()]

    if not available_food:
        return []

    api_url = "https://api.spoonacular.com/recipes/findByIngredients"
    api_key = "384d82ba1c5f4de6ab9ac01bbdeb31cd"  #  Spoonacular API key
    params = {
        "ingredients": ",".join(available_food),
        "number": 5,
        "apiKey": api_key
    }

    response = requests.get(api_url, params=params)
    return response.json() if response.status_code == 200 else []

# Streamlit UI
st.title("🥦 Smart Refrigerator AI")

# 📸 Upload Image for Barcode Detection
st.subheader("Upload Food Image (Barcode Detection)")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    barcodes = decode(image)
    if barcodes:
        barcode = barcodes[0].data.decode('utf-8')
        st.write(f"**Detected Barcode:** `{barcode}`")

        # Get food details using API
        food_data = get_food_details(barcode)
        if food_data:
            st.success("Food Item Found!")
            st.write(f"**Name:** {food_data['item_name']}")
            st.write(f"**Calories per 100g:** {food_data['calories']}")
            st.write(f"**Ingredients:** {food_data['ingredients']}")

            # Ask for expiry date
            expiry_date = st.date_input("Select Expiry Date", min_value=datetime.date.today())
            if st.button("Add to Inventory"):
                insert_food(food_data["item_name"], expiry_date)
                st.success(f"{food_data['item_name']} added to inventory!")

        else:
            st.error("Food not found in database!")
    else:
        st.error("No barcode found in image.")

# 📌 Manually Add Food Item
st.subheader("Manually Enter Food Details")
food_name = st.text_input("Food Name")
expiry_date_manual = st.date_input("Select Expiry Date", min_value=datetime.date.today())

if st.button("Add Manually"):
    if food_name:
        insert_food(food_name, expiry_date_manual)
        st.success(f"{food_name} added to inventory!")
    else:
        st.error("Please enter a valid food name.")

# ⏳ Check Expiring Food Items
st.subheader("⚠️ Expiring Food Items")
expiring_items = get_expiring_food()
if expiring_items:
    st.table(expiring_items)
else:
    st.info("No expiring food items found.")

# 🍽️ Recipe Suggestions
st.subheader("🍳 Suggested Recipes")
if st.button("Get Recipes"):
    recipes = suggest_recipes()
    if recipes:
        for recipe in recipes:
            st.write(f"**{recipe['title']}**")
            st.image(recipe["image"], width=200)
            st.write(f"🔗 [View Recipe](https://spoonacular.com/recipes/{recipe['title'].replace(' ', '-')}-{recipe['id']})")
            st.write("---")
    else:
        st.error("No recipes found!")
