import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from CSV
anime_df = pd.read_csv('anime_details.csv')

# Select features (genres and studio) as input variables (X) and rating as target variable (y)
X = anime_df[['Genres', 'Studio']]
y = anime_df['My Rating']

# Perform one-hot encoding on categorical variables (Genres and Studio)
X_encoded = pd.get_dummies(X, drop_first=True)
X_encoded_columns = X_encoded.columns  # Store the columns for later use

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize and train the K-Nearest Neighbors model
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

# Example usage: Predict the rating for a new anime
def scrape_anime_data(anime_name):
    # Format the anime name for the URL
    formatted_name = anime_name.replace(' ', '%20')

    # Send a GET request to the search page of MyAnimeList
    search_url = f'https://myanimelist.net/anime.php?q={formatted_name}'
    response = requests.get(search_url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the search results container
    results_container = soup.find('div', class_='js-categories-seasonal')

    if results_container:
        # Find the first search result and extract the anime link
        anime_links = results_container.find_all('a', class_='hoverinfo_trigger')
        anime_match = None

        # Find the anime that matches the exact name
        for anime_link in anime_links:
            if anime_link.text.strip().lower() == anime_name.lower():
                anime_match = anime_link
                break

        if anime_match:
            anime_url = anime_match['href']
            # Send a GET request to the anime page using the extracted URL
            anime_response = requests.get(anime_url)
            anime_soup = BeautifulSoup(anime_response.content, 'html.parser')

            # Extract the desired data
            anime_title = anime_soup.find('h1').text.strip()
            anime_rating = anime_soup.find('div', class_='score-label').text

            # Find the genre tags and extract the genre names
            genre_tags = anime_soup.find_all('span', itemprop='genre')
            anime_genres = [tag.text.strip() for tag in genre_tags]

            # Find the anime studio
            studio_tag = anime_soup.find('span', string='Studios:')
            if studio_tag:
                anime_studio = studio_tag.find_next('a').text.strip()
            else:
                anime_studio = "Studio information not found."

            return anime_title, anime_rating, anime_genres, anime_studio

    return None, None, None, None

# Example usage: Predict the rating for a new anime given by the user
def predict_new_anime_rating():
    # Get anime name from user input
    anime_name = input("Enter the name of the anime: ")

    # Scrape anime data from MyAnimeList
    anime_title, anime_rating, anime_genres, anime_studio = scrape_anime_data(anime_name)

    if anime_title:
        # Create a DataFrame with the new anime data
        new_anime = pd.DataFrame({'Genres': [", ".join(anime_genres)], 'Studio': [anime_studio]})

        # Align the columns of the new anime DataFrame with the training data columns
        new_anime_aligned = new_anime.reindex(columns=X_encoded_columns, fill_value=0)

        if not new_anime_aligned.empty:
            # Predict the rating using the trained model
            rating_prediction = model.predict(new_anime_aligned)

            print("Title:", anime_title)
            print("Rating:", anime_rating)
            print("Predicted Rating:", rating_prediction[0])
            print("Genres:",anime_genres)
            print("Studio:",anime_studio)
        else:
            print("Unable to predict the rating. The input data is empty.")
    else:
        print("Anime not found or data could not be retrieved.")

predict_new_anime_rating()
