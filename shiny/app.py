import shiny
from shiny import App, ui, reactive, render
import requests
import base64
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
import re
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from math import pi
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
import plotly.graph_objs as go
import nltk


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# Spotify BLACK & GREEN
spotify_green = "#179B44"
spotify_black = "#191414"

# API 
CLIENT_ID = "9f4661edc3c44ab98f58dad78e28e111"
CLIENT_SECRET = "a9a2fc4e13514726847e9a44fddbea17"


# LOAD Model and DICTIONARY
lda_model = LdaModel.load("lda_model.gensim")
dictionary = Dictionary.load('lda_model.gensim.id2word')

# LOAD podcast database
all_podcasts = pd.read_csv("shiny_database.gz")


TOPIC_NAMES = {
    0: 'NFL',
    1: 'Technology',
    2: 'Finance',
    3: 'News',
    4: 'Personal Development',
    5: 'Sports and Games',
    6: 'Crime',
    7: 'Education'
}



# Helper Functions: Generate Access Token
def get_access_token():
    """ 通过 Client Credentials Flow 获取 Spotify Access Token """
    url = "https://accounts.spotify.com/api/token"
    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()  

    headers = {
        "Authorization": f"Basic {b64_auth_str}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        access_token = response.json().get("access_token")
        return access_token
    else:
        raise Exception(f"Error fetching access token: {response.status_code} - {response.json()}")
    

# Helper Function: Validate Podcast
def validate_episode(episode):
    """check if english and if longer than 1 minute """
    languages = episode.get("languages", [])
    duration_ms = episode.get("duration_ms", 0)
    contains_english = any("en" in lang for lang in languages if isinstance(lang, str))
    return contains_english and duration_ms > 60000

# Helper Function: Check discription length
def check_description_length(description):
    """ Check if at least 20 words in description """
    if not description:
        return False
    word_count = len(description.split())
    return word_count >= 20

# Helper function: Preprocess Text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    custom_stopwords = set([
        "topic", "topics", "sponsor", "sponsors", "disclaimer", "follow", "support", "website", 
        "spotify", "watch", "lex", "exclusive", "free", "youtube", "check", "join", "shopify",
        "listen", "links", "subscribe", "new", "click", "timestamps", "timestamp", "transcript", 
        "rss", "expressvpn", "netsuite", "introduction", "coupon", "code", "offer", "discount", 
        "promo", "deal", "episode", "episodes", "patreon", "linkedin", "instagram", "twitter", 
        "facebook", "podcast", "podcasts", "visit", "ad", "adchoices", "na", "youd", "dont", "podcast", 
        "app", "acast", "hopeny", "expire", "nextstep", "select", "online", "apply", "youll", 
        "call", "text", "expires", "terms", "gametime", "nonwithdrawable", "powered", "voicefeed", 
        "conditions", "bonus", ''
    ])

    text = emoji.replace_emoji(text, '')  
    text = text.lower()
    text = re.sub(r'\b\w*[\./]\w*\b', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\w*\d+\w*', '', text)
    text = re.sub(r'\b[!@$]\w*\b', '', text)
    text = re.sub(r'na$', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    all_stopwords = stop_words.union(custom_stopwords)
    tokens = [token for token in tokens if token not in all_stopwords]
    
    return tokens

# Function to plot the topic distribution
def plot_topic_distribution(topic_distribution, podcast_name):
    topics, probabilities = zip(*topic_distribution)
    categories = [TOPIC_NAMES.get(topic, f"Topic {topic}") for topic, _ in topic_distribution]
    values = list(probabilities)
    values += values[:1]  # Repeat the first value to close the circle

    # Radar plot setup
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]  # Close the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

     # Set background color to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Plot data
    ax.fill(angles, values, color='skyblue', alpha=0.4)
    ax.plot(angles, values, color='blue', linewidth=2)

   # Customize the appearance for a dark theme with green font
    ax.tick_params(axis='both', which='both', colors='#179B44', labelsize=14)  # Larger tick font
    ax.set_xticks(angles[:-1])  # Remove the last angle to avoid duplication
    ax.set_xticklabels(categories, color='#179B44', fontsize=16)  # Larger category labels
    ax.set_yticklabels([])  # Hide y-tick labels for a cleaner look
    ax.set_yticks([0.1, 0.2, 0.3])  # Adjust these values based on your range
    
    # Set title with green font
    ax.set_title(f"Topic Distribution for Podcast: {podcast_name}", color='#179B44')

    # Save the plot to a BytesIO object (in memory, no file creation)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True)  # Use transparent background for saving
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()  # Close the buffer after usage

    return img_str



# Define the helper function
def get_nearest_podcasts(new_podcast):

    similarities = cosine_similarity(new_podcast.reshape(1, -1), all_podcasts[['Topic0', 'Topic1', 'Topic2', 'Topic3', 'Topic4','Topic5','Topic6', 'Topic7']].values.astype('float32'))
    
    sorted_indices = similarities[0].argsort()[::-1][1:3]  # Skip the first as it's the self-similarity
    
    return sorted_indices

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define the helper function
def plot_get_nearest_podcasts(new_podcast, num_podcasts=20):
    try:

        new_podcast_reshaped = new_podcast.reshape(1, -1)
        all_podcast_vectors = all_podcasts[['Topic0', 'Topic1', 'Topic2', 'Topic3', 'Topic4', 'Topic5', 'Topic6', 'Topic7']].values.astype('float32')
        similarities = cosine_similarity(new_podcast_reshaped, all_podcast_vectors)[0]  
        sorted_indices = similarities.argsort()[::-1]  # Sort from highest to lowest
        top_indices = sorted_indices[1:num_podcasts+1]  # Index from 1
        nearest_podcasts = [(index, similarities[index]) for index in top_indices]
        return nearest_podcasts  
    except Exception as e:
        print(f"Error occurred in get_nearest_podcasts: {str(e)}")
        return []





##############################
######      UI      ##########
##############################
app_ui = ui.page_fluid(
     ui.tags.head(
        ui.tags.title("Spotify Podcast Explorer")  # This sets the browser tab title
    ),
    ui.tags.style(
        f"""
        body {{
            background-color: {spotify_black};
        }}
        .page-title {{
            color: {spotify_green};  /* Deep Green for the title */
            font-size: 2em;  /* Adjust the font size as needed */
            text-align: center;  /* Center the title */
            margin-top: 20px;
        }}
        .spotify-logo {{
            width: 150px;
            padding: 10px;
        }}
        .border-green {{
            border: 2px solid {spotify_green} !important;
        }}
        .btn-spotify {{
            background-color: {spotify_green} !important;
            border-color: {spotify_green} !important;
            color: #fff !important;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .spotify-icon {{
            width: 1em; 
            height: 1em; 
            margin-right: 10px;
        }}
        .footer {{
            color: white;
            text-align: center;
            padding: 20px;
            border-top: 2px solid {spotify_green};
        }}
        .spotify-card {{
            background-color: {spotify_black} !important;
            color: {spotify_green} !important;
            border: 2px solid {spotify_green} !important;
        }}
        .spotify-search-results {{
        background-color: {spotify_black} !important;
        color: {spotify_green} !important;
        }}
        .spotify-dropdown {{
        background-color: #191414 !important;
        color: #179B44 !important;
        border: 2px solid #179B44 !important;
        }}
        .hidden {{
            display: none;
        }}
        .scatter-plot-text {{
            color: #179B44 !important; 
        }}
        """
    ),
    
    # Spotify logo and search 
    ui.tags.div(
        ui.tags.img(src="https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg", 
                    class_="spotify-logo"),
        # Adding a custom title at the top of the page
    ui.tags.div(
        ui.h1("Spotify Podcast Explorer", class_="page-title"),  # This creates a page title
        style="text-align: center;"
    ),
        style="text-align: left;"
    ),
    
    # Page Design
    ui.page_sidebar(
        ui.sidebar(
            ui.h3("Enter Podcast Episode Name", style=f"color: {spotify_green};"),
            ui.input_text("podcast_name", "Podcast Name", placeholder="Please enter the Podcast Episode Name"),
            ui.input_action_button(
                "search_button", 
                ui.tags.div(
                    ui.tags.img(
                        src="https://upload.wikimedia.org/wikipedia/commons/8/84/Spotify_icon.svg", 
                        class_="spotify-icon"
                    ),
                    ui.tags.span("Search"),
                    style="display: flex; align-items: center; justify-content: center;"
                ),
                class_="btn btn-spotify"
            ),
            ui.input_dark_mode(mode="dark"),
            class_="border-green"
        ),
        
        ui.card(
            ui.navset_card_tab(
                ui.nav_panel("LDA Topic Visualization", 
                    # ui.h1("Data Visualization Content", style=f"color: {spotify_green}; text-align: center;"),
                    ui.output_ui("render_lda_viz")  # Render the LDA visualization here
                ),
                ui.nav_panel("Search Results",
                    ui.tags.div(  
                        ui.tags.p("Podcast Analysis: ", style="color: #179B44;"),
                        ui.output_ui("analyze_podcast")
                    ),
                    ui.tags.div(
                        ui.tags.div(
                            ui.input_select(
                                "topic_x", 
                                "Select X-axis Topic", 
                                choices=[f"Topic {i}: {name}" for i, name in TOPIC_NAMES.items()], 
                                selected="Topic 0: NFL",
                            ),
                            style="display: block;" 
                        ),
                        ui.tags.div(
                            ui.input_select(
                                "topic_y", 
                                "Select Y-axis Topic", 
                                choices=[f"Topic {i}: {name}" for i, name in TOPIC_NAMES.items()], 
                                selected="Topic 1: Technology",
                            ),
                            style="display: block;" 
                        ),
                        style="display: flex; justify-content: space-between; gap: 10px;", 
                        class_="spotify-dropdown",
                        
                    ),
                    
                    
                    ui.input_action_button("render_scatter_plot", "Plot", class_="btn btn-spotify"),
                    ui.output_ui("plot_scatter"),
                )
            ),
            class_="spotify-card"
        )
    ),
    
     # Contact info at the bottom
    ui.tags.div(
        ui.tags.p("Copyright @ Fall2024-Stat628-Module4-Group4 . If you have any questions about the app, feel free to contact us at szhang655@wisc.edu / rming@wisc.edu", style="color: #179B44;"),
        class_="footer"
    )
)


##############################
######SERVER LOGIC############
##############################
def server(input, output, session):

    search_result = reactive.value(None)


    access_token = get_access_token()  # Get access token
    
    @reactive.event(input.search_button)
    def search_podcast():
        podcast_name = input.podcast_name()
        message = ""
        if not podcast_name:
            message += ("Please enter a Podcast Episode name.\n")
            return None, message
        
        headers = {"Authorization": f"Bearer {access_token}"}
        search_url = "https://api.spotify.com/v1/search"
        params = {
            "q": podcast_name,
            "type": "episode",
            "limit": 1  # Only take one search result
        }
        
        response = requests.get(search_url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            episodes = data.get("episodes", {}).get("items", [])
            
            if episodes:
                episode = episodes[0]
                #  validation
                languages = episode.get("languages", [])
                duration_ms = episode.get("duration_ms", 0)
                description = episode.get("description", "")
                preview_url = episode.get("audio_preview_url", "")
                cover_image = episode.get("images", [{}])[0].get("url", "")
                if not (validate_episode(episode) and check_description_length(description)):
                    message += "Episode either not in English, less than a minute, or description is too short. Please enter again.\n"
                    result = None
                    return result, message
                else:
                    result = {
                        "name": episode.get("name"),
                        "description": episode.get("description"),
                        "languages": languages,
                        "duration_ms": duration_ms, 
                        "preview_url": preview_url,
                        "cover_image": cover_image
                    }
                    message += ""

                    search_result.set(result)  # Store the result
                    return result, message
                
            else:
                message += "No Episode found, Please enter a valid episode name.\n"
                result = None
                return result, message
        else:
            message += "Error calling Spotify API. Please try later..\n"
            result = None
            return result, message
        
   
    # Render LDA visualization in the UI
    @render.ui
    def render_lda_viz():
        with open(os.path.join("www", "spotify_lda_topics.html"), 'r') as file: 
            html_as_string = file.read()
        
        html_with_styles = f"""
        <div style="transform: scale(0.7); transform-origin: 0 0; width: 100%; height: 100%;">
            {html_as_string}
        </div>
        """
        return ui.HTML(html_as_string)
    
    @output
    @render.text
    @reactive.event(input.search_button)
    def analyze_podcast():
        try:
            result, message = search_podcast()

            if not result:
                return message
            
            new_text = result['description']
            new_bow = dictionary.doc2bow(preprocess_text(new_text))

            # Get new episode's topic distribution
            lda_distribution = lda_model.get_document_topics(new_bow)
            topic_dict = {i: 0 for i in range(8)}

            for topic, prob in lda_distribution:
                topic_dict[topic] = prob
            topic_distribution = [(topic, topic_dict[topic]) for topic in sorted(topic_dict.keys())]
            

            # Print Topic Distribution
            output = ""
            for topic, prob in lda_distribution:
                category_name = TOPIC_NAMES.get(topic, f"Topic {topic}")
                percentage = f"{prob * 100:.2f}%"
                output += (f"{category_name}: {percentage}\n\n")

            
            
            new_podcast_vector = np.array([prob for _, prob in topic_distribution])
            # Get the nearest podcasts using the helper function
            nearest_indices = get_nearest_podcasts(new_podcast_vector)

            # Fetch the nearest podcasts' details from your data
            nearest_podcasts = all_podcasts.iloc[nearest_indices]

            # Prepare the output for the nearest podcasts
            nearest_output = "The two closest podcasts to the new podcast are:\n\n"

            
            for idx, podcast in nearest_podcasts.iterrows():
                nearest_output += f"        Name: {podcast['name']}.\n"


            # Radar Plot
            img_str = plot_topic_distribution(topic_distribution, result['name'])

            # Generate HTML output with side-by-side layout
            html_output = ui.tags.div(
                # Header with podcast image and audio preview
                # Header with podcast name, image and audio preview
                ui.tags.h3(f"Podcast Found: {result['name']}", style="color: #179B44; text-align: center; margin-bottom: 20px;"),  # Add podcast name here
                ui.tags.div(
                    ui.tags.img(src=result['cover_image'], style="display: block; margin: 0 auto; width: 100%; max-width: 300px; height: auto;"),
                    style="text-align: center; margin-bottom: 20px;"
                ),
                ui.tags.h5("Preview Audio:"),
                ui.tags.audio(src=result['preview_url'], controls=True, style="width: 100%;"),
                
                # Side-by-side layout
                ui.tags.div(
                    ui.tags.div(
                        ui.HTML(f'<img src="data:image/png;base64,{img_str}" style="width: 100%; max-width: 400px; height: auto;">'),
                        style="width: 50%; padding: 10px;"
                    ),
                    ui.tags.div(
                        ui.tags.h5("Topic Distribution as Percentages:"),
                        ui.tags.pre(output, style="color: #179B44; white-space: pre-wrap;"),
                        style="width: 50%; padding: 10px;"
                    ),
                    style="display: flex; justify-content: space-around;"
                ),
                
                # Nearest podcasts
                ui.tags.h5("Two Nearest Podcasts:"),
                ui.tags.pre(nearest_output, style="color: #179B44;"),
                style="background-color: #191414; color: #179B44;"
            )

        
            return html_output
            


        except Exception as e:
            message = f"Error occurred: {str(e)}"
            return ui.tags.pre(message, style="color: #179B44;")
    
    @output
    @render.ui
    @reactive.event(input.render_scatter_plot)
    def plot_scatter():

        try:
            result = search_result.get()  # Retrieve the search result from the reactive value
            if not result:
                return "No podcast results found. Please search for a podcast first."
            
            # Process the selected topics from the dropdown
            selected_x = input.topic_x()
            selected_y = input.topic_y()
            
            topic_x_index = int(re.search(r"Topic (\d+):", selected_x).group(1))
            topic_y_index = int(re.search(r"Topic (\d+):", selected_y).group(1))
            
            # Generate the topic distribution for the podcast
            new_text = result['description']
            new_bow = dictionary.doc2bow(preprocess_text(new_text))
            lda_distribution = lda_model.get_document_topics(new_bow)

            topic_dict = {i: 0 for i in range(8)}
            for topic, prob in lda_distribution:
                topic_dict[topic] = prob
            
            topic_distribution = [(topic, topic_dict[topic]) for topic in sorted(topic_dict.keys())]
            new_podcast_vector = np.array([prob for _, prob in topic_distribution])

            # Nearest Podcast
            nearest_podcast_tuples = plot_get_nearest_podcasts(new_podcast_vector)
            
            nearest_indices = [item[0] for item in nearest_podcast_tuples]
            nearest_distances = [item[1] for item in nearest_podcast_tuples]
            nearest_podcasts = all_podcasts.iloc[nearest_indices].copy()

            # Data for Plot
            scatter_data = nearest_podcasts[['Topic0', 'Topic1', 'Topic2', 'Topic3', 'Topic4', 'Topic5', 'Topic6', 'Topic7']].values[:20]
            
            x_values = scatter_data[:, topic_x_index]
            y_values = scatter_data[:, topic_y_index]
            nearest_podcasts['distance'] = nearest_distances
            nearest_podcasts['rank'] = range(1, len(nearest_distances) + 1)  # 排名
            nearest_podcasts['x'] = x_values
            nearest_podcasts['y'] = y_values

            #  Color density to reflect nearest rank
            max_rank = max(nearest_podcasts['rank'])
            colors = ['rgba(30, 144, 255, {:.2f})'.format(0.3 + 0.7 * (max_rank - rank) / max_rank) 
                    for rank in nearest_podcasts['rank']]  # light to dark blue

            # Input Point 
            input_x = new_podcast_vector[topic_x_index]
            input_y = new_podcast_vector[topic_y_index]

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=nearest_podcasts['x'],
                y=nearest_podcasts['y'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=colors  
                ),
                text=[f"Name: {row['name']}<br>Rank: {row['rank']}<br>Distance: {row['distance']:.3f}" 
                    for _, row in nearest_podcasts.iterrows()], 
                hoverinfo='text'
            ))

            # Input point is RED 
            fig.add_trace(go.Scatter(
                x=[input_x],
                y=[input_y],
                mode='markers',
                marker=dict(
                    size=14,
                    color='red',
                    symbol='star'
                ),
                text=['Input Podcast'],
                hoverinfo='text'
            ))

            fig.update_layout(
                title=f"Scatter plot of Topic {topic_x_index} vs Topic {topic_y_index} With 20 Nearest Podcasts",
                xaxis_title=f"Topic {topic_x_index}",
                yaxis_title=f"Topic {topic_y_index}",
                template="plotly_dark",
                showlegend=False
            )

            # HTML Render
            return ui.HTML(pio.to_html(fig, full_html=False))

        except Exception as e:
            message = f"Error occurred: {str(e)}"
            return ui.tags.pre(message, style="color: #179B44;")
                

############### APP #################
app = App(app_ui, server)