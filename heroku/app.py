import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.graph_objs as go
import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

########### Define your variables ######

tabtitle = 'Game Start to GG'
sourceurl = 'https://www.kaggle.com/xtyscut/video-games-sales-as-at-22-dec-2016csv?select=Video_Games_Sales_as_at_22_Dec_2016.csv'
githublink = 'https://github.com/tzhang900/ds_capstone/tree/main/heroku'
genre_options = ['Sports', 'Platform', 'Racing', 'Role-Playing', 'Puzzle', 'Misc', 'Shooter', 'Simulation', 'Action', 'Fighting', 'Adventure', 'Strategy']
platform_options = ['Wii', 'NES', 'GB', 'DS', 'X360', 'PS3', 'PS2', 'SNES', 'GBA', 'PS4', '3DS', 'N64', 'PS', 'XB', 'PC', '2600', 'PSP', 'XOne', 'WiiU', 'GC', 'GEN', 'DC', 'PSV', 'SAT', 'SCD', 'WS', 'NG', 'TG16', '3DO', 'GG', 'PCFX']
image1='characters.jpg'
image2='consoles.jpg'
background='background.jpg'
########## Set up the chart

import pandas as pd
games = pd.read_csv('assets/Video_Games.csv')

########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle

##### prepate data
games.drop(['Developer'] ,inplace =True, axis =1)
games.drop(['Critic_Count'] ,inplace =True, axis =1)
games.drop(['User_Count'] ,inplace =True, axis =1)
games.drop(['Name'], inplace=True, axis = 1)
games.drop(['Publisher'], inplace=True, axis = 1)

# AO has only 1 game, which sales extremly too well (Grand Theft Auto 4), this will skew the analysis, let's put it into M cateogry which was its initial rating
# Also K-A (Kid to Aldut) is the same as E (everyone)
games['Rating'] = games['Rating'].replace("AO", "M")
games['Rating'] = games['Rating'].replace("K-A", "E")
games['Rating'] = games['Rating'].fillna("RP")

# Replace'tbd' with nan
games['User_Score'] = games['User_Score'].replace('tbd', np.nan)
# Now change User_Score to numeric type
games['User_Score'] = games['User_Score'].astype('float64')

#fill in mean value for User_Score and Critic_Score
games['User_Score'].fillna(value=games['User_Score'].mean(), inplace=True)
games['Critic_Score'].fillna(value=games['Critic_Score'].mean(), inplace=True)

#  Still a few row missing data on Name, Genre and Year_of_Release, but not that much compared to the whole dataset. Drop them
games.dropna(inplace=True)

# Transform categorized variable ('Genre')
games['Genre'], label_genre = pd.factorize(games['Genre'])
games['Platform'], label_platform = pd.factorize(games['Platform'])

### building model
y_sales = games["EU_Sales"].copy()
X_sales = games.drop(['Global_Sales','JP_Sales', 'EU_Sales', 'Other_Sales', 'Year_of_Release', 'User_Score', 'Critic_Score', 'Rating'], axis = 1)

# Now, split both X and y data into training and testing sets.
X_sales_train, X_sales_test, y_sales_train, y_sales_test = train_test_split(X_sales, y_sales, 
                                       test_size=0.2, 
                                       random_state=42)

# Create a local instance of the sklearn class
lin_reg_sales = LinearRegression()

# Fit your instance to the training dataset
lin_reg_sales.fit(X_sales_train, y_sales_train)

########### Set up the layout

app.layout = html.Div(
    style={
    },
    children=[
        html.H1('Game Sales in Euro Prediction'),
        html.Div([
            html.Div([
                    html.H6("Copies Sold in North America in millions, eg 3.3"),
                    dcc.Input(id='na-sales', type="number"),
                    html.H6('Select Genre:'),
                    dcc.Dropdown(
                        id='genres-drop',
                        options=[{'label': i, 'value': i} for i in genre_options],
                        value='genre'
                    ),
                    html.H6('Select Platform:'),
                    dcc.Dropdown(
                        id='platforms-drop',
                        options=[{'label': i, 'value': i} for i in platform_options],
                        value='platform'
                    ),
                    html.Img(src=app.get_asset_url(image2), style={'width': '100%', 'height': '10%'}),
            ], className='four columns'),
            html.Div([
                html.Div(id='your-output-here', children='',  style={'color': 'blue', 'fontSize': 40}),
                html.Img(src=app.get_asset_url(image1), style={'width': '80%', 'height': '10%'}),
            ], className='eight columns'),
        ], className='twelve columns'),
        html.P(),
        html.Br(),
        html.A('Code on Github', href=githublink),
        html.Br(),
        html.A("Data Source", href=sourceurl),
    ]
)

def listToString(f): 
    
    # initialize an empty string
    str1 = " " 
    s = ['{:.3f}'.format(x) for x in f]
    
    # return string  
    return (str1.join(s))
    
# make a function that can intake any varname and produce a map.
@app.callback(Output('your-output-here', 'children'),
             [Input('na-sales', 'value'),
              Input('genres-drop', 'value'),
              Input('platforms-drop', 'value')])
def make_figure(sales,genre,platform):
    if (sales is None or genre is None or genre == "genre" or platform is None or platform == "platform"):
       return f'Please enter all values to predict the result'
    int_genre = list(label_genre).index(genre)
    int_platform = list(label_platform).index(platform)
    mygraphtitle = f'Exports of {genre} and {platform} in 2011'
    mycolorscale = f'genre {int_genre} and platform {int_platform}'
    input_value = [[int_genre, int_platform, sales]]
    prediction = lin_reg_sales.predict(input_value)
    return f'Predict {platform} {genre} game sales in Europe:' +   listToString(prediction)
    #return  mycolorscale
    #return mygraphtitle

############ Deploy
if __name__ == '__main__':
    app.run_server(debug=True)
