import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv("acvitkovic/filtered_properties.csv")

# Loop through each column
for column in data.columns[:-1]:  # Exclude the 'anomaly' column
    # Calculate the top and bottom 10%
    bottom_10_percent_value = data[column].quantile(0.1)
    top_10_percent_value = data[column].quantile(0.9)

    #'anomaly' columns red, normal ones blue
    colors = data['anomaly'].map({-1: 'red', 1: 'blue'})
    
    fig = px.scatter(data, x=column, y=[0] * len(data[column]), 
                     color=colors, color_discrete_map='identity',
                     hover_data={'index': data.index, column: True, 'anomaly': True},
                     labels={column: column, 'y': 'Fixed Y-axis Value'},
                     title=f'{column} Values Over X-Axis Range')
    
    fig.add_shape(
        type="line",
        x0=bottom_10_percent_value,
        y0=-1,
        x1=bottom_10_percent_value,
        y1=1,
        line=dict(color="Red", width=2, dash="dash"),
        name="Bottom 10%"
    )
    
    fig.add_shape(
        type="line",
        x0=top_10_percent_value,
        y0=-1,
        x1=top_10_percent_value,
        y1=1,
        line=dict(color="Red", width=2, dash="dash"),
        name="Top 10%"
    )
    
    fig.update_layout(
        height=400, 
        yaxis=dict(visible=False),  # Hide the Y-axis - don't need it
        xaxis_title=column,
        yaxis_title='',
        showlegend=False
    )
    
    fig.show()
