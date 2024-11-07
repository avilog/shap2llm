from shap2llm.explainer import Explainer
import pandas as pd

bike_df = pd.read_csv('../data/bike_sharing_dataset/day.csv', index_col ='dteday')
bike_df.drop('instant', axis=1, inplace=True)
bike_df['temp'] = bike_df['temp']*41
bike_df['atemp'] = bike_df['atemp']*50
bike_df['hum'] = bike_df['hum']*100
bike_df['windspeed'] = bike_df['windspeed']*67

print(bike_df.describe())

# separate input and output in each dataset
input_columns = ['season', "holiday", 'yr', 'mnth', 'weekday', 'workingday','temp', 'hum', 'windspeed']
output_column =  'cnt'


from sklearn.ensemble import RandomForestRegressor

rforest = RandomForestRegressor(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
rforest.fit(bike_df[input_columns], bike_df[output_column])

explainer = Explainer(dataset_context="Bike-sharing rental process is highly correlated to the environmental and seasonal settings. For instance, weather conditions, precipitation, day of week, season, hour of the day, etc. can affect the rental behaviors. The core data set is related to the two-year historical log corresponding to years 2011 and 2012 from Capital Bikeshare system, Washington D.C., USA which is publicly available in http://capitalbikeshare.com/system-data. We aggregated the data on two hourly and daily basis and then extracted and added the corresponding weather and seasonal information. Weather information are extracted from http://www.freemeteo.com.",
        model_task_context="the model predicts total daily count of total rental bikes",
        model=rforest,
        X_dataset=bike_df[input_columns],
        features_descriptions={"season" : "season (1:springer, 2:summer, 3:fall, 4:winter)",
                                "yr" : "year (0: 2011, 1:2012)",
                                "mnth" : "month (1 to 12)",
"holiday" : "weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)",
"weekday" : "day of the week",
"workingday" : "if day is neither weekend nor holiday is 1, otherwise is 0.",
"weathersit" :"""
    1: Clear, Few clouds, Partly cloudy, Partly cloudy
    2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
    3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
    4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog""",
"temp" : "temperature in Celsius",
"atemp": "feeling temperature in Celsius",
"hum": "humidity",
"windspeed": "wind speed"},
        shap_algorithm="tree")

print(explainer.describe_graph(feature_name="mnth"))
#tasks=[Task(explanation='The general pattern of the graph exhibits a gradual increase in SHAP values as the month progresses from January (1) to December (12). This indicates a positive influence on bike rentals during the later months of the year, particularly peaking toward the end of the year.', output='There is a clear upward trend in the SHAP values as the months advance.'), Task(explanation='Surprisingly, while most of the data points are concentrated with positive SHAP values for later months, the values for the early months (especially February) show notably higher negative SHAP values compared to other months. This suggests that conditions during these months greatly reduce the number of bike rentals, likely due to colder weather and fewer outdoor activities typical of winter months. This behavior contrasts with the general intuition that bike rentals might be more consistent across the year.', output='The sharp decrease in SHAP values for early months is unexpected, suggesting seasonal effects.'), Task(explanation="The dependence scatter plot illustrates the SHAP values associated with the feature 'mnth', which corresponds to the month of the year in the bike-sharing dataset. As observed, there’s a significant positive trend in SHAP values from January to December, indicating that rental counts tend to be higher in the later months. Notably, the early months, particularly February, show a pronounced dip into negative SHAP values, suggesting adverse factors influencing bike rentals during winter. This trend highlights the seasonal nature of bike-sharing, with demand dropping in colder months and rising again as the year progresses. The distribution histogram at the bottom indicates a concentration of data points in the mid-range of months, supporting the observed rental patterns.", output='')] final_answer='The graph shows how the month (mnth) affects the SHAP values, which indicate the impact of the month on bike rentals. There is a clear upward trend in SHAP values as months progress, suggesting an increase in bike rentals later in the year. However, early months, particularly February, are marked by lower, negative SHAP values, indicating a decrease in bike usage during colder months. The lower rentals in winter contrast with the expectations of consistent year-round usage, emphasizing the seasonality in bike-sharing behavior. The distribution histogram reveals a concentration of data during the middle of the year, aligning with the peak rental trends.'
print(explainer.describe_graph(feature_name="season"))
#tasks=[Task(explanation="The graph depicts how the 'season' feature impacts the SHAP values, which reflect the effect of season on the predicted total daily count of bike rentals. The x-axis represents the numeric value of the seasons (1 for spring, 2 for summer, 3 for fall, and 4 for winter), while the y-axis indicates the SHAP values for those predictions. As we observe, SHAP values are generally lower for spring, moderate for summer, and rise significantly in fall and winter, indicating the strength of correlation between the season and rental predictions.", output='The general trend shows that summer has a moderate positive effect on bike rentals, while fall and winter have strong positive SHAP values, indicating a stronger influence on bike rentals compared to spring.'), Task(explanation='A surprising aspect of the plot is the significantly high SHAP values for fall and winter compared to spring and summer. Intuitively, one might expect warmer seasons to drive higher bike rental counts due to favorable weather conditions. However, the data suggests that these colder seasons may have factors, such as holiday activities in winter or pleasant fall weather, that promote bike usage. This could reflect variations in cultural activities or promotional strategies during these seasons that aren’t immediately apparent.', output='The higher SHAP values for fall and winter when compared to spring point to an unexpected positive impact on bike rentals during colder months.'), Task(explanation='The dependence scatter plot illustrates how the season affects bike rentals based on SHAP values. The SHAP value is low (around -400) during spring, indicative of lower rental activity, while it spikes to about 600 in fall and similarly high in winter. Summer, while warmer, has moderate SHAP values around 200. Notably, the data points in fall and winter are concentrated at the upper end of the SHAP scale, emphasizing a strong positive correlation of these seasons with bike rentals. This is contrary to typical expectations where warmer weather might be predicted to drive higher rentals.', output="In summary, the scatter plot reveals a strong positive effect of fall and winter on bike rentals, with surprising results that differ from expectations of the influence of warmer seasons like spring and summer. The histogram at the bottom shows the distribution of season values, confirming that the model's predictions are impacted significantly by fall and winter.")] final_answer="The dependence scatter plot indicates that the 'season' feature plays a crucial role in predicting bike rentals, with a notable positive influence from fall and winter compared to spring and summer. The SHAP values for spring are low (~-400), while fall and winter exhibit strong positive values (~600), suggesting that these seasons might promote bike usage more than what one might expect. Interestingly, summer has a modest positive effect (~200), highlighting a counterintuitive trend that colder seasons may attract more rentals. Overall, the graph effectively demonstrates how seasonal variations directly impact bike-sharing demand."
print(explainer.describe_graph(feature_name="holiday"))
#tasks=[Task(explanation="The graph displays a dependence scatter plot analyzing the impact of the 'holiday' feature on the model's predictions. The x-axis represents the values of the 'holiday' feature, which take binary values (0 for non-holiday and 1 for holiday). The y-axis shows the SHAP values, indicating how much the 'holiday' feature influences bike rental predictions. Predominantly, data points are concentrated around the non-holiday value (0) and a few extend to the holiday value (1), demonstrating a clear distinction in model response based on this feature.", output='The plot indicates that when it is a holiday, the SHAP values are dramatically negative, suggesting that holidays lead to a decrease in total bike rentals according to the model.'), Task(explanation='The surprising aspect of this graph is the stark contrast in SHAP values when comparing holidays and non-holidays. It appears counterintuitive that bike rentals would decrease on holidays, as one might expect more people to ride bikes on days off. This behavior could be attributed to factors like reduced availability of bikes, changes in commuter behavior, or an increase in options for recreational activities on holidays, leading to fewer bike rentals.', output='This behavior may be surprising because holidays are typically seen as times when leisure activities, such as bike-sharing, would increase.'), Task(explanation="In summary, the dependence scatter plot reveals a distinct effect of the 'holiday' feature on bike rental predictions. For non-holiday days, the SHAP values cluster around positive values, indicating that bike rentals increase. Conversely, for holiday days, a dramatic drop in SHAP values suggests a significant decrease in rentals, contrary to intuitive expectations. This insight reflects how holidays influence user behavior and the overall rental count, potentially tied to reduced bike availability or competition from other recreational options.", output='')] final_answer="The dependence scatter plot illustrates the impact of the 'holiday' feature on bike rental predictions. Non-holidays show positive SHAP values, indicating increased rentals, while holidays yield significantly negative SHAP values, suggesting a decrease in rentals. This observation runs counter to the common expectation that bike rentals would rise during holidays, potentially due to alternative recreational choices or limited bike availability on such days."
print(explainer.get_improved_metadata())
#ParsedChatCompletionMessage[Explainer.get_improved_metadata.<locals>.ColumnsDescriptions](content='{"descriptions":[{"name":"season","new_name":"Season of the Year","new_description":"Indicates the current season when the bike rentals are happening. It can be one of four options: spring, summer, fall, or winter."},{"name":"yr","new_name":"Year","new_description":"Denotes the year during which the rental took place. It can either be 2011 or 2012."},{"name":"mnth","new_name":"Month","new_description":"Represents the month of the rental, ranging from January (1) to December (12)."},{"name":"holiday","new_name":"Holiday Indicator","new_description":"This feature shows whether the rental day falls on a recognized public holiday."},{"name":"weekday","new_name":"Day of the Week","new_description":"Specifies which day of the week the rental occurred, ranging from Monday (0) to Sunday (6)."},{"name":"workingday","new_name":"Working Day Status","new_description":"Indicates if the rental day is a working day or not. A day is considered a working day if it is neither a weekend nor a holiday."},{"name":"weathersit","new_name":"Weather Situation","new_description":"Describes the weather conditions during the rentals. It includes categories such as clear skies, misty, light snow, and heavy rain."},{"name":"temp","new_name":"Temperature (Celsius)","new_description":"Shows the actual temperature in degrees Celsius at the time of the rental."},{"name":"atemp","new_name":"Feels Like Temperature (Celsius)","new_description":"Represents the \'feels like\' temperature in degrees Celsius, which indicates how the temperature feels to a person due to factors like humidity and wind."},{"name":"hum","new_name":"Humidity Level","new_description":"Indicates the percentage of humidity in the air during the rental."},{"name":"windspeed","new_name":"Wind Speed","new_description":"Shows the speed of the wind in meters per second during the time of rental."}]}', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], parsed=ColumnsDescriptions(descriptions=[ColumnsDescription(name='season', new_name='Season of the Year', new_description='Indicates the current season when the bike rentals are happening. It can be one of four options: spring, summer, fall, or winter.'), ColumnsDescription(name='yr', new_name='Year', new_description='Denotes the year during which the rental took place. It can either be 2011 or 2012.'), ColumnsDescription(name='mnth', new_name='Month', new_description='Represents the month of the rental, ranging from January (1) to December (12).'), ColumnsDescription(name='holiday', new_name='Holiday Indicator', new_description='This feature shows whether the rental day falls on a recognized public holiday.'), ColumnsDescription(name='weekday', new_name='Day of the Week', new_description='Specifies which day of the week the rental occurred, ranging from Monday (0) to Sunday (6).'), ColumnsDescription(name='workingday', new_name='Working Day Status', new_description='Indicates if the rental day is a working day or not. A day is considered a working day if it is neither a weekend nor a holiday.'), ColumnsDescription(name='weathersit', new_name='Weather Situation', new_description='Describes the weather conditions during the rentals. It includes categories such as clear skies, misty, light snow, and heavy rain.'), ColumnsDescription(name='temp', new_name='Temperature (Celsius)', new_description='Shows the actual temperature in degrees Celsius at the time of the rental.'), ColumnsDescription(name='atemp', new_name='Feels Like Temperature (Celsius)', new_description="Represents the 'feels like' temperature in degrees Celsius, which indicates how the temperature feels to a person due to factors like humidity and wind."), ColumnsDescription(name='hum', new_name='Humidity Level', new_description='Indicates the percentage of humidity in the air during the rental.'), ColumnsDescription(name='windspeed', new_name='Wind Speed', new_description='Shows the speed of the wind in meters per second during the time of rental.')]))
