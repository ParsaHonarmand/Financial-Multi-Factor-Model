
# get ticker data
# df = yf.download("PLD", start='2018-10-25', group_by='ticker').dropna() 
# sml = yf.download("SPY", start='2018-10-25', group_by='ticker').dropna() #[map(str.title, ['open', 'close', 'low', 'high', 'volume'])]

# df = yf.download("PLD", period="5yr", interval="1m", group_by='ticker').dropna() 
# sml = yf.download("SPY", period="5yr", interval="1m", group_by='ticker').dropna() 

# sml = pd.read_csv('SML.csv', index_col=0)

# print(sml)
# exit()

# df.columns = [x.lower() for x in df.columns]
# sml.columns = [x.lower() for x in df.columns]

# df = df[["close"]]

# df['sml'] = sml[['close']]

# print(df)

# df['offset'] = df['close'].shift(-1)

# df['percent_returns'] = df['offset'] / df['close'] - 1

# df['ema_10'] = ta.ema(df['close'], length=10)

# calculate MACD values
# df.ta.macd(close='close', append=True)
# df = df.dropna()

# print(df)
# exit()


# X_train, X_test, y_train, y_test = train_test_split(df[['percent_returns']], df[['MACDh_12_26_9']], test_size=.3)
# X_train, X_test, y_train, y_test = train_test_split(df[['close']], df[['ema_10']], test_size=.3)
# X_train, X_test, y_train, y_test = train_test_split(df[['sml']], df[['close']], test_size=.3)

# Create Regression Model
# model = LinearRegression()

# Train the model
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)


# Printout relevant metrics

# plt.scatter(X_test, y_test, color="black")
# plt.plot(X_test, y_pred, color="blue", linewidth=3)

# plt.xticks(X_test)
# plt.yticks(y_pred)

# plt.show()

# fig = py.subplots.make_subplots(rows=2, cols=1)
# fig = go.Figure()
# Fast Signal (%k)
#   fig.add_trace(
#       go.Scatter(
#           x=df.index,
#           y=df['macd_12_26_9'],
#           line=dict(color='#ff9900', width=2),
#           name='macd',
#           # showlegend=False,
#           legendgroup='2',
#       )
#   )
# Slow signal (%d)
# fig.add_trace(
#     go.Scatter(
#         x=df.index,
#         y=df['macds_12_26_9'],
#         line=dict(color='#000000', width=2),
#         # showlegend=False,
#         legendgroup='2',
#         name='signal'
#     )
# )
# Colorize the histogram values
# colors = np.where(df['macdh_12_26_9'] < 0, '#ff0000', '#00ff00')
# Plot the histogram
# fig.add_trace(
#     go.Bar(
#         x=df.index,
#         y=df['macdh_12_26_9'],
#         name='histogram',
#         marker_color=colors,
#     )
# )
# # Make it pretty
# layout = go.Layout(
#     title=ticker,
#     plot_bgcolor='#efefef',
#     # Font Families
#     font_family='Monospace',
#     font_color='#000000',
#     font_size=20,
#     xaxis=dict(
#         rangeslider=dict(
#             visible=False
#         )
#     )
# )
# Update options and show plot
# fig.update_layout(layout)
#   fig.write_html("./figures/" + ticker + "_MACD.html")
#   fig.write_image("./macds/" + ticker + ".png", width=1920, height=1080)
# fig.show()