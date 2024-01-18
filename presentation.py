import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import simpledialog
from tkinter.filedialog import askopenfilename
from sklearn.linear_model import LogisticRegression
import pickle
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Load the pre-trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# List of teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Gujarat Titans', 'Lucknow Super Giants', 'Punjab Kings', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 'Jaipur', 'Chennai',
          'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur',
          'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']

# Create the main window
main = tk.Tk()
main.title("IPL Match Win Predictor")

# Set the window size and position
window_width = 800
window_height = 600
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
x_position = (screen_width // 2) - (window_width // 2)
y_position = (screen_height // 2) - (window_height // 2)
main.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

def plot_probabilities(prob_win, prob_loss):
    teams = [bat.get(), bowl.get()]
    probabilities = [prob_win, prob_loss]

    plt.figure(figsize=(6, 4))  # Decrease the figsize to (6, 4)
    bars = plt.bar(teams, probabilities, color=['green', 'red'], width=0.4)
    plt.xlabel('Team')
    plt.ylabel('Probability')
    plt.title('Win Probability')

    # Set the y-axis limits to be the same for both teams
    plt.ylim(0, 1)

    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{round(prob*100, 2)}%', ha='center', va='bottom')

    # Create a FigureCanvasTkAgg object to display the plot inside a Tkinter window
    canvas = FigureCanvasTkAgg(plt.gcf(), master=main)
    canvas.draw()

    # Position the canvas at the top-right corner
    canvas.get_tk_widget().place(x=650, y=10)  # Adjust x and y coordinates as per your preference

def predict_probability():
    try:
        total_runs = int(entry1.get())
        current_score = int(entry2.get())
        current_overs = float(entry3.get())
        current_wickets = int(entry4.get())

        if current_overs <= 0 or current_overs >= 20:
            raise ValueError("Overs should be between 0.1 and 19.6")

        if current_wickets < 0 or current_wickets > 10:
            raise ValueError("Wickets should be between 0 and 10.")

        if total_runs < 0 or total_runs < current_score:
            raise ValueError("Score cannot be greater than the target.")

        if current_score < 0:
            raise ValueError("Score cannot be negative.")

        runs_left = total_runs - current_score
        balls_left = (120 - (int(current_overs) * 6)) or 1  # To avoid zero division error
        wickets_left = 10 - current_wickets
        crr = current_score / current_overs
        rrr = (runs_left * 6) / balls_left
        rr_score = crr * 20

        input_df = pd.DataFrame({
            'batting_team': [bat.get()],
            'bowling_team': [bowl.get()],
            'city': [stadium.get()],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [total_runs],
            'crr': [crr],
            'rrr': [rrr]
        })

        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        header_text = f"{bat.get()} - {round(win * 100, 2)}%"
        header_label.config(text=header_text)

        header_text_bowl = f"{bowl.get()} - {round(loss * 100, 2)}%"
        header_label_bowl.config(text=header_text_bowl)
        plot_probabilities(win, loss)

    except ValueError as ve:
        # Display error message
        header_label.config(text=f"Error: {str(ve)}")
        header_label_bowl.config(text="")

# Create style
style = ttk.Style(main)
style.configure('TLabel', font=('Arial', 18), foreground='black', background='lightgreen')
style.configure('TButton', font=('Arial', 16), foreground='black', background='white')

# Labels and Entry widgets
bat_label = ttk.Label(main, text="Batting Team:")
bat_label.grid(row=0, column=0, padx=10, pady=5)

bat = tk.StringVar()
bat.set(teams[0])
bat_dropdown = ttk.OptionMenu(main, bat, *teams)
bat_dropdown.grid(row=0, column=1, padx=10, pady=5)

bowl_label = ttk.Label(main, text="Bowling Team:")
bowl_label.grid(row=1, column=0, padx=10, pady=5)

bowl = tk.StringVar()
bowl.set(teams[0])
bowl_dropdown = ttk.OptionMenu(main, bowl, *teams)
bowl_dropdown.grid(row=1, column=1, padx=10, pady=5)

stadium_label = ttk.Label(main, text="City:")
stadium_label.grid(row=2, column=0, padx=10, pady=5)

stadium = tk.StringVar()
stadium.set(cities[0])
stadium_dropdown = ttk.OptionMenu(main, stadium, *cities)
stadium_dropdown.grid(row=2, column=1, padx=10, pady=5)

target_label = ttk.Label(main, text="Target:")
target_label.grid(row=3, column=0, padx=10, pady=5)

entry1 = tk.Entry(main)
entry1.grid(row=3, column=1, padx=10, pady=5)

score_label = ttk.Label(main, text="Score:")
score_label.grid(row=4, column=0, padx=10, pady=5)

entry2 = tk.Entry(main)
entry2.grid(row=4, column=1, padx=10, pady=5)

overs_label = ttk.Label(main, text="Overs:")
overs_label.grid(row=5, column=0, padx=10, pady=5)

entry3 = tk.Entry(main)
entry3.grid(row=5, column=1, padx=10, pady=5)

wickets_label = ttk.Label(main, text="Wickets:")
wickets_label.grid(row=6, column=0, padx=10, pady=5)

entry4 = tk.Entry(main)
entry4.grid(row=6, column=1, padx=10, pady=5)

# Prediction button
predict_button = ttk.Button(main, text="Predict Probability", command=predict_probability)
predict_button.grid(row=7, column=0, columnspan=2, padx=50, pady=20)  # Increase padding for centering

# Labels to show the prediction results
header_label = ttk.Label(main, text="", font=("Arial", 24, "bold"))
header_label.grid(row=8, column=0, columnspan=2, padx=50, pady=20)  # Increase padding for centering

header_label_bowl = ttk.Label(main, text="", font=("Arial", 24, "bold"))
header_label_bowl.grid(row=9, column=0, columnspan=2, padx=50, pady=20)  # Increase padding for centering

main.config(bg="lightgreen")
main.mainloop()
