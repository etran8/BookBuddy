import os
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

import matplotlib.dates as dates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import openai
import pandas as pd
import requests
import seaborn as sns
from encryptoenv.EnvFile import EnvFile
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import FuncFormatter
from serpapi import GoogleSearch


# from dotenv import load_dotenv, find_dotenv
def initialize_database():
    # Connect to MySQL server
    EnvFile().create_environment_variables()
    passw = os.environ["PASSWORD"]
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password=passw
    )

    cursor = connection.cursor()

    # Create database if not exists
    create_database_query = "CREATE DATABASE IF NOT EXISTS `FINANCE`"
    cursor.execute(create_database_query)

    # Switch to the 'FINANCE' database
    use_database_query = "USE `FINANCE`"
    cursor.execute(use_database_query)

    # Create the table if not exists
    create_table_query = """
    CREATE TABLE IF NOT EXISTS `TRANSACTIONS` (
        `ID` INT AUTO_INCREMENT PRIMARY KEY,
        `AMT` DECIMAL(10, 2),
        `CATEGORY` ENUM('revenue', 'expense'),
        `DATE_INPUTED` DATE
    )
    """
    cursor.execute(create_table_query)

    # Commit and close
    connection.commit()
    cursor.close()
    connection.close()


# Function to create a new dashboard based on the function clicked
def create_new_dashboard(root, function_name):
    # Function to bring the window to the front after 2 seconds
    if function_name == "Trends Graph":
        create_trends_generation_ui(root)  # Call function to create UI for image generation
    elif function_name == "Image Generation":
        create_image_generation_ui(root)  # Call function to create UI for image generation
    elif function_name == "Text Generation":
        create_text_generation_ui(root)  # Call function to create UI for text generation
    elif function_name == "SQL Deletion":
        create_sql_delete_ui(root)  # Call function to create UI for SQL input
    elif function_name == "SQL Input":
        create_sql_input_ui(root)  # Call function to create UI for SQL delete


# ------------------------------------------------------------------------------------------------
# Function to get the path to the desktop and create required folders if they don't exist
def get_desktop_path():
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    main_folder = os.path.join(desktop_path, 'Book_Buddy')
    generative_text_folder = os.path.join(main_folder, 'Generative_Text')
    generative_image_folder = os.path.join(main_folder, 'Generative_Image')
    current_trends_folder = os.path.join(main_folder, 'Current_Trends')

    messages = []  # List to store messages about folder creation

    # Create the main folder if it doesn't exist
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
        messages.append(f"The main folder 'Book_Buddy' has been created at:\n{main_folder}")

    # Create the generative text folder if it doesn't exist
    if not os.path.exists(generative_text_folder):
        os.makedirs(generative_text_folder)
        messages.append(f"The folder 'Generative_Text' has been created at:\n{generative_text_folder}")

    # Create the generative image folder if it doesn't exist
    if not os.path.exists(generative_image_folder):
        os.makedirs(generative_image_folder)
        messages.append(f"The folder 'Generative_Image' has been created at:\n{generative_image_folder}")

    # Create the current trends folder if it doesn't exist
    if not os.path.exists(current_trends_folder):
        os.makedirs(current_trends_folder)
        messages.append(f"The folder 'Current_Trends' has been created at:\n{current_trends_folder}")

    if messages:
        # Display all messages in a single pop-up with improved formatting
        messagebox.showinfo("Folders Created", "\n\n".join(messages))


# Function to write text to a file
def txt_to_file(path, time, string):
    new_file_path = os.path.join(path, time + "GeneratedText.txt")
    with open(new_file_path, "w", encoding="utf-8") as file:
        file.write(string)
    messagebox.showinfo("File Saved", f"Text saved at: {new_file_path}")


# Function to convert time string format (replace ':' with '-')
def convert_time_string(string):
    return string.replace(":", "-")


# fetch data for current month
def fetch_data_for_current_month():
    try:
        # Connect to MySQL server
        EnvFile().create_environment_variables()
        passw = os.environ["PASSWORD"]
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password=passw,
            database="FINANCE"
        )

        cursor = connection.cursor()

        # Get current month and year
        now = datetime.now()
        current_month = now.month
        current_year = now.year

        # Fetch revenue and expenses data for the current month
        select_query = f"""
                SELECT DATE(`DATE_INPUTED`) AS DAY_INPUTED, SUM(CASE WHEN CATEGORY = 'revenue' THEN AMT ELSE -AMT END) AS NET_AMOUNT
                FROM `TRANSACTIONS`
                WHERE MONTH(`DATE_INPUTED`) = {current_month}
                AND YEAR(`DATE_INPUTED`) = {current_year}
                GROUP BY DAY_INPUTED
                ORDER BY DAY_INPUTED
            """
        cursor.execute(select_query)
        data = cursor.fetchall()

        cursor.close()
        connection.close()

        return data

    except mysql.connector.Error:
        # Display a pop-up window for the database connection error
        messagebox.showerror("Database Error", "Could not connect to the Database.")
        return None


def dollar_formatter(x, pos):
    return f"${x:.2f}"


# noinspection PyBroadException
def plot_total_finances(root):
    root.geometry("960x557")
    clear_frame(root)

    # Create and configure the hyperlink label
    hyperlink_label = tk.Label(root, text="Back to Financial Database", fg="blue", cursor="hand2")
    hyperlink_label.place(anchor="nw", x=-20, y=-20)  # Adjust x and y coordinates as needed
    hyperlink_label.bind("<Button-1>",
                         lambda event: business_finances_ui(root))  # Bind click event to call main_dashboard

    try:
        # Fetch data from the database
        data = fetch_data_for_current_month()

        if data is not None:
            # Process the fetched data
            days = [row[0] for row in data]
            net_finances = [row[1] for row in data]

            # Calculate cumulative total
            cumulative_total = np.cumsum(net_finances)

            # Plot the line graph
            fig, ax = plt.subplots(figsize=(12, 6))  # Set the figure size to be larger

            # Plot net finances
            ax.plot(days, cumulative_total, label='Cumulative Total', marker='o', color='b')

            # Add horizontal line at y=0 for reference
            ax.axhline(y=0, color='black', linestyle='--')

            # Add light dotted lines on Y axis
            ax.yaxis.grid(color='gray', linestyle=':', linewidth=0.5)

            # Format X-axis to display every day
            if len(days) == 1:
                ax.set_xticks(days)  # Show only the single day
                ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))  # Format as year-month-day
            else:
                ax.xaxis.set_major_locator(dates.DayLocator(interval=1))  # Display every day
                ax.xaxis.set_major_formatter(dates.DateFormatter('%d'))  # Format as day number

            # Format Y-axis to display dollar sign
            ax.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))

            # Add labels and title
            ax.set_xlabel('Day')
            ax.set_ylabel('Cumulative Total')
            ax.set_title('Cumulative Total Net Finances by Day for Current Month')

            # Add legend
            ax.legend()

            # Set tight layout to prevent clipping of labels
            plt.tight_layout()

            # Create a canvas to display the plot
            canvas = FigureCanvasTkAgg(fig, master=root)
            canvas.draw()

            # Add the canvas to the GUI
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        else:
            # Display a message if there are no values inside the database
            no_values_label = tk.Label(root, text="There are no values inside the database", bg="white",
                                       font=("Helvetica", 12))
            no_values_label.pack(expand=True)
    except Exception as e:
        # Display a message if data retrieval fails
        no_values_label = tk.Label(root, text="Connection Error Check SQL Database", bg="white",
                                   font=("Helvetica", 12))
        no_values_label.pack(expand=True)
        messagebox.showerror("Database Error", "Could not connect to the Database.")


# Function to generate text using OpenAI API
def generative_text(prompt_entry, use_pre_determined_prompt, selected_prompt):
    # load_dotenv(find_dotenv())
    if prompt_entry == "" or prompt_entry.strip(" ") == "":
        messagebox.showerror("Error", "Please input a word inside the text box")
    else:
        EnvFile().create_environment_variables()
        prompt = ""
        # k = os.getenv("SecretKey")
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
        main_folder = os.path.join(desktop_path, 'Book_Buddy')
        generative_text_folder = os.path.join(main_folder, 'Generative_Text')
        model = "gpt-3.5-turbo-instruct"
        try:
            if use_pre_determined_prompt != "custom":
                if selected_prompt == "Twitter":
                    prompt = "Twitter social media post for " + prompt_entry
                elif selected_prompt == "Instagram":
                    prompt = "Instagram social media post for " + prompt_entry
                elif selected_prompt == "Facebook":
                    prompt = "Facebook social media post for " + prompt_entry
            elif use_pre_determined_prompt == "custom":
                prompt = prompt_entry
            # Use instruct Models since they work with Completion Command.
            # Using the OpenAI API v1.0.0 interface
            openai.api_key = os.environ["OpenAIKey"]
            response = openai.Completion.create(
                engine=model,
                # models found here: https://platform.openai.com/docs/models/gpt-3-5-turbo
                prompt=prompt,
                max_tokens=250,  # more tokens means longer posts but also means increase in cost.
                n=1,
                stop=None,
                temperature=0.7,
            )

            generated_text = response.choices[0].text
            # Write generated text to file
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            txt_to_file(generative_text_folder, current_time, generated_text)
        except openai as e:
            # Handle specific error indicating insufficient funds
            if e.code == "insufficient-funds":
                messagebox.showerror("Error", "Insufficient funds. Please add credits to your OpenAI account.")
            else:
                messagebox.showerror("Error", f"OpenAI API Error: {e}")


# Create image from OpenAI
def generative_image(prompt_entry, use_pre_determined_prompt, selected_prompt):
    # load_dotenv(find_dotenv())
    if prompt_entry == "" or prompt_entry.strip(" ") == "":
        messagebox.showerror("Error", "Please input a word inside the text box")
    else:
        EnvFile().create_environment_variables()
        prompt = ""
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
        main_folder = os.path.join(desktop_path, 'Book_Buddy')
        generative_image_folder = os.path.join(main_folder, 'Generative_Image')
        try:
            if use_pre_determined_prompt != "custom" and selected_prompt:
                prompt = f"{selected_prompt} image of " + prompt_entry
            elif use_pre_determined_prompt == "custom":
                prompt = prompt_entry
            openai.api_key = os.environ["OpenAIKey"]
            # a string would work here too
            response = openai.Image.create(
                model="dall-e-3",
                # models can be found at: https://platform.openai.com/docs/models/dall-e
                prompt=prompt,
                size="1024x1024",  # Set the desired size
                # Usable sizes: 1024x1024, 1024x1792, or 1792x1024
                n=1,
            )
            image_url = response.get('data', [])[0].get('url')

            if image_url:
                current_time = datetime.now()
                fileName = current_time.strftime("%Y-%m-%d_%H-%M-%S")
                # Download and save the image directly as a PNG file
                image_path = os.path.join(generative_image_folder, fileName + 'GeneratedImage.png')
                response = requests.get(image_url, stream=True)
                with open(image_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=128):
                        file.write(chunk)

                messagebox.showinfo("Image Saved", f"Image Saved: {image_path}")
            else:
                messagebox.showerror("Error", "Image URL not found in the response.")
        except openai as e:
            # Handle specific error indicating insufficient funds
            if e.code == "insufficient-funds":
                messagebox.showerror("Error", "Insufficient funds. Please add credits to your OpenAI account.")
            else:
                messagebox.showerror("Error", f"OpenAI API Error: {e}")


# Function to create and save a trend image
def create_trend_image(keyword, time_span):
    if keyword == "" or keyword.strip(" ") == "":
        messagebox.showerror("Error", "Please input a word inside the text box")
    else:
        try:
            current_time = datetime.now()
            fileName = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            timeframe = ""
            if time_span == "1 Week":
                timeframe = "now 7-d"
            elif time_span == "1 Month":
                timeframe = "today 1-m"
            elif time_span == "2 Month":
                timeframe = "today 2-m"
            elif time_span == "6 Month":
                timeframe = "today 6-m"

            desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
            main_folder = os.path.join(desktop_path, 'Book_Buddy')
            current_trends_folder = os.path.join(main_folder, 'Current_Trends')
            path_location = current_trends_folder
            EnvFile().create_environment_variables()
            # API used https://serpapi.com/
            Serp = os.environ["SerpAPI"]
            params = {
                "engine": "google_trends",
                "q": keyword,
                "date": timeframe,
                "tz": "-540",
                "data_type": "TIMESERIES",
                "api_key": Serp
            }
            GoogleTrendsDict = {
                'interest_over_time': {},
                'compared_breakdown_by_region': [],
                'interest_by_region': [],
                'related_topics': {},
                'related_queries': {}
            }
            # Make the API request
            search = GoogleSearch(params)
            results = search.get_dict()
            timeseries = []
            interest_over_time = results["interest_over_time"]
            GoogleTrendsDict['interest_over_time'] = interest_over_time
            # Extracting data
            for result in GoogleTrendsDict['interest_over_time']['timeline_data']:
                for value in result['values']:
                    query = value['query']
                    extracted_value = value['extracted_value']

                    timeseries.append({
                        'timestamp': result['timestamp'],
                        'query': query,
                        'extracted_value': extracted_value,
                    })
            # Plotting data
            df = pd.DataFrame(data=timeseries)

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')

            # Drop rows with NaT (Not a Time) values, if any
            df = df.dropna(subset=['timestamp'])

            sns.set(rc={'figure.figsize': (13, 5)})

            ax = sns.lineplot(
                data=df,
                x='timestamp',
                y='extracted_value',
                hue='query',
                color='blue',
            )
            ax.grid(True, which='both', linestyle='--', color='lightgrey')

            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
            ax.set(ylabel='Interest over time')
            ax.set(xlabel='Date')

            # Format x-axis labels with percentage sign
            def format_percent(x, pos):
                return f"%{x:.0f}"

            # Set the frequency of x-axis labels based on the time span
            if time_span == "1 Week":
                ax.xaxis.set_major_locator(dates.DayLocator(interval=1))
            elif time_span == "1 Month":
                ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=dates.MO))
            elif time_span == "2 Month":
                ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=dates.MO))
            elif time_span == "6 Month":
                ax.xaxis.set_major_locator(dates.MonthLocator())

            # Format x-axis labels
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_percent))

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)

            # Format x-axis labels
            ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
            plt.tight_layout()

            # Save the figure as a PNG file in the same location as the text file
            png_file_path = os.path.join(path_location, fileName + 'BookInterest.png')
            plt.savefig(png_file_path)

            messagebox.showinfo("File Saved", f"File Saved: {png_file_path}")

            """
            # Plot using pandas plotting
            plt.figure(figsize=(12, 10))
            data.plot(x='date', y=interest, figsize=(12, 10), title='Keyword Web Search Interest Over Time')
            plt.ylabel('Percentage of interest', fontsize=14)
            plt.xlabel('Days', fontsize=14)


            """
        except KeyError:
            messagebox.showinfo("Check ENV file",
                                "ENV file not setup correctly, check training documentation.")
        except Exception as e:
            messagebox.showinfo("Exception Occurred", f"An exception occurred: {str(e)}")


def delete_transaction(transaction_id):
    # Perform deletion of transaction with given ID
    try:
        EnvFile().create_environment_variables()
        passw = os.environ["PASSWORD"]
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password=passw,
            database="FINANCE"
        )

        cursor = connection.cursor()

        delete_query = f"DELETE FROM `TRANSACTIONS` WHERE `ID` = {transaction_id}"
        cursor.execute(delete_query)
        connection.commit()

        messagebox.showinfo("Deleted", "Transaction deleted successfully.")

        cursor.close()
        connection.close()

    except mysql.connector.Error as e:
        messagebox.showerror("Database Error", f"Failed to connect to the database: {e}")


# Function to handle Trends Graph button click
def trends_graph_clicked(root):
    create_new_dashboard(root, "Trends Graph")


# Function to handle Image Generation button click
def image_gen_clicked(root):
    create_new_dashboard(root, "Image Generation")


# Function to handle Text Generation button click
def text_gen_clicked(root):
    create_new_dashboard(root, "Text Generation")


def sql_input_clicked(root):
    create_new_dashboard(root, "SQL Input")


# Function to handle Text Generation button click
def sql_del_clicked(root):
    create_new_dashboard(root, "SQL Deletion")


def create_sql_input_ui(root):
    clear_frame(root)
    root.geometry("287x299")  # Adjust the size as needed

    # Clear the old dashboard
    # button = tk.Button(root, text="Get Window Size", command=lambda:get_window_size(root))
    # button.pack()
    # Create and configure the hyperlink label
    hyperlink_label = tk.Label(root, text="Back to Financial Database", fg="blue", cursor="hand2")
    hyperlink_label.place(anchor="nw", x=-20, y=-20)  # Adjust x and y coordinates as needed
    hyperlink_label.bind("<Button-1>",
                         lambda event: business_finances_ui(root))  # Bind click event to call main_dashboard

    # Create frame for radio buttons
    radio_frame = tk.Frame(root, bg="white")
    radio_frame.pack(side="top", padx=5, pady=5)

    # Create a shared StringVar for transaction type selection
    transaction_type = tk.StringVar(value="revenue")  # Default to "Revenue"

    # Function to update the style of the selected radio button
    def update_style():
        revenue_radio.config(bg="white", fg="black")
        expenses_radio.config(bg="white", fg="black")
        if transaction_type.get() == "revenue":
            revenue_radio.config(bg='white', fg='black')
        elif transaction_type.get() == "expenses":
            expenses_radio.config(bg='white', fg='black')

    # Create radio button group for transaction type selection
    revenue_radio = tk.Radiobutton(radio_frame, text="Revenue", variable=transaction_type, value="revenue",
                                   bg="white", fg="black", indicatoron=False, command=update_style)
    revenue_radio.pack(side="left", padx=5, pady=5)
    expenses_radio = tk.Radiobutton(radio_frame, text="Expenses", variable=transaction_type, value="expenses",
                                    bg="white", fg="black", indicatoron=False, command=update_style)
    expenses_radio.pack(side="left", padx=5, pady=5)

    # Create and configure entry for transaction amount
    amount_label = tk.Label(root, text="Enter amount:", bg="white", fg="black")
    amount_label.pack(side="top", padx=5, pady=5)
    amount_entry = tk.Entry(root)
    amount_entry.pack(side="top", padx=5, pady=5)

    # Function to validate the input format
    def validate_input():
        try:
            amount = float(amount_entry.get())
            if not 0 <= amount <= 9999999.99:  # Customize the range as needed
                raise ValueError
            else:
                # Connect to MySQL server
                EnvFile().create_environment_variables()
                passw = os.environ["PASSWORD"]
                connection = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password=passw,
                    database="FINANCE"
                )

                cursor = connection.cursor()

                # Get current date
                now = datetime.now()
                current_date = now.strftime('%Y-%m-%d')

                # Get transaction category based on selected radio button
                category = "revenue" if transaction_type.get() == "revenue" else "expense"

                # Insert the validated input into the database
                insert_query = f"""
                    INSERT INTO `TRANSACTIONS` (DATE_INPUTED, AMT, CATEGORY)
                    VALUES ('{current_date}', {amount}, '{category}')
                """
                cursor.execute(insert_query)
                connection.commit()

                cursor.close()
                connection.close()

                messagebox.showinfo("Success", "Transaction successfully added to the database.")
                return True
        except ValueError:
            messagebox.showerror("Error", "Incorrect format. Please input a number with up to 2 decimal places.")
            return False

    # Create and configure "Submit" button
    submit_button = ttk.Button(root, text="Submit", command=lambda: validate_input())
    submit_button.pack(side="bottom", expand=True, fill='both', padx=5, pady=5)


def create_sql_delete_ui(root):
    root.geometry("537x490")  # Adjust the size as needed
    clear_frame(root)

    # Create and configure the hyperlink label
    hyperlink_label = tk.Label(root, text="Back to Financial Database", fg="blue", cursor="hand2")
    hyperlink_label.place(anchor="nw", x=-20, y=-20)  # Adjust x and y coordinates as needed
    hyperlink_label.bind("<Button-1>",
                         lambda event: business_finances_ui(root))  # Bind click event to call main_dashboard

    try:
        # Connect to the database
        EnvFile().create_environment_variables()
        passw = os.environ["PASSWORD"]
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password=passw,
            database="FINANCE"
        )

        cursor = connection.cursor()

        # Fetch transactions ordered by date for the current month
        current_month = datetime.now().month
        select_query = f"SELECT `ID`, `AMT`, `DATE_INPUTED`, `CATEGORY` FROM `TRANSACTIONS` WHERE MONTH(`DATE_INPUTED`) = {current_month} ORDER BY `DATE_INPUTED`"
        cursor.execute(select_query)
        transactions = cursor.fetchall()

        if not transactions:
            # Display message if there are no transactions for the current month
            no_values_label = tk.Label(root, text="No values inside the database", bg="white", font=("Helvetica", 12))
            no_values_label.pack(expand=True)
            messagebox.showinfo("No Values", "There are no values inside the database for this current month.")
        else:
            # Create a frame to hold the list of transactions
            frame = tk.Frame(root, bg="white")
            frame.pack(side="top", fill="both", expand=True, padx=5, pady=5)

            # Create a canvas with scrollbar
            canvas = tk.Canvas(frame, bg="white")
            scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
            transaction_frame = tk.Frame(canvas, bg="white")

            canvas.create_window((0, 0), window=transaction_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            # Display transactions with delete buttons
            current_day = None
            for transaction in transactions:
                transaction_id, amt, date_inputed, category = transaction
                transaction_date = date_inputed.strftime("%Y-%m-%d")

                # Display day if it's a new day
                if transaction_date != current_day:
                    current_day = transaction_date
                    day_label = tk.Label(transaction_frame, text=f"-------Date: {transaction_date}-------", bg="white",
                                         font=("Helvetica", 12))
                    day_label.pack(fill="x", padx=5, pady=(5, 0))

                # Display transaction info and delete button
                category_text = "Expense" if category == "expense" else "Revenue"
                transaction_info = f"Amount: {amt}, Category: {category_text}"

                frame = tk.Frame(transaction_frame, bg="white")
                frame.pack(fill="x", padx=5, pady=5)

                delete_button = tk.Button(frame, text="Delete",
                                          command=lambda id_of_transaction=transaction_id: delete_and_refresh(root, id_of_transaction))
                delete_button.pack(side="left")

                label = tk.Label(frame, text=transaction_info, bg="white", font=("Helvetica", 12))
                label.pack(side="left")

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            def on_configure():
                canvas.configure(scrollregion=canvas.bbox("all"))

            transaction_frame.bind("<Configure>", str(on_configure))

        cursor.close()
        connection.close()

    except mysql.connector.Error as e:
        messagebox.showerror("Database Error", f"Failed to connect to the database: {e}")


def delete_and_refresh(root, transaction_id):
    delete_transaction(transaction_id)
    create_sql_delete_ui(root)


def clear_frame(frame):
    # Clear the frame before adding new widgets
    for widget in frame.winfo_children():
        widget.destroy()


# Function to create UI for image generation
def create_image_generation_ui(root):
    # clears the old dashboard
    root.geometry("300x315")  # Adjust the size as needed

    clear_frame(root)
    # Create and configure the hyperlink label
    hyperlink_label = tk.Label(root, text="Back to AI Generators", fg="blue", cursor="hand2")
    hyperlink_label.place(anchor="nw", x=-20, y=-20)  # Adjust x and y coordinates as needed

    hyperlink_label.bind("<Button-1>", lambda event: ai_content_ui(root))  # Bind click event to call main_dashboard

    radio_frame = tk.Frame(root, bg="white")
    radio_frame.pack(side="top", padx=5, pady=5)

    # Create a shared StringVar for prompt selection
    prompt_selection = tk.StringVar(value="custom")  # Default to "Custom Prompt"

    # Function to update the style of the selected radio button
    def update_style():
        custom_prompt_radio.config(bg="white", fg="black")
        pre_filled_prompt_radio.config(bg="white", fg="black")
        if prompt_selection.get() == "custom":
            custom_prompt_radio.config(bg='white', fg='black')
            current_prompt_label.config(text="Current Prompt: Custom")
        elif prompt_selection.get() == "pre-filled":
            pre_filled_prompt_radio.config(bg='white', fg='black')
            current_prompt_label.config(text="Current Prompt: Pre-filled")

    # Create radio button group for prompt selection
    custom_prompt_radio = tk.Radiobutton(radio_frame, text="Custom Prompt", variable=prompt_selection, value="custom",
                                         bg="white", fg="black", indicatoron=False, command=update_style, name="custom")
    custom_prompt_radio.pack(side="left", padx=5, pady=5)
    pre_filled_prompt_radio = tk.Radiobutton(radio_frame, text="Pre-filled Prompt", variable=prompt_selection,
                                             value="pre-filled", bg="white", fg="black", indicatoron=False,
                                             command=update_style, name="pre-filled")
    pre_filled_prompt_radio.pack(side="left", padx=5, pady=5)

    # Create frame for pre-filled prompt selection
    pre_filled_frame = tk.Frame(root, bg="white")
    pre_filled_frame.pack(side="top", padx=5, pady=5)

    # Create and configure label to display current prompt
    current_prompt_label = tk.Label(root,
                                    text="Current Prompt: " + prompt_selection.get() if prompt_selection.get() else "",
                                    bg="white", fg="black")
    current_prompt_label.pack(side="top", padx=5, pady=5)

    # Create and configure entry for custom prompt
    prompt_label = tk.Label(root, text="Enter prompt:", bg="white", fg="black")
    prompt_label.pack(side="top", padx=5, pady=5)
    prompt_entry = tk.Entry(root)
    prompt_entry.pack(side="top", padx=5, pady=5)

    # Create and configure drop-down list for pre-filled prompts
    pre_filled_prompt_var = tk.StringVar(value="Realistic")
    pre_filled_prompt_label = tk.Label(pre_filled_frame, text="Select pre-filled prompt:", bg="white", fg="black")
    pre_filled_prompt_label.pack(side="left", padx=5, pady=5)
    pre_filled_prompt_options = ["Realistic", "Abstract", "Water Painting", "Realistic"]
    pre_filled_prompt_menu = ttk.OptionMenu(pre_filled_frame, pre_filled_prompt_var, *pre_filled_prompt_options)
    pre_filled_prompt_menu.pack(side="left", padx=5, pady=5)

    # Create and configure "Generate" button
    generate_button = ttk.Button(root, text="Generate",
                                 command=lambda: generative_image(prompt_entry.get(), prompt_selection.get(),
                                                                  pre_filled_prompt_var.get()))
    generate_button.pack(side="bottom", expand=True, fill='both', padx=5, pady=5)


# Function to create UI for Trends Graph
def create_trends_generation_ui(root):
    # clears the old dashboard
    clear_frame(root)
    root.geometry("302x296")  # Adjust the size as needed
    # Create and configure the hyperlink label
    hyperlink_label = tk.Label(root, text="Back to Dashboard", fg="blue", cursor="hand2")
    hyperlink_label.place(anchor="nw", x=-20, y=-20)  # Adjust x and y coordinates as needed

    hyperlink_label.bind("<Button-1>", lambda event: main_dashboard(root))  # Bind click event to call main_dashboard

    prompt_label = tk.Label(root, text="Enter Keyword:", bg="white", fg="black")
    prompt_label.pack(side="top", padx=5, pady=5)
    prompt_entry = tk.Entry(root)
    prompt_entry.pack(side="top", padx=5, pady=5)

    # Create frame for pre-filled prompt selection
    pre_filled_frame = tk.Frame(root, bg="white")
    pre_filled_frame.pack(side="top", padx=5, pady=5)

    # Create and configure label to display current choice for pre-filled prompt
    current_pre_filled_label = tk.Label(root, text="Current Pre-filled Prompt: 1 Month",
                                        bg="white", fg="black")
    current_pre_filled_label.pack(side="top", padx=5, pady=5)

    # Create and configure drop-down list for pre-filled prompts
    pre_filled_prompt_var = tk.StringVar(value="1 Month")
    pre_filled_prompt_options = ["1 Month", "2 Months", "1 Week", "6 months", "1 Month"]

    # Function to update the current choice label
    def update_current_choice(*args):
        current_pre_filled_label.config(text="Current Pre-filled Prompt: " + pre_filled_prompt_var.get())

    pre_filled_prompt_var.trace("w", update_current_choice)  # Add trace to call update_current_choice on write

    pre_filled_prompt_label = tk.Label(pre_filled_frame, text="Select pre-filled prompt:", bg="white", fg="black")
    pre_filled_prompt_label.pack(side="left", padx=5, pady=5)
    pre_filled_prompt_menu = ttk.OptionMenu(pre_filled_frame, pre_filled_prompt_var, *pre_filled_prompt_options)
    pre_filled_prompt_menu.pack(side="left", padx=5, pady=5)

    # Create and configure "Generate" button
    generate_button = ttk.Button(root, text="Generate Graph",
                                 command=lambda: create_trend_image(prompt_entry.get(), pre_filled_prompt_var.get()))
    generate_button.pack(side="bottom", expand=True, fill='both', padx=5, pady=5)


# Function to create UI for text generation
def create_text_generation_ui(root):
    root.geometry("293x315")  # Adjust the size as needed
    clear_frame(root)

    # Create and configure the hyperlink label
    hyperlink_label = tk.Label(root, text="Back to AI Generators", fg="blue", cursor="hand2")
    hyperlink_label.place(anchor="nw", x=-20, y=-20)  # Adjust x and y coordinates as needed

    hyperlink_label.bind("<Button-1>", lambda event: ai_content_ui(root))  # Bind click event to call main_dashboard

    # Create frame for radio buttons
    radio_frame = tk.Frame(root, bg="white")
    radio_frame.pack(side="top", padx=5, pady=5)

    # Create a shared StringVar for prompt selection
    prompt_selection = tk.StringVar(value="custom")  # Default to "Custom Prompt"

    # Function to update the style of the selected radio button
    def update_style():
        custom_prompt_radio.config(bg="white", fg="black")
        pre_filled_prompt_radio.config(bg="white", fg="black")
        if prompt_selection.get() == "custom":
            custom_prompt_radio.config(bg='white', fg='black')
            current_prompt_label.config(text="Current Prompt: Custom")
        elif prompt_selection.get() == "pre-filled":
            pre_filled_prompt_radio.config(bg='white', fg='black')
            current_prompt_label.config(text="Current Prompt: Pre-filled")

    # Create radio button group for prompt selection
    custom_prompt_radio = tk.Radiobutton(radio_frame, text="Custom Prompt", variable=prompt_selection, value="custom",
                                         bg="white", fg="black", indicatoron=False, command=update_style, name="custom")
    custom_prompt_radio.pack(side="left", padx=5, pady=5)
    pre_filled_prompt_radio = tk.Radiobutton(radio_frame, text="Pre-filled Prompt", variable=prompt_selection,
                                             value="pre-filled", bg="white", fg="black", indicatoron=False,
                                             command=update_style, name="pre-filled")
    pre_filled_prompt_radio.pack(side="left", padx=5, pady=5)
    
    # Create frame for pre-filled prompt selection  
    pre_filled_frame = tk.Frame(root, bg="white")
    pre_filled_frame.pack(side="top", padx=5, pady=5)
    # Create and configure label to display current prompt
    current_prompt_label = tk.Label(root,
                                    text="Current Prompt: " + prompt_selection.get() if prompt_selection.get() else "",
                                    bg="white", fg="black")
    current_prompt_label.pack(side="top", padx=5, pady=5)

    # Create and configure entry for custom prompt
    prompt_label = tk.Label(root, text="Enter prompt:", bg="white", fg="black")
    prompt_label.pack(side="top", padx=5, pady=5)
    prompt_entry = tk.Entry(root)
    prompt_entry.pack(side="top", padx=5, pady=5)

    # Create and configure drop-down list for pre-filled prompts
    pre_filled_prompt_var = tk.StringVar(value="Twitter")
    pre_filled_prompt_label = tk.Label(pre_filled_frame, text="Select pre-filled prompt:", bg="white", fg="black")
    pre_filled_prompt_label.pack(side="left", padx=5, pady=5)
    pre_filled_prompt_options = ["Twitter", "Instagram", "Facebook", "Twitter"]
    pre_filled_prompt_menu = ttk.OptionMenu(pre_filled_frame, pre_filled_prompt_var, *pre_filled_prompt_options)
    pre_filled_prompt_menu.pack(side="left", padx=5, pady=5)

    # Create and configure "Generate" button for text generation
    generate_button = ttk.Button(root, text="Generate", command=lambda: generative_text(
        prompt_entry.get(),
        prompt_selection.get(), pre_filled_prompt_var.get()))
    generate_button.pack(side="bottom", expand=True, fill='both', padx=5, pady=5)


def ai_content_ui(root):
    root.geometry("420x146")  # Adjust the size as needed
    clear_frame(root)
    button_frame = tk.Frame(root, bg="white")
    button_frame.pack(anchor="nw", padx=5, pady=5)
    # Create and configure the hyperlink label
    hyperlink_label = tk.Label(root, text="Back to Dashboard", fg="blue", cursor="hand2")
    hyperlink_label.place(anchor="nw", x=-20, y=-20)  # Adjust x and y coordinates as needed

    hyperlink_label.bind("<Button-1>", lambda event: main_dashboard(root))  # Bind click event to call main_dashboard

    image_button = ttk.Button(button_frame, text="Image Generation", command=lambda: image_gen_clicked(root))
    image_button.grid(row=0, column=1, sticky="nw", padx=5, pady=5)

    text_button = ttk.Button(button_frame, text="Text Generation", command=lambda: text_gen_clicked(root))
    text_button.grid(row=0, column=2, sticky="nw", padx=5, pady=5)


def get_finance_values():
    # noinspection PyBroadException
    try:
        # Load environment variables from .env file
        EnvFile().create_environment_variables()
        passw = os.environ["PASSWORD"]
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password=passw,
            database="FINANCE"
        )

        cursor = connection.cursor()

        # Get current month and year
        now = datetime.now()
        current_month = now.month
        current_year = now.year

        # Fetch total expenses
        cursor.execute("""
            SELECT FORMAT(SUM(AMT), 2) FROM TRANSACTIONS WHERE CATEGORY = 'expense'
        """)
        total_expenses = cursor.fetchone()[0]

        # Fetch total revenue
        cursor.execute("""
            SELECT FORMAT(SUM(AMT), 2) FROM TRANSACTIONS WHERE CATEGORY = 'revenue'
        """)
        total_revenue = cursor.fetchone()[0]

        # Fetch monthly expenses
        cursor.execute(f"""
            SELECT FORMAT(SUM(AMT), 2) FROM TRANSACTIONS
            WHERE CATEGORY = 'expense' AND MONTH(DATE_INPUTED) = {current_month} AND YEAR(DATE_INPUTED) = {current_year}
        """)
        monthly_expenses = cursor.fetchone()[0]

        # Fetch monthly revenue
        cursor.execute(f"""
            SELECT FORMAT(SUM(AMT), 2) FROM TRANSACTIONS
            WHERE CATEGORY = 'revenue' AND MONTH(DATE_INPUTED) = {current_month} AND YEAR(DATE_INPUTED) = {current_year}
        """)
        monthly_revenue = cursor.fetchone()[0]

        # Fetch entire database total
        cursor.execute("""
            SELECT FORMAT(SUM(AMT), 2) FROM TRANSACTIONS
        """)
        entire_total = cursor.fetchone()[0]

        cursor.close()
        connection.close()

        return entire_total, monthly_revenue, total_revenue, monthly_expenses, total_expenses

    except mysql.connector.Error as e:
        messagebox.showerror("Database Error", f"Failed to connect to the database: {e}")
        return 0, 0, 0, 0, 0
    except:
        return 0, 0, 0, 0, 0


def format_currency(value):
    # Format currency with commas for thousands
    return '{:,.2f}'.format(value)


# debugcode for window size
def get_window_size(root):
    width = root.winfo_width()
    height = root.winfo_height()
    print("Window size: {}x{}".format(width, height))


# noinspection PyBroadException
def business_finances_ui(root):
    root.geometry("699x263")  # Adjust the size as needed
    try:
        initialize_database()
        clear_frame(root)
        total, m_revenue, t_revenue, m_expenses, t_expenses = get_finance_values()
        button_frame = tk.Frame(root, bg="white")
        button_frame.pack(anchor="nw", padx=5, pady=5)

        # Create and configure the hyperlink label
        hyperlink_label = tk.Label(root, text="Back to Dashboard", fg="blue", cursor="hand2")
        hyperlink_label.place(anchor="nw", x=-20, y=-20)  # Adjust x and y coordinates as needed

        # Total Label
        total_label = tk.Label(root, text="Total: $" + str(total), fg="black", bg="white", cursor="hand2",
                               font=("Helvetica", 12, "bold"))
        total_label.place(anchor="nw", x=250, y=200)  # Adjust x and y coordinates as needed

        # Monthly Revenue Label
        monthly_revenue_label = tk.Label(root, text="Monthly Revenue: $" + str(m_revenue), bg="white", fg="black",
                                         cursor="hand2")
        monthly_revenue_label.place(anchor="nw", x=420, y=140)  # Adjust x and y coordinates as needed
        # Monthly Expenses Label
        monthly_expenses_label = tk.Label(root, text="Monthly Expenses: $" + str(m_expenses), bg="white", fg="black",
                                          cursor="hand2")
        monthly_expenses_label.place(anchor="nw", x=420, y=90)  # Adjust x and y coordinates as needed
        # Total Expenses Label
        total_expenses_label = tk.Label(root, text="Total Expenses: $" + str(t_expenses), bg="white", fg="black",
                                        cursor="hand2")
        total_expenses_label.place(anchor="nw", x=70, y=90)  # Adjust x and y coordinates as needed
        # Total Revenue Label
        total_revenue_label = tk.Label(root, text="Total Revenue: $" + str(t_revenue), bg="white", fg="black",
                                       cursor="hand2")
        total_revenue_label.place(anchor="nw", x=70, y=140)  # Adjust x and y coordinates as needed
        hyperlink_label.bind("<Button-1>",
                             lambda event: main_dashboard(root))  # Bind click event to call main_dashboard

        finances_input_button = ttk.Button(button_frame, text="Input Finances", command=lambda: sql_input_clicked(root))
        finances_input_button.grid(row=0, column=3, sticky="nw", padx=5, pady=5)

        finances_delete_button = ttk.Button(button_frame, text="Delete Finances", command=lambda: sql_del_clicked(root))
        finances_delete_button.grid(row=0, column=4, sticky="nw", padx=5, pady=5)

        finances_plot_button = ttk.Button(button_frame, text="Plot Finances",
                                          command=lambda: plot_total_finances(root))
        # button = tk.Button(root, text="Get Window Size", command=lambda:get_window_size(root))
        # button.pack()

        finances_plot_button.grid(row=0, column=5, sticky="nw", padx=5, pady=5)
        for button in [finances_input_button, finances_delete_button, finances_plot_button]:
            button.config(width=17)

    except mysql.connector.Error:
        messagebox.showerror("Database Error", "Failed to connect to the database: Please Check your password.")
        main_dashboard(root)
    except:
        messagebox.showerror("Database Error", "ENV file is not setup correctly, refer to training documentation")
        main_dashboard(root)


def main_dashboard(root):
    root.geometry("665x146")  # Adjust the size as needed
    clear_frame(root)
    # Create a frame to contain the buttons
    button_frame = tk.Frame(root, bg="white")
    button_frame.pack(anchor="nw", padx=5, pady=5)
    # Create and configure buttons for other functionalities
    AI_Content_Generator_button = ttk.Button(button_frame, text="AI Content Generator",
                                             command=lambda: ai_content_ui(root))
    AI_Content_Generator_button.grid(row=0, column=0, sticky="nw", padx=5, pady=5)

    # Create and configure buttons for other functionalities
    trends_button = ttk.Button(button_frame, text="Web Scraper", command=lambda: trends_graph_clicked(root))
    trends_button.grid(row=0, column=1, sticky="nw", padx=5, pady=5)

    Business_Finance_Tracker_button = ttk.Button(button_frame, text="Business Finance Tracker",
                                                 command=lambda: business_finances_ui(root))
    Business_Finance_Tracker_button.grid(row=0, column=2, sticky="nw", padx=5, pady=5)


"""
    # Create and configure buttons for other functionalities
    trends_button = ttk.Button(button_frame, text="Trends Graph", command=lambda: trends_graph_clicked(root))
    trends_button.grid(row=0, column=0, sticky="nw", padx=5, pady=5)

    image_button = ttk.Button(button_frame, text="Image Generation", command=lambda: image_gen_clicked(root))
    image_button.grid(row=0, column=1, sticky="nw", padx=5, pady=5)

    text_button = ttk.Button(button_frame, text="Text Generation", command=lambda: text_gen_clicked(root))
    text_button.grid(row=0, column=2, sticky="nw", padx=5, pady=5)

    finances_input_button = ttk.Button(button_frame, text="Input Finances", command=lambda: sql_input_clicked(root))
    finances_input_button.grid(row=0, column=3, sticky="nw", padx=5, pady=5)

    finances_delete_button = ttk.Button(button_frame, text="Delete Finances", command=lambda: sql_del_clicked(root))
    finances_delete_button.grid(row=0, column=4, sticky="nw", padx=5, pady=5)

    finances_plot_button = ttk.Button(button_frame, text="Plot Revenue vs Expenses", command=lambda: plot_total_finances(root))
    finances_plot_button.grid(row=0, column=5, sticky="nw", padx=5, pady=5)
    """


# Function to handle GUI inputs and run the main loop
def main():
    # Start GUI
    get_desktop_path()
    root = tk.Tk()
    root.title("Book Buddy")
    root.configure(bg="white", padx=30, pady=30)  # Set background color to white

    # Define styles for the buttons
    style = ttk.Style()
    style.configure('TButton', font=('Arial', 12), foreground="black", background='white',
                    padding=20, )  # Black foreground
    main_dashboard(root)

    # Function to handle closing of the main dashboard
    def on_close():
        root.quit()  # Quit the main loop when the main dashboard is closed

    # Bind the closing event of the root window to the on_close function
    root.protocol("WM_DELETE_WINDOW", on_close)

    # Run the GUI
    root.mainloop()


# Run the main function if the script is executed
if __name__ == "__main__":
    main()