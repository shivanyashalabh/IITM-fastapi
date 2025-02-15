import os
import subprocess
import json
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import shutil
import base64
import sqlite3
import speech_recognition as sr
import pandas as pd
from io import BytesIO
from PIL import Image
import librosa
import soundfile as sf
import markdown
import sys
from pathlib import Path

tasks = """A1. Install uv (if required) and run https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py with ${user.email} as the only argument. (NOTE: This will generate data files required for the next tasks.) (MAKE SURE TO SEND IN THE USER EMAIL WITH TASK CODE LIKE A1 ABC@XYZ.com)
A2. Format the contents of /data/format.md using prettier@3.4.2, updating the file in-place
A3. The file /data/dates.txt contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to /data/dates-wednesdays.txt
A4. Sort the array of contacts in /data/contacts.json by last_name, then first_name, and write the result to /data/contacts-sorted.json
A5. Write the first line of the 10 most recent .log file in /data/logs/ to /data/logs-recent.txt, most recent first
A6. Find all Markdown (.md) files in /data/docs/. For each file, extract the first occurrance of each H1 (i.e. a line starting with # ). Create an index file /data/docs/index.json that maps each filename (without the /data/docs/ prefix) to its title (e.g. {"README.md": "Home", "path/to/large-language-models.md": "Large Language Models", ...})
A7. /data/email.txt contains an email message. Pass the content to an LLM with instructions to extract the sender’s email address, and write just the email address to /data/email-sender.txt
A8. /data/credit-card.png contains a credit card number. Pass the image to an LLM, have it extract the card number, and write it without spaces to /data/credit-card.txt
A9. /data/comments.txt contains a list of comments, one per line. Using embeddings, find the most similar pair of comments and write them to /data/comments-similar.txt, one per line
A10. The SQLite database file /data/ticket-sales.db has a tickets with columns type, units, and price. Each row is a customer bid for a concert ticket. What is the total sales of all the items in the “Gold” ticket type? Write the number in /data/ticket-sales-gold.txt
Developers will call the /run?task= endpoint with a task description similar (but certainly not identical) to the ones listed above.
TAKS B AND TASK A1 MIGHT REQUIRE ADDITIONAL INFO WITH RESPONSE, INCLUDE IT AFTER A SPACE. FOR EXAMPLE "B3 {URL}"
B3. Fetch data from an API and save it
B4. Clone a git repo and make a commit
B5. Run a SQL query on a SQLite or DuckDB database (example "B5 SELECT * FROM tickets WHERE type = 'Gold'")
B6. Extract data from (i.e. scrape) a website
B7. Compress or resize an image (Include dimensions, if provided, in the response in JSON format, with property names in double quotes, like "B7 {URL} Dimensions:{"height": 100, "width": 100}")
B8. Transcribe audio from an MP3 file
B9. Convert Markdown to HTML
B10. Write an API endpoint that filters a CSV file and returns JSON data
"""

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")


def execute_function(task):
    # since the task is in enlgish, we will use the GPT-4o-Mini model to convert the task into code
    task_code = task_translation(task)
    print(f"Task code: {task_code}")
    # task_code = 'B9 https://gotranscript.com/audios/practice/transcribing_1.mp3'
    if task_code.startswith("B"):
        if task_code.startswith("B5"):
            query = task_code[task_code.find("SELECT"):]
            return task_list[task_code.split(" ")[0]](query)
        URL = task_code.split(" ")[1]
        if task_code.startswith("B7"):
            dimensions = json.loads(task_code.split("Dimensions:")[1])
            task_code = task_code.split(" ")[0]
            task_code = task_code.strip()
            return task_list[task_code](URL, dimensions.get("height"), dimensions.get("width"))
        task_code = task_code.split(" ")[0]
        return task_list[task_code](URL)
    if task_code.startswith("A1") and not task_code.startswith("A10"):
        user_email = task_code.split(" ")[1]
        print(f"{user_email=}")
        task_code = task_code.split(" ")[0]
        return task_list[task_code](user_email)
    if task_code in task_list:
        task_list[task_code]()
    else:
        raise ValueError("Invalid task")
    return


def task_translation(task):
    print("Translating task...")
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "understand natural language and choose most suitable function to be called from the function list... " + tasks},
            {"role": "user", "content": task}
        ],
        "max_tokens": 200
    }

    try:
        # Add timeout to ensure request waits
        response = requests.post(url, headers=headers, json=data, timeout=30)

        # Check for HTTP errors
        response.raise_for_status()

        # Parse JSON response
        result = response.json()
        print("API Response:", result)  # Debugging line to check full response

        return result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

    except requests.Timeout:
        print("Request timed out. Try increasing timeout or checking API response time.")
        return "Error: Request timed out"

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return f"Error: {e}"

    except Exception as e:
        print(f"Unexpected error: {e}")
        return f"Error: {e}"


def install_and_run_datagen(user_email):
    # Check if `uv` is installed; if not, install it
    print("Checking if uv is installed...")
    if not shutil.which("uv"):
        print("Installing uv...")
        subprocess.run([sys.executable, "-m", "pip",
                       "install", "uv"], check=True)

    # Download `datagen.py`
    url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    script_path = "/tmp/datagen.py"  # Temporary location to store the script

    response = requests.get(url)
    if response.status_code == 200:
        with open(script_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded datagen.py to {script_path}")
    else:
        print("Failed to download datagen.py")
        return

    # Run `datagen.py` with the provided email
    print(f"Running datagen.py with argument: {user_email}")
    subprocess.run([sys.executable, script_path, user_email], check=True)

    # Cleanup (optional)
    os.remove(script_path)
    print("Script execution completed and cleaned up.")
# ------------------------------------------------------
# A2: Done


def format_markdown():
    """Format the contents of /data/format.md using Prettier."""
    file_path = "/data/format.md"
    try:
        subprocess.run(
            ["npx", "prettier@3.4.2", "--write", file_path], check=True)
        print(f"Formatted {file_path} successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error formatting {file_path}: {e}")

# ------------------------------------------------------
# A3: Done


def count_wednesdays():
    with open("/data/dates.txt", "r") as file:
        lines = file.readlines()

    # List of possible date formats
    date_formats = [
        "%d-%b-%Y",
        "%Y-%m-%d",
        "%b %d, %Y",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%d/%m/%Y",
    ]

    wednesday_count = 0

    for line in lines:
        date_str = line.strip()
        for fmt in date_formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                if date_obj.weekday() == 2:  # Wednesday is represented by 2
                    wednesday_count += 1
                break  # Stop checking formats if successfully parsed
            except ValueError:
                continue  # Try the next format
    with open("/data/dates-wednesdays.txt", "w") as file:
        file.write(str(wednesday_count))

    return
# -----------------------------------------------------
# A4: Done


def sort_contacts():
    with open("/data/contacts.json", "r") as f:
        contacts = json.load(f)
    contacts.sort(key=lambda x: (x["last_name"], x["first_name"]))
    with open("/data/contacts-sorted.json", "w") as f:
        f.write(json.dumps(contacts))
    return
# -----------------------------------------------------
# A5: Done


def get_last_modified_files(directory: str, count: int = 10):
    """
    Returns the 'count' most recently modified files in the given directory.

    :param directory: Path to the directory.
    :param count: Number of files to return (default is 10).
    :return: List of file paths sorted by modification time (most recent first).
    """
    try:
        # Convert to Path object for flexibility
        directory_path = Path(directory)

        if not directory_path.is_dir():
            raise ValueError(f"Invalid directory: {directory}")

        # Get list of all files (ignoring subdirectories)
        files = [f for f in directory_path.iterdir() if f.is_file()]

        # Sort by last modified time (most recent first)
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Return the top 'count' files
        return [str(f) for f in files[:count]]

    except Exception as e:
        print(f"Error: {e}")
        return []


def write_recent_logs():
    # find 10 most recent log files in /data/logs/ using their metadata and write the first line of each to /data/logs-recent.txt
    # they are not correclt named so 21 might me the most recent log and 40 could be the oldest
    log_files = os.listdir("/data/logs/")

    with open("/data/logs-recent.txt", "w") as f:
        ten_most_recent_files = get_last_modified_files(
            "/data/logs/", count=10)
        for filename in ten_most_recent_files:
            with open(filename, "r") as log_file:
                first_line = log_file.readline()
                f.write(first_line)
    return
# -----------------------------------------------------
# A6: Done


def extract_h1():
    # Find all Markdown (.md) files in /data/docs/. For each file, extract the first occurrance of each H1 (i.e. a line starting with # ). Create an index file /data/docs/index.json that maps each filename (without the /data/docs/ prefix) to its title (e.g. {"README.md": "Home", "path/to/large-language-models.md": "Large Language Models", ...})
    # note that docs folder has folders within it that contain markdown files
    h1_index = {}
    for root, _, files in os.walk("/data/docs/"):
        for file in files:
            if file.endswith(".md"):
                with open(os.path.join(root, file), "r") as f:
                    for line in f:
                        if line.startswith("# "):
                            h1_index[os.path.relpath(os.path.join(
                                root, file), "/data/docs/")] = line[2:].strip()
                            break
    with open("/data/docs/index.json", "w") as f:
        json.dump(h1_index, f)
    return
# -----------------------------------------------------
# A7: Done


def extract_email():
    # /data/email.txt contains an email message. Pass the content to an LLM with instructions to extract the sender’s email address, and write just the email address to /data/email-sender.txt
    with open("/data/email.txt", "r") as f:
        email = f.read()
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Extract the sender's email address from the given email message and give me the output of JUST THE SENDERS EMAIL ADDRESS"},
            {"role": "user", "content": email}
        ],
        "max_tokens": 100
    }
    response = requests.post(url, headers=headers, json=data)
    print(response.json())
    email_sender = response.json()["choices"][0]["message"]["content"].strip()
    with open("/data/email-sender.txt", "w") as f:
        f.write(email_sender)
    return
# -----------------------------------------------------
# A8: Done


def extract_credit_card():
    # Read the image as binary
    with open("/data/credit_card.png", "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")

    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "I am working on an OCR-based assignment where I need to extract text from an image. Please analyze the text in this image and return any numeric sequences found, preserving the original order and format."},
            {"role": "user", "content": f"data:image/png;base64,{image_data}"}
        ],
        "max_tokens": 100
    }

    response = requests.post(url, headers=headers, json=data)

    try:
        response_json = response.json()
        print(response_json)  # Debugging line

        credit_card_number = response_json["choices"][0]["message"]["content"].strip(
        ).replace(" ", "")
        with open("/data/credit-card.txt", "w") as f:
            f.write(credit_card_number)

    except json.JSONDecodeError:
        print("Error: Could not decode JSON response.")
    except KeyError:
        print("Error: Unexpected response format.", response_json)

# -----------------------------------------------------

# A9: Done


def similar_comments():
   # contains a list of comments, one per line. Using embeddings, find the most similar pair of comments and write them to /data/comments-similar.txt, one per line
   # do this without using openai but you can use any other library but try to keep it lightweeight you can use scikit-learn
   # Got this error: Internal Server Error: np.matrix is not supported. Please convert to a numpy array with np.asarray. For more information see:
    with open("/data/comments.txt", "r") as f:
        comments = f.readlines()
    # use chatgpt for it
    content = " ".join(comments)
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system",
                "content": "Find the most similar pair (two comments) of comments from the given list. JUST OUTPUT THE TWO COMMENTS."},
            {"role": "user", "content": content}
        ],
        "max_tokens": 100
    }
    response = requests.post(url, headers=headers, json=data)
    most_similar_comments = response.json(
    )["choices"][0]["message"]["content"].strip().split("\n")

    with open("/data/comments-similar.txt", "w") as f:
        f.write("\n".join(most_similar_comments))
    return
# -----------------------------------------------------
# A10: Done


def total_sales():
    conn = sqlite3.connect("/data/ticket-sales.db")
    c = conn.cursor()
    c.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
    total = c.fetchone()[0]
    with open("/data/ticket-sales-gold.txt", "w") as f:
        f.write(str(total))
    return


# -----------------------------------------------------
"""Starting B tasks
B3. Fetch data from an API and save it
B4. Clone a git repo and make a commit
B5. Run a SQL query on a SQLite or DuckDB database
B6. Extract data from (i.e. scrape) a website
B7. Compress or resize an image
B8. Transcribe audio from an MP3 file
B9. Convert Markdown to HTML
B10. Write an API endpoint that filters a CSV file and returns JSON data
"""
# -----------------------------------------------------
# B3: Done


def fetch_data_from_api(url):
    print(url)
    response = requests.get(url)
    data = response.json()
    with open("/data/api_data.json", "w") as f:
        json.dump(data, f)
    return data
# -----------------------------------------------------
# B4: Done


def clone_git_repo(url):
    # Clone a git repo and make a commit
    subprocess.run(["git", "clone", url])
    # Make a commit
    subprocess.run(["git", "commit", "-m", "Initial commit"])
    return
# -----------------------------------------------------
# B5: Done


def run_sql_query(query):
    # Run a SQL query on a SQLite or DuckDB database
    conn = sqlite3.connect("/data/ticket-sales.db")
    c = conn.cursor()
    c.execute(query)
    result = c.fetchall()
    with open("/data/query_result.json", "w") as f:
        json.dump(result, f)
    return json.dumps(result)

# -----------------------------------------------------
# B6: Done
# B6. Extract data from (i.e. scrape) a website using BeautifulSoup


def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    with open("/data/website_data.txt", "w") as f:
        f.write(soup.prettify())
    return soup.prettify()
# -----------------------------------------------------
# B7: Done
# B7. Compress/ resize an image


def compress_image(image_path, height=None, width=None):
    isURL = image_path.startswith("http")
    if isURL:
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    if height and width:
        image = image.resize((width, height))
    else:
        image = image.resize((100, 100))
    image.save("/data/compressed_image.jpg")
    return
# -----------------------------------------------------
# B8: Done


def transcribe_audio(audio_path):
    is_url = audio_path.startswith("http")

    # Load audio from URL or file
    if is_url:
        response = requests.get(audio_path)
        audio_data, sample_rate = librosa.load(
            BytesIO(response.content), sr=16000)
    else:
        audio_data, sample_rate = librosa.load(audio_path, sr=16000)

    # Convert to WAV
    wav_buffer = BytesIO()
    sf.write(wav_buffer, audio_data, sample_rate, format="WAV")
    wav_buffer.seek(0)

    # Debugging: Ensure sr is the correct module
    print(type(sr))  # Should be <module 'speech_recognition'>

    # Transcribe using speech_recognition
    recognizer = sr.Recognizer()  # ✅ Use a new variable to avoid conflicts
    with sr.AudioFile(wav_buffer) as source:
        audio = recognizer.record(source)
    text = recognizer.recognize_google(audio)

    # Save transcription
    with open("/data/transcribed_text.txt", "w") as f:
        f.write(text)

    return text

# -----------------------------------------------------
# B9: Done


def convert_markdown_to_html(markdown_path):
    # Convert Markdown to HTML
    isURL = markdown_path.startswith("http")
    if isURL:
        response = requests.get(markdown_path)
        markdown_text = response.text
    else:
        with open(markdown_path, "r") as f:
            markdown_text = f.read()
    html = markdown.markdown(markdown_text)
    with open("/data/converted_html.html", "w") as f:
        f.write(html)
    return html

# -----------------------------------------------------
# B10: Done


def filter_csv_and_return_json(csv_path, filter_column, filter_value):
    # Write an API endpoint that filters a CSV file and returns JSON data
    isURL = csv_path.startswith("http")
    if isURL:
        response = requests.get(csv_path)
        csv = response.text
    else:
        with open(csv_path, "r") as f:
            csv = f.read()
    df = pd.read_csv(csv)
    filtered_data = df[df[filter_column] ==
                       filter_value].to_dict(orient="records")
    with open("/data/filtered_data.json", "w") as f:
        json.dump(filtered_data, f)
    return json.dumps(filtered_data)


task_list = {
    "A1": install_and_run_datagen,
    "A2": format_markdown,
    "A3": count_wednesdays,
    "A4": sort_contacts,
    "A5": write_recent_logs,
    "A6": extract_h1,
    "A7": extract_email,
    "A8": extract_credit_card,
    "A9": similar_comments,
    "A10": total_sales,
    "B3": fetch_data_from_api,
    "B4": clone_git_repo,
    "B5": run_sql_query,
    "B6": scrape_website,
    "B7": compress_image,
    "B8": transcribe_audio,
    "B9": convert_markdown_to_html,
    "B10": filter_csv_and_return_json
}
