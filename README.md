# Medical-Chatbot-using-Llama2

## Demo Of The Bot
<a href="https://drive.google.com/file/d/1G_dunG0WVngfArAJYvlpYziVhz0I19k_/view?usp=sharing">Click Here</a>


## Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- Pinecone

## STEPS:

Clone the repository

```bash
Project repo: https://github.com/apk471/ChatBot
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -p venv python==3.12 -y
```

```bash
conda activate venv
```

### STEP 02- install the requirements

```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Download the quantize model from the link provided in model folder & keep the model in the model directory:

```ini
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```

```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,

```bash
open up localhost:
```
