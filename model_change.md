# Changing the LLM Backend to Ollama (Phi3)

Agar aapko aage chalkar Groq se wapas **Ollama (Phi3)** par switch karna ho, toh aapko sirf apni `.env` file mein kuch minor changes karne honge. 

Uske baad aapka Chatbot offline waale local model ka use karne lagega.

Neeche diye gaye steps ko follow karein:

### Step 1: Open the `.env` file
Apne project folder mein `.env` file ko open karein.

### Step 2: Update the LLM_BACKEND
Line 2 par, `LLM_BACKEND` ki value `groq` se hata kar wapas `ollama` kar dein:

```diff
- LLM_BACKEND=groq
+ LLM_BACKEND=ollama
```

### Step 3: Verify the Ollama Model Name
Line 5 par check karein ki aapka `OLLAMA_MODEL` sahi set hai (by default `phi3` tha):

```ini
OLLAMA_MODEL=phi3
OLLAMA_HOST=http://localhost:11434
```

> [!NOTE]
> Agar aap Ollama me Llama3 ya Mistral use karna chahte hain, toh bas `OLLAMA_MODEL=mistral` ya `llama3` set kar sakte hain (Lekin uske liye aapko terminal me pehle `ollama pull mistral` run karna hoga).

### Step 4: Make Sure Ollama is Running
Kyunki Ollama ek local tool hai, app ko run karne se pehle make sure aapke system me Ollama background me chal raha ho. 
Agar wo on nahi hai toh terminal me command chala kar check karein:
```bash
ollama serve
```

### Step 5: Restart the Streamlit App
Agar aapka Streamlit server pehle se chal raha tha, toh use terminal me `Ctrl + C` dabakar band karein. 
Fir dobara se run karein:
```bash
streamlit run app.py
```

Bas itna hi karna hai! Ab aapka chatbot wapas se local Phi3 model se answer generate karne lagega.

---

# Setting up Mistral via Ollama

Agar aapko **Mistral** model use karna hai (jo ki assignment ka default tha), toh aapko in additional steps ko follow karna hoga:

### Step 1: Download the Mistral Model
Mistral model ko apne PC mein pehli baar download karne ke liye terminal me yeh command run karein:
```bash
ollama pull mistral
```
*(Yeh model thoda bada hota hai (~4GB), aaram se download hone dein).*

### Step 2: Update `.env` to use Mistral
Apne `.env` file mein `OLLAMA_MODEL` ko change karke `mistral` kar dein:

```diff
- OLLAMA_MODEL=phi3
+ OLLAMA_MODEL=mistral
```
(Sath me make sure karein ki `LLM_BACKEND=ollama` ho).

### Step 3: Run the Server
Phir se apna streamlit app check kar lein. Model ab Groq ya Phi3 ki jagah locally `mistral` ko point karega!
```bash
streamlit run app.py
```
