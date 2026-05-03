tep 1 — Python environment (run these in your terminal)                                                                                                                                    
                                                                                                                                                                                              
  cd ~/Documents/projects/roadmap-llm-systems-engineering                                                                                                                                     
  python3 -m venv .venv                                                                                                                                                                       
  source .venv/bin/activate                                                                                                                                                                   
  pip install transformers tiktoken datasets torch jupyter pandas                                                                                                                             
                                                                                                                                                                                              
  ---
  Step 2 — Get the Kaggle dataset
                                                                                                                                                                                              
  1. Go to https://www.kaggle.com/datasets/kanchana1990/uae-real-estate-2024-dataset
  2. Download the CSV manually and put it in layers/layer_01/data/ (the data/ folder is gitignored so it won't be committed)                                                                  
                                                                                                                                                                                              
  ---                                                                                                                                                                                         
  Step 3 — Exercise 1.1: Tokenizer comparison notebook                                                                                                                                        
                                                      
  Create layers/layer_01/tokenizer_comparison.ipynb. In it:
  - Load the CSV, pick 100 rows that have both an English and Arabic description                                                                                                              
  - Tokenize each with 4 tokenizers: tiktoken (gpt-4o), cl100k_base, bert-base-multilingual-cased, and AraGPT2 or aubmindlab/bert-base-arabertv2                                              
  - Count tokens per text, build a comparison table, compute cost estimates at $0.0001/1k tokens                                                                                              
                                                                                                                                                                                              
  ---                                                                                                                                                                                         
  Step 4 — Exercises 1.2 & 1.3: Ollama experiments                                                                                                                                            
                                                                                                                                                                                              
  You already have deepseek-r1:1.5b — use that instead of qwen2.5.
                                                                                                                                                                                              
  Temperature (1.2): Run the same property description prompt 4 times each at temperature=0, 0.3, 0.8, 1.2 using the Ollama REST API (POST localhost:11434/api/generate) and note how outputs 
  differ.                                                                                                                                                                                     
                                                                                                                                                                                              
  TTFT (1.3): Send prompts of ~200, ~2000, and ~10000 tokens. Record time-to-first-token for each. You'll see it grow as context grows.                                                       
   
  import requests, time                                                                                                                                                                       
                                                                                                                                                                                              
  def ttft(prompt):
      start = time.time()                                                                                                                                                                     
      resp = requests.post("http://localhost:11434/api/generate",
          json={"model": "deepseek-r1:1.5b", "prompt": prompt, "stream": True},                                                                                                               
          stream=True)
      for chunk in resp.iter_lines():                                                                                                                                                         
          first_token_time = time.time() - start                                                                                                                                              
          break
      return first_token_time                                                                                                                                                                 
                  
  ---
  Step 5 — Tiny transformer
                                                                                                                                                                                              
  Create layers/layer_01/tiny_transformer.py. Use Karpathy's minGPT as reference (https://github.com/karpathy/minGPT). Train a character-level or word-level GPT on 1000 English property
  descriptions from the CSV. It doesn't need to be good — it needs to train and generate text.                                                                                                
                  
  ---                                                                                                                                                                                         
  Step 6 — COMPLETION.md
                        
  Copy templates/layer_completion_template.md to layers/layer_01/COMPLETION.md and answer the 3 gate questions in your own words.
                                                                                                                                                                                              
  ---



okay lets start i have deepseek on my local machine


🤖start🤖

Nice, you're all set with deepseek! Let's work through this layer step by step.

**Step 1** is setting up your Python environment. Have you already created the venv and installed the packages (`transformers tiktoken datasets torch jupyter pandas`)? Or do you need to do that first?

Let's make sure the foundation is solid before we move on to the dataset and exercises.

🤖end🤖

why are we using ollama 

🤖start🤖

Great question! Ollama lets you run LLMs **locally on your own machine** — which is exactly what you need for these exercises because:

1. **Temperature experiment (1.2)** — you need to tweak generation parameters like temperature and compare outputs. With a local model you have full control and zero cost per call.
2. **TTFT experiment (1.3)** — you're measuring *time-to-first-token* latency at different prompt sizes. That measurement only makes sense on a local setup where network latency isn't a confounding factor.

Since you already have `deepseek-r1:1.5b` running locally via Ollama, you're good to go for both exercises. You just hit its REST API at `localhost:11434`.

Does that make sense? Ready to tackle Step 1 (the venv setup)?

🤖end🤖

why not llm api


🤖start🤖

Good thinking — you *could* use a cloud API (OpenAI, Anthropic, etc.), but for these specific exercises local has advantages:

- **Free** — no API credits needed. You'll be making dozens of calls across the temperature and TTFT experiments.
- **TTFT measurement** — with a cloud API, your "time to first token" includes network round-trip time, which swamps the actual model processing time you're trying to observe. Locally, you isolate the *model's* latency.
- **Learning goal** — part of this layer is understanding what happens *inside* the inference pipeline. Running it locally makes that more tangible.

In later layers you'll likely use cloud APIs too, but for now local gives you cleaner data and more control.

Shall we get your environment set up?

🤖end🤖

okay what is next


🤖start🤖

Step 1 — set up your Python environment. Run these in your terminal:

```bash
cd ~/Documents/projects/roadmap-llm-systems-engineering
python3 -m venv .venv
source .venv/bin/activate
pip install transformers tiktoken datasets torch jupyter pandas
```

Let me know when that's done (or if you hit any errors) and we'll move to grabbing the dataset.

🤖end🤖

okay

🤖start🤖

Great! Next is **Step 2** — get the dataset.

1. Go to [https://www.kaggle.com/datasets/kanchana1990/uae-real-estate-2024-dataset](https://www.kaggle.com/datasets/kanchana1990/uae-real-estate-2024-dataset)
2. Download the CSV
3. Put it in `layers/layer_01/data/`

Let me know once you've got it!

🤖end🤖

okay done

🤖start🤖

Nice! Now **Step 3** — the tokenizer comparison notebook.

Create `layers/layer_01/tokenizer_comparison.ipynb` and start with the first piece: loading the CSV and picking 100 rows that have both an English and Arabic description.

Can you take a look at the CSV first (load it with pandas) and figure out which columns contain the English and Arabic descriptions? That'll tell us how to filter the rows.

🤖end🤖

about the librariyts we installed what are they and why are we using thme


🤖start🤖

Great question! Here's what each one does and why you need it:

- **`tiktoken`** — OpenAI's fast tokenizer library. You'll use it to tokenize text with GPT-4o and cl100k_base tokenizers in the comparison exercise.

- **`transformers`** — Hugging Face's library for loading pre-trained models and tokenizers. You need it for the BERT multilingual and Arabic tokenizers (bert-base-multilingual-cased, AraGPT2/AraBERT).

- **`datasets`** — Hugging Face's data loading library. Handy for working with datasets, though here you'll mainly use pandas for the CSV.

- **`torch`** — PyTorch, the deep learning framework. Needed by `transformers` under the hood, and you'll use it directly for the tiny transformer in Step 5.

- **`jupyter`** — So you can create and run the `.ipynb` notebooks for the exercises.

- **`pandas`** — For loading and manipulating the CSV data (filtering rows, building comparison tables, etc.).

Make sense? Ready to look at that CSV?

🤖end🤖

does jupyter take time to Download


🤖start🤖

It can take a minute or two since it pulls in quite a few dependencies, but it shouldn't be too bad. If it's still installing, just let it finish — the rest of the packages are smaller.

Did the `pip install` complete OK, or is it still running?

🤖end🤖

i have added the csv how should do that


🤖start🤖

First, check what columns are in the CSV. In a new notebook cell, try:

```python
import pandas as pd
df = pd.read_csv("layers/layer_01/data/YOUR_FILENAME.csv")
df.columns.tolist()
```

Replace `YOUR_FILENAME` with the actual filename. What columns do you see?

🤖end🤖

['title',
 'displayAddress',
 'bathrooms',
 'bedrooms',
 'addedOn',
 'type',
 'price',
 'verified',
 'priceDuration',
 'sizeMin',
 'furnishing',
 'description']
 

🤖start🤖

Looks like there's only one `description` column — no separate English and Arabic columns. Let's see what the descriptions actually look like. Try this:

```python
df["description"].head(10)
```

Some rows might have mixed English/Arabic text, or maybe some are in English and others in Arabic. Let's see what we're working with!

🤖end🤖

nstalling collected packages: webencodings, pytz, pure-eval, ptyprocess, nvidia-cusparselt-cu13, mpmath, fastjsonschema, cuda-toolkit, xxhash, widgetsnbextension, websocket-client, webcolors, wcwidth, urllib3, uri-template, tzdata, typing-extensions, triton, traitlets, tqdm, tornado, tomli, tinycss2, sympy, soupsieve, six, shellingham, send2trash, safetensors, rpds-py, rfc3986-validator, regex, pyzmq, pyyaml, python-json-logger, pygments, pycparser, pyarrow, psutil, propcache, prometheus-client, platformdirs, pexpect, parso, pandocfilters, packaging, overrides, nvidia-nvtx, nvidia-nvshmem-cu13, nvidia-nvjitlink, nvidia-nccl-cu13, nvidia-curand, nvidia-cufile, nvidia-cuda-runtime, nvidia-cuda-nvrtc, nvidia-cuda-cupti, nvidia-cublas, numpy, networkx, nest-asyncio, mdurl, MarkupSafe, lark, jupyterlab_widgets, jupyterlab-pygments, jsonpointer, json5, idna, hf-xet, h11, fsspec, frozenlist, fqdn, filelock, executing, dill, defusedxml, decorator, debugpy, cuda-pathfinder, comm, click, charset_normalizer, certifi, bleach, babel, attrs, async-timeout, asttokens, annotated-doc, aiohappyeyeballs, terminado, stack_data, rfc3987-syntax, rfc3339-validator, requests, referencing, python-dateutil, prompt-toolkit, nvidia-cusparse, nvidia-cufft, nvidia-cudnn-cu13, multiprocess, multidict, mistune, matplotlib-inline, markdown-it-py, jupyter-core, jinja2, jedi, httpcore, exceptiongroup, cuda-bindings, cffi, beautifulsoup4, async-lru, aiosignal, yarl, tiktoken, rich, pandas, nvidia-cusolver, jupyter-server-terminals, jupyter-client, jsonschema-specifications, ipython, arrow, argon2-cffi-bindings, anyio, typer, jsonschema, isoduration, ipywidgets, ipykernel, httpx, argon2-cffi, aiohttp, torch, nbformat, jupyter-console, huggingface-hub, tokenizers, nbclient, jupyter-events, datasets, transformers, nbconvert, jupyter-server, notebook-shim, jupyterlab-server, jupyter-lsp, jupyterlab, notebook, jupyter
why did all these installed

🤖start🤖

Good observation! You only asked for 6 packages, but each one has **dependencies** — other libraries it needs to work. For example:

- **`torch`** pulled in all those `nvidia-*` packages (GPU/CUDA support)
- **`transformers`** needs `tokenizers`, `safetensors`, `huggingface-hub`, etc.
- **`jupyter`** needs a whole web server stack (`tornado`, `websocket-client`, `jupyterlab`, etc.)

This is totally normal with `pip`. You only need to think about the 6 you installed — the rest are just behind-the-scenes plumbing.

Everything look like it installed without errors?

🤖end🤖

all lib are installed how do icreate a notebook and access it


🤖start🤖

In your terminal (with the venv activated), run:

```bash
cd ~/Documents/projects/roadmap-llm-systems-engineering
jupyter notebook
```

This will open Jupyter in your browser. From there, navigate to `layers/layer_01/` and click **New → Python 3** to create a new notebook. Rename it to `tokenizer_comparison`.

Give that a go and let me know when you're in!

🤖end🤖



🤖start🤖

How's it going — did you get the notebook open? If so, the next step is loading the CSV and checking out those descriptions:

```python
import pandas as pd
df = pd.read_csv("data/YOUR_FILENAME.csv")
df["description"].head(10)
```

What do you see?🧠🧠🧠🧠🧠🧠🧠

How's it going — did you get the Jupyter notebook open? Were you able to load the CSV and check `df["description"].head(10)`?

🤖end🤖

0     OWN A CHALET IN EL GOUNA WITH A PRIME LOCATION...
1     For sale, a villa with immediate delivery in C...
2     With a down payment of EGP 1,513,000, a fully ...
3     Own an apartment in New Cairo with a minimal d...
4     Project: Granville\nLocation: Fifth Settlement...
5     Chalet with Marina and Lake View in The Island...
6     Penthouse 4BR for sale with Installments over ...
7     Stand-alone villa in NYOUM Compound, October\n...
8     **La Vista Ras El Hikma seaview twinhouse - Re...
9     A 195-square-meter duplex + a 42-square-meter ...
10    With Down payment 1 million only Fully Finishe...
11    • Chalet In || Gouna ||:\n- Layout: 1 Bedroom,...
12    Fully finished chalet with air conditioning fo...
13    Luxury Chalet 3 Bedroom With Full Sea View In ...
14    Looking for elegant apartment with luxurious f...
15    Unit Type: Duplex\n2 Bedrooms\nBuilt Up Area: ...
16    Taj City Compound - New Cairo\nFifth Settlemen...
17    Developer: Orascom\nProject: Makadi Heights\nU...
18    Chalet for sale in Hacienda Waters, North Coas...
19    CAESAR BY OGAMI\n_____________________________...
20    Area : New Heliopolis\nProject : Talala\nType:...
21    apartment for Sale – Club Hills Residence by H...
22    Solare – Ras El Hekma: A world-class coastal e...
23    Chalet 3BR Lagoon View – The Med Ras El Hekma ...
24    nnouncing Oro Beach Project – Hurghada, Al Ahy...
25    Resale ground duplex in Ivoire Compound\nbua 2...
26    Own a stunning one-story Twinhouse in Olive Oa...
27    Prime Studio for Sale in One of the Most Prest...
28    Apartment for sale with facilities in Bloomfie...
29    RTM townhouse in Badya Palm Hills, 6th of Octo...
30    have a fully finished penthouse  for you in pr...
31    Enjoy a European lifestyle in Mountain View, s...
32    Why Choose to Live in Cavali Residence?\nA pri...
33    Get your Villa in a prime location In Hear Of ...
34    Looking for elegant apartment with luxurious f...
35    standalone Villa In Belle Vie By Emaar Misr\nP...
36    last chalet first row lagoon with lowest down ...
37    Location: Amwaj North Coast – KM 136 Alexandri...
38    ONE7\nONE7 is a prestigious new development by...
39    OWN YOUR SEA VIEW UNIT FULLY FINISHED ON SANDY...
40    Zone : Aven\nType Town House Middle\nBUA : 261...
41    Apartment 80 meter at the lowest price, direct...
42    Step into design-forward living in a beach hou...
43    Townhouse 4 Bedrooms\nMost Prime Location – VY...
44    Lake view residence 2\nApartment for sale\n5th...
45    Apartment Limited Offer Book immediately in Ev...
46    Own a 115 sqm Apartment with a Garden View in ...
47    LIMITTED OFFER _5% DOWNPAYMENT + 5% AFTER 6 MO...
48    Fully Finished Chalet with Panoramic Sea & Lag...
49    شاليه مميز للبيع في مارسيليا بيتش 5 ( راس الحك...
Name: description, dtype: object

🤖start🤖

Interesting! Most descriptions are in English, but look at row **49** — that's in Arabic. There might be more Arabic ones further down.

Let's find out how many Arabic descriptions there are. Try this:

```python
df["description"].str.contains("[\u0600-\u06FF]").sum()
```

This checks for Arabic characters. What number do you get?

🤖end🤖

1783 but what is this filltering


🤖start🤖

1783 Arabic descriptions — plenty to work with!

That line filters by checking each description for **Arabic Unicode characters**. Here's what it does:

- `\u0600-\u06FF` is the Unicode range for Arabic script characters (like ع، م، ك, etc.)
- `.str.contains(...)` checks if each row's description has at least one character in that range
- `.sum()` counts how many rows matched (`True` = 1, `False` = 0)

So it's basically asking: "How many descriptions contain at least one Arabic letter?"

Now, the exercise asks for 100 rows that have **both** English and Arabic text. Can you think of how to filter for rows that contain *both* Arabic and English characters?

🤖end🤖

we can seperate them

🤖start🤖

Good idea! You could grab 50 English-only and 50 Arabic-only descriptions. That would let you compare how each tokenizer handles both languages.

How would you filter for rows that are **only English** (no Arabic characters)? You already know how to find the Arabic ones — what's the opposite?

🤖end🤖

df["description"].str.contains("[\u0600-\u06FF]")we use not_contains


🤖start🤖

Close! There's no `not_contains` in pandas, but you can negate with `~` (the tilde). So:

```python
~df["description"].str.contains("[\u0600-\u06FF]")
```

This flips `True` to `False` and vice versa — giving you rows with **no** Arabic characters. Try using that to grab 50 English and 50 Arabic rows!

🤖end🤖

english[] = df["description"].str.contains("[\u0600-\u06FF]").sum()


🤖start🤖

Almost! A couple of things to fix:

1. Use `~` to get the English (non-Arabic) ones
2. `.sum()` gives you a count — but you want the actual rows, not just a number
3. Variable names can't have `[]` in Python

Try something like:

```python
english = df[~df["description"].str.contains("[\u0600-\u06FF]")]
```

This filters the dataframe to only rows **without** Arabic characters. Can you do the same for Arabic rows?

🤖end🤖

---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[9], line 1
----> 1 english = df[~df["description"].str.contains("[\u0600-\u06FF]")]

File ~/Documents/projects/roadmap-llm-systems-engineering/layers/layer_01/.venv/lib/python3.10/site-packages/pandas/core/generic.py:1574, in NDFrame.__invert__(self)
   1570 if not self.size:
   1571     # inv fails with 0 len
   1572     return self.copy(deep=False)
-> 1574 new_data = self._mgr.apply(operator.invert)
   1575 res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   1576 return res.__finalize__(self, method="__invert__")

File ~/Documents/projects/roadmap-llm-systems-engineering/layers/layer_01/.venv/lib/python3.10/site-packages/pandas/core/internals/managers.py:361, in BaseBlockManager.apply(self, f, align_keys, **kwargs)
    358             kwargs[k] = obj[b.mgr_locs.indexer]
    360 if callable(f):
--> 361     applied = b.apply(f, **kwargs)
    362 else:
    363     applied = getattr(b, f)(**kwargs)

File ~/Documents/projects/roadmap-llm-systems-engineering/layers/layer_01/.venv/lib/python3.10/site-packages/pandas/core/internals/blocks.py:395, in Block.apply(self, func, **kwargs)
    389 @final
    390 def apply(self, func, **kwargs) -> list[Block]:
    391     """
    392     apply the function to my values; return a block if we are not
    393     one
    394     """
--> 395     result = func(self.values, **kwargs)
    397     result = maybe_coerce_values(result)
    398     return self._split_op_result(result)

TypeError: bad operand type for unary ~: 'float'

🤖start🤖

This error happens because some `description` values are `NaN` (missing), and you can't negate `NaN` with `~`. Add `na=False` to handle that:

```python
english = df[~df["description"].str.contains("[\u0600-\u06FF]", na=False)]
```

Try that!

🤖end🤖

okay next

🤖start🤖

Now do the same for Arabic rows, and then sample 50 from each:

```python
arabic = df[df["description"].str.contains("[\u0600-\u06FF]", na=False)]
```

Then pick 50 from each:

```python
en_sample = english.sample(50, random_state=42)
ar_sample = arabic.sample(50, random_state=42)
```

(`random_state=42` makes it reproducible — you'll get the same 50 every time you run it.)

Try that and let me know what you get!

🤖end🤖

okay next

🤖start🤖

Now combine them into one dataframe:

```python
sample = pd.concat([en_sample, ar_sample])
```

Next up is tokenizing! You'll need to load the 4 tokenizers. Start with tiktoken:

```python
import tiktoken
gpt4o_enc = tiktoken.encoding_for_model("gpt-4o")
```

Try that and let me know if it works.

🤖end🤖

okay what did we do here

🤖start🤖

Good question! Let's recap:

1. **`pd.concat([en_sample, ar_sample])`** — combines your 50 English and 50 Arabic rows into a single 100-row dataframe called `sample`. This is your working dataset for the tokenizer comparison.

2. **`tiktoken.encoding_for_model("gpt-4o")`** — loads the tokenizer that GPT-4o uses. A tokenizer breaks text into **tokens** (small chunks like words, subwords, or characters) that the model actually processes. Different models use different tokenizers, which is why you're comparing them.

The goal is to see how each tokenizer splits the same text — some might use more tokens for Arabic, some fewer. That directly affects cost and performance.

Make sense? Did the tiktoken lines run OK?

🤖end🤖

yes

🤖start🤖

Now load the other 3 tokenizers. Next one — cl100k_base:

```python
cl100k_enc = tiktoken.get_encoding("cl100k_base")
```

This is the tokenizer used by GPT-3.5 and GPT-4 (the older ones). Try that, then we'll load the two Hugging Face ones.

🤖end🤖

why do we need these

🤖start🤖

The exercise is about **comparing** tokenizers to see how they handle the same text differently. Here's why these 4 specifically:

- **gpt-4o** — the newest OpenAI tokenizer, optimized for multilingual text
- **cl100k_base** — the older GPT-4/3.5 tokenizer, so you can see how OpenAI improved
- **bert-base-multilingual-cased** — a Google tokenizer designed for 100+ languages
- **AraBERT/AraGPT2** — a tokenizer specifically built for Arabic

By comparing them on the same English and Arabic descriptions, you'll see things like: Does the Arabic-specific tokenizer use fewer tokens for Arabic text? Does GPT-4o handle Arabic better than the older cl100k? Fewer tokens = lower cost and faster processing.

That's why tokenizer choice matters in real systems!

🤖end🤖

next

🤖start🤖

Load the two Hugging Face tokenizers:

```python
from transformers import AutoTokenizer
bert_tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
```

Try that — the first time it will download the model files, which may take a moment. Let me know when it's done!

🤖end🤖

from transformers import AutoTokenizer
bert_tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


from transformers import AutoTokenizer
bert_tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


🤖start🤖

Did it download and run OK? If so, now load the Arabic one:

```python
ara_tok = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
```

Let me know when it's ready!

🤖end🤖

next

🤖start🤖

Now let's tokenize! Try counting tokens for each description with all 4 tokenizers:

```python
sample["gpt4o_tokens"] = sample["description"].apply(lambda x: len(gpt4o_enc.encode(x)))
```

This adds a column with the GPT-4o token count for each row. Can you do the same for the other 3 tokenizers?

🤖end🤖

give to me 

🤖start🤖

Here you go:

```python
sample["cl100k_tokens"] = sample["description"].apply(lambda x: len(cl100k_enc.encode(x)))
sample["bert_tokens"] = sample["description"].apply(lambda x: len(bert_tok.encode(x)))
sample["ara_tokens"] = sample["description"].apply(lambda x: len(ara_tok.encode(x)))
```

Run that and then try `sample[["description", "gpt4o_tokens", "cl100k_tokens", "bert_tokens", "ara_tokens"]].head(10)` to see the results!

🤖end🤖

[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (630 > 512). Running this sequence through the model will result in indexing errors
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (551 > 512). Running this sequence through the model will result in indexing errors

🤖start🤖

That warning is fine — it just means a couple of descriptions are longer than BERT's max input size (512 tokens). It doesn't affect the **counting**, which is all we're doing here. The warning would only matter if you were running text *through* the model for predictions.

Did the token counts get added? Try:

```python
sample[["description", "gpt4o_tokens", "cl100k_tokens", "bert_tokens", "ara_tokens"]].head(10)
```

🤖end🤖

description	gpt4o_tokens	cl100k_tokens	bert_tokens	ara_tokens
18873	Chalet for Sale – Salt by Tatweer Misr | North...	277	281	280	440
13311	Code GMVs6\nApartment for sale in Galleria moo...	168	171	177	311
12388	Ultra super deluxe detached villa with air con...	167	169	189	274
6809	Scandic Resort – Exclusive Seafront Living in ...	292	301	334	551
2794	Your apartment in a luxurious compound built o...	108	109	119	195
13224	Penthouse for Sale – Mountain View, Hyde Park\...	99	106	104	164
3454	Limited offer Own now your fully finished chal...	182	187	205	326
6421	Chalet in fouka bay Ras Elhekma kilo 212 Lagoo...	234	239	230	346
4292	Chalet 90m + Garden 120\n2bedroom + 2bathroom\...	121	128	108	168
15596	Apartment with a garden for sale in Beverly Hi...	200	205	289	388



🤖start🤖

Great results! Notice how the **AraBERT tokenizer** (`ara_tokens`) uses the most tokens for these English descriptions — that makes sense since it was built for Arabic, not English.

Now let's see how they compare on the **Arabic** rows. Try:

```python
sample[["description", "gpt4o_tokens", "cl100k_tokens", "bert_tokens", "ara_tokens"]].tail(10)
```

That should show some Arabic descriptions. What do you notice about the token counts?

🤖end🤖


description	gpt4o_tokens	cl100k_tokens	bert_tokens	ara_tokens
9298	apartment 205m for sale in westown Beverly Hil...	355	759	423	364
8072	للبيع\nفيلا فاخرة فى كمبوند الجريا - بيفرلى هي...	663	1259	757	574
3384	Amwaj North Coast Sidi Abdel Rahman\nChalet\n1...	184	292	200	225
5909	شقة للبيع178م بالتجمع الخامس القاهرة الجديدة\n...	183	333	177	141
13623	شاليه للبيع في بورتو جولف\nالمساحة : 80 متر\nا...	197	700	269	153
18176	دوبلكس 240م بحري\nبيزمنت + ارضي *بحديقة خاصة 1...	104	186	110	83
5189	شقق للبيع في افضل موقع في التجمع الخامس - القا...	363	432	455	638
18753	For sale in Mountain View i_city new cairo\nUn...	362	779	421	375
18557	شقة للبيع في اليكس ويست\n- المساحة : ١٨٣م\n- ا...	222	756	293	169
479	تاون هاوس للبيع في باديه بالم هيلز, كمبوندات 6...	525	1077	694	592


🤖start🤖

Interesting! Look at the Arabic rows (like rows 8072, 5909, 13623, 18176) — notice how:

- **`cl100k_tokens`** is massive compared to the others (e.g. 1259 vs 574 for row 8072)
- **`ara_tokens`** is the lowest for Arabic text (e.g. 153 vs 700 for row 13623)

This shows exactly why tokenizer choice matters — cl100k (older GPT tokenizer) is very inefficient with Arabic, while AraBERT handles it much better. More tokens = higher cost.

Now let's build the **cost comparison**. Try computing the average token count per tokenizer, split by language:

```python
sample["lang"] = ["en"]*50 + ["ar"]*50
sample.groupby("lang")[["gpt4o_tokens", "cl100k_tokens", "bert_tokens", "ara_tokens"]].mean()
```

What do you see?

🤖end🤖


gpt4o_tokens	cl100k_tokens	bert_tokens	ara_tokens
lang				
ar	247.16	451.26	289.24	257.12
en	213.90	220.36	246.68	375.90


🤖start🤖

This tells a clear story:

- **For Arabic text**: `cl100k` averages **451 tokens** vs AraBERT's **257** — almost twice as many! GPT-4o (247) is much better than cl100k for Arabic.
- **For English text**: GPT-4o (214) and cl100k (220) are similar and most efficient. AraBERT (376) struggles with English, as expected.

Key takeaway: **using the wrong tokenizer can nearly double your costs**.

Now let's add actual cost estimates. The exercise says $0.0001 per 1k tokens. Can you figure out how to add a cost column for each tokenizer?

🤖end🤖

how 

🤖start🤖

The formula is: **cost = (tokens / 1000) × $0.0001**

Try it for one tokenizer first:

```python
sample["gpt4o_cost"] = sample["gpt4o_tokens"] / 1000 * 0.0001
```

Can you do the same for the other 3?

🤖end🤖

okay how

🤖start🤖

Same pattern — just swap the column names:

```python
sample["cl100k_cost"] = sample["cl100k_tokens"] / 1000 * 0.0001
sample["bert_cost"] = sample["bert_tokens"] / 1000 * 0.0001
sample["ara_cost"] = sample["ara_tokens"] / 1000 * 0.0001
```

Then check the average cost by language:

```python
sample.groupby("lang")[["gpt4o_cost", "cl100k_cost", "bert_cost", "ara_cost"]].mean()
```

What numbers do you get?

🤖end🤖


gpt4o_cost	cl100k_cost	bert_cost	ara_cost
lang				
ar	0.000025	0.000045	0.000029	0.000026
en	0.000021	0.000022	0.000025	0.000038


🤖start🤖

Now you can see the cost difference clearly:

- **Arabic text**: cl100k costs **$0.000045** vs GPT-4o at **$0.000025** — nearly **double**!
- **English text**: AraBERT costs **$0.000038** vs GPT-4o at **$0.000021**

So if you're building a system handling Arabic real estate listings, choosing the wrong tokenizer literally doubles your costs at scale.

That completes **Exercise 1.1**! 🎉 You've got the tokenizer comparison with a cost table.

Ready to move on to **Exercise 1.2** — the temperature experiment with deepseek?

🤖end🤖

okay please can we take a whole view of what we did here and how it help us 

🤖start🤖

Sure! Here's a recap of what you did in Exercise 1.1 and why it matters:

**What you did:**

1. Loaded a real estate dataset with English and Arabic descriptions
2. Filtered and sampled 50 English + 50 Arabic rows
3. Tokenized every description with 4 different tokenizers (GPT-4o, cl100k, BERT multilingual, AraBERT)
4. Counted tokens and calculated cost per description

**What you learned:**

- A **tokenizer** breaks text into chunks (tokens) that an LLM can process. Different tokenizers split text differently.
- **Language matters** — tokenizers built for one language are inefficient for another. AraBERT was great for Arabic but expensive for English, and vice versa for cl100k.
- **Tokenizer choice directly affects cost** — cl100k needed nearly double the tokens for Arabic compared to GPT-4o, which means double the API cost at scale.

**Why this matters in real systems:**

If you're building an LLM-powered app (like a real estate chatbot for the UAE market), picking the right tokenizer/model for your language mix could save significant money and improve performance.

Ready for Exercise 1.2 — the temperature experiment? 🔥

🤖end🤖

next

🤖start🤖

**Exercise 1.2 — Temperature experiment** 🔥

You'll send the same prompt to your local deepseek model at 4 different temperatures (0, 0.3, 0.8, 1.2) and observe how the outputs change.

First, make sure Ollama is running. Then try this in a new notebook cell:

```python
import requests

def generate(prompt, temperature):
    resp = requests.post("http://localhost:11434/api/generate",
        json={"model": "deepseek-r1:1.5b", "prompt": prompt, "temperature": temperature, "stream": False})
    return resp.json()["response"]
```

This is a helper function that sends a prompt to deepseek and returns the response. Try running it — any errors?

🤖end🤖

no errors

🤖start🤖

Now test it! Pick a property description prompt and run it at temperature 0:

```python
prompt = "Describe a luxury 3-bedroom apartment in Dubai Marina with sea view"
print(generate(prompt, temperature=0))
```

What does it output?

🤖end🤖

 tge gebereate function is just sending a one prompt responce and waiteing for responce to retur

🤖start🤖

Exactly right! The `generate` function:

1. **Sends** a single prompt to deepseek via Ollama's REST API (`POST` to `localhost:11434`)
2. **Waits** for the full response (`"stream": False` means it waits until the model finishes generating)
3. **Returns** just the text response

It's a simple request-response cycle — like asking someone a question and waiting for their full answer before moving on.

Did the temperature=0 output look good?

🤖end🤖

<think>
Okay, so the user wants me to describe a luxury three-bedroom apartment in Dubai Marina with sea views. Hmm, where do I start? First, I need to think about Dubai Marina itself— it's that big city center right near the beaches and all that luxury can come with. But since the focus is on a 3-bedroom apartment, I should make sure it’s both stylish and practical.

I remember that luxury apartments usually have high-end finishes like marble or wood, but maybe I can mention something else to add uniqueness. Maybe a modern twist on traditional designs? Oh, wait, the example response mentioned a "stunning modern design," so that's a good starting point.

Sea views are essential here. I should describe how they’re accessed— perhaps through glass doors or skylights. It’s important to make it feel exclusive and luxurious. Maybe mention some amenities like a private gym, tennis courts, or a rooftop deck? That would add a nice touch for the luxury aspect.

The price range is a big part of this. Dubai Marina is pricey because it's such a trendy area. I should note that prices are typically high, maybe in $500k to $1 million, but I can also mention the quality of materials and workmanship for authenticity.

I need to make sure the apartment not only has views but feels like a modern destination. Maybe include elements that show it's been thought through carefully by top design firms, ensuring everything is both stylish and functional. Including something about being close to amenities would add practical value without compromising on luxury.

Wait, I should avoid sounding too generic. The example response did that well with "stunning modern design" and specific features like glass doors and private areas. So I can structure mine similarly, starting with the location, then moving into design elements, views, features, and final touches to make it a true destination.

I also want to highlight how this place is not just great for relaxation but also for entertaining guests. Maybe mention a bar, dining options, and maybe even a fitness room or lounge area if there’s one.

Oh, and the location— Dubai Marina isn't just about beaches; it's part of theCBD area, which has its own cultural aspects. So perhaps adding something about the vibe, like the atmosphere of the area where you live could add to the luxury feel.

I should make sure all these elements are woven into a cohesive description without sounding too piecemeal. It needs to flow smoothly and show each feature contributing to why it's considered luxurious.

Alright, putting this together: start with the location and amenities, mention the views through glass doors or skylights, talk about modern design with high-end finishes, include specific features like a private gym, tennis courts, and maybe even a rooftop deck for relaxation. Mention the price range as high but achievable, highlight the quality of materials and craftsmanship, talk about being close to attractions like the beaches, and conclude by emphasizing it's a destination that combines luxury, modernity, and exclusivity.

Wait, did I cover everything? Maybe include something about the interior finishes or color schemes if they were part of the example. But since I'm just thinking, maybe not necessary. Focus on the outside features as the main points for the description.

I think that's a solid structure. Now, let me make sure it flows well and covers all the important aspects without being too wordy.
</think>

Certainly! Here’s an elegant and comprehensive description of a luxury 3-bedroom apartment in Dubai Marina with sea views:

---

In the heart of Dubai Marina, nestled between the bustling CBD and the stunning beaches of Al Khor and Marina Bay Sands, you can find this exquisite 3-bedroom apartment designed to be both modern and luxurious. This property is located on the highest floor of the luxury building, offering exclusive access to breathtaking views of the Dubai Marina and the surrounding city skyline.

With stunning sea views stretching from the open kitchen to the roof, the apartment is a perfect blend of sophistication and natural light. The exterior features stylish glass doors that seamlessly project into the living area, enhancing the sense of exclusivity. The interior is a testament to high-end design, with high-quality wood floors, marble countertops, and modern finishes like lacquered tiles and polished concrete.

The apartment boasts a private gym, state-of-the-art tennis courts, and an expansive rooftop deck where you can relax on the beach or enjoy a drink by the water. Additionally, there is a private lounge area for entertainment, complete with a comfortable sofa and armchairs, perfect for guests or enjoyable use by locals.

The price range for this luxury property typically ranges from $500,000 to $1 million, reflecting Dubai Marina's premium real estate market. The apartment has been meticulously crafted with thoughtfulness by top design firms, ensuring not only its aesthetic appeal but also practicality and functionality in a bustling city center.

Moreover, this location is just steps away from a variety of cultural and entertainment options, making it an ideal spot for both relaxation and lively socializing. Perfect for those seeking the perfect blend of luxury, modernity, and exclusivity, this apartment in Dubai Marina is sure to stand out as one of the most desirable properties available.

--- 

This description captures the essence of a luxurious apartment with sea views in Dubai Marina while highlighting its practical features, design, and proximity to amenities.

🤖start🤖

Interesting! A few things to notice about this output at **temperature=0** (most deterministic):

- The model did a lot of "thinking" in the `<think>` block before answering — that's how deepseek-r1 works, it reasons step by step
- The actual response is structured and detailed

Now let's see how **higher temperatures** change things. Run the same prompt at all 4 temperatures:

```python
for temp in [0, 0.3, 0.8, 1.2]:
    print(f"\n{'='*50}")
    print(f"Temperature: {temp}")
    print(f"{'='*50}")
    print(generate(prompt, temperature=temp))
```

This might take a minute since it's generating 4 responses. What differences do you notice as temperature increases?

🤖end🤖

what is the tempertures 

🤖start🤖

**Temperature** controls how "random" or "creative" the model's output is.

- **Temperature 0** — the model always picks the most likely next token. Output is very consistent and predictable. Run it 10 times, you'll get nearly the same answer.
- **Temperature 0.3** — a little randomness, but still pretty focused.
- **Temperature 0.8** — more variety and creativity, but can sometimes drift off topic.
- **Temperature 1.2** — very creative/random, might produce surprising or even nonsensical text.

Think of it like asking someone to write a property listing:
- Temp 0 = a careful professional who sticks to the facts
- Temp 1.2 = a wild creative writer who improvises

Are the 4 outputs generating? Let's see how they compare!

🤖end🤖


==================================================
Temperature: 0
==================================================
<think>
Okay, so I'm trying to describe a luxurious three-bedroom apartment in Dubai Marina with a sea view. Hmm, where do I start? Well, I know Dubai Marina is a big city area with all sorts of views and landmarks. But what exactly makes an apartment luxurious there?

First, the main points are luxury, three bedrooms, and a view of the marina. Luxury usually means high-end materials, maybe some international architecture, and probably a good layout that's comfortable for residents or tourists. The sea view adds to the ambiance, so I should think about what kind of views you'd get.

I remember seeing pictures of apartments with pools on the marina, which is nice. Also, the marina itself is bustling with activity, but maybe it's better to focus on the apartment's features rather than the surrounding area. The apartment should be quiet since there are so many activities around Dubai Marina.

So, for the first bedroom, maybe a king-size bed with a queen-sized pillow for sleep. It gives that luxurious feel. I've heard of units where one room is smaller, but having both bedrooms comfortable could make it more functional. Maybe the second and third bedrooms can be slightly smaller, but still comfortable, so each room has its own space without feeling cramped.

The kitchen should be top-notch with everything you'd need: a cozy eat-in area with dishes that look fancy. Cooking in the apartment would be great for a luxurious lifestyle. Then the bathroom needs to be spotless and well-lit, maybe even with a spa-like shower because that's a big part of luxury.

Ceiling height is important too—it should be high enough to see into the room without looking down. The walls are solid to keep the space feeling sturdy and modern. Maybe some unique features like mirrored ceilings or high-end furniture can add to the sophistication.

Leaving no stone unturned, I should think about amenities beyond just the kitchen and bathroom. A fitness center might be nice if it's on-site, offering a warm environment. The pool in the evening is a great luxury too, giving peace of mind for those who like to relax after work.

The sea view from the first bedroom would be stunning. Maybe there are some glass doors or windows that let in natural light and风景 outside, which adds to the ambiance without being too intrusive.

Lighting should be bright enough for reading or cooking, but soft where people aren't looking. Maybe a mix of natural sunlight with some ambient lighting from candles or plants. The air conditioning is important for comfortable sleep at night.

I also need to make sure the description isn't too long and stays true to Dubai Marina's vibe—luxurious, historic, stylish, with all the modern touches that make it special. So combining all these elements should create a vivid picture of this luxurious apartment.
</think>

A luxurious three-bedroom apartment in Dubai Marina is a perfect blend of luxury, sophistication, and natural beauty. Here's a detailed description:

**First Bedroom:** 
- A king-size bed with a queen-sized pillow for comfort and style, offering a cozy yet luxurious sleep space.

**Second Bedroom:** 
- A slightly smaller bedroom to balance the roomy feel of Dubai Marina while providing a comfortable retreat with ample storage and modern amenities.

**Third Bedroom:** 
- Another small room designed to be functional without compromising on comfort. It features a well-placed bed and plenty of storage options, ensuring each space is both enjoyable and practical.

**Kitchen:** 
- A cozy eat-in kitchenette that's both functional and luxurious, complete with high-end appliances and a variety of dishes available for cooking, making it ideal for dining or relaxation.

**Bathroom:**
- A spa-like shower with modern fixtures and materials, designed to enhance the comfort and elegance of each bathroom.

**Ceiling Height:** 
- High ceiling height allows natural light to flow into each room while maintaining a feeling of warmth and control, perfect for a luxurious lifestyle.

**Wall Materials:** 
- Solid walls ensure structural stability and an authentic feel in a modern setting, reflecting Dubai Marina's blend of history and innovation.

**Amenities:**
- A fitness center on-site with warm surroundings, offering both comfort and fitness. Additionally, a pool available for entertaining, providing peace of mind for those seeking downtime.

**Sea View:** 
- The apartment benefits from stunning sea views from the first bedroom, offering an breathtaking view into Dubai Marina's vibrant landscape and city life.

**Lighting:** 
- Bright natural light is distributed evenly, with soft ambient lighting provided by candles or plants to create a comfortable yet inviting atmosphere.

**Air Conditioning:** 
- A well-insulated system ensures a comfortable sleep environment throughout the day, with the added benefit of less heat during evenings when relaxing outside is preferred.

This apartment in Dubai Marina is a testament to its mix of luxury, history, and modernity, offering both comfort for residents and a delightful escape for visitors.

==================================================
Temperature: 0.3
==================================================
<think>
Okay, so I need to describe a luxurious three-bedroom apartment in Dubai Marina with sea views. Hmm, where do I start? I know Dubai is known for its luxury, but the exact location of Dubai Marina is pretty far from the city center. It's in the Middle East, right? So the apartment should be on the outskirts or maybe near some popular tourist spots?

First, the structure. Since it's a 3-bedroom apartment, I guess it would have three different rooms: master suite, primary bedroom, and secondary bedroom. The master suite should be the most luxurious part because of the sea views. Maybe the master suite has a spa-like bathroom with a big shower, maybe even an en-suite, to make that feel special.

Then, for the primary bedroom, I can imagine a large, quiet space that's a good fit for someone who likes their privacy but still wants some space to unwind or do something nice like a bath. The secondary bedroom could be another comfortable space with a view, maybe an en-suite as well if it's separate.

When it comes to the views themselves, Dubai Marina has a stunning sea view. I think the apartment would have a front yard or maybe a balcony that faces towards the ocean or the skyline. Including a beach access point makes sense because then you can relax by the water. Maybe even some sunbathing areas for people who like to take breaks.

Lighting is important too. The apartment should have plenty of natural light, especially in the morning and evening when the sun is high. Maybe using a lot of window treatments or blackout curtains so it's fully visible from the bedroom floor.

For aesthetics, color plays a big role. Darker grays and warm tones can create a luxurious look. Maybe include some chandeliers or glass doors that let light in without being intrusive. Also, maybe an open concept kitchenette with high-end appliances to make cooking convenient for guests.

Hydrotherapy could be a nice touch. A spa-like bathroom with a steam shower and perhaps some therapeutic candles would add a luxurious touch. The master suite can have a view of the pool or marina, which is a big plus for relaxation.

Soundproofing is another consideration. The apartment should have soundproof windows to avoid noise干扰ing the sea views. Maybe include some privacy features like tall ceilings or a spa-like bathroom with its own separate entrance.

I also want to note that while Dubai Marina has beautiful views, there are areas where it's not as stunning. Maybe mention some key spots for viewing without making it seem unimportant.

Putting all this together, I can structure the description by starting with an overview of Dubai Marina and then moving into the specific features of the apartment: layout, lighting, aesthetics, interior details, and final touches like soundproofing.

Wait, should I include any specific examples or names? Maybe something about a particular building for clarity. Also, maybe touch on how easy it is to access amenities from there, like the beach or marina services.

I think that covers most of the key points. Now, let me make sure each section flows well and highlights all these elements without sounding too cluttered.
</think>

Certainly! Here's a comprehensive description of a luxurious 3-bedroom apartment in Dubai Marina with sea views:

---

**Described Apartment in Dubai Marina**

Dubai Marina stands as a testament to luxury, offering breathtaking sea views and vibrant nightlife. Imagine living in an apartment nestled among the city's serene streets, just steps from Dubai Beach. This property, designed for elegance and convenience, offers a perfect blend of modern amenities and luxurious living.

**Layout and Features:**
- **3-Bedroom Apartment:** The apartment features a master suite, primary bedroom, and secondary bedroom. The master suite is the heart of the space, featuring a grand spa-like bathroom with an en-suite shower, a large tub, and a stunning view of Dubai Marina's oceanfront.

**Lighting:**
- Natural light dominates throughout the apartment, ensuring every room is illuminated, from the bedrooms to the living areas. blackout curtains or window treatments are used to maximize visibility while offering privacy in each room.

**Aesthetics:**
- Dark grays and warm tones create a luxurious atmosphere, with chandeliers adding depth and glass doors granting open views of the ocean or Marina's marina. The interior is inviting, combining modern aesthetics with timeless elegance.

**Interior Details:**
- **Homey feel:** A high-end kitchenette is seamlessly integrated into the living space, equipped with top-tier appliances, making entertaining trips easy for guests.
- **Hydrotherapy and Relaxation:** The master suite features a steam shower in a spa-like bathroom with therapeutic candles, while the primary bedroom includes privacy amenities such as tall ceilings and a comfortable bed.

**Soundproofing:**
- Soundproof windows ensure clear visibility of the sea or Marina from all angles. Additionally, private areas like the master suite's spa-like bathroom are equipped with dedicated exits for ease of access.

**Location and Amenities:**
- Nestled in Dubai Marina, this apartment is close to key attractions: beach access points, marina services, and public transport links. It offers easy access to vibrant dining spots, entertainment options, and a range of private amenities.

**Special Features:**
- The apartment includes a sunbathing area with shade, ensuring comfort for all residents or guests. Perfect for those seeking both views and convenience in their beach relaxation.

**Final Touches:**
- Ensuring soundproofing throughout the space enhances the serene experience. A walkway leading into the pool adds another layer of luxury to the property.

This apartment in Dubai Marina is a testament to modern design, privacy, and luxurious living, offering everything a discerning guest or neighbor would desire. Whether you're seeking views or convenience, every detail ensures an unforgettable experience.

---

This description encapsulates the essence of Dubai Marina's beauty while highlighting the unique features of the apartment.

==================================================
Temperature: 0.8
==================================================
<think>
Okay, so I'm trying to imagine a luxurious three-bedroom apartment in Dubai Marina that has a view of the sea. I want this place to be not just amazing on its own but also a part of Dubai's vibrant community. Let me break down what I think are important aspects for such an apartment.

First, the size and layout. Dubai Marina is known for being modern and with great views, so a 3-bedroom apartment needs to be both functional and stylish. Maybe I can go from one to two stories, but perhaps more like four or five feet? That would give good natural light without being too narrow. Each bedroom should have its own bed, maybe even a shared bathroom for privacy, but that's optional depending on the brand.

The design should reflect Dubai's architectural trends. I've heard of smart homes and modern architecture. Maybe incorporating some tiles or glass panes in the walls to create an open space. The interior should be elegant with wooden floors or marble countertops, which would make it sound luxurious. I also want a sea view, so placing the bedroom on the southern side could give that stunning ocean view.

The amenities are important too. Smart home technology makes sense for convenience and privacy. There might be a pool area and a garden to provide green space. A gym seems essential for relaxation or exercise during the day. The kitchen should have all the necessary appliances, like a range, coffee machine, and perhaps some storage. Maybe there's even a Juliette bar for a relaxed night's toast.

The location in Dubai Marina should be prime for dining and shopping. I'm thinking of areas like Finsley Square or Tengel Heights. These areas are bustling with restaurants, shops, and entertainment. Plus, being close to the Marina would mean easy access to sea views from the apartment.

I also want the space to feel private yet comfortable. Maybe have a master suite with a big bed, perhaps double beds for privacy, but maybe two separate rooms for comfort instead of sharing. The bathroom should be top-notch, with clean water and maybe some smart fixtures or a spa-like experience.

The decor should blend modern elements with desert influences. Maybe some futuristic tiles that look like desert sand in the shower or decorative curtains that give an authentic feel to each room. Using color palettes that are vibrant yet timeless could help with the sea view theme.

Thinking about the amenities, having a pool and garden is a plus. The beach itself is popular there, so it's not too far from where I live. Maybe even some rooftop gardens or pools for summer relaxation. If we're in the summer, outdoor seating would be ideal.

Budget-wise, Dubai Marina is known as one of the most expensive cities in the world. So the apartment should be well-maintained with everything you’d need, including a washing machine, dryer, a car wash (if possible), and access to nearby amenities. Plus, maybe some local services like taxis or elevators.

I'm also considering whether to include a separate kitchen area for privacy, especially if it's in the master suite. Maybe having two smaller kitchens adjacent to each other could offer more space but not necessarily add much value beyond convenience. Alternatively, a dedicated kitchen with access points from other areas might be nice for privacy and convenience.

Thinking about the views, the southern bedroom should dominate the sea view. The neighboring bedrooms can have good views too, maybe slightly less than the main one. This would create a balanced and visually appealing layout. I also want the walls to reflect Dubai's modern design, which could include elements like high-tech doors or skylights that let in natural light.

I'm also wondering about the potential for future expansion. Since Dubai Marina is so dynamic, maybe adding some future rooms or developments adjacent to this apartment would be a nice touch. Plus, considering the proximity to the Marina, it's within easy access to various amenities and attractions.

Wait, I should make sure the design isn't too modern. It should still have a touch of desert beauty without seeming forced. Maybe using minimalist materials but incorporating elements that mimic desert architecture, like curved lines or textures that look like sand.

I also need to think about the utilities. The kitchen needs everything from the fridge and stove to a range and coffee machine. Maybe even some storage for clothing, personal items, etc., which would be practical without being cluttered. The bathroom should have easy access taps with some smart features, maybe even a spa-like shower with water pressure control.

For the garden area, perhaps a small rooftop deck that can be used year-round. It could have seats, plants, and maybe some lighting to make the space feel cozy at night. Maybe there are even stairs leading up from the main level to a rooftop terrace for more views.

I also want to think about the size of each room. The master bedroom should be sizable but not too cramped. The second and third bedrooms can have slightly smaller areas, with good storage options like cabinets or drawers. This way, the space feels spacious without feeling crowded.

Additionally, having a private concierge service would add another layer of convenience for guests, especially in a bustling area like Dubai Marina. It would handle everything from checking into the apartment to transporting luggage around.

In terms of design elements, maybe using high-tech materials like double panes of glass or advanced insulation to keep energy use low while offering natural light. Soft lighting that's dimmed according to the time of day could make the space feel more comfortable and inviting at night.

I should also consider any local regulations or building codes in Dubai Marina regarding luxury units, but since Dubai is a major city with its own regulations, I think it's acceptable as long as everything is designed properly and functional within the structure.

Putting it all together, the apartment would be a blend of modern design, desert aesthetics, and premium amenities. It'd have great views, private spaces, and access to everything Dubai Marina has to offer, making it a desirable home in an expensive city.
</think>

A luxurious three-bedroom apartment in Dubai Marina with sea views would blend modern design with desert beauty and premium amenities. Here's a structured description:

### Location and Design:
- **Position:** Located on the southern side of Dubai Marina, offering prime access to the vibrant area and stunning ocean view.
- **Architecture:** Modern yet stylish, incorporating elements like smart home technology, glass walls for natural light, and minimalist design that mimics desert architecture.

### Room Layout:
- ** master suite:** Dominant bedroom with a large double bed, ensuring comfort and privacy. Adjacent rooms offer slightly less prominent views, creating a balanced layout.
- **Bedrooms:** Each room is designed to be both functional and stylish, with modern aesthetics that reflect Dubai's architectural trends.
- **Bathrooms:** Top-notch convenience includes easy access taps and smart fixtures or spa-like experiences.

### Amenities:
- **Gym:** Well-equipped for relaxation or physical activity.
- **Pool and Garden:** Ideal for outdoor activities, offering seating and a serene retreat from the city heat.
- **Sunset Bar:** A stylish bar with a rooftop garden for summer relaxation and an open kitchen dining area on the main level.

### Functional Features:
- **Apartment:** Equipped with all necessary appliances, including washer/dryer, and access to nearby amenities.
- **Pool and Elevator:** Easy access to the Marina's pool and elevator, providing convenience.
- **Efficient Management:** Modern maintenance schedule ensures smooth operations and security.

### Decor and Aesthetics:
- **Wall tiles:** Reflect desert sand in the shower for an authentic feel.
- **Floral decor:** Incorporates vibrant yet timeless design elements.
- **Color Palette:** Intense hues that convey sea views while maintaining modern elegance.

### Budget Considerations:
- **Accommodation:** Well-maintained with all necessary facilities, including car washes and nearby amenities.
- **Future Expansion:** Potential for future developments adjacent to the apartment enhancing its appeal.

This apartment in Dubai Marina offers a blend of luxury, design, and convenience, making it an ideal destination in one of the city's most dynamic areas.

==================================================
Temperature: 1.2
==================================================
<think>
Okay, so I need to describe a luxurious 3-bedroom apartment in Dubai Marina with sea views. Hmm, where do I start? Well, first, I should think about what makes Dubai Marina special and what features would make an apartment there stand out.

I know Dubai is known for its luxury, high-end architecture, and vibrant nightlife. The Marina is a big landmark with great views of the city, so that's probably a key element. Plus, sea views are a must-visit feature. I should include that in my description to highlight the appeal.

Next, the location. Dubai Marina is on the edge of the city, right? So an apartment there would be convenient for both commuting and shopping. Maybe mention nearby attractions like the Dubai Mall or the Al Mawga market. That way, it shows how easy it is to access other parts of the city from the apartment.

When thinking about the architecture, luxury often involves high-quality materials and modern designs. So perhaps describe a smart home or something with energy-efficient features. I should also mention the size—three bedrooms are good because they can offer more space for comfort but still have enough functionality. Maybe include details on the layout, like how the rooms are arranged to maximize natural light and privacy.

The amenities would be important too. High-end breakfast options like sandwiches or coffee would add value. There's also probably a swimming pool, which is great for relaxation. Other facilities like entertainment centers and maybe even a gym could complement the sea views.

In terms of costs, Dubai is pricey, but luxury apartments are usually more expensive than ordinary ones. I should mention that prices can vary based on location, age, and desired amenities. Also, considering proximity to attractions makes it easier to enjoy Dubai's cultural scene without long commutes.

I need to structure this in a way that flows well. Maybe start with an overview of the location and what to expect, then move into the features like sea views, architecture, size, layout, amenities, and costs. Conclude by summarizing how it all puts you in the luxury category.

Wait, did I cover everything? Let me check: location (near marina, nearby attractions), sea views, modern architecture, three bedrooms, smart home features, high-end breakfast, swimming pool, entertainment facilities, gym, proximity to landmarks, and cost considerations. Yeah, that seems comprehensive.

I should make sure the language is sophisticated but not too complex. Maybe use terms like "exquisite design," "high-quality materials," "smart home technology," and "convenience shopping." Also, highlight how these elements combine to create a luxurious living space.

Okay, I think I have a good outline. Now, time to put it all together into a coherent description.
</think>

The luxury 3-bedroom apartment situated in Dubai Marina is a gem of contemporary design and sophistication. Nestled on the edge of the city, this apartment offers unparalleled sea views, making it a must-visit destination. With its prime location near the Dubai Marina and bustling Al Mawga market, residents have easy access to all essential amenities.

At the heart of the apartment lies an exquisite architecture with modern design elements, integrating high-quality materials and smart home technology for maximum convenience. The three-bedroom layout provides ample space for comfort while ensuring functionality—each room benefits from natural light and privacy through efficient layouts.

A private swimming pool is a standout feature, offering relaxation in an ideal setting. Complementing this, entertainment facilities like a fitness center are included, enhancing the overall experience. The apartment also features a gym and modern amenities, making it a hub for both relaxation and activity.

The cost of luxury apartments in Dubai varies based on location, age, and desired features, but they are certainly within reach for those willing to invest. Proximity to Dubai's vibrant cultural scene adds an extra layer of convenience, ensuring a luxurious yet enjoyable lifestyle.


🤖start🤖

Now you can see the difference! Notice how:

- **Temp 0 & 0.3** — long, structured, detailed. Pretty similar to each other.
- **Temp 0.8** — started getting more creative (mentioned a "Juliette bar", "Finsley Square", "Tengel Heights" — places that don't exist!)
- **Temp 1.2** — shorter and more confident, but also invented things like "Al Mawga market"

Key takeaway: **higher temperature = more creative but less accurate**. The model starts "hallucinating" — making up facts that sound plausible but aren't real.

In a real system:
- Use **low temperature** (0–0.3) for factual tasks like answering questions or extracting data
- Use **higher temperature** (0.7–1.0) for creative writing or brainstorming

That completes **Exercise 1.2**! 🎉 Ready for **Exercise 1.3** — the TTFT (time-to-first-token) experiment?

🤖end🤖

okay next 

🤖start🤖

**Exercise 1.3 — Time to First Token (TTFT)** ⏱️

The idea here is to see how the model's response time changes as you send it longer prompts. You'll test with prompts of ~200, ~2000, and ~10,000 tokens.

First, let's set up the TTFT measurement function:

```python
import requests, time

def ttft(prompt):
    start = time.time()
    resp = requests.post("http://localhost:11434/api/generate",
        json={"model": "deepseek-r1:1.5b", "prompt": prompt, "stream": True},
        stream=True)
    for chunk in resp.iter_lines():
        first_token_time = time.time() - start
        break
    return first_token_time
```

Do you understand what this function does differently from the `generate` one we used before?

🤖end🤖

what is inter_lines()

🤖start🤖

`iter_lines()` reads the response **line by line** as it arrives, instead of waiting for the whole thing to finish.

Since we set `"stream": True`, the model sends back tokens one at a time. `iter_lines()` lets us grab the **very first chunk** that arrives — then we immediately `break` out of the loop and record the time.

So we're measuring: **how long from sending the request until the model starts producing output?** That's the TTFT.

Did that make sense? Ready to create the test prompts?

🤖end🤖

okay 

🤖start🤖

Now you need 3 prompts of different lengths: ~200, ~2000, and ~10,000 tokens. A simple way is to repeat some text. Try:

```python
short = "Describe a luxury apartment. " * 10
medium = "Describe a luxury apartment. " * 100
long = "Describe a luxury apartment. " * 500
```

Then test each one:

```python
print(f"Short: {ttft(short):.3f}s")
print(f"Medium: {ttft(medium):.3f}s")
print(f"Long: {ttft(long):.3f}s")
```

What times do you get?

🤖end🤖

how is the prompts biggeer when we just * multiply them 


🤖start🤖

Multiplying with `*` repeats the string. So:

- `"hello " * 3` gives `"hello hello hello "`

Same idea here — `"Describe a luxury apartment. " * 100` just repeats that sentence 100 times, making a much longer prompt. The model has to **process all of it** before it can start generating a response.

That's why TTFT should increase with longer prompts — more text for the model to read through first.

Go ahead and run it — what times do you get?

🤖end🤖

Short: 30.171s
Medium: 12.318s
Long: 66.707s

🤖start🤖

Interesting results! Notice how **Long (66s)** took more than double **Medium (12s)** — that shows how TTFT grows with prompt length.

The **Short (30s)** being slower than Medium is a bit surprising though. That could be because the model was "cold starting" (loading into memory) on the first call. Try running all three again — the second run should give more consistent results since the model will already be loaded.

🤖end🤖

okay 

Short: 1.521s
Medium: 12.236s
Long: 80.852s


🤖start🤖

Much better! Now you can see the pattern clearly:

- **Short (1.5s)** → fast
- **Medium (12s)** → ~8x slower
- **Long (81s)** → ~53x slower

The TTFT grows because the model has to **read and process the entire prompt** before generating even one token. Longer prompt = more work = longer wait.

This is why in real systems you want to keep prompts concise — every extra token in your input adds latency for the user.

That completes **Exercise 1.3**! 🎉 Ready for **Step 5** — the tiny transformer?

🤖end🤖

next

🤖start🤖

**Step 5 — Tiny Transformer** 🧠

This is the fun one! You'll build a small GPT-like model that trains on the English property descriptions from your CSV and learns to generate new ones.

Before we start coding, do you know what a **transformer** is and how it relates to the LLMs you've been using (like deepseek)?

🤖end🤖

Okay lets goo

🤖start🤖

Let's start! First — do you have a rough idea of what a **transformer** is? That'll help me know where to begin explaining before we code it.

🤖end🤖

lets start from zero 

🤖start🤖

OK, from zero!

A **transformer** is just a type of neural network — a program that learns patterns from data. All modern LLMs (GPT, deepseek, Claude) are transformers.

The core idea is simple: given some text, **predict the next word**.

For example, if the model sees: *"Luxury apartment with sea..."* — it should predict **"view"** is likely next.

It learns this by reading thousands of examples and getting better at guessing. That's basically what training is — lots of guessing and correcting.

Does that make sense so far?

🤖end🤖

yes

🤖start🤖

Great! Now the key question is: **how does it decide which word comes next?**

The transformer looks at all the previous words and figures out which ones are **most important** for predicting the next one. This mechanism is called **attention**.

For example in: *"The apartment has 3 bedrooms and a beautiful sea ___"*

The word "sea" pays a lot of **attention** to "apartment", "beautiful", and "bedrooms" to figure out the next word should be "view" — not "food" or "car".

Make sense? Any questions before we start coding?

🤖end🤖

okay

🤖start🤖

Let's start coding! Create a new file `layers/layer_01/tiny_transformer.py`.

The first thing we need is to **prepare the data** — load the property descriptions and convert them into characters the model can work with.

Can you write the code to load the CSV and grab the first 1000 English descriptions? You've done something similar already in the tokenizer exercise!

🤖end🤖

import pandas as pd
df = pd.read_csv("data/egypt_real_estate_listings.csv")
english = df[~df["description"].str.contains("[\u0600-\u06FF]", na=False)]
en_sample = english.sample(100, random_state=42)

🤖start🤖

Good start! Two small tweaks:

1. You want **1000** descriptions, not 100
2. You just need the description text, not the whole dataframe

Can you fix the sample size and then extract just the `"description"` column into a list?

🤖end🤖

okay i chagned to 1000

🤖start🤖

Now extract just the description text. Try:

```python
texts = en_sample["description"].tolist()
```

Then combine them all into one big string that the model can learn from. How do you think you could join them together?

🤖end🤖

we use a for loop 

🤖start🤖

You could, but there's a simpler way! Python has a `.join()` method for strings:

```python
text = "\n".join(texts)
```

This joins all 1000 descriptions into one big string, with a newline between each. Try it, then check how long it is with `len(text)` — what do you get?

🤖end🤖

---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[8], line 6
      4 en_sample = english.sample(1000, random_state=42)
      5 texts = en_sample["description"].tolist()
----> 6 text = "\n".join(texts)

TypeError: sequence item 280: expected str instance, float found

🤖start🤖

Some descriptions are `NaN` (missing values), which pandas stores as `float`. Filter those out first:

```python
texts = en_sample["description"].dropna().tolist()
text = "\n".join(texts)
```

Try that!

🤖end🤖

```python
texts = en_sample["description"].dropna().tolist()
text = "\n".join(texts)
```

why does it learn form one string

🤖start🤖

Great question! The model learns at the **character level** — it looks at sequences of characters and learns to predict the next one. So it doesn't care about individual descriptions, it just needs a long stream of text to find patterns in.

By joining everything into one big string, we're giving it one continuous sequence like:

`...sea view\nLuxury apartment with 3 bed...`

It will learn patterns like: after "sea " comes "view", after "bed" comes "room", after "\n" comes the start of a new listing, etc.

Does that make sense? Ready for the next step — building the **vocabulary** (the set of unique characters)?

🤖end🤖

okay

🤖start🤖

Next step: build the vocabulary. You need to find all the **unique characters** in your text.

Python has a built-in for this — can you think of a way to get all unique characters from a string?

🤖end🤖

how

🤖start🤖

Try using `set()` — it gives you all unique items:

```python
chars = sorted(set(text))
print(len(chars), chars)
```

What do you get?

🤖end🤖

127 ['\t', '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '\xa0', '°', '²', '»', '¾', 'ç', 'é', '\u200b', '\u200e', '‑', '–', '—', '’', '“', '”', '•', '…', '\u202a', '\u202c', '\u202f', '\u2060', '€', '→', '▪', '▫', '◾', '✳', '⸻', '\uf0b7', 'ﬁ', '️', '\ufeff']

🤖start🤖

127 unique characters! That's your vocabulary — every character the model will learn to work with.

Now you need to convert characters to numbers (since neural networks work with numbers, not text). Create two mappings:

```python
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
```

The first converts a character to a number, the second converts back. Try it — what does `char_to_idx['a']` give you?

🤖end🤖

i didn't get what are thsews uniques


🤖start🤖

Those 127 characters are every **distinct symbol** that appears in your 1000 property descriptions. That includes:

- Letters: `a-z`, `A-Z`
- Numbers: `0-9`
- Common symbols: spaces, commas, periods, `$`, `%`, etc.
- Special characters: things like `€`, `°`, `²` (from "m²"), etc.

This is the model's **alphabet** — the only characters it will ever know about. If a character isn't in this list, the model can't read or write it.

The two mappings we just made are like a **dictionary**:
- `char_to_idx` → "a" = 66, "b" = 67, etc.
- `idx_to_char` → 66 = "a", 67 = "b", etc.

We need numbers because neural networks can only do math with numbers, not letters. Does that make sense?

🤖end🤖

okay next


🤖start🤖

Now encode the entire text into numbers:

```python
import torch
data = torch.tensor([char_to_idx[ch] for ch in text])
```

This turns your big string into a list of numbers that PyTorch can work with. What does `data.shape` give you?

🤖end🤖

why do we nee to encode it what what extacly are we doiect 

🤖start🤖

Great question! Here's what we're building, step by step:

**The goal:** Train a small neural network to write property descriptions by learning patterns from real ones.

**The problem:** Neural networks only understand numbers, not letters. So we need to convert our text into numbers first.

**What we've done so far:**
1. Got 1000 property descriptions → joined into one big string
2. Found all 127 unique characters (our alphabet)
3. Made a mapping: each character gets a number (like `'a' = 66`)
4. **Now:** converting the entire text into a sequence of numbers using that mapping

So `"sea view"` might become something like `[85, 69, 67, 2, 88, 73, 69, 89]` — same information, just in a format the network can process.

Make sense?

🤖end🤖

torch.Size([966938])

🤖start🤖

Almost a million characters! That's plenty of data for our tiny model to learn from.

Now we need to split this into **training data** and **validation data** — training to learn from, validation to check if it's actually learning. A common split is 90/10:

```python
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

Try that — what sizes do you get?

🤖end🤖

trian torch.Size([870244])

🤖start🤖

And what about `val_data.shape`? It should be around 96,694.

Do you understand why we split the data? Why not just train on all of it?

🤖end🤖

torch.Size([96694])

🤖start🤖

That's the one! So ~870k for training, ~97k for validation.

The reason we split is: if you only test on data the model already trained on, you can't tell if it actually **learned patterns** or just **memorized everything**. Validation data is text it's never seen during training — so if it does well there, it's genuinely learning.

Ready for the next step — building the actual model?

🤖end🤖

okay

🤖start🤖

Now we need a function to create **training batches** — small chunks of text the model learns from one at a time.

The idea is: pick a random spot in the text, grab a sequence of characters as **input**, and the next character as the **target** (what it should predict).

For example, with a sequence length of 8:
- Input: `"Luxury a"` → Target: `"p"` (next character is "p" from "apartment")

First, let's set a **block size** — how many characters the model looks at to make a prediction:

```python
block_size = 64
```

Can you think of how you'd grab a random chunk of `train_data` that's `block_size` long?

🤖end🤖

What should i do next

🤖start🤖

We just set `block_size = 64`. Now you need to grab a random starting position in `train_data` and slice out a chunk. 

Hint: you can use `torch.randint()` to pick a random index. What range should that random index be in, so you don't go past the end of the data?

🤖end🤖

torch.randint()?

🤖start🤖

Yes! `torch.randint(low, high, size)` gives you random integers. For example:

```python
ix = torch.randint(len(train_data) - block_size, (4,))
```

This picks 4 random starting positions. We subtract `block_size` so we don't go past the end. The `4` is our **batch size** — how many examples we train on at once.

From those positions, can you figure out how to slice out the input (`x`) and target (`y`) sequences?

🤖end🤖

i don''t get it

🤖start🤖

Think of it like a sliding window. Say your data is:

`[10, 45, 66, 72, 85, 67, 90, 12, 33, 55]`

And `ix = 3` (a random starting position), with `block_size = 4`:

- **Input (x):** `[72, 85, 67, 90]` ← 4 characters starting at position 3
- **Target (y):** `[85, 67, 90, 12]` ← the same thing shifted one step forward

So for each character in `x`, the model should predict the **next** character in `y`. It's like a guessing game — given what you've seen so far, what comes next?

Does that click?

🤖end🤖

okay

🤖start🤖

Great! Now let's put it into code. Try this:

```python
x = train_data[ix[0] : ix[0]+block_size]
y = train_data[ix[0]+1 : ix[0]+block_size+1]
```

This grabs one pair. But we want a whole batch at once. Can you think of how to do this for all 4 positions in `ix`?

🤖end🤖

how

🤖start🤖

Let's use a simple example. Say your data is:

`[10, 45, 66, 72, 85, 67, 90, 12]`

And the starting position is 2, with `block_size = 4`:

- `x = data[2:6]` → `[66, 72, 85, 67]`
- `y = data[3:7]` → `[72, 85, 67, 90]`

See? Every element in `y` is the **next character** after the corresponding element in `x`. The model's job is: given `x[0]=66`, predict `y[0]=72`. Given `x[1]=72`, predict `y[1]=85`. And so on.

Does that make sense now?

🤖end🤖

okay what now

🤖start🤖

Now wrap it into a function that creates a full batch:

```python
batch_size = 4

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x, y
```

Try running `x, y = get_batch('train')` then check `x.shape`. What do you get?

🤖end🤖

what exactly randint and stack do 

🤖start🤖

**`torch.randint(len(d) - block_size, (batch_size,))`**

Picks random numbers. Like rolling dice. If `len(d)` is 870,244 and `block_size` is 64, it picks 4 random numbers between 0 and 870,180. These are the starting positions for our text chunks.

**`torch.stack([...])`**

Takes a list of tensors and stacks them into one bigger tensor. Like stacking 4 rows on top of each other to make a table.

So if you have 4 chunks of 64 characters each, `stack` turns them into one tensor of shape `(4, 64)` — 4 rows, 64 columns.

Did you try `x.shape`? What does it show?

🤖end🤖

torch.Size([4, 64])

🤖start🤖

4 rows of 64 characters — exactly what we expected. 4 batches, each 64 characters long.

Now it's time to build the actual model! The simplest starting point is an **embedding layer** — it converts each character number into a vector (a list of numbers) that the network can learn with.

```python
import torch.nn as nn
```

Have you used `nn` before, or is this new to you?

🤖end🤖

no

🤖start🤖

`torch.nn` is PyTorch's module for building neural networks. It gives you building blocks like:

- **Embedding** — turns numbers into vectors
- **Linear** — a basic layer that does math (multiply + add)
- **Module** — a base class you use to define your own model

Think of it like LEGO pieces — you combine them to build your model. We'll use a few of these to build our tiny transformer.

Ready to define the model class?

🤖end🤖

what is a model class

🤖start🤖

A **class** in Python is a template for creating objects. A **model class** is a class that defines your neural network — what layers it has and how data flows through them.

In PyTorch, you create one by extending `nn.Module`:

```python
class TinyGPT(nn.Module):
```

This gives your model all the built-in PyTorch functionality (like training, saving, etc.) for free. You just need to define two things:

1. **`__init__`** — what layers the model has
2. **`forward`** — how data flows through those layers

Have you worked with Python classes before?

🤖end🤖

yes

🤖start🤖

Great! Then let's start building. Here's the skeleton:

```python
class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
```

`nn.Embedding(127, 64)` means: for each of our 127 characters, create a vector of 64 numbers that the model will learn.

Can you guess what the `forward` method needs to do?

🤖end🤖

whyd did we do 127 64

🤖start🤖

- **127** — that's your vocabulary size! The number of unique characters you found earlier. Each character gets its own embedding.
- **64** — the **embedding dimension**. Each character gets represented as a vector of 64 numbers. This is a choice we make — bigger means the model can learn richer patterns, but is slower to train.

Think of it like this: instead of representing "a" as just the number `66`, the model learns a richer description like `[0.23, -0.5, 0.81, ...]` (64 numbers). This lets it capture relationships between characters.

Make sense?

🤖end🤖

no i diddn't uderstand what are we starign here

🤖start🤖

OK let me back up. Here's the big picture of what we're building:

**Our goal:** Given some characters, predict the next character.

**The problem:** The model can't work with raw character numbers (like `66` for "a"). A single number doesn't carry enough information.

**The solution — Embedding:** Turn each character into a **list of 64 numbers** instead. These numbers start random, but as the model trains, it learns useful values. For example, it might learn that "a" and "e" (both vowels) have similar vectors.

So the embedding is just the **first step** — converting characters into something the model can actually learn from.

Does that help?

🤖end🤖

okay what we did exactly in this class

🤖start🤖

In the `TinyGPT` class so far, we've done just one thing:

**Created an embedding table** — a big lookup table with 127 rows (one per character) and 64 columns. When the model sees character number `66` ("a"), it looks up row 66 and gets back 64 numbers.

Right now those 64 numbers are **random**. During training, the model will adjust them so that characters that behave similarly in property descriptions end up with similar numbers.

That's all we've done so far — just set up the first building block. Next we need to add more layers and the `forward` method. Ready?

🤖end🤖

okay what does the forward method need 

🤖start🤖

The `forward` method defines **what happens when data passes through the model**. It needs to:

1. **Take in** a batch of character indices (like our `x` with shape `[4, 64]`)
2. **Look up** the embeddings for each character
3. **Produce** a prediction — for each position, what's the most likely next character?

So the simplest version would look up the embeddings, then use a **Linear layer** to convert those 64-number vectors back into 127 numbers — one score per possible next character. The highest score is the model's best guess.

Want to try writing it?

🤖end🤖


    def forward(self, x):
        for i in x:
           

🤖start🤖

Good instinct to iterate, but the nice thing about PyTorch is you don't need a loop! The embedding layer can handle the whole batch at once:

```python
emb = self.embedding(x)
```

This takes your entire `x` of shape `[4, 64]` and returns shape `[4, 64, 64]` — each character is now a vector of 64 numbers.

Try starting your forward method with that line. What do you think should come next?

🤖end🤖

we need to produce a predection

🤖start🤖

Exactly! You need to convert those 64-number vectors into 127 scores (one per character in your vocabulary). That's what `nn.Linear` does.

First, add it to `__init__`:

```python
self.head = nn.Linear(64, 127)
```

Then in `forward`, after the embedding line, apply it:

```python
logits = self.head(emb)
```

`logits` is the model's raw scores — the highest score is its best guess for the next character. Try adding both lines!

🤖end🤖

    
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding= nn.Embedding(vocab_size, 64)
        self.head = nn.Linear(64,127)

    def forward(self, x):
        emb = self.embedding(x)
        logits = self.head(emb)
            

🤖start🤖

Looks great! Just add `return logits` at the end of `forward`:

```python
    def forward(self, x):
        emb = self.embedding(x)
        logits = self.head(emb)
        return logits
```

Then create the model:

```python
model = TinyGPT(len(chars))
```

Try it — any errors?

🤖end🤖

can you explain the forward method  

🤖start🤖

Sure! Here's what happens step by step when data flows through `forward`:

**`emb = self.embedding(x)`**
- `x` is shape `[4, 64]` — 4 batches of 64 character indices
- The embedding looks up each character index in the table
- Result: `[4, 64, 64]` — each character is now a vector of 64 numbers

**`logits = self.head(emb)`**
- The Linear layer takes each 64-number vector and converts it to 127 numbers
- Result: `[4, 64, 127]` — for each of the 64 positions, we get 127 scores (one per possible next character)

**`return logits`**
- Returns those scores so we can pick the highest one as our prediction

So the shape goes: `[4, 64]` → `[4, 64, 64]` → `[4, 64, 127]`

Make sense?

🤖end🤖

i didn't get it what is batch of 64 charchter indices and how to is becoming a vector of 64 an dthen  each 64 number vector is becoming 127

🤖start🤖

Let me break it down with a tiny example. Forget about batches for now — imagine just **one sequence of 5 characters**: `"sea v"`

**Step 1 — Start:** We have character indices:
`[85, 69, 67, 2, 88]` — that's 5 numbers

**Step 2 — Embedding:** Each number gets replaced by a vector of 64 numbers from the lookup table:
- `85` → `[0.23, -0.5, 0.81, ... ]` (64 numbers)
- `69` → `[0.11, 0.34, -0.2, ... ]` (64 numbers)
- ...and so on for all 5

So 5 numbers become a **5 × 64** table.

**Step 3 — Linear:** Each row of 64 numbers gets converted to 127 numbers (one score per character in our alphabet). The highest score is the prediction for what comes next.

So for position where we have `"v"`, the model might give the highest score to character `"i"` (hoping the word is "view").

Does that part make more sense now?

🤖end🤖

why and how is the convesoin happening 

🤖start🤖

**Why?**

A single number (like `85` for "s") tells the model nothing about what "s" *means* or how it relates to other characters. But a vector of 64 numbers can encode rich information — like "this character often appears in words" or "this character usually follows vowels."

**How?**

The **embedding** is literally a table stored in memory:

| Character index | Vector |
|---|---|
| 0 (\t) | [0.12, -0.3, 0.55, ...] |
| 1 (\n) | [0.44, 0.21, -0.8, ...] |
| ... | ... |
| 85 (s) | [0.23, -0.5, 0.81, ...] |

When the model sees `85`, it just goes to **row 85** and grabs those 64 numbers. That's it — it's a lookup, like a dictionary.

The **Linear** layer works differently — it does actual math. It multiplies the 64 numbers by a **weight matrix** (64 × 127) to produce 127 output scores. This is learned during training.

Does the difference between the two steps make sense?

🤖end🤖

no 

🤖start🤖

Let me try an even simpler analogy.

Imagine you have 127 students (our characters). You want to describe each student. You could use just their **student ID number** (like `85`) — but that tells you nothing about them.

Instead, you give each student a **report card with 64 grades** — math score, reading score, sports score, etc. Now you can compare students meaningfully. Two students with similar grades are "similar."

That's what the embedding does — gives each character a richer description (64 numbers) instead of just an ID number.

Which part is still unclear — the embedding (lookup table), or the Linear layer (the math part)?

🤖end🤖

the liear par

🤖start🤖

OK! Imagine you have **3 numbers** and you want to turn them into **5 numbers**. The Linear layer does this with simple multiplication and addition.

Say your 3 numbers are `[2, 1, 3]`.

The Linear layer has a table of weights (learned during training):

```
        out1  out2  out3  out4  out5
num1:    0.5   0.3  -0.1   0.8   0.2
num2:    0.1   0.7   0.4  -0.3   0.5
num3:   -0.2   0.1   0.6   0.4   0.3
```

For `out1`: `(2×0.5) + (1×0.1) + (3×-0.2) = 1.0 + 0.1 - 0.6 = 0.5`

It does this for all 5 outputs. That's it — just multiply and add!

In our model: 64 numbers in → multiply by a 64×127 weight table → 127 numbers out. Each of those 127 numbers is a **score** for each character being the next one.

Does that make sense now?

🤖end🤖

why are we multiplying them

🤖start🤖

The multiplication is how the model **makes decisions**.

Think of it this way: those 127 output scores represent "how likely is each character to be next?" The weights control **which input numbers matter** for each prediction.

For example, if the weight connecting input #3 to output "v" is very high, that means: "when input #3 is big, 'v' is more likely to be the next character."

During training, the model **adjusts these weights** so the multiplications produce the right answers. That's what "learning" actually is — finding the right weights.

So: **multiplication = the model's way of combining information to make a prediction.**

Does that click?

🤖end🤖

okay

🤖start🤖

Now try creating the model and testing it:

```python
model = TinyGPT(len(chars))
out = model(x)
print(out.shape)
```

What shape do you get?

🤖end🤖

---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Cell In[37], line 3
      1 model = TinyGPT(len(chars))
      2 out = model(x)
----> 3 print(out.shape)

AttributeError: 'NoneType' object has no attribute 'shape'

🤖start🤖

The model returned `None` — that means you forgot the `return logits` line at the end of `forward`. Can you add that and try again?

🤖end🤖

torch.Size([4, 64, 127])

🤖start🤖

That's the shape we expected:

- **4** — batches
- **64** — positions in each sequence
- **127** — scores for each possible next character

The model is working! Now we need to **train it** — which means teaching it to give high scores to the *correct* next character.

For that we need a **loss function** — it measures how wrong the model's predictions are. Have you heard of a loss function before?

🤖end🤖

is when we compare it wiht something else

🤖start🤖

Exactly! The loss function compares the model's predictions against the **actual next character** (our `y` targets). 

If the model predicted "v" and the answer was "v" → low loss (good!). If it predicted "z" and the answer was "v" → high loss (bad!).

The goal of training is to **make the loss as small as possible**. We use a loss function called `CrossEntropyLoss` — it's the standard one for "pick one out of many" problems like ours (picking 1 character out of 127).

Want to try adding it?

🤖end🤖

how

🤖start🤖

Try this:

```python
loss_fn = nn.CrossEntropyLoss()
```

Then to compute the loss, you need to reshape the logits and targets. Give it a go:

```python
loss = loss_fn(out.view(-1, 127), y.view(-1))
print(loss.item())
```

What number do you get?

🤖end🤖

5.042339324951172

🤖start🤖

That makes sense! A random guess across 127 characters would give a loss of about `ln(127) ≈ 4.84`. You got 5.04, which is close — the model is basically guessing randomly right now.

Training will push this number **down**. The lower it goes, the better the model is at predicting the next character.

Do you know what an **optimizer** is? That's the next piece we need.

🤖end🤖

that optimizes the tianging maybe

🤖start🤖

Close! The optimizer is what **adjusts the weights** to reduce the loss. Remember those weight numbers in the Linear layer? After each batch, the optimizer nudges them slightly in the direction that makes the loss smaller.

It's like tuning a radio dial — each step gets you a little closer to the right frequency.

The most common one is called **Adam**:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

`lr=0.001` is the **learning rate** — how big each adjustment step is. Too big = overshoots, too small = takes forever. 0.001 is a good default.

Make sense?

🤖end🤖

okay when is this done after trainging

🤖start🤖

The optimizer works **during** training, not after. Here's what one training step looks like:

1. **Get a batch** of data
2. **Forward pass** — model makes predictions
3. **Compute loss** — how wrong was it?
4. **Backward pass** — figure out which direction to adjust weights
5. **Optimizer step** — actually adjust the weights

You repeat this thousands of times. Each time, the model gets a little better.

Want to write the training loop?

🤖end🤖

yes

🤖start🤖

Start with a simple `for` loop. How many steps do you think we should train for? Try:

```python
for step in range(1000):
```

What do you think should go inside the loop? Think back to those 5 steps I just listed.

🤖end🤖

for step in range(1000):
    model = TinyGPT(len(chars))
    out = model(x)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(out.view(-1,127), y.view(-1))
    print(loss.item())   
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

🤖start🤖

Good attempt! But there are a few issues:

1. **`model`, `loss_fn`, and `optimizer`** should be created **before** the loop, not inside it. Right now you're creating a brand new model every step — so it never learns!
2. You're missing the **backward pass** and **optimizer step**
3. You need a **fresh batch** each step

Try moving model/loss_fn/optimizer **above** the loop. Then inside the loop, what 5 things should happen? (Hint: look at my list from earlier)

🤖end🤖

.

🤖start🤖

Here's a hint. Before the loop:

```python
model = TinyGPT(len(chars))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

Then inside the loop, the 5 steps are:
1. Get a batch → `x, y = get_batch('train')`
2. Forward pass → `out = model(x)`
3. Compute loss → `loss = loss_fn(out.view(-1,127), y.view(-1))`
4. Backward pass → `?`
5. Optimizer step → `?`

Can you figure out steps 4 and 5?

🤖end🤖

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for step in range(1000):
    x, y = get_batch('train')
    out = model(x)
    loss = loss_fn(out.view(-1,127),y.view(-1))
    if loss < 5:
        optimizer(x)

🤖start🤖

Good thinking — but the backward pass and optimizer step aren't quite right. You don't need the `if` check. Here's what steps 4 and 5 should be:

```python
    loss.backward()
    optimizer.step()
```

- `loss.backward()` — calculates which direction to adjust the weights
- `optimizer.step()` — actually adjusts them

There's also one more line needed **before** `loss.backward()`:

```python
    optimizer.zero_grad()
```

This clears out the old adjustments from the previous step. Can you put it all together?

🤖end🤖

for step in range(1000):
    x, y = get_batch('train')
    out = model(x)
    loss = loss_fn(out.view(-1,127),y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

🤖start🤖

That's the training loop! Add a print line so you can see progress:

```python
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
```

Put that inside the loop after `optimizer.step()`, then run it. Is the loss going down?

🤖end🤖

Step 0, Loss: 2.6355
Step 100, Loss: 2.1148
Step 200, Loss: 2.7477
Step 300, Loss: 2.6460
Step 400, Loss: 2.5649
Step 500, Loss: 2.4672
Step 600, Loss: 2.8317
Step 700, Loss: 2.5169
Step 800, Loss: 2.5345
Step 900, Loss: 2.4926

🤖start🤖

The loss dropped from ~5.0 (random) to ~2.5 — the model is learning! It's not going much lower because our model is still very simple (no attention yet).

Let's see what it can generate! Try this:

```python
def generate(model, start_char, length=200):
    idx = torch.tensor([char_to_idx[start_char]])
    result = [start_char]
    for _ in range(length):
        logits = model(idx.unsqueeze(0))
        probs = torch.softmax(logits[0, -1], dim=0)
        next_idx = torch.multinomial(probs, 1).item()
        result.append(idx_to_char[next_idx])
        idx = torch.cat([idx, torch.tensor([next_idx])])
    return ''.join(result)

print(generate(model, 'A'))
```

What does it produce? 😄

🤖end🤖

Alumoviaxe Cosa  fid, ance kem,0
Fooonaro
2 pasatheane________________fis, mads t Tan sc cte besthrss We Cg ppous LLof a
PrntentIns 2, id ditotorerianges jemadin--Re: a
TTogant  yos a onvint oym ARes
N

🤖start🤖

Haha — it's mostly gibberish, but look closely! It's already learned some things:

- Starts words with capital letters (`Alumoviaxe`, `Fooonaro`, `Tan`)
- Uses commas, spaces, and newlines in roughly the right places
- Produces letter combinations that *almost* look like English words

Not bad for a tiny model with no attention! Remember, this is a **character-level** model — it's predicting one letter at a time with no real understanding of words.

Want to try training for more steps (say 5000) to see if it improves?

🤖end🤖

how do we do that and can it affect my computer

🤖start🤖

Training for more steps is easy — just change `range(1000)` to `range(5000)` and run the loop again.

As for your computer — this tiny model is very small, so it won't cause any problems. It might take a couple of minutes longer, but it won't overheat or slow down your machine noticeably. Real transformer training (like GPT-4) needs thousands of GPUs — yours is tiny by comparison! 😄

Want to give it a try?

🤖end🤖

Alu ts eles, roul iede ( 172 Tatisrome squkil achita Ratit cl yom Zaw RAve Itee w sig :
Facloresed  Lorth ico Prhake ooff  ouste malay + ndieces fifre wa dit by oces, re aifon Bin Rew cules
51,00 bixp 

🤖start🤖

It's getting better! Notice now it's producing things like:

- **Numbers that look like prices**: `51,00`
- **Real estate-ish words**: `squkil` (squarish?), `achita` (architecture?)
- **Formatting patterns**: colons, parentheses, `+` signs

It's still not writing real words, but it's clearly learning the **structure** and **patterns** of property listings.

This is about as good as it'll get without adding **attention** — the mechanism that lets the model look at which earlier characters are important for predicting the next one. That's what makes real transformers so powerful.

But for the exercise, this is great — the goal was just to get it to train and generate text. ✅

Ready for **Step 6** — writing your COMPLETION.md?

🤖end🤖

okay


🤖start🤖

Step 6 says to copy `templates/layer_completion_template.md` to `layers/layer_01/COMPLETION.md` and answer 3 gate questions in your own words.

Can you first check what's in the template? Try opening it to see what the 3 questions are.

🤖end🤖

can you ask me 

🤖start🤖

Sure! Based on what you learned in Layer 1, here are some questions — answer in your own words:

**1. What is a tokenizer, and why does tokenizer choice matter for multilingual applications?**

Take your time — think back to the tokenizer comparison exercise. What did you observe?

🤖end🤖

okay a tokenizer is where we take the letters in word and turn it to to tokens and we compared hwo old and new modles are better in tokenizing and some are better in multi languages that can effect in the consumputino of token from the model we are usieng

🤖start🤖

That's a solid answer! You've got the key points:

- Tokenizer converts text into tokens
- Different models tokenize differently
- Some handle multilingual text better
- It affects cost (token consumption)

You could also mention the specific example you saw — cl100k needed **almost double** the tokens for Arabic compared to GPT-4o. That's a concrete way to show why it matters.

**2. How does temperature affect LLM output, and when would you use low vs high temperature?**

🤖end🤖

okay low temp give consies and starigh answers while higher are more creative and random

🤖start🤖

Good! You could also add that very high temperatures can cause **hallucinations** — remember how at 1.2 the model invented "Al Mawga market" which doesn't exist?

**3. What is a transformer, and what is the basic process of training one?**

🤖end🤖

a transformer is the backbone of hwo llm work they predect the next letter that will be produced by the llm for our example inthe tiny llm we created a traing data and a val data with sepcifice special character and of same size and we trained is using torch nn which is a neaural network whre the the data will be given as embedding the forwarded and then we se the loss if there the loss is big we optimize and then test it 

🤖start🤖

That's a great answer! You've captured the key ideas:

- Transformer predicts the next token/character
- You split data into training and validation
- Data goes through embedding → forward pass → loss calculation
- If loss is high, optimize the weights to improve

You could also mention that training is a **loop** — you repeat the process thousands of times, and each time the model gets a little better at predicting.

Those are solid answers for your COMPLETION.md! Now copy the template and write them up:

```bash
cp templates/layer_completion_template.md layers/layer_01/COMPLETION.md
```

Then fill in your answers. You've completed Layer 1! 🎉

🤖end🤖