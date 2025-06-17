# 📁 Examples and CLI Walkthrough

This folder will include example scripts (Python and shell) and Jupyter notebooks demonstrating how to use the different interfaces provided by the package — from the powerful `deleanbm` CLI to the higher-level Python API (`DeLeAnBatchManager`), or even the low-level modules for full flexibility.

While examples will be added soon, below you can already find a full step-by-step guide on how to use the CLI to annotate prompts using the ADeLe framework.

---

## 🧭 Step-by-Step CLI Guide

Let's assume you're working on a project where you want to follow the ADeLe framework to evaluate and profile a large language model (LLM) for a given benchmark — or simply need demand-level annotations for your prompts. Start by navigating to your project directory:

```bash
cd path/to/your/project
```

Your project directory might look like this:

```bash
project/
├── data/
│   └── prompt_data.jsonl      # Input prompts to annotate — one per line
│                              # Each entry must include "prompt" and "custom_id" keys
│                              # You can also use CSV or Parquet with according columns.
├── experiments/
├── rubrics/
│   ├── AS.txt
│   └── ...
└── .env
```

> ℹ️ You can place the `rubrics/` folder wherever you like. It may include only the official DeLeAn v1.0 rubrics or any custom rubrics you define, following the format explained in the main README.

---

## ⚙️ Preparing the Environment

Before running any commands, make sure your environment variables are set correctly.

- For Azure OpenAI:
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_ENDPOINT`
- For OpenAI (non-Azure):
  - `OPENAI_API_KEY`

You can export them manually:

```bash
export AZURE_OPENAI_API_KEY=<your_key>
export AZURE_OPENAI_ENDPOINT=<your_endpoint>
export OPENAI_API_KEY=<your_key>
```

Or store them in a `.env` file in your project directory.
> If you do this, always run the `deleanbm` command from that directory, as it will look for the file in the current working directory.

For more flexibility you can store them in the `.env` file that comes with this package after cloning it, as the program will always fallback to that file when trying to setup the environment.

---

## 🆕 Step 1 – Setup the Run

The first step is to **initialize a new run**, which acts as a persistent reference to your configuration (data, rubrics, model, paths...). It will also prepare all internal folders automatically.

```bash
deleanbm --run-name example_run setup \
  --source-data-file ./data/prompt_data.jsonl \
  --rubrics-folder ./rubrics \
  --base-folder ./experiments/batch_runs/ \
  --annotations-folder ./data/annotations \
  --azure \
  --openai-model gpt-4o-global-batch \
  --max-completion-tokens 1000
```

> `--azure` flag tells the program to work on with Azure Environment. If this is set, your `--openai_model` must be the name of a model deployed on your Azure environment.
>
> If you are working without Azure, just omit this flag or speicify `--no-azure`. In this case your `--openai-model` is a common identifier of the model in the OpenAI API.
>
> For more information about how to setup a new run or better understand each option, run `deleanbm setup --help`

After this step, both `base-folder` and `annotations-folder` will be created if they don't already exist. Your project structure might now look like this:

```bash
project/
├── data/
│   ├── annotations/
│   └── prompt_data.jsonl
├── experiments/
│   └── batch_runs/
├── rubrics/
│   ├── AS.txt
│   └── ...
└── .env
```

---

## 📝 Step 2 – Generate Batch Input Files

Each batch job requires a `.jsonl` input file describing all requests. These are built by combining:

- The full rubric content
- The prompt to annotate
- A standardized instruction template

To automatically generate these files run:

```bash
deleanbm -r example_run create-input-files
```

By default, input files will be generated for **all rubrics** found in your `rubrics/` folder. 
> You can also create files for specific demands providing its names as arguments (without any option flag). \
> See `deleanbm create-input-files --help` for details.

This will produce one subfolder per rubric in your `base-folder`, each containing an `input.jsonl` file:

```bash
project/
├── experiments/
│   └── batch_runs/
│       ├── AS/
│       │   └── input.jsonl
│       └── ...
```

> Files will be splitted if needed to respect Batch API size and length limits.
> If this happens, you will find one subfolder for each part. This is, if AS input file was splitted into two parts, you will find both `AS_part1` and `AS_part2` subfolders with its own input files.

---

## 🚀 Step 3 – Launch Batch Jobs

Now you're ready to submit your input files to the (Azure) OpenAI Batch API:

```bash
deleanbm -r example_run launch
```

This will submit one job for each available `input.jsonl` file inside the base folder and save its metadata as a JSON file within each subfolder:

```bash
project/
├── experiments/
│   └── batch_runs/
│       ├── AS/
│       │   ├── input.jsonl
│       │   └── batch_metadata.jsonl
│       └── ...
```

> You can also launch jobs for specific demands. See `deleanbm launch --help` for details.

---

## 🔍 Step 4 – Manually Check and Download Results

Once you've launched your batch jobs, you can manually monitor their progress and download the results when they're ready. This gives you full control over the checking process.

### ✅ Check the status of your jobs

You can check the current status of all jobs associated with your run:

```bash
deleanbm -r example_run check
```

This will show the state of each batch job (e.g., `validating`, `in_progress`, `completed`, `failed`), and a general summary of all their statuses.

> You can filter by demand(s) if you only want to check a subset. See `deleanbm check --help` for more information.

For example, after executing this command you could see something like:

```bash
2025-06-16 23:45:10 [INFO] ...
2025-06-16 23:45:10 [INFO] =========================
2025-06-16 23:45:10 [INFO] Batch Job Status Summary:
2025-06-16 23:45:10 [INFO] - completed: 5 (27.78%)
2025-06-16 23:45:10 [INFO] - in_progress: 12 (66.67%)
2025-06-16 23:45:10 [INFO] - validating: 1 (5.56%)
2025-06-16 23:45:10 [INFO] Status checks complete: 18 total, 0 errors.
```

### ⬇️ Download results

Once all jobs are marked as `completed`, you can download the outputs:

```bash
deleanbm -r example_run download
```

This will download all `output.jsonl` files, save them in their respective folders, generate a `summary.txt` for each batch job and a general `sumamry.txt` at the `base-folder`.

```bash
project/
├── experiments/
│   └── batch_runs/
│       ├── AS/
│       │   ├── input.jsonl
│       │   ├── batch_metadata.jsonl
│       │   ├── output.jsonl
│       │   └── summary.txt
│       ├── ...
│       └── summary.txt
```

> Again, you could specify demand(s) providing them as arguments. See `deleanbm download --help` for more information

Note that this will only download `output.jsonl` (and `summary.txt`) files if the jobs were `completed`. You could also find one more file named as `errors.jsonl` even if the job was `completed`. This file will contain one line for each failed request within the original `input.jsonl`. This could happen if some requests contain prompts that violate their Terms of Service, or if something was wrongly specified in the request object.

> Additionaly, once a job is downloaded, the `batch_metadata.jsonl` will be updated by saving the batch object at the time of the download.

---

## 🔄 Step 4 (Alternative) – Automatic Tracking and Download

Instead of checking manually, you can automate the process with a single command that:

1. Periodically checks the status of all pending (not finalized) jobs
2. Downloads results as soon as they appear as completed
3. Generates summaries and updates the batch metadata
4. Generates the final, general summary once everything is finalized (completed or failed).

Just run:

```bash
deleanbm -r example_run track-and-download-loop
```

This command will keep running until all jobs are either completed or failed.

You can customize:

- `--check-interval` (default: 600 seconds)
- `--n-jobs` for parallelism when checking and downloading

This is ideal if you're running a large batch and want to leave it unattended while everything is handled automatically.

---

## 🧾 Step 5 – Parse the Results

Once you've downloaded the batch results, you can use the `parse-output-files` command to extract the annotated demand levels and save them in your preferred format.

Each entry is parsed using regex and structured format matching to extract the level predicted by the model. If this extraction fails (e.g., malformed response, unexpected format, incomplete answer), the annotation is marked as a **failure** and saved as NaN. This can happen even if the batch job itself or specific request was completed successfully.

This command supports a wide range of output configurations to adapt to different analysis needs. You can:

- Choose between `jsonl`, `csv` or `parquet` as output format
- Format the data in `long` mode (one row per annotation) or `wide` mode (one row per prompt)
- Filter results by status (only successful or only failed annotations)
- Filter by specific `finish_reason`

    > ℹ️ `finish_reason` values:
    >
    > - `stop`: the model ended naturally (usually good)
    > - `length`: the model was cut off due to reaching `max_completion_tokens`
    > - `other`: rare cases, e.g., safety filters or unexpected failures
    > \
    > \
    > These can affect extraction reliability. It's often useful to treat `length` failures separately to distinguish between *“model didn’t know”* and *“model ran out of room.”*

- Include or exclude additional metadata such as the full model response, finish reason, or original prompt
- Output a single file or one per demand
- Use subfolders for better organization

You can mix these options to produce exactly the view you need — whether for manual inspection, statistical analysis, loading into external systems or retrying later.

Parsed files will be saved in the annotations folder defined during setup, optionally inside a subfolder if you use the `--folder` flag.

Run `deleanbm parse-output-files --help` to see all available options.

### ✨ Example usages

#### 🔹 Basic default: general parsing without filters
Outputs all annotations in a single long-format JSONL file with model responses and finish reasons:

```bash
deleanbm -r example_run parse-output-files
```

#### 🔹 Wide-format CSV
This produces one row per prompt and one column per demand. Missing annotations (due to failed extractions) will appear as NaNs:

```bash
deleanbm -r example_run parse-output-files \
  --file-type csv --format wide
```

#### 🔹 Failed annotations (only `stop` finish reason)
This helps analyze when the model failed to give an extractable annotation, even though it finished its response before hitting the token limit:

```bash
deleanbm -r example_run parse-output-files \
  --only-failed --finish-reason stop
```

#### 🔹 Failed annotations due to token limit (`length`)
Useful to identify prompts where the model was cut off before completing the instruction. Also includes original prompts and saves results in a subfolder so they can be used to retry later with more `max_completion_tokens` setting up new dedicated runs:

```bash
deleanbm -r example_run parse-output-files \
  --only-failed --finish-reason length \
  --include-prompts --only-levels \
  --split-by-demand --folder failed_length
```

#### 🔹 Parquet output for scalable pipelines
Long-format files can become large (`n_prompts × n_rubrics` rows). Parquet offers better performance and compression:

```bash
deleanbm -r example_run parse-output-files \
  --file-type parquet --only-succeed
```
 
---

After running all the previous commands your proyect structure will look like:

```bash
project/
├── data/
│   ├── annotations/
│   │   ├── failed_length/
│   │   │   ├── example_run_annotations_AS_length_failed_only_levels_w_prompts.jsonl
│   │   │   └── ...
│   │   ├── example_run_annotations_long.jsonl
│   │   ├── example_run_annotations_wide.csv
│   │   ├── example_run_annotations_stop_failed.jsonl
│   │   └── example_run_annotations_succeed.parquet
│   └── prompt_data.jsonl
├── rubrics/
│   ├── AS.txt
│   └── ...
├── experiments/
│   └── batch_runs/
│       ├── AS/
│       │   ├── input.jsonl
│       │   ├── batch_metadata.jsonl
│       │   ├── output.jsonl
│       │   └── summary.txt
│       ├── ...
│       └── summary.txt
└── .env
```