#------------------------------------------
# Imports
#------------------------------------------

import os,sys
import json
import numpy as np
import os
import random
import subprocess
#------------------------------------------
# Read Solutions
#------------------------------------------

def install(package):
    # Add print para mostrar que se está instalando
    subprocess.check_call([sys.executable, "-m", "pip", "install","-q", package])

def save_score(score_dir, scores):
    """ Save scores as JSON file. """
    
    score_file = os.path.join(score_dir, 'scores.json')

    with open(score_file, 'w') as f_score:
        json.dump(scores, f_score, indent=4)

def print_bar():
    """ Display a bar ('----------'). """
    print('-' * 10)

def prepare_dataset_data(dataset, reference_dir, prediction_dir):
    """
    Performs steps 1 to 10:
      1. Searches for the run file for the dataset in prediction_dir.
      2. Verifies that the dataset folder exists in reference_dir.
      3. Searches for the qrels file within that folder.
      4. Searches for the "mappings" folder and loads mapping files (corpus and queries).
      5. Reads the run DataFrame.
      6. Renames columns (assuming TREC format).
      7. Remaps the q_id and doc_id fields if mappings exist.
      8. Ensures they are strings.
      9. Creates the Qrels object from the file.
      10. Creates the Run object from the filtered DataFrame.
      
    Returns (qrels, run) or (None, None) if any required file is missing or if the data is empty.
    """
    # --- Step 1: Search for the run_* file for the dataset in prediction_dir ---
    all_pred_files = os.listdir(prediction_dir)
    run_files = [
        os.path.join(prediction_dir, file)
        for file in all_pred_files
        if file.startswith(f"run_{dataset}")
    ]
    if not run_files:
        print(f"No run_{dataset} files found in '{prediction_dir}'.")
        return None, None
    run_path = run_files[0]
    print(f"Run file found: {run_path}")

    # --- Step 2: Verify that the dataset folder exists in reference_dir ---
    dataset_dir = os.path.join(reference_dir, dataset)
    if not os.path.isdir(dataset_dir):
        print(f"Folder '{dataset}' does not exist in '{reference_dir}'.")
        return None, None

    # --- Step 3: Search for the qrels file within the dataset folder ---
    ref_files = os.listdir(dataset_dir)
    qrels_files = [
        os.path.join(dataset_dir, file)
        for file in ref_files
        if file.startswith(f"qrels_{dataset}")
    ]
    if not qrels_files:
        print(f"No qrels_{dataset} files found in {dataset_dir}. This stage does not contain this information")
        return None, None
    qrels_path = qrels_files[0]
    print(f"Qrels file found: {qrels_path}")

    # --- Step 4: Search for the "mappings" folder inside the dataset folder and load mappings ---
    mappings_dir = os.path.join(dataset_dir, "mappings")
    ce_part_to_original = {}
    q_part_to_original = {}

    if os.path.isdir(mappings_dir):
        mapping_files = os.listdir(mappings_dir)
        # Look for a file starting with "corpus"
        corpus_files = [f for f in mapping_files if f.startswith("corpus")]
        if corpus_files:
            ce_elements_mapping_path = os.path.join(mappings_dir, corpus_files[0])
            ce_map = pd.read_csv(ce_elements_mapping_path, sep="\t", 
                                 names=["original_job_id", "part_job_id", "job_name"])
            ce_part_to_original = dict(zip(ce_map["part_job_id"], ce_map["original_job_id"]))
        # Look for a file starting with "queries"
        queries_files = [f for f in mapping_files if f.startswith("queries")]
        if queries_files:
            queries_mapping_path = os.path.join(mappings_dir, queries_files[0])
            q_map = pd.read_csv(queries_mapping_path, sep="\t", 
                                names=["original_job_id", "part_job_id", "job_name"])
            q_part_to_original = dict(zip(q_map["part_job_id"], q_map["original_job_id"]))



    # --- Step 5: Load the run DataFrame ---
    # Assume TREC format (it may have 5 or 6 columns)
    run_df = pd.read_csv(run_path, sep=r"\s+")
    column_names = ["q_id", "Q0", "doc_id", "rank", "score"]
    if run_df.shape[1] > 5:
        column_names.append("tag")
    run_df.columns = column_names

    # --- Step 6 & 7: Remap q_id and doc_id if mappings exist ---
    if ce_part_to_original and q_part_to_original:
      run_df["q_id"] = run_df["q_id"].map(q_part_to_original).dropna()
      run_df["doc_id"] = run_df["doc_id"].map(ce_part_to_original).dropna()

    # --- Step 8: Ensure q_id and doc_id are strings ---
    run_df["q_id"] = run_df["q_id"].astype(str)
    run_df["doc_id"] = run_df["doc_id"].astype(str)

    # --- Step 9: Create Qrels object ---
    qrels = Qrels.from_file(qrels_path, kind="trec")

    # --- Step 10: Create Run object from the DataFrame ---
    run_name = f"run_{dataset}"
    if "tag" in run_df.columns and not run_df["tag"].dropna().empty:
        run_name = run_df["tag"].iloc[0]
    run = Run.from_df(
        df=run_df,
        q_id_col="q_id",
        doc_id_col="doc_id",
        score_col="score",
        name=run_name
    )

    return qrels, run

def prepare_dataset_data_gender(dataset, reference_dir, prediction_dir, gender):
    # Get prexises
    if gender == "male":
        prefix = "m"
    elif gender == "female":
        prefix = "f"
    elif gender == "neutral":
        prefix = "n"

    # --- Step 1: Search for the run_* file for the dataset in prediction_dir ---
    all_pred_files = os.listdir(prediction_dir)
    run_files = [
        os.path.join(prediction_dir, file)
        for file in all_pred_files
        if file.startswith(f"run_{dataset}")
    ]
    if not run_files:
        print(f"No run_{dataset} files found in '{prediction_dir}'.")
        return None, None
    run_path = run_files[0]
    print(f"Run file found: {run_path}")

    # --- Step 2: Verify that the dataset folder exists in reference_dir ---
    dataset_dir = os.path.join(reference_dir, dataset)
    if not os.path.isdir(dataset_dir):
        print(f"Folder '{dataset}' does not exist in '{reference_dir}'.")
        return None, None

    # --- Step 3: Search for the qrels file within the dataset folder ---
    ref_files = os.listdir(dataset_dir)
    qrels_files = [
        os.path.join(dataset_dir, file)
        for file in ref_files
        if file.startswith(f"qrels_{dataset}")
    ]
    if not qrels_files:
        print(f"No qrels_{dataset} files found in {dataset_dir}. This stage does not contain this information")
        return None, None
    qrels_path = qrels_files[0]
    print(f"Qrels file found: {qrels_path}")

    # --- Step 4: Search for the "mappings" folder inside the dataset folder and load mappings ---
    mappings_dir = os.path.join(dataset_dir, "mappings")
    ce_part_to_original = {}
    q_part_to_original = {}

    if os.path.isdir(mappings_dir):
        mapping_files = os.listdir(mappings_dir)
        # Look for a file starting with "corpus"
        corpus_files = [f for f in mapping_files if f.startswith("corpus")]
        if corpus_files:
            ce_elements_mapping_path = os.path.join(mappings_dir, corpus_files[0])
            ce_map = pd.read_csv(ce_elements_mapping_path, sep="\t", 
                                  names=["original_job_id", "part_job_id", "job_name"])
            ce_part_to_original = dict(zip(ce_map["part_job_id"], ce_map["original_job_id"]))
        # Look for a file starting with "queries"
        queries_files = [f for f in mapping_files if f.startswith("queries")]
        if queries_files:
            queries_mapping_path = os.path.join(mappings_dir, queries_files[0])
            q_map = pd.read_csv(queries_mapping_path, sep="\t", 
                                names=["original_job_id", "part_job_id", "job_name"])
            q_part_to_original = dict(zip(q_map["part_job_id"], q_map["original_job_id"]))



    # --- Step 5: Load the run DataFrame ---
    # Assume TREC format (it may have 5 or 6 columns)
    run_df = pd.read_csv(run_path, sep=r"\s+")
    column_names = ["q_id", "Q0", "doc_id", "rank", "score"]
    if run_df.shape[1] > 5:
        column_names.append("tag")
    run_df.columns = column_names

    # --- Step 6 & 7: Remap q_id and doc_id if mappings exist ---
    if ce_part_to_original and q_part_to_original:
      run_df["q_id"] = run_df["q_id"].map(q_part_to_original).dropna()
      run_df["doc_id"] = run_df["doc_id"].map(ce_part_to_original).dropna()

    # --- Step 8: Ensure q_id and doc_id are strings ---
    run_df["q_id"] = run_df["q_id"].astype(str)
    run_df["doc_id"] = run_df["doc_id"].astype(str)

    # Filter q_id and doc_id considering only those ids startign by the prefix. 
    # If is crosslingual, only filter those elements with gender info (in docs)
    if dataset in ["en-es","en-de"]:
        run_df = run_df[run_df["doc_id"].str.startswith(prefix)]
    else:
        run_df = run_df[run_df["q_id"].str.startswith(prefix) & run_df["doc_id"].str.startswith(prefix)]
    
    # Check that there is data after filtering.
    if run_df.empty:
        print(f"[Warning] No data left after filtering by gender prefix '{prefix}'. Skipping dataset {dataset} for gender {gender}.")
        return None, None

    # Process qrels to maintain only gender specific data:
    # Assuming a standard TREC qrels file with columns: q_id, iter, doc_id, rel
    qrels_df = pd.read_csv(qrels_path, sep=r"\s+", header=None, names=["q_id", "iter", "doc_id", "rel"])
    # Ensure the q_id and doc_id are strings
    qrels_df["q_id"] = qrels_df["q_id"].astype(str)
    qrels_df["doc_id"] = qrels_df["doc_id"].astype(str)
    

    if dataset in ["en-es","en-de"]:
        qrels_df = qrels_df[qrels_df["doc_id"].str.startswith(prefix) ]
    else:
        qrels_df = qrels_df[ qrels_df["q_id"].str.startswith(prefix) & qrels_df["doc_id"].str.startswith(prefix) ]

    if qrels_df.empty:
        print(f"[Warning] No data left after filtering by gender prefix '{prefix}'. Skipping dataset {dataset} for gender {gender}.")
        return None, None

    # Step 10 - Transform to Run and Qrels. 
    run_name = f"run_{dataset}"
    if "tag" in run_df.columns and not run_df["tag"].dropna().empty:
        run_name = run_df["tag"].iloc[0]

    run = Run.from_df(
          df=run_df,
          q_id_col="q_id",
          doc_id_col="doc_id",
          score_col="score",
          name=run_name
      )
    
    qrels = Qrels.from_df(qrels_df, q_id_col="q_id", doc_id_col="doc_id", score_col="rel")

    return qrels, run
    
        
def evaluate_dataset(qrels, run, dataset, metrics):
    """
    Evaluates the given dataset and returns a dictionary with the metrics.
    """
    try:
        results = evaluate(qrels, run, metrics)
        #print(f"Metrics for {dataset}: {results}")
        return results
    except AssertionError as e:
        if "Qrels and Run query ids do not match" in str(e):
            print(f"[Warning] Qrels and Run query IDs do not match for dataset {dataset}. This might be because some doc_ids for this gender are not linked to any query_id.")
            results = evaluate(qrels, run, metrics, make_comparable=True)
            return results
        else:
            print(f"[Error] Unexpected error while evaluating dataset {dataset}: {str(e)}")
        return {}
        
def write_file(file, content):
    """ Write content to a file. """
    with open(file, 'a', encoding="utf-8") as f:
        f.write(content)


def main():
    #------------------------------------------
    # Read Args
    #------------------------------------------
    
    input_dir = '/app/input'    # Input from ingestion program
    output_dir = '/app/output/' # To write the scores
    score_file = os.path.join(output_dir, 'scores.json')
    html_file = os.path.join(output_dir, 'detailed_results.html')

    if not os.path.exists(input_dir):
        print(f"[-] Input directory not found: {input_dir}")
        return
    if not os.path.exists(output_dir):
        print(f"[-] Output directory not found: {output_dir}")
        return

    print("Input Directory:", input_dir)
    print("Contents:", os.listdir(input_dir))
    print("Output Directory:", output_dir)
    print("Contents:", os.listdir(output_dir))

    reference_dir = os.path.join(input_dir, 'ref')
    prediction_dir = os.path.join(input_dir, 'res')

    if not os.path.exists(reference_dir) or not os.path.exists(prediction_dir):
        print("[-] Reference or Prediction directory is missing.")
        return

    print("Reference Directory:", reference_dir)
    print("Contents:", os.listdir(reference_dir))
    print("Prediction Directory:", prediction_dir)
    print("Contents:", os.listdir(prediction_dir))
    # ------------------------------
    # Define constants
    # ------------------------------------
    REQUIRED_LANG_COMBS = ["es-es", "en-en", "de-de"]
    AVAILABLE_LANG_COMBS = ["es-es", "en-en", "de-de", "zh-zh", "en-es", "en-de", "en-zh"]
    AVAILABLE_LANG_COMBS_GENDER = ["es-es","de-de","en-es","en-de"]
    GENERAL_METRICS = ["map", "mrr", "precision@5", "precision@10", "precision@100", "ndcg"]
    GENDER_METRICS = ["map"]
    scores = {}
    
    # Validate that required run files exist
    existing_run_files = os.listdir(prediction_dir)
    required_files_missing = []
    for lang_comb in REQUIRED_LANG_COMBS:
        expected_file_prefix = f"run_{lang_comb}"
        if not any(file.startswith(expected_file_prefix) for file in existing_run_files):
            required_files_missing.append(expected_file_prefix)
    
    if required_files_missing:
        print("[-] Missing required run files for languages:", required_files_missing)
        return

    # Create and open html file
    write_file(html_file, '<h1>Detailed results</h1>') # Create the file to give real-time feedback

    for dataset in AVAILABLE_LANG_COMBS:
        print_bar()
        print(f"Processing dataset: {dataset}")
        qrels, run = prepare_dataset_data(dataset, reference_dir, prediction_dir)
        
        if qrels is None or run is None:
            print(f"[-] Skipping evaluation for '{dataset}'")
            if run is None:
                print("[-] Missing run files for these languages. If you have uploaded, check that you follow the naming rules.")
            continue

        # Evaluate the dataset (external function)
        results = evaluate_dataset(qrels, run, dataset,GENERAL_METRICS)

        for m in results:
            scores[f"{m}_{dataset}"] = results[m]
        
        #try:
            # Generate report to detailed results
            #report = compare(
            #    qrels=qrels,
            #    runs = [run],
            #    metrics = GENERAL_METRICS,
            #    rounding_digits = 4,
            #    make_comparable=True
            #    )
        
            #write_file(html_file, f'<h2>Results for {dataset}</h2>')
            #write_file(html_file, report.to_dataframe().to_html())
            #write_file(html_file, f'<br><br>')
        #except:
            #print(f"Report not-generated for {dataset}")
	
        try:
            # Print map in the output
            print(f"MAP value for {dataset} computed") #: {scores[f'map_{dataset}']}")
        except:
            print(f"Metrics not computed for {dataset}")
            
    for gender_dataset in AVAILABLE_LANG_COMBS_GENDER:
        for gender in ["male","female","neutral"]:
            print_bar()
            print(f"Processing dataset: {gender_dataset} for gender {gender}")
            qrels, run = prepare_dataset_data_gender(gender_dataset, reference_dir, prediction_dir, gender)
            
            if qrels is None:
                print(f"[-] Skipping gender evaluation for '{gender_dataset}'. In this phase this information is not provided.")
                continue
            if run is None:
                print(f"[-] Skipping gender evaluation for '{gender_dataset}' due to missing run files for these languages.")
                continue
            results = evaluate_dataset(qrels, run, gender_dataset,GENDER_METRICS)
            print(f"MAP value for {gender_dataset}-{gender} computed") 
            scores[f"map_{gender_dataset}_{gender}"] = results
             
    print_bar()
    print("Scoring program finished.")
    
    #------------------------------------------
    # Write Score
    #------------------------------------------
    write_file(html_file, f'The results will be announced at the end of the task evaluation period.')
    save_score(output_dir, scores)


if __name__ == '__main__':
    print("Installing ranx library... Please wait.")
    install("ranx==0.3.20")
    from ranx import Qrels, Run, evaluate, compare
    print("Ranx library is ready")

    import pandas as pd
    print("Pandas Version:", pd.__version__)

    main()

